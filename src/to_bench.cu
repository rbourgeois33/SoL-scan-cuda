#include "to_bench.cuh"

#include "cuda_tools/cuda_error_checking.cuh"

#include <raft/core/device_span.hpp>

#include <rmm/device_uvector.hpp>
#include <rmm/device_scalar.hpp>

#include <cuda/atomic>

// see https://research.nvidia.com/sites/default/files/pubs/2016-03_Single-pass-Parallel-Prefix/nvr-2016-002.pdf

// Launch params
constexpr int WARP_SIZE = 32; //Nvidia constant
constexpr int BLOCK_SIZE = 768; //Optimized for occupancy
constexpr int WPT = 8; //Work per thread, each thread deals with WPT elements of the array
//constexpr int WPT4 =WPT/4;
constexpr int WPT_WARPS_PER_BLOCK = WPT*BLOCK_SIZE/WARP_SIZE; //WARPS PER BLOCK AS IF the bloc was WPT*BLOCK_SIZE wide
constexpr int WARPS_PER_BLOCK = BLOCK_SIZE/WARP_SIZE;


// State constant for Decoupled Lookback (DLB)
using state_type = int;
constexpr state_type X = 0;
constexpr state_type A = 1;
constexpr state_type P = 2;

// Descriptor struct
template <typename T>
struct descriptor {
    T aggregate;            // Holds an accumulated or aggregate value (of type T)
    T inclusive_prefix;     // Holds an inclusive prefix value (of type T)
    state_type status_flag; // state indicator 
};

// Warp level primitives with human readable name
template <typename T>
__inline__ __device__
T warpReduceSum(T val) {
    #pragma unroll
    for (int offset = WARP_SIZE/2; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

__inline__ __device__
bool warpAllTrue(bool predicate){
    return __all_sync(0xffffffff, predicate);
}

__inline__ __device__
unsigned warpWhichAreTrue(bool predicate){
    return __ballot_sync(0xffffffff, predicate);
}

__inline__ __device__
int warpMaxIndexTrue(bool predicate){

    unsigned mask =  warpWhichAreTrue(predicate);
    return (mask != 0) ? (31 - __clz(mask)) : -1;
}

//Fig 1.b of https://research.nvidia.com/sites/default/files/pubs/2016-03_Single-pass-Parallel-Prefix/nvr-2016-002.pdf
template <typename T>
__device__ __forceinline__ int warp_scan_kogge_stone(T val) {
    const int lane = threadIdx.x % WARP_SIZE;
    #pragma unroll
    for (int offset = 1; offset < WARP_SIZE; offset *= 2) {
        int temp = __shfl_up_sync(0xFFFFFFFF, val, offset);
        // Use predicate instead of if
        val += (lane >= offset) ? temp : 0;
    }
    return val;
}


//compute prefix via sequential lookback (only thread 0 does it). Prefix must be shared.
//Algo 4.1.4 de https://research.nvidia.com/sites/default/files/pubs/2016-03_Single-pass-Parallel-Prefix/nvr-2016-002.pdf
template <typename T>
__inline__ __device__
void sequential_lookback(T& prefix, T* sdata, int DLB_blockIdx, raft::device_span<descriptor<T>> descriptors){

    if ((threadIdx.x==0)&&(DLB_blockIdx!=0)){
    
        auto& block_descriptor = descriptors[DLB_blockIdx];
        cuda::atomic_ref<state_type, cuda::thread_scope_device> ref_status_flag(block_descriptor.status_flag);
        cuda::atomic_ref<int, cuda::thread_scope_device> ref_inclusive_prefix(block_descriptor.inclusive_prefix);

        int previous_index = DLB_blockIdx-1;
        state_type s;

        while (true) {

            cuda::atomic_ref<state_type, cuda::thread_scope_device> ref_prev_status_flag(descriptors[previous_index].status_flag);
            cuda::atomic_ref<int, cuda::thread_scope_device> ref_prev_inclusive_prefix(descriptors[previous_index].inclusive_prefix);
            cuda::atomic_ref<int, cuda::thread_scope_device> ref_prev_aggregate(descriptors[previous_index].aggregate);

            ref_prev_status_flag.wait(X);
                
            s = ref_prev_status_flag.load();
            
            assert(s!=X);
            assert((s==A)||(s==P));

            if (s == P) {
                prefix += ref_prev_inclusive_prefix.load();
                break;
            }else{
                prefix += ref_prev_aggregate.load();
            }

            previous_index--;
        }

        assert(s==P);
        assert(previous_index>=0);
            
        ref_inclusive_prefix.store(prefix+sdata[WPT_WARPS_PER_BLOCK-1]);
        ref_status_flag.store(P);
    }
    __syncthreads(); //Crucial, all threads must know the prefix value
}

//compute prefix via warp parrallel lookback (only thread 0-32 do it). Prefix must be shared.
//Algo 4.1.4 de https://research.nvidia.com/sites/default/files/pubs/2016-03_Single-pass-Parallel-Prefix/nvr-2016-002.pdf
//With parallel optimisation (section 4.4)
template <typename T>
__inline__ __device__
void warp_parallel_lookback(T& prefix, T* sdata, int DLB_blockIdx, raft::device_span<descriptor<T>> descriptors){
    
    //Parallel SIMD lookback
    if ((threadIdx.x<WARP_SIZE)&&(DLB_blockIdx!=0)){
        
        auto& block_descriptor = descriptors[DLB_blockIdx];
        cuda::atomic_ref<state_type, cuda::thread_scope_device> ref_status_flag(block_descriptor.status_flag);
        cuda::atomic_ref<int, cuda::thread_scope_device> ref_inclusive_prefix(block_descriptor.inclusive_prefix);

        int previous_index = DLB_blockIdx-WARP_SIZE+threadIdx.x;

        state_type s;

        while ((true)) {

            int avoid_OOB = max(0, previous_index);
            cuda::atomic_ref<state_type, cuda::thread_scope_device> ref_prev_status_flag(descriptors[avoid_OOB].status_flag);
            cuda::atomic_ref<int, cuda::thread_scope_device> ref_prev_inclusive_prefix(descriptors[avoid_OOB].inclusive_prefix);
            cuda::atomic_ref<int, cuda::thread_scope_device> ref_prev_aggregate(descriptors[avoid_OOB].aggregate);

            ref_prev_status_flag.wait(X);
            
            s = ((previous_index<0) ? A:ref_prev_status_flag.load());

            bool AllA = warpAllTrue((s==A));
            
            assert(s!=X);

            if (AllA){
                assert(s==A);
                assert(previous_index>0);//block 0 is never A
                int thread_agg=ref_prev_aggregate.load();
                int warp_agg = warpReduceSum<int>(thread_agg);   //Thread 0 gets the result   
                if (threadIdx.x==0) prefix+=warp_agg; //only Thread 0 gets the result
            }else{
                int lane_maxP = warpMaxIndexTrue(s==P);
                //threadIdx.x=lane
                assert(lane_maxP!=-1);
                assert(lane_maxP>=0);
                assert(lane_maxP<32);
                if (s==P){
                    assert(threadIdx.x<=lane_maxP);
                }
                int thread_prefix;
                if (threadIdx.x == lane_maxP) {
                    thread_prefix = ref_prev_inclusive_prefix.load();
                } else if (threadIdx.x > lane_maxP) {
                    thread_prefix = ref_prev_aggregate.load();
                } else {
                    thread_prefix = 0;
                }
                int warp_prefix = warpReduceSum<int>(thread_prefix);   //Thread 0 gets the result   
                if (threadIdx.x==0) prefix+=warp_prefix; //only Thread 0 gets the resul
                break;
            }
            __syncwarp();   
            previous_index-=WARP_SIZE;
        }

        if (threadIdx.x==0){
            ref_inclusive_prefix.store(sdata[WPT_WARPS_PER_BLOCK-1] + prefix);
            ref_status_flag.store(P);
        }
    }
    __syncthreads(); //Crucial, all threads must know the prefix value
}

// void inline __device__ vectorized_load_from_gmem_to_smem(int DLB_blockIdx, raft::device_span<int> buffer, int *sdata, int size){
    
//     assert(WPT%4==0); //Obligé pour la vecto
//     int global_thread_base = DLB_blockIdx * WPT4 * BLOCK_SIZE + threadIdx.x;// No stride between threads !!!

//     int4* buffer_vec = reinterpret_cast<int4*>(buffer.data());
//     int4* sdata_vec = reinterpret_cast<int4*>(sdata);
    
//     if (DLB_blockIdx==blockDim.x-1){ //Only last block needs to be careful
//         #pragma unroll
//         for (int k = 0; k < WPT4; k++) {
//             const int thread_offset = k*BLOCK_SIZE;
//             const int i_global = global_thread_base + thread_offset;
//             const int i_local = threadIdx.x + thread_offset;
            
//             if ((i_global * 4 + 3) < size) {
//                 // Fully within bounds - direct vectorized load
//                 sdata_vec[i_local] = buffer_vec[i_global];
//             } else {
//                 // Handle boundary case - scalar loads
//                 int4 temp = make_int4(0, 0, 0, 0);
//                 int base_idx = i_global * 4;
//                 if (base_idx + 0 < size) temp.x = buffer.data()[base_idx + 0];
//                 if (base_idx + 1 < size) temp.y = buffer.data()[base_idx + 1];
//                 if (base_idx + 2 < size) temp.z = buffer.data()[base_idx + 2];
//                 if (base_idx + 3 < size) temp.w = buffer.data()[base_idx + 3];
//                 sdata_vec[i_local] = temp;
//             }
//         }
//     } else {// Fully within bounds - direct vectorized load
//         #pragma unroll
//         for (int k = 0; k < WPT4; k++) {
//             const int thread_offset = k*BLOCK_SIZE;
//             const int i_global = global_thread_base + thread_offset;
//             const int i_local = threadIdx.x + thread_offset;

//             sdata_vec[i_local] = buffer_vec[i_global];
//         }
//     }
//     __syncthreads(); //Post load sync
// }

// void inline __device__ vectorized_store_from_smem_to_gmem(int DLB_blockIdx, raft::device_span<int> buffer, int* sdata, int size, int prefix){
    
//     assert(WPT%4==0); //Obligé pour la vecto
//     int global_thread_base = DLB_blockIdx * WPT4 * BLOCK_SIZE + threadIdx.x; // No stride between threads !!!

//     int4* buffer_vec = reinterpret_cast<int4*>(buffer.data());
//     int4* sdata_vec = reinterpret_cast<int4*>(sdata);

//     // Write back scanned data from smem to gmem and adding prefix
//     if (DLB_blockIdx==blockDim.x-1){  //Only last block needs to be careful
//         #pragma unroll
//         for (int k = 0; k < WPT4; k++) {
//             const int thread_offset = k*BLOCK_SIZE;
//             const int i_global = global_thread_base + thread_offset;
//             const int i_local = threadIdx.x + thread_offset;
            
//             if ((i_global * 4 + 3) < size) {
//                 // Fully within bounds - vectorized load, add prefix, vectorized store             
//                 int4 temp = sdata_vec[i_local];
//                 temp.x += prefix;
//                 temp.y += prefix;
//                 temp.z += prefix;
//                 temp.w += prefix;
//                 buffer_vec[i_global] = temp;
//             } else {
//                 // Handle boundary case - scalar operations
//                 int base_idx = i_global * 4;
//                 if (base_idx + 0 < size) buffer[base_idx + 0] = sdata[base_idx + 0] + prefix;
//                 if (base_idx + 1 < size) buffer[base_idx + 1] = sdata[base_idx + 1] + prefix;
//                 if (base_idx + 2 < size) buffer[base_idx + 2] = sdata[base_idx + 2] + prefix;
//                 if (base_idx + 3 < size) buffer[base_idx + 3] = sdata[base_idx + 3] + prefix;
//             }
//         }
//     } 
//     else {
//         #pragma unroll
//         for (int k = 0; k < WPT4; k++) {
//             const int thread_offset = k*BLOCK_SIZE;
//             const int i_global = global_thread_base + thread_offset;
//             const int i_local = threadIdx.x + thread_offset;

//             int4 temp = sdata_vec[i_local];
//             temp.x += prefix;
//             temp.y += prefix;
//             temp.z += prefix;
//             temp.w += prefix;
//             buffer_vec[i_global] = temp;
//         }
//     }
// }

//Load WPT elements per threads into registers in a coalescing way
void inline __device__ load_from_gmem_to_registers(int DLB_blockIdx, raft::device_span<int> buffer, int* thread_value, int size){
    int global_thread_base = DLB_blockIdx * WPT * BLOCK_SIZE + threadIdx.x; // No stride between threads !!!
    #pragma unroll
    for (int k=0; k<WPT ; k++){
        const int thread_offset = k*BLOCK_SIZE;
        const int i_global = global_thread_base + thread_offset;
        thread_value[k] =(i_global < size) ? buffer[i_global] : 0;
    }
    __syncthreads();
}


//Store back to gmem in the same way
void inline __device__ store_from_registers_to_gmem(int DLB_blockIdx, raft::device_span<int> buffer, int* thread_value, int size, int prefix){
    int global_thread_base = DLB_blockIdx * WPT * BLOCK_SIZE + threadIdx.x; // No stride between threads !!!
    #pragma unroll
    for (int k=0; k<WPT ; k++){
        const int thread_offset = k*BLOCK_SIZE;
        const int i = global_thread_base+thread_offset;
        if (i < size) buffer[i] = thread_value[k]+ prefix;
    }
}

//Algo from fig2 (a) of https://research.nvidia.com/sites/default/files/pubs/2016-03_Single-pass-Parallel-Prefix/nvr-2016-002.pdf
__inline__ __device__
void block_scan_kogge_stone(int* thread_value, int* sdata, /*for debug only*/ int global_thread_base, int size, int DLB_blockIdx){

    //Indexes
    int thread_index_within_warp = threadIdx.x & 31;        // Same as threadIdx.x % 32 (faster with bitwise AND)
    int warp_id = threadIdx.x >> 5;                        // Same as threadIdx.x / 32 (faster with bit shift)


    //(top of the drawing)
    //First, each warp scans it's WPT*WARP_SIZE elements with warp primitives 
    //Perform warp-scan, values in thread_value are now scanned 
    #pragma unroll
    for (int k=0; k<WPT ; k++){
        thread_value[k]=warp_scan_kogge_stone(thread_value[k]);
    }

    //Debug check warp-scan
    #ifndef NDEBUG  
    for (int k=0; k<WPT ; k++){
        const int thread_offset = k*BLOCK_SIZE;
        const int i_global = global_thread_base + thread_offset;
        if (i_global<size){
            assert(thread_value[k]==thread_index_within_warp+1);
        }
    }
    #endif  


    //(middle of the drawing)
    //The block now needs to scan WPT*BLOCK_SIZE/WARP_SIZE = WPT*WARP_PER_BLOCK = WPT_WARP_PER_BLOCK values
    //We index these values with warp_idx, starting with the current warp id, strided by WARPS_PER_BLOCK.
    int warp_idx[WPT];
    #pragma unroll
    for (int k=0; k<WPT ; k++){
        warp_idx[k]= warp_id + k * WARPS_PER_BLOCK;
    }

    //Store the aggregate of each warp into shared memory to be scanned lower. 
    //Done by one thread per warp, the last because it is the one who holds the aggregate in it's thread_value
    if (thread_index_within_warp==WARP_SIZE-1){ 
        #pragma unroll
        for (int k=0; k<WPT ; k++){
            sdata[warp_idx[k]]=thread_value[k];
        }
    }
    __syncthreads(); // sync since we touch shared memory

    //Debug check store warp aggregate in sdata
    #ifndef NDEBUG 
    int base_block =DLB_blockIdx*WPT*BLOCK_SIZE; 
    if (threadIdx.x==0){
        for(int warp=0; warp<WPT_WARPS_PER_BLOCK; warp++){
            //Check the bound of the warp window
            int begin = min(base_block+ warp    *WARP_SIZE  , size-1);
            int end   = min(base_block+(warp+1) *WARP_SIZE-1, size-1);

            //expected result
            int expected_result = end-begin+1;
            //Inf the window is out of scope, 0
            if (begin==size-1){
                expected_result=0;
            }
            assert(sdata[warp]==expected_result);
        }
    }
    #endif

    //Scan the values in shared memory
    //Only 32 threads work. We choose to make the last thread of each wapr work
    //[TODO // more ! ]
    #pragma unroll
    for (int s=1; s < WPT_WARPS_PER_BLOCK; s*=2){
        int tmp[WPT];
        #pragma unroll

        for (int k=0; k<WPT ; k++){
            if (thread_index_within_warp==WARP_SIZE-1) tmp[k] = ((warp_idx[k] >= s) ? sdata[warp_idx[k] - s] : 0);
        }
        __syncthreads(); //Avoid RAW
                
        #pragma unroll
        for (int k=0; k<WPT ; k++){
            if (thread_index_within_warp==WARP_SIZE-1) sdata[warp_idx[k]] += tmp[k];
        }
        __syncthreads();
    }
    
    #ifndef NDEBUG  
    //Debug scan shared memory
    if (threadIdx.x==0){
        for(int warp=0; warp<WPT_WARPS_PER_BLOCK; warp++){
            //Check the bound of the warp window
            int begin = min(base_block+ warp    *WARP_SIZE  , size-1);

            //In the window is out of scope, DONT CHECK
            if (begin<size-1){
                 assert(sdata[warp]==(warp+1)*WARP_SIZE);
            }
        }
    }
    #endif

    //(low part of the drawing)
    //Propagate aggregates from shared memory into registers
    for (int k=0; k<WPT ; k++){
        if (warp_idx[k]>0) thread_value[k] += sdata[warp_idx[k]-1];
    }

    #ifndef NDEBUG  
    //Debug check on Propagate aggregates from shared memory into registers
    //at this point, threads should hold block-wide scanned data
    for (int k=0; k<WPT ; k++){
        const int thread_offset = k*BLOCK_SIZE;
        const int i_global = global_thread_base + thread_offset;
        if (i_global<size){
            assert(thread_value[k]==threadIdx.x+thread_offset+1);
        }  
    }
    #endif
}

template <typename T>
__global__
void kernel_decoupled_lookback(raft::device_span<T> buffer, 
                              raft::device_span<descriptor<T>> descriptors,
                              raft::device_span<int> DLB_counter, 
                              int size)
{
    assert(BLOCK_SIZE==blockDim.x);

    __shared__ unsigned int DLB_blockIdx;

    //Compute blockIdx
    cuda::atomic_ref<int, cuda::thread_scope_device> ref_DLB_counter(DLB_counter[0]);
    //Only thread 0 of the block reads and increment the global counter
    if (threadIdx.x==0){
        DLB_blockIdx=ref_DLB_counter.fetch_add(1, cuda::memory_order_relaxed);
    }
    __syncthreads();//Sync since DLB is shared

    //Atomic references, 1 per group of WPT blocks
    auto& block_descriptor = descriptors[DLB_blockIdx];
    cuda::atomic_ref<state_type, cuda::thread_scope_device> ref_status_flag(block_descriptor.status_flag);
    cuda::atomic_ref<int, cuda::thread_scope_device> ref_inclusive_prefix(block_descriptor.inclusive_prefix);
    cuda::atomic_ref<int, cuda::thread_scope_device> ref_aggregate(block_descriptor.aggregate);

    // No one is done initially
    if (threadIdx.x==0){
        ref_status_flag.store(X);
    }
    
    //WPT * WARP_PER_BLOCK sdata allocation, will be filled and scanned according to fig.2(a)
    extern __shared__ T sdata[];
    //Load elems in smem
    T thread_value[WPT];

    load_from_gmem_to_registers(DLB_blockIdx, buffer, thread_value, size);

    // //Debug check on data loaded into registers
    int global_thread_base = DLB_blockIdx * WPT * BLOCK_SIZE + threadIdx.x;
#ifndef NDEBUG  
    for (int k=0; k<WPT ; k++){
        const int thread_offset = k*BLOCK_SIZE;
        const int i_global = global_thread_base + thread_offset;
        if (i_global<size){
            assert(thread_value[k]==1);
        }
        else{
            assert(thread_value[k]==0);
        }
    }
#endif


    //Scan the values in thread_values using sdata (fig 2.a)
    block_scan_kogge_stone(thread_value, sdata, /*for debug only*/ global_thread_base, size, DLB_blockIdx);

    __shared__ T prefix;
    //Update the descriptor’s status_flag to A (or P if first block)
    //And storing aggregate (prefix if first block)
    //Chaque bloc a fait son scan, le dernoer thread du block connais le prefix, c'est son dernier thread_value.
    if (threadIdx.x==BLOCK_SIZE-1){  
        prefix = 0;
        auto ref = ((DLB_blockIdx==0) ? ref_inclusive_prefix : ref_aggregate);
        ref.store(thread_value[WPT-1]);
        ref_status_flag.store(((DLB_blockIdx==0) ? P:A));
        ref_status_flag.notify_all();
    }
    __syncthreads(); // Sync since we touched prefix

    //compute prefix via parallel lookback (only thread 0-31 do it). Prefix must be shared.
    warp_parallel_lookback(prefix, sdata, DLB_blockIdx, descriptors);

#ifndef NDEBUG
    //Debug check on lookback
    //at this point, threads should hold buffer-wide scanned data
    for (int k=0; k<WPT ; k++){
        const int thread_offset = k*BLOCK_SIZE;
        const int i_global = global_thread_base + thread_offset;
        if (i_global<size){
            assert(thread_value[k]+prefix==i_global+1);
        }  
    }
#endif

    //vectorized_store_from_smem_to_gmem(DLB_blockIdx, buffer, sdata, size, prefix); //does not apply anymore
    store_from_registers_to_gmem(DLB_blockIdx, buffer, thread_value, size, prefix);
}

void DLB(rmm::device_uvector<int>& buffer)
{
    int size = buffer.size();

    assert(BLOCK_SIZE<=1024);
    assert(BLOCK_SIZE%WARP_SIZE==0);
    assert(BLOCK_SIZE>=WARP_SIZE);
   //assert((BLOCK_SIZE & (BLOCK_SIZE - 1)) == 0); 

    //Number of blocks - less since each block handles WPT*BLOCK_SIZE elements
    unsigned int NBLOCKS=(size + WPT*BLOCK_SIZE - 1)/(WPT*BLOCK_SIZE);
    
    rmm::device_scalar<int> DLB_counter(0, buffer.stream());

    rmm::device_uvector<descriptor<int>> descriptors(NBLOCKS, buffer.stream());

    //Shared memory now needs WPT*BLOCK_SIZE elements
    kernel_decoupled_lookback<int><<<NBLOCKS, BLOCK_SIZE, WPT_WARPS_PER_BLOCK*sizeof(int), buffer.stream()>>>(
        raft::device_span<int>(buffer.data(), buffer.size()),
        raft::device_span<descriptor<int>>(descriptors.data(), descriptors.size()),
        raft::device_span<int>(DLB_counter.data(), 1),
        size);

    CUDA_CHECK_ERROR(cudaStreamSynchronize(buffer.stream()));
}

/*
Leaderboard on a 1024^3 buffer(SOL=980 GB/s)
Base DLB: 40.6 Warp stall: barrières. Il faut des "plus gros blocks"
More WPT(=8): 185.4 (x4.6) Warp stall: barrières. On peut pas faire + gros bloc, mais on peut enlever de la latence en faisant // lookbacl
Unroll the block scan: 188(+1%)
parallel lookback 234 (x1.24) Warp stall:: MIO throttle: trop de I/O dans la shared memory. Faut un scan moin shared intensive
better scann attempt 1 249 (+6%) encore meme pb. Tentative de scan dans les registres. mais nuc montre que c'est mieux qd meme
better occupancy, 272 (+9%) merci ncu qui ma dit que block size 768 serait mieux !!
*/