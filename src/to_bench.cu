#include "to_bench.cuh"

#include "cuda_tools/cuda_error_checking.cuh"

#include <raft/core/device_span.hpp>

#include <rmm/device_uvector.hpp>
#include <rmm/device_scalar.hpp>

#include <cuda/atomic>

// Launch params
constexpr int WARP_SIZE = 32;
constexpr int BLOCK_SIZE = 1024;
constexpr int WPT =8;
constexpr int WPT4 =WPT/4;


// State constant for DLB
using state_type = int;
constexpr state_type X = 0;
constexpr state_type A = 1;
constexpr state_type P = 2;

// Descriptor struct
template <typename T>
struct descriptor {
    T aggregate;           // Holds an accumulated or aggregate value (of type T)
    T inclusive_prefix;    // Holds an inclusive prefix value (of type T)
    state_type status_flag; // Some sort of state indicator (likely an enum or bitflag)
};

// Warp level primitive with human readable name
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

//compute prefix via sequential lookback (only thread 0 does it). Prefix must be shared.
template <typename T>
__inline__ __device__
void sequential_lookback(T& prefix, T* sdata, int tid, int DLB_blockIdx, raft::device_span<descriptor<T>> descriptors){

    if ((tid==0)&&(DLB_blockIdx!=0)){
    
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
            
        //5. Compute and record the partition-wide inclusive prefixes.
        //Note: moi je l'ai fais avant le DLB le scan je pense que c'est pareil.
        ref_inclusive_prefix.store(prefix+sdata[WPT*BLOCK_SIZE-1]);
        ref_status_flag.store(P);
    }
    __syncthreads(); //Crucial, all threads must know the prefix value
}

//compute prefix via warp parrallel lookback (only thread 0-32 do it). Prefix must be shared.
template <typename T>
__inline__ __device__
void warp_parallel_lookback(T& prefix, T* sdata, int tid, int DLB_blockIdx, raft::device_span<descriptor<T>> descriptors){
    
    //Parallel SIMD lookback
    if ((tid<WARP_SIZE)&&(DLB_blockIdx!=0)){
        
        auto& block_descriptor = descriptors[DLB_blockIdx];
        cuda::atomic_ref<state_type, cuda::thread_scope_device> ref_status_flag(block_descriptor.status_flag);
        cuda::atomic_ref<int, cuda::thread_scope_device> ref_inclusive_prefix(block_descriptor.inclusive_prefix);

        int previous_index = DLB_blockIdx-WARP_SIZE+tid;

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
                if (tid==0) prefix+=warp_agg; //only Thread 0 gets the result
            }else{
                int lane_maxP = warpMaxIndexTrue(s==P);
                //tid=lane
                assert(lane_maxP!=-1);
                assert(lane_maxP>=0);
                assert(lane_maxP<32);
                if (s==P){
                    assert(tid<=lane_maxP);
                }
                int thread_prefix;
                if (tid == lane_maxP) {
                    thread_prefix = ref_prev_inclusive_prefix.load();
                } else if (tid > lane_maxP) {
                    thread_prefix = ref_prev_aggregate.load();
                } else {
                    thread_prefix = 0;
                }
                int warp_prefix = warpReduceSum<int>(thread_prefix);   //Thread 0 gets the result   
                if (tid==0) prefix+=warp_prefix; //only Thread 0 gets the resul
                break;
            }
            __syncwarp();   
            previous_index-=WARP_SIZE;
        }

        if (tid==0){
            ref_inclusive_prefix.store(sdata[WPT*BLOCK_SIZE-1] + prefix);
            ref_status_flag.store(P);
        }
    }
    __syncthreads(); //Crucial, all threads must know the prefix value
}

//compute block_level_scan with kogge-stone algo
template <typename T>
__inline__ __device__
void block_scan_kogge_stone(T* sdata, int tid){
    #pragma unroll
    for (int s=1; s < WPT*BLOCK_SIZE; s*=2){
        int idx[WPT];
        T tmp[WPT];

        #pragma unroll
        for (int k=0; k<WPT ; k++){
            idx[k] = tid+k*BLOCK_SIZE;

            if (idx[k] >= s)
                tmp[k] = sdata[idx[k] - s];
        }     
        __syncthreads(); //Avoid RAW
        #pragma unroll
        for (int k=0; k<WPT ; k++){
            if (idx[k] >= s)
                sdata[idx[k]]+=tmp[k];
        } 
        __syncthreads();
    }
}


void inline __device__ vectorized_load_from_gmem_to_smem(int DLB_blockIdx, raft::device_span<int> buffer, int *sdata, int size){
    
    assert(WPT%4==0); //Obligé pour la vecto

    int global_base = DLB_blockIdx * WPT * BLOCK_SIZE;
    int4* buffer_vec = reinterpret_cast<int4*>(buffer.data());
    int4* sdata_vec = reinterpret_cast<int4*>(sdata);
    int tid = threadIdx.x;
    int global_base4 = (global_base / 4);
    
    if (DLB_blockIdx==blockDim.x-1){ //Only last block needs to be careful
        #pragma unroll
        for (int k = 0; k < WPT4; k++) {
            const int i_local = tid * WPT4 + k;
            const int i_global = global_base4 + i_local;
            
            if ((i_global * 4 + 3) < size) {
                // Fully within bounds - direct vectorized load
                sdata_vec[i_local] = buffer_vec[i_global];
            } else {
                // Handle boundary case - scalar loads
                int4 temp = make_int4(0, 0, 0, 0);
                int base_idx = i_global * 4;
                if (base_idx + 0 < size) temp.x = buffer[base_idx + 0];
                if (base_idx + 1 < size) temp.y = buffer[base_idx + 1];
                if (base_idx + 2 < size) temp.z = buffer[base_idx + 2];
                if (base_idx + 3 < size) temp.w = buffer[base_idx + 3];
                sdata_vec[i_local] = temp;
            }
        }
    } else {
        int i0 = DLB_blockIdx * WPT4 * BLOCK_SIZE + threadIdx.x;
        #pragma unroll
        for (int k = 0; k < WPT4; k++) {
            const int i = i0+k*BLOCK_SIZE;
            // Fully within bounds - direct vectorized load
            sdata_vec[threadIdx.x+k*BLOCK_SIZE] = buffer_vec[i];
        }
    }
    __syncthreads(); //Post load sync
}

void inline __device__ load_from_gmem_to_smem(int DLB_blockIdx, raft::device_span<int> buffer, int *sdata, int size){
    int global_thread_base = DLB_blockIdx * WPT * BLOCK_SIZE + threadIdx.x;
    #pragma unroll
    for (int k=0; k<WPT ; k++){
         const int thread_offset = k*BLOCK_SIZE;
         const int i = global_thread_base+thread_offset;
        sdata[threadIdx.x+thread_offset] = (i < size) ? buffer[i] : 0;
    }
    __syncthreads();
}


void inline __device__ vectorized_store_from_smem_to_gmem(int DLB_blockIdx, raft::device_span<int> buffer, int *sdata, int size, int prefix){
    
    assert(WPT%4==0); //Obligé pour la vecto

    int global_base = DLB_blockIdx * WPT * BLOCK_SIZE;
    int4* buffer_vec = reinterpret_cast<int4*>(buffer.data());
    int4* sdata_vec = reinterpret_cast<int4*>(sdata);
    int tid = threadIdx.x;
    int global_base4 = (global_base / 4);

    // Write back scanned data from smem to gmem and adding prefix
    if (DLB_blockIdx==blockDim.x-1){  //Only last block needs to be careful
        #pragma unroll
        for (int k = 0; k < WPT4; k++) {
            const int i_local = tid * (WPT4) + k;
            const int i_global = global_base4 + i_local;
            
            if ((i_global * 4 + 3) < size) {
                // Fully within bounds - vectorized load, add prefix, vectorized store
                int4 temp = sdata_vec[i_local];
                temp.x += prefix;
                temp.y += prefix;
                temp.z += prefix;
                temp.w += prefix;
                buffer_vec[i_global] = temp;
            } else {
                // Handle boundary case - scalar operations
                int base_idx = i_global * 4;
                if (base_idx + 0 < size) buffer[base_idx + 0] = sdata[base_idx + 0] + prefix;
                if (base_idx + 1 < size) buffer[base_idx + 1] = sdata[base_idx + 1] + prefix;
                if (base_idx + 2 < size) buffer[base_idx + 2] = sdata[base_idx + 2] + prefix;
                if (base_idx + 3 < size) buffer[base_idx + 3] = sdata[base_idx + 3] + prefix;
            }
        }
    } else {
        #pragma unroll
        for (int k = 0; k < WPT4; k++) {
            const int i_local = tid * WPT4 + k;
            const int i_global = global_base4 + i_local;
        
            // Fully within bounds - vectorized load, add prefix, vectorized store
            int4 temp = sdata_vec[i_local];
            temp.x += prefix;
            temp.y += prefix;
            temp.z += prefix;
            temp.w += prefix;
            buffer_vec[i_global] = temp;
        }
    }
}

void inline __device__ store_from_smem_to_gmem(int DLB_blockIdx, raft::device_span<int> buffer, int *sdata, int size, int prefix){
    int global_thread_base = DLB_blockIdx * WPT * BLOCK_SIZE + threadIdx.x;
    #pragma unroll
    for (int k=0; k<WPT ; k++){
        const int thread_offset = k*BLOCK_SIZE;
        const int i = global_thread_base+thread_offset;
        if (i < size) buffer[i] = sdata[threadIdx.x+thread_offset] + prefix;
    }
}


template <typename T>
__global__
void kernel_decouple_lookback(raft::device_span<T> buffer, 
                              raft::device_span<descriptor<T>> descriptors,
                              raft::device_span<int> DLB_counter, 
                              int size)
{
    //Check that block size is a power of 2
    assert((BLOCK_SIZE & (BLOCK_SIZE - 1)) == 0); 
    assert(BLOCK_SIZE==blockDim.x);
    assert(BLOCK_SIZE>=WARP_SIZE);

    unsigned int tid = threadIdx.x;
    __shared__ unsigned int DLB_blockIdx;

    cuda::atomic_ref<int, cuda::thread_scope_device> ref_DLB_counter(DLB_counter[0]);
    //Only thread 0 of the block reads and increment the global counter
    if (tid==0){
        DLB_blockIdx=ref_DLB_counter.fetch_add(1, cuda::memory_order_relaxed);
    }
    __syncthreads();


    //Atomic references, 1 per group of WPT blocks
    auto& block_descriptor = descriptors[DLB_blockIdx];
    cuda::atomic_ref<state_type, cuda::thread_scope_device> ref_status_flag(block_descriptor.status_flag);
    cuda::atomic_ref<int, cuda::thread_scope_device> ref_inclusive_prefix(block_descriptor.inclusive_prefix);
    cuda::atomic_ref<int, cuda::thread_scope_device> ref_aggregate(block_descriptor.aggregate);

    // No one is done initially
    if (tid==0){
        ref_status_flag.store(X);
    }
    
    //Load WPT elements per thread into shared memory
    extern __shared__ T sdata[];

    //Load elems in smem
    //vectorized_load_from_gmem_to_smem(DLB_blockIdx, buffer, sdata, size);
    load_from_gmem_to_smem(DLB_blockIdx, buffer, sdata, size);
    //Debug check, smem load
    #ifdef DEBUG
    if (tid==0){
        for (int i_local=0; i_local<WPT*BLOCK_SIZE ; i_local++){
            const int i_global = global_base + i_local;
            if (i_global<size){
                assert(sdata[i_local]==1);
            }else{
                assert(sdata[i_local]==0);
            }
        }
    }
    #endif

    //3. Each processor computes and records its partition-wide aggregate to the corresponding partition descriptor.
    //Perform scan on WPT*BLOCK_SIZE elements
    
    block_scan_kogge_stone(sdata, tid);

    //Debug check, local scan
    #ifdef DEBUG
    if (tid==0){
        for (int i_local=0; i_local<WPT*BLOCK_SIZE ; i_local++){
            const int i_global = global_base + i_local;
            if (i_global<size){
                assert(sdata[i_local]==i_local+1);
            }  
        }
    }
    #endif

    __shared__ T prefix;
    //It then executes a memory fence and updates the descriptor’s status_flag to A Furthermore, the processor owning the
    //first partition copies aggregate to the inclusive_prefix field, updates status_flag to P, and skips to Step 6 below. 
    if (tid==0){
        prefix = 0;
        auto ref = ((DLB_blockIdx==0) ? ref_inclusive_prefix : ref_aggregate);
        ref.store(sdata[WPT*BLOCK_SIZE-1]);
        
        ref_status_flag.store(((DLB_blockIdx==0) ? P:A));
        ref_status_flag.notify_all();
    }
    __syncthreads(); // Sync since we touched prefix

    //compute prefix via sequential lookback (only thread 0 does it). Prefix must be shared.
    //sequential_lookback(prefix, sdata, tid, DLB_blockIdx, descriptors);
    //compute prefix via parallel lookback (only thread 0-31 do it). Prefix must be shared.
    warp_parallel_lookback(prefix, sdata, tid, DLB_blockIdx, descriptors);


    //Check debug
    #ifdef DEBUG
    if (tid==0){
        for (int i_local=0; i_local<WPT*BLOCK_SIZE ; i_local++){
            const int i_global = global_base + i_local;
             if (i_global<size){
                assert(sdata[i_local]+prefix==i_global+1);
            }  
        }
    }
    #endif

    //vectorized_store_from_smem_to_gmem(DLB_blockIdx, buffer, sdata, size, prefix);
    store_from_smem_to_gmem(DLB_blockIdx, buffer, sdata, size, prefix);

}

void DLB(rmm::device_uvector<int>& buffer)
{
    int size = buffer.size();

    assert(BLOCK_SIZE<=1024);
    assert(BLOCK_SIZE%WARP_SIZE==0);
    assert(BLOCK_SIZE>=WARP_SIZE);
    assert((BLOCK_SIZE & (BLOCK_SIZE - 1)) == 0); 

    //Number of blocks - less since each block handles WPT*BLOCK_SIZE elements
    unsigned int NBLOCKS=(size + WPT*BLOCK_SIZE - 1)/(WPT*BLOCK_SIZE);
    
    rmm::device_scalar<int> DLB_counter(0, buffer.stream());

    rmm::device_uvector<descriptor<int>> descriptors(NBLOCKS, buffer.stream());

    //Shared memory now needs WPT*BLOCK_SIZE elements
    kernel_decouple_lookback<int><<<NBLOCKS, BLOCK_SIZE, WPT*BLOCK_SIZE*sizeof(int), buffer.stream()>>>(
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
*/