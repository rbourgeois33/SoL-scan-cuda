#include "to_bench.cuh"

#include "cuda_tools/cuda_error_checking.cuh"

#include <raft/core/device_span.hpp>

#include <rmm/device_uvector.hpp>
#include <rmm/device_scalar.hpp>

#include <cuda/atomic>

// Launch params
constexpr int WARP_SIZE = 32;
constexpr int BLOCK_SIZE = 1024;
constexpr int WPT = 8;

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
    return (mask != 0) ? (31 - __clz(mask)) : 0;
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

        int previous_index = DLB_blockIdx-tid-1;

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
                int my_prefix=((previous_index<0) ? 0:ref_prev_aggregate.load());
                int loc_prefix = warpReduceSum<int>(my_prefix);   //Thread 0 gets the result   
                if (tid==0) my_prefix+=loc_prefix; //only Thread 0 gets the result
            }else{
                int lane_maxP = warpMaxIndexTrue(s==P);
                int my_prefix;
                if (previous_index<lane_maxP){
                    my_prefix=0;
                }else{
                    my_prefix=(tid==lane_maxP) ? ref_prev_inclusive_prefix.load():ref_prev_aggregate.load();
                }
                int loc_prefix = warpReduceSum<int>(my_prefix);   //Thread 0 gets the result   
                if (tid==0) my_prefix+=loc_prefix; //only Thread 0 gets the resul
                break;
            }
            __syncwarp();   
            previous_index-=WARP_SIZE;

            __syncwarp();
        }

        if (tid==0){
            ref_inclusive_prefix.store(sdata[WPT*BLOCK_SIZE-1] + prefix);
            ref_status_flag.store(P);
        }
    }
    __syncthreads(); //Crucial, all threads must know the prefix value
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

    extern __shared__ T sdata[];
    unsigned int tid = threadIdx.x;
    __shared__ unsigned int DLB_blockIdx;

    cuda::atomic_ref<int, cuda::thread_scope_device> ref_DLB_counter(DLB_counter[0]);
    //Only thread 0 of the block reads and increment the global counter
    if (tid==0){
        DLB_blockIdx=ref_DLB_counter.fetch_add(1, cuda::memory_order_relaxed);
    }
    __syncthreads();
    
    //global indices
    unsigned int i[WPT];
    i[0] = DLB_blockIdx * WPT * BLOCK_SIZE + threadIdx.x;
    #pragma unroll
    for (int k=1; k<WPT ; k++){
        i[k]=i[k-1]+ BLOCK_SIZE;
    }

    //Atomic references, 1 per group of WPT blocks
    auto& block_descriptor = descriptors[DLB_blockIdx];
    cuda::atomic_ref<state_type, cuda::thread_scope_device> ref_status_flag(block_descriptor.status_flag);
    cuda::atomic_ref<int, cuda::thread_scope_device> ref_inclusive_prefix(block_descriptor.inclusive_prefix);
    cuda::atomic_ref<int, cuda::thread_scope_device> ref_aggregate(block_descriptor.aggregate);

    // No one is done initially
    if (tid==0){
        ref_status_flag.store(X);
    }
    //2. Synchronize. All processors synchronize to ensure a consistent view of initialized partition descriptors.
    
    //Load WPT elements per thread into shared memory
    T val[WPT];
    #pragma unroll
    for (int k=0; k<WPT ; k++){
        val[k]=(i[k] < size) ? buffer[i[k]] : 0;
    }

    //Load elems in smem
    #pragma unroll
    for (int k=0; k<WPT ; k++){
        sdata[tid +k*BLOCK_SIZE] = val[k];
    }
    __syncthreads();

    //Perform scan on WPT*BLOCK_SIZE elements
    //3. Each processor computes 
    #pragma unroll
    for (int s=1; s < WPT*BLOCK_SIZE; s*=2){
        int idx[WPT];
        T toto[WPT];

        #pragma unroll
        for (int k=0; k<WPT ; k++){
            idx[k] = tid+k*BLOCK_SIZE;
            toto[k] = (idx[k] >= s) ? sdata[idx[k] - s] : 0;
        }     
        __syncthreads(); //Avoid RAW
        
        for (int k=0; k<WPT ; k++){
            sdata[idx[k]]+=toto[k];
        } 
        __syncthreads();
    }
    
    __shared__ T prefix;
    if (tid==0){
        prefix = 0;
        auto ref = ((DLB_blockIdx==0) ? ref_inclusive_prefix : ref_aggregate);
        ref.store(sdata[WPT*BLOCK_SIZE-1]); //and records its partition-wide aggregate to the corresponding partition descriptor.
         //It then executes a memory fence and updates the descriptor’s status_flag to A Furthermore, the processor owning the
         //first partition copies aggregate to the inclusive_prefix field, updates status_flag to P, and skips to Step 6 below. 
        ref_status_flag.store(((DLB_blockIdx==0) ? P:A));
        ref_status_flag.notify_all();
    }
    __syncthreads();

    //compute prefix via sequential lookback (only thread 0 does it). Prefix must be shared.
    sequential_lookback(prefix, sdata, tid, DLB_blockIdx, descriptors);
    //compute prefix via parallel lookback (only thread 0-31 do it). Prefix must be shared.
    //warp_parallel_lookback(prefix, sdata, tid, DLB_blockIdx, descriptors);


    //6. Perform a partition-wide scan seeded with the partition’s exclusive prefix
    #pragma unroll
    for (int k=0; k<WPT ; k++){
        if (i[k] < size) buffer[i[k]] = sdata[tid+k*BLOCK_SIZE] + prefix;
    }
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
Base DLB: 40.6 
More WPT(=8): 185.4 (x4.6)
Unroll the block scan: 188(+1%)
*/