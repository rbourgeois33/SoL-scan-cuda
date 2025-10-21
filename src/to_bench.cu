#include "to_bench.cuh"

#include "cuda_tools/cuda_error_checking.cuh"

#include <raft/core/device_span.hpp>

#include <rmm/device_uvector.hpp>
#include <rmm/device_scalar.hpp>

#include <cuda/atomic>


constexpr int WARP_SIZE = 32;
constexpr int BLOCK_SIZE = 1024;
constexpr int WPT = 8;

using state_type = int;
constexpr state_type X = 0;
constexpr state_type A = 1;
constexpr state_type P = 2;

__global__ void kernel_print(raft::device_span<int> buffer, int size)
{
    unsigned int tid = blockIdx.x*blockDim.x+threadIdx.x;
    if (tid > size) return;
    printf("i = %u, value = %d\n", tid, static_cast<int>(buffer[tid]));
}

template <typename T>
__global__
void kernel_scan_baseline(raft::device_span<T> buffer)
{
    for (int i = 1; i < buffer.size(); ++i)
        buffer[i] += buffer[i - 1];
}

void baseline_scan(rmm::device_uvector<int>& buffer)
{
	kernel_scan_baseline<int><<<1, 1, 0, buffer.stream()>>>(
        raft::device_span<int>(buffer.data(), buffer.size()));

    CUDA_CHECK_ERROR(cudaStreamSynchronize(buffer.stream()));
}

template <typename T>
__global__
void kernel_kogge_stone(raft::device_span<T> buffer, raft::device_span<T> sum_per_block, int size)
{
    //Check that block size is a power of 2
    //Notre dessin marche que si c'est le cas
    assert((blockDim.x & (blockDim.x - 1)) == 0); 
    assert(blockDim.x>=WARP_SIZE);

    extern __shared__ int sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*blockDim.x+threadIdx.x;

    //Check if tid is out of bound. If it is, fill with 0's to not change resut.
    //we use the input size and not buffer.size() as our intermediate buffer is too large
    sdata[tid] = (i < size) ? buffer[i]:0;

    __syncthreads();

    for (int s=1; s<blockDim.x; s*=2){
        int toto = (tid >= s) ? sdata[tid-s]:0;
        __syncthreads();
        sdata[tid]+= toto;
        __syncthreads();
    }
    
    if (i < size) buffer[i] = sdata[tid];

    if (tid==blockDim.x-1) sum_per_block[blockIdx.x] = sdata[tid];
}

template <typename T>
__global__
void kernel_propagate(raft::device_span<T> buffer, raft::device_span<T> sum_per_block, int size)
{
    //Check that block size is a power of 2
    //Notre dessin marche que si c'est le cas
    assert((blockDim.x & (blockDim.x - 1)) == 0); 
    assert(blockDim.x>=WARP_SIZE);

    unsigned int i = blockIdx.x*blockDim.x+threadIdx.x;

    if ((blockIdx.x>0)&&(i<size)) buffer[i] += sum_per_block[blockIdx.x-1];
    //printf("propagate i = %u, block= %u ,  blockvalue= %u value = %d\n", i, blockIdx.x, sum_per_block[blockIdx.x-1],static_cast<int>(buffer[i]));

}

void kogge_stone(rmm::device_uvector<int>& buffer)
{

    int size = buffer.size();


    assert(BLOCK_SIZE<=1024);
    assert(BLOCK_SIZE%WARP_SIZE==0);
    assert(BLOCK_SIZE>=WARP_SIZE);
    assert((BLOCK_SIZE & (BLOCK_SIZE - 1)) == 0); 

    //Number of blocks to touch the whole array
    unsigned int NBLOCKS=(size+BLOCK_SIZE-1)/BLOCK_SIZE;

    assert(NBLOCKS<=1024); //For now

    rmm::device_uvector<int> sum_per_block(NBLOCKS, buffer.stream());
    rmm::device_uvector<int> sum_per_block_out(NBLOCKS, buffer.stream());

	kernel_kogge_stone<int><<<NBLOCKS, BLOCK_SIZE, BLOCK_SIZE*sizeof(int), buffer.stream()>>>(
        raft::device_span<int>(buffer.data(), buffer.size()),
        raft::device_span<int>(sum_per_block.data(), sum_per_block.size()),
        size);

    //kernel_print<<<1, BLOCK_SIZE>>>(raft::device_span<int>(buffer.data(), buffer.size()), size);

    kernel_kogge_stone<int><<<1, BLOCK_SIZE, BLOCK_SIZE*sizeof(int), buffer.stream()>>>(
        raft::device_span<int>(sum_per_block.data(), sum_per_block.size()),
        raft::device_span<int>(sum_per_block_out.data(), sum_per_block_out.size()),
        NBLOCKS);
    
   //     std::cout<<"NBLOCK="<<NBLOCKS<<std::endl;
    //kernel_print<<<1, BLOCK_SIZE, BLOCK_SIZE*sizeof(int), buffer.stream()>>>( raft::device_span<int>(sum_per_block.data(), sum_per_block.size()), NBLOCKS);

    kernel_propagate<int><<<NBLOCKS, BLOCK_SIZE, BLOCK_SIZE*sizeof(int), buffer.stream()>>>(
         raft::device_span<int>(buffer.data(), buffer.size()),
         raft::device_span<int>(sum_per_block.data(), sum_per_block.size()),
        size);

    CUDA_CHECK_ERROR(cudaStreamSynchronize(buffer.stream()));
}

template <typename T>
struct descriptor {
    T aggregate;           // Holds an accumulated or aggregate value (of type T)
    T inclusive_prefix;    // Holds an inclusive prefix value (of type T)
    state_type status_flag; // Some sort of state indicator (likely an enum or bitflag)
};

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

    extern __shared__ int sdata[];
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
    cuda::atomic_ref<state_type, cuda::thread_scope_device> ref_status_flag(descriptors[DLB_blockIdx].status_flag);
    cuda::atomic_ref<int, cuda::thread_scope_device> ref_inclusive_prefix(descriptors[DLB_blockIdx].inclusive_prefix);
    cuda::atomic_ref<int, cuda::thread_scope_device> ref_aggregate(descriptors[DLB_blockIdx].aggregate);

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
    
    
    //4. Determine the partition’s exclusive prefix using decoupled look-back. 
    if ((tid==0)&&(DLB_blockIdx!=0)){
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

    __syncthreads();

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
Rename stuff like 2016 paper: 183(-1%)
*/