#include "to_bench.cuh"

#include "cuda_tools/cuda_error_checking.cuh"

#include <raft/core/device_span.hpp>

#include <rmm/device_uvector.hpp>
#include <rmm/device_scalar.hpp>

#include <cuda/atomic>


constexpr int WARP_SIZE = 32;
constexpr int BLOCK_SIZE = 64;

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
__global__
void kernel_decouple_lookback(raft::device_span<T> buffer, 
                              raft::device_span<T> partial_sums,
                              raft::device_span<T> prefixes,
                              raft::device_span<state_type> states, 
                              raft::device_span<int> DLB_counter, 
                              int size)
{
    //Check that block size is a power of 2
    //Notre dessin marche que si c'est le cas
    assert((blockDim.x & (blockDim.x - 1)) == 0); 
    assert(blockDim.x>=WARP_SIZE);

    extern __shared__ int sdata[];
    unsigned int tid = threadIdx.x;
    __shared__ unsigned int DLB_blockIdx;

    cuda::atomic_ref<int, cuda::thread_scope_device> ref_DLB_counter(DLB_counter[0]);
    //Only thread 0 of the block reads and increment the global counter
    if (tid==0){
        DLB_blockIdx=ref_DLB_counter.fetch_add(1, cuda::memory_order_relaxed); //result is old value, before the +=1
    }
    __syncthreads(); //Ensure that DLB_blockIdx has the right value for all threads
    
    //Global index
    unsigned int i = DLB_blockIdx*blockDim.x+threadIdx.x;

    //Atomic references
    cuda::atomic_ref<state_type, cuda::thread_scope_device> ref_my_state(states[DLB_blockIdx]);
    cuda::atomic_ref<int, cuda::thread_scope_device> ref_my_prefix(prefixes[DLB_blockIdx]);
    cuda::atomic_ref<int, cuda::thread_scope_device> ref_my_partial_sum(partial_sums[DLB_blockIdx]);

    // No one is done initially
    if (tid==0){
        ref_my_state.store(X);
    }
    
    //Check if tid is out of bound. If it is, fill with 0's to not change resut.
    sdata[tid] = (i < size) ? buffer[i]:0;

    //sync the shared mem access
    __syncthreads();

    for (int s=1; s<blockDim.x; s*=2){
        int toto = (tid >= s) ? sdata[tid-s]:0;
        __syncthreads(); //Avoid RAW
        sdata[tid]+= toto;
        __syncthreads();
    }
    
    // Block 0 is done
    if (DLB_blockIdx==0){
        if (tid==0){ //Only thread 0 does atomic
            ref_my_prefix.store(sdata[blockDim.x-1]); //Store my prefix
            ref_my_partial_sum.store(sdata[blockDim.x-1]); //Store my prefix
            ref_my_state.store(P); //Let the world know I'm done
            ref_my_state.notify_all();
        }
        if (i < size) buffer[i] = sdata[tid]; //store result and return
        return;
    }

     __shared__ T prefix; // We are going to compute this prefix via lookback, 1 per block
     
    if (tid==0){  //Only thread 0 does atomic
        prefix=0;

        ref_my_partial_sum.store(sdata[blockDim.x-1]); //Store the local partial sum
        ref_my_state.store(A); //Let the world know I'm done
        ref_my_state.notify_all();       
        
        int previous_index = DLB_blockIdx-1; // Our lookback index
        state_type s;

        while (true) {
            cuda::atomic_ref<state_type, cuda::thread_scope_device> ref_previous_state(states[previous_index]);
            cuda::atomic_ref<int, cuda::thread_scope_device> ref_previous_partial_sum(partial_sums[previous_index]);
            cuda::atomic_ref<int, cuda::thread_scope_device> ref_previous_prefix(prefixes[previous_index]);

            // Wait until previous block is not X anymore
            ref_previous_state.wait(X);
            
            s = ref_previous_state.load();
        
            assert(s!=X); // S is A or P;
            assert((s==A)||(s==P));

            if (s == P) {
                prefix += ref_previous_prefix.load();
                break;
            }else{
                prefix += ref_previous_partial_sum.load();
            }

            previous_index--;

        }

        assert(s==P);
        assert(previous_index>=0);
        
        ref_my_prefix.store(prefix+sdata[blockDim.x-1]);
        //int n = blockIdx.x;
        //int expected_prefix = (n*(n+1))/2;
        //assert(prefix == expected_prefix);
        ref_my_state.store(P); //Let the world know I'm done
        ref_my_state.notify_all();  
    }    

    __syncthreads(); // Crucial ! attendre the thread 0 ait fini de calculer le préfix

    if (i < size) buffer[i] = sdata[tid]+prefix;
    
}

void DLB(rmm::device_uvector<int>& buffer)
{

    int size = buffer.size();

    assert(BLOCK_SIZE<=1024);
    assert(BLOCK_SIZE%WARP_SIZE==0);
    assert(BLOCK_SIZE>=WARP_SIZE);
    assert((BLOCK_SIZE & (BLOCK_SIZE - 1)) == 0); 

    //Number of blocks to touch the whole array
    unsigned int NBLOCKS=(size+BLOCK_SIZE-1)/BLOCK_SIZE;
    
    rmm::device_scalar<int> DLB_counter(0, buffer.stream());

    rmm::device_uvector<state_type> states(NBLOCKS, buffer.stream());
    rmm::device_uvector<int> prefixes(NBLOCKS, buffer.stream());
    rmm::device_uvector<int> partial_sums(NBLOCKS, buffer.stream());

	kernel_decouple_lookback<int><<<NBLOCKS, BLOCK_SIZE, (BLOCK_SIZE+2)*sizeof(int), buffer.stream()>>>( //2 ints for DLB counter and prefix
        raft::device_span<int>(buffer.data(), buffer.size()),
        raft::device_span<int>(partial_sums.data(), partial_sums.size()),
        raft::device_span<int>(prefixes.data(), prefixes.size()),
        raft::device_span<state_type>(states.data(), states.size()),
        raft::device_span<int>(DLB_counter.data(), 1),
        size);


    CUDA_CHECK_ERROR(cudaStreamSynchronize(buffer.stream()));
}
