#include "to_bench.cuh"

#include "cuda_tools/cuda_error_checking.cuh"

#include <raft/core/device_span.hpp>

#include <rmm/device_uvector.hpp>

constexpr int WARP_SIZE = 32;
constexpr int BLOCK_SIZE = 1024;

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
