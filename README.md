# Homework "Scan"

## Description
This is my attempt at [the "scan" homework](https://github.com/Kh4ster/tp_irgpua/tree/master/tp_scan_fetch_content) taught by [Nicolas Blin](https://github.com/Kh4ster) (Nvidia). The goal is to implement a fast scan operation on an array of integers. One can build my solution with a classical `cmake` process.

## My solution

My solution implements the [decoupled lookback](https://research.nvidia.com/sites/default/files/pubs/2016-03_Single-pass-Parallel-Prefix/nvr-2016-002.pdf) method of Meririll and Garland as strongly suggested by the teacher with extra optimisations:

- Increased work per threads, each threads handles `WPT` elements to reduce the amount of blocks (virtually, blocks are `WPT` times bigger). This mainly helps because it shortens the lookback lenght, and therefore reduces the lookback latency.
- Occupancy maximizing block size of 768. Large blocks are needed to reduce the lookback latency, but blocks of 1024 hurt the occupancy. The maximal size that ensures 100% occupancy is 768 as indicated by `ncu`.
- All unrollable loops are unrolled.
- Parallel lookback: each block uses it's first 32 threads (first warp) to lookback a 32-blocks wide window in a SIMD fashion, dramatically reducing the lookback latency. This implies the use of warp-level intrinsics to perform reductions.
- Implementing a radix-32 Brent-Kung scan-then-propagate
strategy (fig 2a of the paper) where the warps-level scan are performed in registers (not shared memory) with warp-level intrinsics.

## Performance

A good implementation should be as expensive as a copy. The Attained bandwith (BW) is therefore computed as twice the size of the array to be sorted, divided by the runtime. This value should be compared to the GPU's peak bandwitdh.

The tests are done on a `size=1024^3` elements array of integers.

| GPU      | Peak BW (TB) | Attained BW |
| ----------- | ----------- | ----------- |
| RTX Ada 6000    | 0.96       | 0.83 (86%)       |
| V100   |   0.89     |      0.69   (77%)
| A100   |   1.56      |    1.07  (68%)    |
| H100   |   2.04     |      1.23 (60%)   |

Still room for improvement on recent server gpus !
