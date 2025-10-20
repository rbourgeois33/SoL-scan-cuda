# TP scan

The goal of the scan TP is to learn how to program scan on GPU and how to program the Decoupled Look-back.

You should first try to have a working scan before going into the Decoupled Look-back.
You will find "TODO" where you need to modify things and add your code.

## Requirements

* [Cuda Toolkit](https://developer.nvidia.com/cuda-downloads)
* C++ compiler ([g++](https://gcc.gnu.org/) for linux,  [MSVC](https://visualstudio.microsoft.com/downloads/) for Windows)
* [GPU supported by CUDA](https://en.wikipedia.org/wiki/CUDA#GPUs_supported)
* [CMake](https://cmake.org/download/)
* [GoogleBenchmark](https://github.com/google/benchmark)

## Build

- To build, execute the following commands :

```bash
mkdir build && cd build
cmake ..
make -j
```

## Run (from ./build directory) :

### Running the benchmarks

```bash
./bench
```

### Running Nsight Compute

- The following command will generate the Nsight Compute report with all kernel information (full).

```bash
ncu -o scan_nsight -f --set full ./bench --bench-nsight
```

You can now open the *.ncu-rep file using Nsight Compute and analyze the results.

### Additional infos

* By default the program **will run in release**. To build in **debug**, do:

```bash
cmake -DCMAKE_BUILD_TYPE=Debug ..
```

* You can specify the "--no-check" option when running the bench binary to disable result checking :
```bash
./bench --no-check
```

* You can specify the "--bench-nsight" option when running the bench binary to forbid Google Benchmark from running the functions multiple times (Nsight will do this job) :
```bash
./bench --bench-nsight
```

## To add and benchmark your scan

In `bench/main.cc`:
* Add the sizes you want to benchmark to "sizes" array
* Add the name / function you want to benchmark to "scan_to_bench" array

In `src/to_bench.cu(h)`:
* Add your functions to benchmark