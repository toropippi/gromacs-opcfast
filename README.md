# GROMACS OPCfast

This is an unofficial research fork of **GROMACS 2026.3** for simulations with four-site Optimal Point Charge (OPC) water. The GROMACS development team has not endorsed this fork. Please report problems with OPCfast in [this repository's issue tracker](https://github.com/toropippi/gromacs-opcfast/issues).

OPC water carries its negative charge on a virtual site. This CUDA implementation constructs that site and redistributes its force on the GPU, so coordinates can stay on the device through integration and constraints. It also includes GPU CMAP evaluation and fusion of OPC force redistribution with force reduction on supported steps. The study used a single GPU and one thread-MPI rank per simulation.

**Source history.** The August 9, 2026 production build and the October 2026 benchmark controls are recorded as two commits above the official [v2026.3 tag](https://github.com/gromacs/gromacs/tree/v2026.3). This publication branch was assembled on November 20, 2026.

| Source | Use in the study |
| --- | --- |
| [Production source, opcfast20260809g](https://github.com/toropippi/gromacs-opcfast/tree/opcfast-20260809g) | Five 5-µs Trp-cage production runs with the modified build. |
| [Publication source with benchmark controls](https://github.com/toropippi/gromacs-opcfast/tree/opcfast-paper-v1) | GLP-1 receptor membrane-system timings. The controls allow GPU CMAP, OPC force fusion and CUDA Graphs to be switched separately. |

[Compare the fork with GROMACS v2026.3](https://github.com/toropippi/gromacs-opcfast/compare/v2026.3...opcfast-paper-v1) · [Inspect the benchmark-control commit](https://github.com/toropippi/gromacs-opcfast/commit/cca945c0b4b18398aa9079cb437d77e8ab94bf51)

**Related GROMACS development.** Lukas Müllender demonstrated GPU virtual-site handling on a 395-molecule TIP4P water box in [GROMACS MR 5977](https://gitlab.com/gromacs/gromacs/-/merge_requests/5977) and integrated site construction and force redistribution with GPU updates in [MR 6119](https://gitlab.com/gromacs/gromacs/-/merge_requests/6119). The folding tests and membrane timings in this study used the OPCfast code linked above.

## Build

Clone the publication revision, then configure a CUDA build from the repository root. The benchmark used mixed precision, thread-MPI, OpenMP, AVX2 and the bundled CPU FFT. The commands below target the RTX 2080 Ti (SM 75) and RTX 3080 (SM 86). Set `CMAKE_CUDA_ARCHITECTURES` for the device on which you intend to run.

```sh
git clone --branch opcfast-paper-v1 https://github.com/toropippi/gromacs-opcfast.git
cd gromacs-opcfast
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DGMX_GPU=CUDA "-DCMAKE_CUDA_ARCHITECTURES=75;86" -DGMX_SIMD=AVX2_256 -DGMX_FFT_LIBRARY=fftpack -DGMX_MPI=OFF -DGMX_THREAD_MPI=ON -DGMX_OPENMP=ON -DBUILD_SHARED_LIBS=OFF -DGMX_BUILD_UNITTESTS=ON -DREGRESSIONTEST_DOWNLOAD=OFF
cmake --build build --target gmx --parallel 4
```

Use a supported host compiler and CUDA toolkit. On Windows, run CMake and Ninja from an x64 Visual Studio developer prompt. The executable is `build/bin/gmx.exe` on Windows and `build/bin/gmx` on Linux. Run it with `--version` to check the `opcfast20260809g` suffix and CUDA support. The study's Windows benchmark used MSVC 19.44 and CUDA 12.6. For other platforms and dependencies, see the [GROMACS 2026.3 installation guide](https://manual.gromacs.org/2026.3/install-guide/index.html).

## Reproduce the GLP-1 receptor timings

The [benchmark instructions](benchmarks/glp1r/README.md) include the production MDP, all 28 device/configuration command lines, environment controls, processor affinity and the checksum of the compiled input. Figure 4 used three timed trials per configuration and device.

## Attribution and license

The GROMACS source modifications retain the GNU Lesser General Public License, version 2.1 or later. See [COPYING](COPYING), the [upstream README](docs/README-upstream.txt), [AUTHORS](AUTHORS) and the [GROMACS citation information](CITATION.cff).
