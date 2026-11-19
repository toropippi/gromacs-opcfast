# GLP-1 receptor membrane benchmark

The timed system contains GLP-1 bound to wild-type GLP-1R in a lipid bilayer (166,826 atoms, including 31,884 OPC waters). The input came from replica 1 in [Kato et al., 2026](https://doi.org/10.1016/j.jmgm.2026.109445).

## Input

[production.mdp](production.mdp) is the parameter file used by GROMACS 2023 to compile the production input. Running the benchmark requires that compiled input, `md_0_1.tpr`, or the corresponding coordinates, topology with includes, and index groups to prepare one. The TPR and molecular data are not distributed in this fork. The recorded TPR has SHA-256 `c03603178d9275bbad2b7e9c6d46de07b542baa8eec816537a67ee14799d3565`.

With the original molecular inputs, the recorded preprocessing command was:

```sh
gmx grompp -f production.mdp -o md_0_1.tpr -c step6.1_equilibration.gro -p topol.top -n index.ndx
```

## Commands and timing

[commands.tsv](commands.tsv) lists the 14 measured configurations on each GPU. `mdrun` contains the arguments to pass to the Reference or Optimized executable. `environment` lists the GROMACS overrides enabled for that run. Start each run with inherited `GMX_*` and `OMP_*` variables cleared, then apply only the overrides in its row. An empty field means no GROMACS overrides. Select one physical device with `CUDA_VISIBLE_DEVICES` so that it is logical device 0. The hexadecimal process-affinity mask is the one used on the measurement host. Adapt that mask to the logical CPU layout of another host.

Use a fresh working directory per trial, with `md_0_1.tpr` present. The following Windows Command Prompt example reproduces the fused, fixed-PME configuration for the RTX 3080 from a clean environment, with the Optimized `gmx.exe` on `PATH` and physical CUDA device 0 selected:

```bat
set "CUDA_VISIBLE_DEVICES=0"
set "GMX_DISABLE_CUDA_GRAPH=1"
start "" /b /wait /affinity 5555 gmx.exe mdrun -s md_0_1.tpr -deffnm md -nsteps 100000 -resetstep 25000 -gpu_id 0 -noconfout -ntmpi 1 -ntomp 6 -pin off -pinoffset 0 -pinstride 2 -nb gpu -pme gpu -pmefft gpu -bonded gpu -update gpu -nstlist 150 -notunepme
```

Each of three randomized blocks included every configuration once per GPU (84 trials). The 2-fs input advances 100,000 steps per trial. GROMACS resets the performance counters after 25,000 steps. Read the throughput in ns/day from the final `Performance:` line of `md.log`. The recorded elapsed times were in whole seconds, equivalent to 0.013 ms/step over the 75,000 timed steps.

Figure 4 follows `custom_listed_cpu_cmap` → `custom_resident_cpu_cmap` → `custom_resident_gpu_cmap` → `custom_fused` → `custom_full_fixed` with PME tuning off. Each change is measured against the preceding configuration, with step-time savings expressed relative to that GPU's initial CPU-update time. The 1.57–1.66× whole-build comparison uses `stock_tuned` and `custom_full_operational`, both with PME tuning on.
