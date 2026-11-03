# Device setup

Two device classes are supported. On both, run `harness rails` before measuring:
it reports the energy rails found, the clock state and any other process holding
GPU memory.

```bash
python -m cslr_cost.harness.cli rails
```

## NVIDIA Jetson Orin (tested: Orin Nano Super developer kit, JetPack 6)

The harness runs natively on the board, not in the container: the container is
x86, and JetPack's own TensorRT and GStreamer-enabled OpenCV are what a
deployment uses.

**Environment.** A virtualenv on the system Python with
`--system-site-packages`, so TensorRT's Python bindings and the system OpenCV are
visible. Install a CUDA-enabled aarch64 PyTorch whose kernels cover the Orin
(sm_87); keep NumPy at the system's 1.x, because the system OpenCV and TensorRT
are built against it. Then run with `PYTHONPATH=src`.

**Before every measuring session** (pinning does not survive a reboot):

```bash
sudo nvpmodel -m 0          # the power mode you report (e.g. 15W on Orin Nano Super)
sudo nvpmodel -q
systemctl get-default       # multi-user.target: the desktop's GPU clients count as contention
sudo jetson_clocks          # pin clocks (P4)
sudo jetson_clocks --show   # MinFreq == MaxFreq on every CPU and the GPU
```

Save the unpinned state first if you want to restore it later
(`sudo jetson_clocks --store <file>`, then `--restore <file>`).

**Energy.** The board's INA3221 monitor, read through hwmon: `VDD_IN` is the
module's total input; `VDD_CPU_GPU_CV` and `VDD_SOC` are a breakdown *inside*
it, so never add them to `VDD_IN`. The monitor refreshes every 160–400 ms and is
sampled at 10 Hz; reads cost a few percent of one core, recorded per window.
NVML answers on the Orin but does not report power, so it is not used.

**Contention (GPU attribution).** Other processes' GPU memory is visible only in
`/sys/kernel/debug/nvmap/iovmm/clients`, which is root-only. Allow the measuring
user to read exactly that file and nothing else:

```bash
echo '<user> ALL=(root) NOPASSWD: /usr/bin/cat /sys/kernel/debug/nvmap/iovmm/clients' \
  | sudo tee /etc/sudoers.d/cslr-nvmap > /dev/null
sudo chmod 0440 /etc/sudoers.d/cslr-nvmap
sudo visudo -cf /etc/sudoers.d/cslr-nvmap      # must print "parsed OK"
sudo -n /usr/bin/cat /sys/kernel/debug/nvmap/iovmm/clients    # as <user>: no password
```

Without it, runs are marked `PARTIALLY CHECKED`, never `clean`.

**Memory.** CPU and GPU share one pool. The harness reports the process's
`nvmap` footprint, which includes TensorRT's allocations.

**Idle drifts with attached hardware.** A USB disk or a running desktop raises
idle by about 1 W. Compare runs from different sessions on over-idle energy, with
each run's own idle subtracted (P2).

## x86 desktop with an NVIDIA GPU (tested: i9-9900K + RTX 3060)

Use the container (`docker build -t cslr-cost-harness:dev .`, then
`scripts/dev.sh ...`); it pins CUDA 11.8, PyTorch 2.0.0, TensorRT 10.3, mmpose,
SCTK and ctcdecode.

**Energy.** NVML's energy counter for the GPU, and Intel RAPL `package-0` for the
CPU. Desktop energy therefore covers the GPU and the CPU package, not the whole
machine. RAPL is needed because decode, crop and postproc run on the CPU and are
invisible on the GPU rail.

RAPL counters are root-only by default (CVE-2020-8694). Make them readable by the
`users` group, not by everyone:

```bash
sudo cp scripts/udev/99-rapl.rules /etc/udev/rules.d/
sudo udevadm control --reload-rules && sudo udevadm trigger --subsystem-match=powercap
cat /sys/class/powercap/intel-rapl:0/energy_uj    # a number, not "Permission denied"
```

`scripts/dev.sh` mounts the powercap tree into the container and adds the
`users` group, so the same counters are readable inside.

**Clocks.** Consumer GPUs and desktop CPUs usually cannot be pinned without root
and vendor tools; rows record `device_power_mode: default`, and desktop energy is
labelled indicative.

**Contention.** NVML lists every process holding GPU memory, and `/proc/stat`
gives other processes' CPU time. On a shared machine, measure when it is quiet:
a busy window marks the run `SHARED MACHINE`.
