# GPU tools

`gpu` provides HTTP access to remote GPU boxes. The local environment and each
GPU box have separate filesystems. `gpu push` and `gpu pull` transfer files;
`gpu run -- PROGRAM [ARG ...]` starts a GPU box with its first command;
the box stops when that command exits. `gpu exec BOX -- PROGRAM [ARG ...]`
runs additional commands while the box is alive.
Run and exec wait unless `--detach` is set.
`gpu --help` lists the interface.
`gpu policy` reports this session's resources, limits and allocated GPU time.
GPU-time accounting starts after readiness, includes idle allocation, and sums
across GPUs. Commands can run concurrently on a box. A stopped or failed box
loses its processes and local files; work is not automatically restored.
