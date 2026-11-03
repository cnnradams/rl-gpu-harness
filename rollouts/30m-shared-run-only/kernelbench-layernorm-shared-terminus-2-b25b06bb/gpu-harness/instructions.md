# GPU tools

`gpu run -- PROGRAM [ARG ...]` starts a GPU box for that command; the box stops
when it exits. Run waits unless `--detach` is set.
GPU programs start only through `gpu run`. `gpu stop BOX` stops a box early;
`gpu status COMMAND` and `gpu wait COMMAND` read command output and state.
CPU `/workspace` writes are committed before each run. The new GPU box mounts
that workspace; GPU writes commit in the background and on shutdown.
`gpu sync` commits and reloads the CPU workspace view. Reload requires no open
files or working directories on the Volume. GPU-local paths outside `/workspace`
are lost when the box stops. `gpu --help` lists commands; `gpu policy` lists limits.
