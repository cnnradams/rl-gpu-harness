# GPU tools

`gpu` provides HTTP access to remote GPU boxes. The agent and GPU boxes mount
the same session-specific Volume at `/workspace`.
`gpu run -- PROGRAM [ARG ...]` starts a GPU box with its first command;
the box stops when that command exits. `gpu exec BOX -- PROGRAM [ARG ...]`
runs additional commands while the box is alive.
Run and exec wait unless `--detach` is set. A new box mounts the latest
committed files. Modal commits GPU writes in the background and on shutdown.
`gpu sync` commits and reloads the agent's CPU-side Volume view; it does not
allocate a GPU or reload a running GPU box. It can run before any GPU exists
and after a GPU box stops. `gpu refresh BOX` reloads a running GPU box's Volume
view. Reload fails while any process holds open files or a working directory
on that Volume. GPU commands default to `/workspace`; the CPU agent starts in `/`.
Sync reports completed steps on failure without stopping processes.
Concurrent same-file edits use last-writer-wins
semantics. Paths outside `/workspace` are separate and GPU-local files there
are lost when the box stops.
`gpu --help` lists the interface; `gpu policy` reports resources and limits.
GPU time starts at readiness, includes idle allocation and sums across GPUs.
Commands can run concurrently. A failed box loses its running processes;
work is not automatically restarted.

This session has an enforced budget of 1292.62 allocated GPU seconds.
