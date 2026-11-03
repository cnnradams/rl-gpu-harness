Write `/workspace/report.json` as a JSON object with these keys:
- `gpu`: the GPU name as a string.
- `sum`: the integer sum of the integers from 0 through 999.

The GPU name and sum are obtained from a remote box. GPU tools are available
through the `gpu` command. The local environment has no GPU. `gpu policy` reports whether `/workspace`
is shared with GPU boxes or files need explicit push/pull.
