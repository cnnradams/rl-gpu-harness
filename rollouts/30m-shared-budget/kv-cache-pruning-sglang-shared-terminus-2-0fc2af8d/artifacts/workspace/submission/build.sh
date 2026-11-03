#!/bin/bash
# Reproduce sglang_kv_prune.patch: install the plugin into a frozen SGLang
# v0.5.17 checkout (default /workspace/sglang) and emit the unified diff.
set -euo pipefail
TREE="${H2O_TREE:-/workspace/sglang}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python3 "$HERE/install_plugin.py" "$TREE" "$HERE/plugin"
git -C "$TREE" add -N python/sglang/srt/mem_cache/h2o_policy.py \
                     python/sglang/srt/mem_cache/h2o_runtime.py
git -C "$TREE" diff -- python/sglang > "$HERE/sglang_kv_prune.patch"
echo "patch written to $HERE/sglang_kv_prune.patch"
