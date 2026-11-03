# Cohort manifests

Build or check a recorded set of task/interface assignments:

```bash
uv run --extra harbor python -m benchmarks.manifest build CATALOG.json --output MANIFEST.json
uv run --extra harbor python -m benchmarks.manifest verify MANIFEST.json
```

The catalog supplies task names, agent horizons, `gpu_image` and `verifier_image`
IDs, the public Harbor agent configuration, service URL/source identity,
`repeats`, `order_seed` and `concurrency`. Credentials belong to the trusted
launcher and its Secret configuration, outside these JSON files.

The builder creates attached, manual-transfer (`remote`) and Shared Volume
cells for each task and repeat. Its shuffle seed controls assignment order;
it is separate from model or benchmark random seeds. Rows include the GPU
settings and factual interface instructions.

The manifest records prepared task files, instructions, scorers, harness source
and dependency hashes. Verification reports changed, added or removed local
inputs. Runtime model availability, the deployed service and derived images
are checked or recorded by the [cohort runner](study.md). Prepare PostTrainBench
for the planned horizon so its prompt and timer agree.

Generating a manifest starts no work. Keep generated catalogs, manifests and
trial outputs under `artifacts/` or another local output directory.
