from benchmarks import export


def test_secret_split_across_scan_chunks(tmp_path, monkeypatch):
    monkeypatch.setattr(export, "_SCAN_CHUNK_BYTES", 16)
    path = tmp_path / "artifact.bin"
    path.write_bytes(b"x" * 13 + b"private-credential" + b"z" * 100)
    assert export.contains_secret(path, ["not-present", "private-credential"])
    assert not export.contains_secret(path, ["not-present", "private-other"])


def test_secret_after_many_chunks_and_single_byte_patterns(tmp_path, monkeypatch):
    monkeypatch.setattr(export, "_SCAN_CHUNK_BYTES", 16)
    path = tmp_path / "artifact.bin"
    path.write_bytes(b"x" * 100 + b"\x00" + "秘密".encode())
    assert export.contains_secret(path, ["\x00"])
    assert export.contains_secret(path, ["秘密"])
    assert not export.contains_secret(path, ["", "absent"])
    assert not export.contains_secret(path, [])


def test_source_capture_uses_packages_when_the_driver_is_relocated(tmp_path, monkeypatch):
    import benchmarks
    import gpu_harness

    runtime = tmp_path / "runtime"
    runtime.mkdir()
    (runtime / "run.py").write_text("relocated Modal driver")
    (runtime / "credentials").write_text("private sentinel outside the source packages")
    for module in (benchmarks, gpu_harness):
        package = runtime / module.__name__
        package.mkdir()
        (package / "__init__.py").write_text("public source")
        (package / "__pycache__").mkdir()
        (package / "__pycache__/cached.pyc").write_bytes(b"cache")
        monkeypatch.setattr(module, "__file__", str(package / "__init__.py"))
    destination = tmp_path / "snapshot"
    export.copy_sources(destination)
    assert sorted(str(p.relative_to(destination)) for p in destination.rglob("*") if p.is_file()) == [
        "benchmarks/__init__.py",
        "gpu_harness/__init__.py",
    ]
