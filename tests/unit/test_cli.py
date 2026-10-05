from __future__ import annotations

from typer.testing import CliRunner

from defectlens import __version__, cli

runner = CliRunner()


def test_versions_command() -> None:
    result = runner.invoke(cli.app, ["versions"])
    assert result.exit_code == 0
    assert __version__ in result.output


def test_verify_missing_manifest(tmp_path) -> None:
    result = runner.invoke(cli.app, ["data", "verify", "visa", "--data-dir", str(tmp_path)])
    assert result.exit_code == 1
    assert "No manifest" in result.output


def test_verify_ok(tmp_path) -> None:
    (tmp_path / "sample.tar").write_bytes(b"abc")
    import defectlens.data.manifest as manifest_mod

    m = manifest_mod.build_manifest(dataset="sample", archive=tmp_path / "sample.tar")
    manifest_mod.write_manifest(m, tmp_path / "sample.manifest.yaml")
    result = runner.invoke(cli.app, ["data", "verify", "sample", "--data-dir", str(tmp_path)])
    assert result.exit_code == 0
    assert "verifies" in result.output


def test_download_requires_archive_for_mvtec(tmp_path) -> None:
    result = runner.invoke(cli.app, ["data", "download", "mvtec", "--dest", str(tmp_path)])
    assert result.exit_code == 1
    assert "requires a manual download" in result.output


def test_download_mvtec_with_local_archive(tmp_path) -> None:
    archive = tmp_path / "mvtec_ad.tar"
    archive.write_bytes(b"mvtec-archive-bytes")
    dest = tmp_path / "out"
    result = runner.invoke(
        cli.app, ["data", "download", "mvtec", "--archive", str(archive), "--dest", str(dest)]
    )
    assert result.exit_code == 0, result.output
    assert (dest / "mvtec_ad.tar").exists()
    assert (dest / "mvtec.manifest.yaml").exists()
    verify = runner.invoke(cli.app, ["data", "verify", "mvtec", "--data-dir", str(dest)])
    assert verify.exit_code == 0, verify.output
    assert "verifies" in verify.output


def test_init_creates_workspace(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("DEFECTLENS_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("DEFECTLENS_ARTIFACTS_DIR", str(tmp_path / "artifacts"))
    monkeypatch.setenv("DEFECTLENS_MLRUNS_DIR", str(tmp_path / "mlruns"))
    result = runner.invoke(cli.app, ["init"])
    assert result.exit_code == 0, result.output
    assert (tmp_path / "data").is_dir()
    assert (tmp_path / "mlruns").is_dir()
