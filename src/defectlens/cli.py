from __future__ import annotations

import json
import shutil
from pathlib import Path

import typer
from rich.console import Console

from defectlens import __version__
from defectlens.config import load_dataset_by_name, load_settings
from defectlens.data.download import download_archive
from defectlens.data.manifest import build_manifest, read_manifest, write_manifest

console = Console()
app = typer.Typer(
    help="DefectLens — defect detection from a single part photo.", no_args_is_help=True
)
data_app = typer.Typer(help="Dataset download, manifest and verification.", no_args_is_help=True)
app.add_typer(data_app, name="data")


@data_app.command("download")
def data_download(
    dataset: str = typer.Argument(..., help="Dataset name from configs/datasets (e.g. visa)."),
    archive: Path = typer.Option(
        None, "--archive", help="Local archive to import (for registration-gated datasets)."
    ),
    dest: Path = typer.Option(None, "--dest", help="Destination directory [default: data/]."),
) -> None:
    settings = load_settings()
    cfg = load_dataset_by_name(dataset)
    dest_dir = dest or settings.data_dir

    if cfg.url is None:
        if archive is None or not archive.exists():
            console.print(
                f"[red]{cfg.name} requires a manual download.[/red]\n"
                "Register at the official page, then re-run with --archive /path/to/file.tar"
            )
            raise typer.Exit(code=1)
        archive_path = dest_dir / archive.name
        dest_dir.mkdir(parents=True, exist_ok=True)
        if archive.resolve() != archive_path.resolve():
            shutil.copyfile(archive, archive_path)
    else:
        with console.status(f"Downloading {cfg.name}..."):
            archive_path = download_archive(
                cfg.url, dest_dir, Path(cfg.url).name, expected_size=cfg.expected_size_bytes
            )

    manifest = build_manifest(
        dataset=cfg.name, archive=archive_path, expected_size=cfg.expected_size_bytes
    )
    manifest_path = write_manifest(manifest, dest_dir / f"{cfg.name}.manifest.yaml")
    console.print(f"[green]Wrote manifest:[/green] {manifest_path}")
    console.print(f"[green]defectlens {__version__}[/green]")


@data_app.command("verify")
def data_verify(
    dataset: str = typer.Argument(..., help="Dataset name."),
    data_dir: Path = typer.Option(None, "--data-dir", envvar="DEFECTLENS_DATA_DIR"),
) -> None:
    settings = load_settings()
    root = data_dir or settings.data_dir
    manifest_path = root / f"{dataset}.manifest.yaml"
    if not manifest_path.exists():
        console.print(f"[red]No manifest at {manifest_path}[/red]")
        raise typer.Exit(code=1)
    issues = read_manifest(manifest_path).verify(root)
    if issues:
        for issue in issues:
            console.print(f"[red]✗ {issue}[/red]")
        raise typer.Exit(code=1)
    console.print(f"[green]✓ {dataset} manifest verifies[/green]")


@app.command()
def init() -> None:
    import mlflow

    settings = load_settings()
    for directory in (settings.data_dir, settings.artifacts_dir, settings.mlruns_dir):
        directory.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(f"file:{settings.mlruns_dir.resolve()}")
    console.print("[green]DefectLens workspace ready[/green]")
    console.print(f"  data:      {settings.data_dir}")
    console.print(f"  artefacts: {settings.artifacts_dir}")
    console.print(f"  mlruns:    {settings.mlruns_dir}")


@app.command()
def fit(
    root: Path = typer.Option(..., "--root", help="Dataset root with <category>/train/good."),
    category: str = typer.Option(..., "--category", help="Category to fit."),
    out: Path = typer.Option(..., "--out", help="Registry directory to write <category>.pt into."),
    coreset_ratio: float = typer.Option(0.01, "--coreset-ratio"),
    target_frr: float = typer.Option(0.01, "--target-frr"),
    seed: int = typer.Option(0, "--seed"),
    backbone: str = typer.Option("wide_resnet50_2", "--backbone"),
    pretrained: bool = typer.Option(
        True, "--pretrained/--no-pretrained", help="--no-pretrained avoids the weights download."
    ),
    review_log: Path = typer.Option(
        None, "--review-log", help="JSONL from the serving review loop; confirmed-good rows fit."
    ),
) -> None:
    """Fit PatchCore, calibrate the threshold, and write a registry artefact.

    This is the rebuild half of the review loop: run it after operators have labelled cases.
    """
    import numpy as np

    from defectlens.eval.benchmark import VALID_EXT, _fit_calibration_splits, _to_tensors
    from defectlens.models.patchcore import PatchCore
    from defectlens.serving.checksums import write_sidecar

    train_good = sorted(
        p for p in (root / category / "train" / "good").iterdir() if p.suffix.lower() in VALID_EXT
    )
    if not train_good:
        raise typer.BadParameter(f"no training images under {root / category / 'train' / 'good'}")
    fit_paths, cal_paths, _ = _fit_calibration_splits(train_good, seed)
    fit_paths = [*fit_paths, *_confirmed_good(review_log, category)]
    model = PatchCore(
        backbone=backbone, coreset_ratio=coreset_ratio, seed=seed, pretrained=pretrained
    )
    model.fit(_to_tensors(fit_paths))
    cal_scores, _ = model.predict(_to_tensors(cal_paths))
    threshold = float(np.quantile(cal_scores.detach().numpy(), 1.0 - target_frr))

    out.mkdir(parents=True, exist_ok=True)
    model.save(
        out / f"{category}.pt",
        category=category,
        threshold=threshold,
        reference_scores=cal_scores.detach().cpu(),
    )
    write_sidecar(out / f"{category}.pt")
    console.print(
        f"  artefact:  {out / f'{category}.pt'}\n"
        f"  fit on:    {len(fit_paths)} nominal images\n"
        f"  threshold: {threshold:.6f} (target FRR {target_frr})"
    )


def _confirmed_good(review_log: Path | None, category: str) -> list[Path]:
    """Spooled images an operator labelled confirmed-good for this category."""
    if review_log is None or not review_log.is_file():
        return []
    rows = [json.loads(line) for line in review_log.read_text().splitlines() if line.strip()]
    return [
        Path(row["image_path"])
        for row in rows
        if row.get("label") == "confirmed-good"
        and row.get("category") == category
        and Path(row["image_path"]).is_file()
    ]


@app.command()
def serve(
    registry: Path = typer.Option(
        None, "--registry", help="Directory of <category>.pt artefacts; enables /ready."
    ),
    host: str = typer.Option(None, "--host"),
    port: int = typer.Option(None, "--port"),
) -> None:
    import uvicorn

    from defectlens.serving.app import create_app
    from defectlens.serving.logs import configure_json_logging
    from defectlens.serving.registry import load_registry

    settings = load_settings()
    configure_json_logging()
    app = create_app(
        models=load_registry(registry) if registry else {},
        spool_dir=settings.artifacts_dir / "queue",
        review_log=settings.artifacts_dir / "reviews.jsonl",
        max_upload_mb=settings.max_upload_mb,
    )
    console.print(f"  registry:  {registry or '(none — /ready will report 503)'}")
    # uvicorn's own access log prints client addresses, which the JSON request log deliberately omits.
    uvicorn.run(app, host=host or settings.host, port=port or settings.port, access_log=False)


@app.command()
def versions() -> None:
    console.print(f"defectlens {__version__}")


eval_app = typer.Typer(help="Evaluation runs.", no_args_is_help=True)
app.add_typer(eval_app, name="eval")


@eval_app.command("benchmark")
def eval_benchmark(
    dataset_root: Path = typer.Option(
        ..., "--dataset-root", help="Root of an extracted category-layout dataset."
    ),
    category: str = typer.Option(..., "--category", help="Category name, e.g. bottle."),
    method: str = typer.Option("patchcore", "--method", help="patchcore | padim."),
    seed: int = typer.Option(0, "--seed"),
    limit_few_shot: bool = typer.Option(
        False, "--quick", help="Skip the few-shot curve for a quick smoke run."
    ),
) -> None:
    from defectlens.eval.benchmark import evaluate_category

    result, _samples = evaluate_category(
        dataset_root,
        category,
        method=method,
        few_shot_counts=() if limit_few_shot else (1, 5, 10, 50),
        seed=seed,
    )
    import json

    console.print_json(json.dumps(result, default=str))
