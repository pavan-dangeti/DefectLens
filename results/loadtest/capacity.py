"""Build the paced capacity table from the committed k6 summaries.

Each level is a constant-arrival-rate run of POST /inspect at a fixed offered load, against a server
on 127.0.0.1 with two inference slots. Latency is for accepted requests only (HTTP 200); refusal is
the share of requests answered 503 by the slots (or 429 by the rate limiter, which is lifted for
these runs). The levels are fractions of the offered load at which capacity was estimated.

    uv run python results/loadtest/capacity.py
"""

from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
LEVELS = [(50, 11.7), (80, 18.7), (100, 23.4)]


def main() -> None:
    lines = [
        "| Offered load | Images/s | Requests | Accepted p50 (ms) | p95 (ms) | p99 (ms) | Refused (503) |",
        "|---|---|---|---|---|---|---|",
    ]
    for pct, rate in LEVELS:
        metrics = json.loads((HERE / f"paced_{pct}.json").read_text())["metrics"]
        accepted = metrics["accepted_ms"]
        lines.append(
            f"| {pct}% of 23.4/s | {rate} | {metrics['http_reqs']['count']} "
            f"| {accepted['med']:.1f} | {accepted['p(95)']:.1f} | {accepted['p(99)']:.1f} "
            f"| {100 * metrics['inference_refused']['value']:.2f}% |"
        )
    table = "\n".join(lines) + "\n"
    (HERE / "capacity.md").write_text(table)
    print(table)


if __name__ == "__main__":
    main()
