"""
Assignment 11 — Monitoring & Alerts starter (TODO).

Tracks block rate, rate-limit hits, judge fail rate.
Fires alerts when thresholds are exceeded.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


def default_metrics_path() -> str:
    """Always resolve to <repo>/outputs/… (safe when cwd is src/)."""
    repo_root = Path(__file__).resolve().parents[2]
    return str(repo_root / "outputs" / "metrics.json")


@dataclass
class Alert:
    metric: str
    value: float
    threshold: float
    message: str


@dataclass
class MonitoringAlert:
    """Aggregate counters from pipeline plugins and emit alerts."""

    block_rate_threshold: float = 0.5
    rate_limit_hit_threshold: int = 5
    judge_fail_rate_threshold: float = 0.3
    error_rate_threshold: float = 0.2
    alerts: list[Alert] = field(default_factory=list)

    # Counters — update these from your pipeline after each request
    total_requests: int = 0
    blocked_requests: int = 0
    rate_limit_hits: int = 0
    judge_checks: int = 0
    judge_fails: int = 0
    redacted_responses: int = 0
    errors: int = 0

    def record(self, *, blocked: bool, layer: str | None) -> None:
        """Update counters after one request went through the pipeline."""
        self.total_requests += 1
        if blocked:
            self.blocked_requests += 1
        if layer == "rate_limit":
            self.rate_limit_hits += 1
        elif layer == "output_redacted":
            self.redacted_responses += 1
        elif layer == "error":
            self.errors += 1

    def check_metrics(self) -> list[Alert]:
        """Recompute alerts from the current counters (idempotent — no duplicates)."""
        snap = self.snapshot()
        error_rate = self.errors / self.total_requests if self.total_requests else 0.0
        checks = [
            ("block_rate", snap["block_rate"], self.block_rate_threshold,
             snap["block_rate"] > self.block_rate_threshold,
             "Block rate above threshold — possible attack wave or over-blocking."),
            ("rate_limit_hits", self.rate_limit_hits, self.rate_limit_hit_threshold,
             self.rate_limit_hits >= self.rate_limit_hit_threshold,
             "Many rate-limit hits — possible flooding / cost attack."),
            ("judge_fail_rate", snap["judge_fail_rate"], self.judge_fail_rate_threshold,
             self.judge_checks > 0 and snap["judge_fail_rate"] > self.judge_fail_rate_threshold,
             "LLM judge rejects too many responses."),
            ("error_rate", error_rate, self.error_rate_threshold,
             error_rate > self.error_rate_threshold,
             "Upstream LLM errors above threshold — check provider / network."),
        ]
        self.alerts = [
            Alert(metric=name, value=float(value), threshold=float(threshold), message=msg)
            for name, value, threshold, fired, msg in checks
            if fired
        ]
        return self.alerts

    def export_json(self, filepath: str | None = None):
        """Write metrics + alerts to JSON under repo-root ``outputs/`` by default."""
        self.check_metrics()
        path = Path(filepath or default_metrics_path())
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.snapshot(), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return path

    def snapshot(self) -> dict:
        block_rate = (
            self.blocked_requests / self.total_requests
            if self.total_requests
            else 0.0
        )
        judge_fail_rate = (
            self.judge_fails / self.judge_checks if self.judge_checks else 0.0
        )
        return {
            "total_requests": self.total_requests,
            "blocked_requests": self.blocked_requests,
            "block_rate": block_rate,
            "rate_limit_hits": self.rate_limit_hits,
            "judge_checks": self.judge_checks,
            "judge_fails": self.judge_fails,
            "judge_fail_rate": judge_fail_rate,
            "redacted_responses": self.redacted_responses,
            "errors": self.errors,
            "alerts": [
                {
                    "metric": a.metric,
                    "value": a.value,
                    "threshold": a.threshold,
                    "message": a.message,
                }
                for a in self.alerts
            ],
        }
