"""Local provider configuration with redaction and file-permission checks."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import stat
from typing import Any
from urllib.parse import urlparse


class LLMConfigError(ValueError):
    """Raised when local LLM credentials or provider settings are invalid."""


@dataclass(frozen=True)
class LLMProviderConfig:
    enabled: bool
    provider: str
    api_key: str = field(repr=False)
    base_url: str
    model: str
    api_mode: str = "responses"
    timeout_seconds: float = 30.0
    max_retries: int = 2
    organization: str | None = None
    project: str | None = None
    extra_headers: dict[str, str] = field(default_factory=dict, repr=False)
    source_path: Path | None = field(default=None, repr=False)

    @classmethod
    def from_json(cls, path: str | Path) -> "LLMProviderConfig":
        source_path = Path(path).resolve()
        payload = json.loads(source_path.read_text(encoding="utf-8"))
        config = cls(
            enabled=bool(payload.get("enabled", False)),
            provider=str(payload.get("provider", "openai_compatible")),
            api_key=str(payload.get("api_key", "")),
            base_url=str(payload.get("base_url", "")).rstrip("/"),
            model=str(payload.get("model", "")),
            api_mode=str(payload.get("api_mode", "responses")),
            timeout_seconds=float(payload.get("timeout_seconds", 30.0)),
            max_retries=int(payload.get("max_retries", 2)),
            organization=payload.get("organization"),
            project=payload.get("project"),
            extra_headers={
                str(name): str(value) for name, value in payload.get("extra_headers", {}).items()
            },
            source_path=source_path,
        )
        config.validate()
        return config

    def validate(self) -> None:
        if self.provider not in {"openai", "openai_compatible"}:
            raise LLMConfigError("provider must be 'openai' or 'openai_compatible'")
        if self.api_mode not in {"responses", "chat_completions"}:
            raise LLMConfigError("api_mode must be 'responses' or 'chat_completions'")
        if self.timeout_seconds <= 0:
            raise LLMConfigError("timeout_seconds must be positive")
        if self.max_retries < 0:
            raise LLMConfigError("max_retries cannot be negative")
        if not self.enabled:
            return
        missing = [
            name
            for name, value in (
                ("api_key", self.api_key),
                ("base_url", self.base_url),
                ("model", self.model),
            )
            if not value or str(value).startswith("PASTE_")
        ]
        if missing:
            raise LLMConfigError(f"Enabled LLM config is missing: {', '.join(missing)}")
        parsed_url = urlparse(self.base_url)
        if parsed_url.scheme not in {"http", "https"} or not parsed_url.netloc:
            raise LLMConfigError("base_url must be an absolute HTTP(S) URL")
        if os.name != "nt" and self.source_path is not None:
            permissions = stat.S_IMODE(self.source_path.stat().st_mode)
            if permissions & 0o077:
                raise LLMConfigError(
                    f"Credential file permissions are {permissions:o}; run "
                    f"'chmod 600 {self.source_path}'"
                )

    def safe_summary(self) -> dict[str, Any]:
        """Return provider settings that are safe to print or expose in diagnostics."""
        return {
            "enabled": self.enabled,
            "provider": self.provider,
            "api_key": "***" if self.api_key else "",
            "base_url": self.base_url,
            "model": self.model,
            "api_mode": self.api_mode,
            "timeout_seconds": self.timeout_seconds,
            "max_retries": self.max_retries,
            "organization": self.organization,
            "project": self.project,
        }
