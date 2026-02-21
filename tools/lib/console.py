"""Shared Rich console for all CLI tools.

Automatically disables markup and color when stdout is not a TTY (e.g.,
when captured by subprocess in tests), so plain-text assertions still pass.
"""

import os
import sys

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich.rule import Rule

_is_tty = sys.stdout.isatty()

console = Console(highlight=False, force_terminal=_is_tty, no_color=not _is_tty)


def enable_hf_offline():
    """Enable HuggingFace offline mode. If a model isn't cached, the caller
    should catch the error and call :func:`disable_hf_offline` to allow a
    one-time download, then retry."""
    os.environ["HF_HUB_OFFLINE"] = "1"


def disable_hf_offline():
    """Temporarily disable HuggingFace offline mode to allow downloading."""
    os.environ.pop("HF_HUB_OFFLINE", None)
    print("Model not cached — downloading...", file=sys.stderr)


def run_with_hf_fallback(fn, *args, **kwargs):
    """Run *fn* in offline mode; if the model is not cached, automatically
    disable offline mode and retry so the download can proceed."""
    try:
        return fn(*args, **kwargs)
    except Exception as exc:
        # Catch HuggingFace offline/cache-miss errors.
        name = type(exc).__name__
        if "OfflineModeIsEnabled" in name or "LocalEntryNotFoundError" in name:
            disable_hf_offline()
            return fn(*args, **kwargs)
        raise


__all__ = ["console", "Table", "Panel", "Text", "Rule",
           "enable_hf_offline", "disable_hf_offline", "run_with_hf_fallback"]
