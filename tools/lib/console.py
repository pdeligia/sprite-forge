"""Shared Rich console for all CLI tools.

Automatically disables markup and color when stdout is not a TTY (e.g.,
when captured by subprocess in tests), so plain-text assertions still pass.
"""

import sys

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text

_is_tty = sys.stdout.isatty()

console = Console(highlight=False, force_terminal=_is_tty, no_color=not _is_tty)

__all__ = ["console", "Table", "Panel", "Text"]
