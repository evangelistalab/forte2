import sys


import sys

import sys


def format_ansi(
    text: str,
    fg: str | None = None,
    bg: str | None = None,
    bold: bool = False,
    use_color: bool | None = None,
) -> str:
    """
    Return `text` formatted with optional ANSI color and bold styling.

    Parameters
    ----------
    text : str
        The text to format.
    fg : str, optional
        Foreground color name (e.g., "red", "bright_cyan").
    bg : str, optional
        Background color name.
    bold : bool, optional
        If True, apply bold styling.
    use_color : bool, optional
        Force color on/off. If None, auto-detect via sys.stdout.isatty().

    Returns
    -------
    str
        Formatted string with ANSI escape sequences if supported.
    """
    COLORS = {
        "black": 30,
        "red": 31,
        "green": 32,
        "yellow": 33,
        "blue": 34,
        "magenta": 35,
        "cyan": 36,
        "white": 37,
        "bright_black": 90,
        "bright_red": 91,
        "bright_green": 92,
        "bright_yellow": 93,
        "bright_blue": 94,
        "bright_magenta": 95,
        "bright_cyan": 96,
        "bright_white": 97,
    }

    if use_color is None:
        use_color = sys.stdout.isatty()

    if not use_color:
        return text

    seq = []
    if bold:
        seq.append("1")
    if fg in COLORS:
        seq.append(str(COLORS[fg]))
    if bg in COLORS:
        seq.append(str(COLORS[bg] + 10))

    if not seq:
        return text

    return f"\033[{';'.join(seq)}m{text}\033[0m"


class AsciiTable:
    COLORS = {
        "black": 30,
        "red": 31,
        "green": 32,
        "yellow": 33,
        "blue": 34,
        "magenta": 35,
        "cyan": 36,
        "white": 37,
        "bright_black": 90,
        "bright_red": 91,
        "bright_green": 92,
        "bright_yellow": 93,
        "bright_blue": 94,
        "bright_magenta": 95,
        "bright_cyan": 96,
        "bright_white": 97,
    }

    def __init__(
        self,
        columns,
        formats=None,
        sep_major="=",
        sep_minor="-",
        padding=3,
        header_fg="bright_white",
        header_bg=None,
        bold_header=True,
        footer_fg="bright_white",
        footer_bg=None,
        row_fg="bright_white",
        row_bg=None,
        use_color=None,
    ):
        """
        Create a color-capable ASCII table with optional bold headers.

        Parameters
        ----------
        columns : list[str]
            Column headers.
        formats : list[str]
            Per-column format strings (e.g., "{:>15.10f}", "{:<10}").
        sep_major, sep_minor : str
            Separator line characters.
        padding : int
            Spaces between columns.
        header_fg, header_bg, footer_fg, footer_bg, row_fg, row_bg : str
            Colors for each section.
        bold_header : bool
            If True, makes header text bold (only in TTY mode).
        use_color : bool, optional
            Force color on/off. If None, auto-detect via sys.stdout.isatty().
        """
        if formats is None:
            formats = ["{:<}"] * len(columns)
        if len(columns) != len(formats):
            raise ValueError("columns and formats must have same length")

        self.columns = list(columns)
        self.formats = list(formats)
        self.padding = " " * padding
        self.sep_major = sep_major
        self.sep_minor = sep_minor

        # Colors
        self.use_color = use_color if use_color is not None else sys.stdout.isatty()
        self.header_fg = header_fg
        self.header_bg = header_bg
        self.footer_fg = footer_fg
        self.footer_bg = footer_bg
        self.row_fg = row_fg
        self.row_bg = row_bg
        self.bold_header = bold_header

        # Compute widths
        self.col_widths = []
        for col, fmt in zip(self.columns, self.formats):
            fmt_width = self._infer_format_width(fmt)
            self.col_widths.append(max(len(str(col)), fmt_width))
        self.total_width = self._total_width()

    # ---------- Public API ----------
    def header(self) -> str:
        """Return a formatted, optionally colored and bold header section."""
        major = self._colored(
            self.sep_major * self.total_width, self.header_fg, self.header_bg
        )
        header_line = self._format_header_line()
        minor = self._colored(
            self.sep_minor * self.total_width, self.header_fg, self.header_bg
        )
        return f"{major}\n{header_line}\n{minor}"

    def row(self, *values, fg=None, bg=None) -> str:
        """Return one formatted row, optionally colored."""
        parts = []
        for val, fmt, width in zip(values, self.formats, self.col_widths):
            cell = self._safe_format(fmt, val)
            if len(cell) < width:
                cell = cell.rjust(width)
            parts.append(cell)
        line = self.padding.join(parts)
        return self._colored(line, fg or self.row_fg, bg or self.row_bg)

    def footer(self) -> str:
        """Return a formatted footer line."""
        return self._colored(
            self.sep_major * self.total_width, self.footer_fg, self.footer_bg
        )

    # ---------- Internals ----------
    def _total_width(self) -> int:
        return sum(self.col_widths) + len(self.padding) * (len(self.columns) - 1)

    def _format_header_line(self) -> str:
        line = self.padding.join(
            col.rjust(w) for col, w in zip(self.columns, self.col_widths)
        )
        if self.use_color and self.bold_header:
            return (
                f"\033[1m{self._colored(line, self.header_fg, self.header_bg)}\033[0m"
            )
        return self._colored(line, self.header_fg, self.header_bg)

    def _safe_format(self, fmt: str, value) -> str:
        try:
            return fmt.format(value)
        except Exception:
            return str(value)

    def _infer_format_width(self, fmt: str) -> int:
        probes = (0, 0.0, "")
        for p in probes:
            try:
                s = fmt.format(p)
                return len(s)
            except Exception:
                continue
        return 0

    def _colored(self, text: str, fg: str = None, bg: str = None) -> str:
        """Apply ANSI colors and background if supported."""
        if not self.use_color:
            return text
        seq = []
        if fg in self.COLORS:
            seq.append(str(self.COLORS[fg]))
        if bg in self.COLORS:
            seq.append(str(self.COLORS[bg] + 10))
        if not seq:
            return text
        return f"\033[{';'.join(seq)}m{text}\033[0m"


# class AsciiTable:
#     def __init__(self, columns, formats=None, sep_major="=", sep_minor="-", padding=3):
#         if formats is None:
#             formats = ["{:<}"] * len(columns)
#         if len(columns) != len(formats):
#             raise ValueError("columns and formats must have same length")

#         self.columns = list(columns)
#         self.formats = list(formats)
#         self.padding = " " * padding
#         self.sep_major = sep_major
#         self.sep_minor = sep_minor

#         # Compute per-column widths = max(header length, inferred format width)
#         self.col_widths = []
#         for col, fmt in zip(self.columns, self.formats):
#             fmt_width = self._infer_format_width(fmt)
#             self.col_widths.append(max(len(str(col)), fmt_width))

#         # Precompute total width for consistent separators
#         self.total_width = self._total_width()

#     # ---------- public API ----------
#     def header(self) -> str:
#         major = self.sep_major * self.total_width
#         header_line = self._format_header_line()
#         minor = self.sep_minor * self.total_width
#         return f"{major}\n{header_line}\n{minor}"

#     def row(self, *values) -> str:
#         parts = []
#         for val, fmt, width in zip(values, self.formats, self.col_widths):
#             cell = self._safe_format(fmt, val)
#             if len(cell) < width:
#                 cell = cell.rjust(width)
#             parts.append(cell)
#         return self.padding.join(parts)

#     def footer(self) -> str:
#         return self.sep_major * self.total_width

#     # ---------- internals ----------
#     def _total_width(self) -> int:
#         return sum(self.col_widths) + len(self.padding) * (len(self.columns) - 1)

#     def _format_header_line(self) -> str:
#         # Right-align headers inside their column widths
#         return self.padding.join(
#             col.rjust(w) for col, w in zip(self.columns, self.col_widths)
#         )

#     def _safe_format(self, fmt: str, value) -> str:
#         try:
#             return fmt.format(value)
#         except Exception:
#             return str(value)

#     def _infer_format_width(self, fmt: str) -> int:
#         probes = (0, 0.0, "")
#         for p in probes:
#             try:
#                 s = fmt.format(p)
#                 return len(s)
#             except Exception:
#                 continue
#         try:
#             s = "{}".format("")
#             return len(s)
#         except Exception:
#             return 0
