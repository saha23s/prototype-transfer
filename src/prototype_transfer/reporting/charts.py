"""Static inline-SVG primitives and HTML snippets shared by the dashboard pages (no JavaScript; hover text is a <title>)."""

from __future__ import annotations

import html
from typing import Callable, Iterable, Sequence

W = 900   # default SVG width; CSS scales it

SPECIES_NAMES = {
    "gcrfin": "Gray-crowned Rosy-Finch", "whcspa": "White-crowned Sparrow", "amepip": "American Pipit",
    "sposan": "Spotted Sandpiper", "rocwre": "Rock Wren", "brebla": "Brewer's Blackbird", "daejun": "Dark-eyed Junco",
    "foxspa": "Fox Sparrow", "clanut": "Clark's Nutcracker", "moublu": "Mountain Bluebird", "casfin": "Cassin's Finch",
    "mallar3": "Mallard", "herthr": "Hermit Thrush", "amerob": "American Robin", "yerwar": "Yellow-rumped Warbler",
    "yelwar": "Yellow Warbler", "dusfly": "Dusky Flycatcher", "mouchi": "Mountain Chickadee",
    "orcwar": "Orange-crowned Warbler", "warvir": "Warbling Vireo", "norfli": "Northern Flicker",
}


def name_of(code: str) -> str:
    return SPECIES_NAMES[code]


def esc(x) -> str:
    return html.escape(str(x), quote=True)


def n(x: int) -> str:
    return f"{x:,}"


def pct(x: float) -> str:
    return f"{100 * x:.0f}%"


def signed(x: float, d: int = 3) -> str:
    shown = f"{abs(x):.{d}f}"
    return ("±" if float(shown) == 0 else "+" if x > 0 else "−") + shown


def linear(d0: float, d1: float, r0: float, r1: float) -> Callable[[float], float]:
    return lambda v: r0 + (v - d0) / (d1 - d0) * (r1 - r0)


def svg(height: float, label: str, body: str, width: float = W) -> str:
    return (f'<svg viewBox="0 0 {width:.0f} {height:.0f}" role="img" aria-label="{esc(label)}" xmlns="http://www.w3.org/2000/svg">'
            f"{body}</svg>")


def text(x: float, y: float, s: str, cls: str = "", anchor: str = "start") -> str:
    klass = f' class="{cls}"' if cls else ""
    return f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}"{klass}>{esc(s)}</text>'


def line(x1: float, y1: float, x2: float, y2: float, cls: str) -> str:
    return f'<line class="{cls}" x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}"/>'


def circle(x: float, y: float, r: float, cls: str, title: str = "") -> str:
    tip = f"<title>{esc(title)}</title>" if title else ""
    return f'<circle class="{cls}" cx="{x:.1f}" cy="{y:.1f}" r="{r}">{tip}</circle>'


def rect(x: float, y: float, w: float, h: float, cls: str, title: str = "", opacity: float | None = None, rx: float = 3) -> str:
    tip = f"<title>{esc(title)}</title>" if title else ""
    op = "" if opacity is None else f' fill-opacity="{opacity:.2f}"'
    return f'<rect class="{cls}" x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{rx}"{op}>{tip}</rect>'


def x_grid(x: Callable[[float], float], ticks: Iterable[float], top: float, bottom: float,
           fmt: Callable[[float], str] = str) -> str:
    return "".join(line(x(t), top, x(t), bottom, "gridline") + text(x(t), bottom + 15, fmt(t), "", "middle") for t in ticks)


def y_grid(y: Callable[[float], float], ticks: Iterable[float], left: float, right: float,
           fmt: Callable[[float], str] = str) -> str:
    return "".join(line(left, y(t), right, y(t), "gridline") + text(left - 7, y(t) + 4, fmt(t), "", "end") for t in ticks)


def table(head: Sequence[str], rows: Sequence[Sequence[str]], cls: str = "mini", row_classes: Sequence[str] | None = None) -> str:
    """HTML table; cells are trusted HTML (callers escape data)."""
    th = "".join(f"<th>{h}</th>" for h in head)
    body = ""
    for i, r in enumerate(rows):
        rc = f' class="{row_classes[i]}"' if row_classes and row_classes[i] else ""
        body += f"<tr{rc}>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>"
    return f'<div class="table-wrap"><table class="{cls}"><thead><tr>{th}</tr></thead><tbody>{body}</tbody></table></div>'


def tiles(items: Sequence[tuple[str, str]]) -> str:
    return '<div class="tiles">' + "".join(
        f'<div class="stat-tile"><span class="value{" long" if len(v) > 9 else ""}">{v}</span><span class="label">{l}</span></div>'
        for v, l in items) + "</div>"


def legend(items: Sequence[tuple[str, str]]) -> str:
    return '<div class="legend">' + "".join(f'<span><i class="swatch {c}"></i>{esc(t)}</span>' for c, t in items) + "</div>"


def figure(fid: str, title: str, hint: str, body: str, key: str = "") -> str:
    h = f'<p class="hint">{hint}</p>' if hint else ""
    return f'<figure class="chart" id="{fid}"><figcaption><h3>{title}</h3>{key}{h}</figcaption>{body}</figure>'


def section(sid: str, title: str, body: str, hint: str = "") -> str:
    h = f'<p class="hint">{hint}</p>' if hint else ""
    return f'<section id="{sid}" aria-labelledby="{sid}-title"><h2 id="{sid}-title">{title}</h2>{h}{body}</section>'


def paper_table(tid: str, label: str, caption: str, head: Sequence[Sequence[str | tuple[str, int]]],
                groups: Sequence[tuple[str, Sequence[Sequence[str]]]], notes: str = "") -> str:
    """A numbered booktabs-style table with the caption above it; cells are trusted HTML.

    head: header rows; a cell is a string or (string, column span). groups: (group title or "", body rows).
    """
    def th(cell) -> str:
        text_, span = cell if isinstance(cell, tuple) else (cell, 1)
        return f'<th colspan="{span}" class="span">{text_}</th>' if span > 1 else f"<th>{text_}</th>"

    top = "".join("<tr>" + "".join(th(c) for c in row) + "</tr>" for row in head)
    width = sum(c[1] if isinstance(c, tuple) else 1 for c in head[-1])
    body = ""
    for title, rows in groups:
        if title:
            body += f'<tr class="group"><td colspan="{width}">{title}</td></tr>'
        body += "".join("<tr>" + "".join(f"<td>{c}</td>" for c in row) + "</tr>" for row in rows)
    foot = f'<p class="tnotes">{notes}</p>' if notes else ""
    return (f'<div class="ptable" id="{tid}"><p class="pcaption"><b>{label}.</b> {caption}</p><div class="table-wrap">'
            f'<table class="paper"><thead>{top}</thead><tbody>{body}</tbody></table></div>{foot}</div>')


def paper_figure(fid: str, label: str, caption: str, body: str, key: str = "") -> str:
    """A numbered figure with the caption below it."""
    return f'<figure class="chart pfigure" id="{fid}">{key}{body}<figcaption class="pcaption"><b>{label}.</b> {caption}</figcaption></figure>'


def takeaways(items: Sequence[str], title: str = "What this page shows") -> str:
    return f'<aside class="takeaways"><h2>{title}</h2><ul>' + "".join(f"<li>{i}</li>" for i in items) + "</ul></aside>"
