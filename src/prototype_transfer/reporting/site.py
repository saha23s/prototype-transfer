"""The multi-page shell: page order, navigation and the HTML skeleton."""

from __future__ import annotations

from dataclasses import dataclass

from prototype_transfer.reporting import notation
from prototype_transfer.reporting.charts import esc

SITE_TITLE = "Prototype transfer: focal → soundscape bird recognition"


@dataclass(frozen=True)
class Page:
    file: str
    nav: str
    title: str
    lede: str
    body: str
    scripts: tuple[str, ...] = ()


PAGE_ORDER = (  # (file, navigation label)
    ("index.html", "Motivation"),
    ("data.html", "Data exploration"),
    ("splits.html", "Splits"),
    ("evaluation.html", "Evaluation"),
    ("formulation.html", "Problem formulation"),
    ("label-curve.html", "Label curve"),
    ("results.html", "Initial results"),
    ("trust.html", "Prototype trust"),
    ("explorer.html", "Species explorer"),
)


def render(page: Page) -> str:
    files = [f for f, _ in PAGE_ORDER]
    i = files.index(page.file)

    def link(j: int) -> str:
        f, label = PAGE_ORDER[j]
        here = ' aria-current="page"' if j == i else ""
        extra = ' class="appendix"' if f == "explorer.html" else ""
        return f'<a href="{f}"{here}{extra}>{label}</a>'

    nav = "".join(link(j) for j in range(len(PAGE_ORDER)))
    prev = f'<a rel="prev" href="{PAGE_ORDER[i - 1][0]}">← {PAGE_ORDER[i - 1][1]}</a>' if i > 0 else "<span></span>"
    nxt = (f'<a rel="next" href="{PAGE_ORDER[i + 1][0]}">{PAGE_ORDER[i + 1][1]} →</a>'
           if i + 1 < len(PAGE_ORDER) else "<span></span>")
    scripts = "".join(f'\n  <script src="{s}"></script>' for s in page.scripts)
    tooltip = '\n  <div id="tooltip" role="tooltip" hidden></div>' if page.scripts else ""
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{esc(page.nav)} · {esc(SITE_TITLE)}</title>
  <link rel="stylesheet" href="style.css">
</head>
<body>
  <p class="site-title">{esc(SITE_TITLE)}</p>
  <nav class="pages" aria-label="Pages">{nav}</nav>
  <header><h1>{page.title}</h1><p class="lede">{page.lede}</p></header>
  <main>
{page.body}
{notation.section(page.file)}
  </main>
  <footer class="pager">{prev}{nxt}</footer>{tooltip}{scripts}
</body>
</html>
"""
