"""Species explorer: the interactive species and prototype views (rendered by app.js from data.js)."""

from __future__ import annotations

from prototype_transfer.reporting.runs import Results
from prototype_transfer.reporting.site import Page

BODY = """
  <section id="summary" class="tiles" aria-label="Summary"></section>
  <section class="charts">
    <figure id="class-auc" class="chart">
      <figcaption>
        <h2>Species-level detection</h2>
        <p class="hint">Class ROC-AUC per species, sorted by drop. Click a row to inspect it.</p>
        <div class="legend"></div>
      </figcaption>
    </figure>
    <figure id="proto-scatter" class="chart">
      <figcaption>
        <h2>Prototype-level detection</h2>
        <p class="hint">One dot per prototype; size = last-layer weight. Below the diagonal = worse on soundscapes.</p>
        <div class="legend"></div>
      </figcaption>
    </figure>
  </section>

  <section aria-labelledby="detail-title">
    <div class="detail-bar">
      <h2 id="detail-title">Species detail</h2>
      <label>Species <select id="species-select"></select></label>
    </div>
    <div id="species-detail"></div>
  </section>

  <section>
    <details>
      <summary>Show data table (all prototypes)</summary>
      <div class="table-wrap">
        <table id="proto-table">
          <thead><tr>
            <th>Species</th><th>Prototype</th><th>Weight</th><th>AUC focal</th><th>AUC soundscape</th>
            <th>ΔAUC</th><th>Median act. focal</th><th>Median act. soundscape</th>
            <th>Share focal</th><th>Share soundscape</th>
          </tr></thead>
          <tbody></tbody>
        </table>
      </div>
    </details>
  </section>

  <section aria-labelledby="caveats-title">
    <h2 id="caveats-title">Caveats</h2>
    <ul id="caveats"></ul>
  </section>
"""


def build(res: Results) -> Page:
    return Page("explorer.html", "Species explorer", "Species explorer",
                "Appendix. The pretrained model on focal and soundscape clips outside the test split, species by species and prototype "
                "by prototype (species with at least 10 soundscape positives there).",
                BODY, ("data.js", "names.js", "app.js"))
