"""Render docs/paper_draft.md to a single PDF with the figures placed inline.

    python tools/build_paper_pdf.py

Writes build/paper_draft.html and build/paper_draft.pdf.

There is no LaTeX or pandoc on this machine, so the path is markdown -> HTML -> LibreOffice.
That constrains what is achievable: LibreOffice's HTML import honours basic block layout, tables
and images, and ignores most of the rest. The output is a readable working draft, not a
typeset manuscript, and it is not what should be sent to a venue — see docs/paper_draft.md §14.

Figures are inserted after the paragraph that first references them, so the reader meets each one
where the argument needs it rather than in an appendix. A figure that is never referenced is
reported rather than appended, because an unreferenced figure is a drafting bug (all eleven are
referenced as of this writing).
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import markdown
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "docs" / "paper_draft.md"
FIGDIR = ROOT / "results" / "figures" / "paper"
OUT = ROOT / "build"

# A4 (210mm) less the 18mm side margins set in CSS, expressed in CSS pixels at 96 dpi because
# the HTML width attribute is unitless-pixel only — LibreOffice silently reads "168mm" as 168px
# and renders the plot at a third of its intended size.
CONTENT_WIDTH_PX = round((210 - 2 * 18) / 25.4 * 96)  # 174mm -> 658px

# Captions are written here rather than scraped from the plots: a figure's title says what it
# draws, a caption says why the reader is looking at it.
CAPTIONS = {
    1: "The harness: an admission layer in front of an unmodified vLLM engine. This is what lets "
       "§9 vary admission policy without touching the engine whose failure §7 characterises.",
    2: "Throughput against concurrency for every workload. The knee is workload-specific, which "
       "is why §6.2 reports that a pooled fit is meaningless.",
    3: "KV occupancy against concurrency for the Experiment A trials. All three die with 71.6% "
       "of the cache unused.",
    4: "Device memory against concurrency, climbing until the allocator has no room for the next "
       "request.",
    5: "The accounting gap. Left: what vLLM reserves at startup. Right: what the device actually "
       "holds when it dies — 5.84 GiB of KV reserved and never touched, while a 742 MiB scorer "
       "request fails with 736 MiB free.",
    6: "Aggregate SLO attainment by policy and arrival rate across the Phase-2 grid.",
    7: "Per-class SLO attainment. In the n = 10 cell `nocap` holds every class above 0.9 while "
       "`fcfs` collapses on all four.",
    8: "The fitted service law against its measurements for `phase1_mixed`, the only workload "
       "that both spans the knee and has a flat plateau.",
    9: "A KV-bound system fails along the horizontal axis, occupancy reaching 100%. This one "
       "fails along the vertical axis, N reaching `max_num_seqs`, at 28.4% ± 0.3% occupancy. The "
       "shaded band is the region KV-based admission control guards; the failures fall outside "
       "it.",
    10: "What moves with model size (KV cost per sequence, left) beside what does not (the "
        "allocation that kills the engine, right), drawn at the same scale.",
    11: "The failing allocation against speculative depth. k = 2 is drawn open because it "
        "survived: there is no failing allocation to measure.",
}

# The two display equations in the draft. A TeX renderer is out of scope for this path, and raw
# TeX in a PDF is worse than plain Unicode, so they are substituted directly.
EQUATIONS = {
    r"$$\mu(N) = \min(N\,r_0,\ \mu_{\max}), \qquad N^* = \mu_{\max}/r_0, \qquad 0 \le N \le N_{\max}$$":
        "<p class=\"eq\">μ(N) = min(N·r₀, μ_max)"
        "&nbsp;&nbsp;&nbsp;&nbsp;N* = μ_max / r₀"
        "&nbsp;&nbsp;&nbsp;&nbsp;0 ≤ N ≤ N_max</p>",
    r"$$M_{\text{scorer}} = N \cdot (k{+}1) \cdot V \cdot 4\ \text{bytes}$$":
        "<p class=\"eq\">M_scorer = N · (k+1) · V · 4 bytes</p>",
}

CSS = """
@page { size: A4; margin: 20mm 18mm; }
body { font-family: "Liberation Serif", Georgia, serif; font-size: 10.5pt; line-height: 1.45;
       color: #111; }
h1 { font-size: 19pt; line-height: 1.2; margin: 0 0 4pt 0; }
h2 { font-size: 13.5pt; margin: 20pt 0 6pt 0; border-bottom: 1px solid #bbb;
     padding-bottom: 3pt; page-break-after: avoid; }
h3 { font-size: 11.5pt; margin: 14pt 0 4pt 0; page-break-after: avoid; }
p { margin: 0 0 7pt 0; text-align: justify; }
code, pre { font-family: "Liberation Mono", monospace; font-size: 9pt; }
pre { background: #f4f4f4; padding: 7pt; border: 1px solid #ddd; white-space: pre-wrap; }
blockquote { margin: 7pt 0 7pt 12pt; padding-left: 10pt; border-left: 3px solid #bbb;
             color: #333; }
table { border-collapse: collapse; margin: 8pt 0; font-size: 9pt; width: 100%; }
th, td { border: 1px solid #bbb; padding: 3pt 5pt; text-align: left; vertical-align: top; }
th { background: #eee; }
.eq { text-align: center; font-family: "Liberation Serif", serif; font-style: italic;
      margin: 9pt 0; }
p.figimg { text-align: center; margin: 12pt 0 3pt 0; page-break-inside: avoid; }
p.cap { font-size: 9pt; color: #333; text-align: left; margin: 0 0 12pt 0; }
"""


def figure_html(n: int) -> str:
    hits = sorted(FIGDIR.glob(f"fig{n:02d}_*.png"))
    if not hits:
        print(f"  WARNING: no image found for Figure {n}")
        return ""
    # LibreOffice's HTML import ignores CSS max-width and lays images out at native pixel size,
    # which for these plots is ~2000px and runs off the page. Explicit width/height attributes in
    # mm are honoured, so compute them here from the real aspect ratio.
    with Image.open(hits[0]) as im:
        px_w, px_h = im.size
    w_px = CONTENT_WIDTH_PX
    h_px = round(w_px * px_h / px_w)
    caption = CAPTIONS.get(n, "")
    # A styled <p> rather than <figure>/<figcaption>: LibreOffice maps the HTML5 elements
    # unpredictably and justifies the caption against instruction.
    return (f'<p class="figimg"><img src="{hits[0].resolve().as_uri()}" alt="Figure {n}" '
            f'width="{w_px}" height="{h_px}"></p>\n'
            f'<p class="cap"><b>Figure {n}.</b> {caption}</p>\n')


def main() -> None:
    text = SRC.read_text()
    for tex, html in EQUATIONS.items():
        if tex not in text:
            print(f"  WARNING: equation not found, TeX will appear raw: {tex[:48]}...")
        text = text.replace(tex, html)

    body = markdown.markdown(
        text, extensions=["tables", "fenced_code", "sane_lists", "md_in_html"])

    # Place each figure after the paragraph that first mentions it. Working on the rendered HTML
    # keeps the paragraph boundaries markdown already decided.
    # Variable-width look-behind is not supported, so split on the closing tags and stitch them
    # back onto the block they belong to.
    paras = re.split(r"(</p>|</table>|</pre>|</blockquote>)\n", body)
    paras = ["".join(paras[i:i + 2]) for i in range(0, len(paras), 2)]
    placed: set[int] = set()
    out_parts = []
    for para in paras:
        out_parts.append(para)
        for n in sorted({int(m) for m in re.findall(r"Figure (\d+)", para)}):
            if n not in placed and n in CAPTIONS:
                out_parts.append(figure_html(n))
                placed.add(n)
    missing = sorted(set(CAPTIONS) - placed)
    if missing:
        print(f"  WARNING: figures never referenced in the text: {missing}")

    OUT.mkdir(exist_ok=True)
    html_path = OUT / "paper_draft.html"
    html_path.write_text(
        f"<!DOCTYPE html><html><head><meta charset='utf-8'>"
        f"<title>KV-cache capacity is not serving capacity</title>"
        f"<style>{CSS}</style></head><body>\n{''.join(out_parts)}\n</body></html>")
    print(f"wrote {html_path}  ({len(placed)}/{len(CAPTIONS)} figures placed)")

    r = subprocess.run(
        ["soffice", "--headless", "--norestore", "--convert-to", "pdf",
         "--outdir", str(OUT), str(html_path)],
        capture_output=True, text=True, timeout=600)
    pdf = OUT / "paper_draft.pdf"
    if r.returncode != 0 or not pdf.exists():
        print(r.stdout, r.stderr, file=sys.stderr)
        sys.exit("LibreOffice conversion failed")
    print(f"wrote {pdf}  ({pdf.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
