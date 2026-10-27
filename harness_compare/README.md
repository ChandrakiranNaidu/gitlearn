# Structural harness PDF comparison (v6)

Compares two electrical harness / wiring-diagram PDFs **structurally**, not pixel by pixel.
Branches may be shifted, stretched, re-bent or rearranged between drawings; the comparison
relies on connectivity, junction identities, terminal identities, globally owned wire numbers
and the ordered content of every unbranched segment. Absolute page coordinates are only used
for extraction, text association and drawing the result images.

| Script | Purpose | Main outputs |
|---|---|---|
| `junction_depth1_ids.py` (Script 1) | Vector extraction engine; junction dots and depth-1 junction IDs such as `DOT1[41\|65\|86]` | `<pdf>_p<page>_junctions.json`, `_wires_text.png`, `_junctions.png` |
| `build_logical_segments.py` (Script 2) | Connectivity graph, logical boundary-to-boundary segments, initial key-based comparison | `pdf_a_segments.json`, `pdf_b_segments.json`, `segment_comparison_initial.json`, images |
| `compare_and_render_segments_v6.py` (Script 3) | Strict global text ownership, terminal enrichment, fuzzy one-to-one matching, final classification | `segment_comparison_v6.json`, corrected segment JSONs, `pdf_a/b_segment_changes_v6.png` |

| `batch_compare.py` | Multi-page driver: pairs the pages of two PDFs, runs Scripts 2 and 3 for every page pair | one folder per page pair, `batch_summary.csv/.json` |
| `harness_compare_app.py` | Desktop application (window) on top of `batch_compare.py`; builds into `HarnessCompare.exe` | same as `batch_compare.py` |

Script 2 imports Script 1 (keep both files in the same folder). Script 3 is a single stand-alone
file that only needs the two PDFs and their segment JSON files.

## Install

```bash
pip install -r requirements.txt        # pymupdf, opencv-python-headless, numpy
```

Tested with Python 3.12 / 3.13, PyMuPDF 2.0, OpenCV 5.0 and NumPy 2.5.

## Desktop application (Windows .exe)

Build once on a Windows PC with 64-bit Python 3.10+ from python.org: double-click
`build_exe.bat` (or run it in a command prompt in this folder). It creates a private virtual
environment, installs the requirements and PyInstaller, and writes `dist\HarnessCompare.exe`,
a single file that runs on other Windows PCs without Python. `build_exe.sh` does the same on
macOS / Linux. Without building, `python harness_compare_app.py` opens the same window.

Using the application:

1. **Old PDF (A)** and **New PDF (B)**: choose the files (single or multi-page). The page count is
   shown next to each. The output folder is filled in automatically (it can be changed).
2. **Page pairing**:
   * *Automatic*: when one PDF has a single page and the other several, that page is compared
     with every page of the other PDF. Otherwise pages are compared page by page (1-1, 2-2, ...),
     and extra pages are listed as "only in A" / "only in B".
   * *Page by page*: always 1-1, 2-2, ...
   * *Every page against every page*: for PDFs whose page order differs.
   Whenever a page is compared with several pages, its closest partner is marked **Best match**.
3. **Compare**: progress and a log are shown, and the work can be cancelled. Each page pair
   then appears as a row: *No differences*, *Differences found* (with the
   changed / review / removed / added counts) or *Skipped* (a page without a harness drawing,
   such as a cover sheet or parts list). Double-click a row to open its two marked pages;
   the buttons open the pair folder, the summary CSV or the output folder.

Output folder: `A_p01_vs_B_p01/` etc. (Script 3 results and marked images of each pair, plus a
`log.txt`), `segments/` (segment JSON of every page, computed once per page), and
`batch_summary.csv` / `batch_summary.json`.

The same runs without a window:

```bash
python batch_compare.py --pdf-a old.pdf --pdf-b new.pdf --out results [--pairing auto|page-by-page|all-pairs] [--zoom 2]
HarnessCompare.exe --pdf-a old.pdf --pdf-b new.pdf --out results        # no console output; see results
```

Notes:

* The single-file .exe unpacks itself on every start, which takes a few seconds. For faster
  starts, replace `--onefile` with `--onedir` in the build script and ship the whole
  `dist\HarnessCompare` folder.
* Some antivirus tools are suspicious of new unsigned PyInstaller programs. If yours is,
  allow the file or sign it.
* Text box geometry is fixed by the scripts themselves (PyMuPDF quad corrections off), so
  results are the same with or without optional PyMuPDF add-ons, in Python and in the .exe.

## Run

```bash
# 1. segments for both PDFs (always regenerate them for the exact PDF pair you compare)
python build_logical_segments.py --pdf-a old.pdf --pdf-b new.pdf --out segments_pair

# 2. final comparison
python compare_and_render_segments_v6.py \
  --pdf-a "old.pdf" \
  --pdf-b "new.pdf" \
  --segments-a "segments_pair/pdf_a_segments.json" \
  --segments-b "segments_pair/pdf_b_segments.json" \
  --out "comparison_output"

# optional: junction diagnostics for one PDF
python junction_depth1_ids.py --pdf old.pdf --out junctions_old
```

Common options: `--page` (zero-based, default 0; `--page-a` / `--page-b` pick different pages of
the two PDFs), `--zoom` (default 3.0),
`--mark-same` (Script 3: also draw unchanged segments). Calibration overrides (all default to
`auto`, measured per PDF): `--wire-width`, `--label-max-side`, `--label-side-h {below,above,on-wire}`,
`--label-side-v {right,left,on-wire}`, and `--no-deskew`.

A segment JSON written by Script 2 carries the SHA-256 of its PDF. Script 3 refuses a JSON
that belongs to another PDF or page. JSON files from another engine (no fingerprint) are checked
geometrically instead: their polylines must lie on the PDF's vectors, and an unrotated or
y-up coordinate frame is detected and converted automatically.

## Sheet borders, title blocks and cropped sheets

Two versions of a drawing may differ only in presentation. One may have the sheet frame,
zone markers and title block while the other has none, or one may be cropped. All scripts
treat this as "no difference":

* The drawing frame is detected as a large axis-aligned rectangle (one path or four lines,
  any stroke width, roughly page-sized, enclosing most of the text). Frame edges and lines in
  the margin band outside it (zone ticks) are never wires. Margin text (zone numbers and
  letters) is never a wire number or terminal name. `--wire-width auto` ignores border lines.
* Wire-width components with no junction dot or circle, only horizontal and vertical lines,
  and either closed cells or all free ends resting on the frame (title block, revision
  table) are dropped from the harness graph.
* Text is read as whole words. A name cut by the crop box is read in full, and text of
  other sheets placed off the page is ignored.
* Script 3 also sets aside segments of a foreign segment JSON that lie on the frame or in
  the margin. They are listed under `inputs.sheet_border_a/b.ignored_segments`.
* The legend is drawn in a strip below the page image, never over the drawing.

## Different drawing styles: per-PDF calibration

No threshold is tied to the point sizes of one drawing. Every PDF is measured first, and each
distance rule is the value tuned on the reference drawing multiplied by (unit of this PDF / unit
of the reference drawing). The two PDFs of a comparison are calibrated independently, so A and B
may differ in style.

| Variation | How it is measured / handled |
|---|---|
| Page scale | All distances are in the PDF's own units. The layout unit (spacing scale) is the median of three independent cues: text size, label placement distance, and pen style (wire pen and dot size combined). So one oddly styled feature, or a fine pen together with small dots, does not distort it. |
| Wire stroke width | The wire pen is the stroke width whose lines carry the numeric labels and meet at junction dots. Frames, zone ticks, tables and title blocks never meet at junction dots. Extra pens (for example bundles drawn thicker) are accepted when they carry labels in the wire-number font. Script 3 uses the pen of the PDF lines under the segment polylines. |
| Text bounding-box size / font | The text unit is the median font size of the numbers sitting next to wires. Zone numbers, table entries and notes in other fonts are ignored, and font box padding does not matter. |
| Text-to-wire distance and side | Measured. Per orientation the label side is the majority side: below / above / on the wire for horizontal wires, right / left / on the wire for vertical ones. A label may be up to 2.4 x the typical label distance away. |
| Text orientation angle | The angle tolerance follows how far the texts deviate from the axes (99th percentile + 3 deg, between 12 and 25 deg). Vertical text may read either way, and "above / below" checks for names and captions use the text's own reading frame. |
| Circle diameter | The junction-dot size is the median outer diameter of filled dots where 3+ wires meet. Dot and element (large circle) limits are relative to it. |
| Vector endpoint positions | The endpoint tolerance comes from the natural break in the gaps between wire ends (capped). Facing free ends with only text between them are bridged. |
| Vector segmentation | Collinear overlapping, split or duplicated pieces are merged, and vertices where the route does not bend are dropped. |
| Drawing rotated off the page axes | The dominant line angle is detected, the analysis runs in a straightened frame, and all outputs are in page coordinates. |
| Leaders vs hatching | A non-wire line counts (leader, tick, symbol) when its pen is at least 4 % of the layout unit or it is long. On the reference drawing this sits midway between hatching (2.5 %) and leaders (6 %), whatever pen the wires use. Closed outlines are never leaders. |
| Terminal names | Ranked by distance, with soft costs for text below the terminal or off its line in the text's own reading frame. There is no hard cut-off, so nearby names do not flip on small shifts. One text names one terminal. |
| Several sheets on one page | The wire-number font is taken from the numbers next to wires, so the zone numbers of other sheets do not set the text unit. |

The measured profile is printed and stored: `profile` in the segment JSON, and
`inputs.calibration_a/b` plus `config_a/b` in `segment_comparison_v6.json`. Any measured value
can be fixed from the command line (see Common options).

Limits:

* **Wire pen:** wires must use a pen that differs from leader / symbol pens by more than about
  6 %. Otherwise a leader is indistinguishable from a wire.
* **Labels near the distance limit:** a label close to the maximum allowed distance can be
  `review` in one drawing and `same` in another whose labels sit generally further from
  their wires.
* **Stacked extremes:** in random stress tests that combine several extremes at once, 112 of
  114 behaved. The two misses were wire ends displaced by about two pen widths on a 0.4 x
  scaled sheet, and a label crammed 1.5 pt below a neighbouring wire with a 25 % smaller font
  on top. Both produced extra findings (false alarms) rather than hiding the planted changes.

## Drawing conventions the rules rely on

* Harness wires are stroked line vectors of one stroke width (0.539 pt in the supplied drawing).
* Junction = small filled dot where 3+ wires leave; large circles (H8P802, P8Q601, ...) are
  circular elements; boxes / ground symbols at wire ends are terminals.
* Horizontal wire: its number is horizontal text **below** the wire (larger y), close to it,
  and its centre projects onto that straight piece (the side is measured per PDF; below is the
  default when a PDF has too few labels to tell).
* Vertical wire: its number is text rotated about 90 degrees on the **right** of the wire
  (likewise measured; right is the default).
* Angled pieces never claim a number; the neighbouring horizontal / vertical piece owns it.
* Annotations (CLAMP, MARKER, ROUTE, LABEL, BRAID, SUPPORT) hang on leader lines and are
  treated as review-only content.

All of these are applied in the displayed page frame (page `/Rotate` taken into account),
straightened when the whole drawing is rotated off the page axes.

## How Script 3 decides

1. Every segment polyline is split into straight pieces: bends, plus breakpoints, leader feet
   and ticks found on the wire. Each piece knows its segment, piece index, orientation, length
   and cumulative position.
2. All numeric text objects are tested against all pieces with the strict side / orientation /
   projection rules. Valid candidates are scored with
   `perpendicular distance / max + 0.5 * centre error / half length`. They are sorted
   page-wide and each text object is assigned to one piece only (global ownership).
   Assignments, scores and rejected alternatives are kept in the JSON.
3. Sequences are rebuilt by cumulative distance along the whole route. The original sequence
   is kept. If the original A/B sequences are identical but the rebuilt ones differ, the pair is
   **same** with the reason "The original extracted sequences are identical. A rebuilt
   extraction difference was ignored."
4. Terminal / element names are re-chosen with global ownership (one text object names one
   endpoint). Numbers, `2M`, `8M`, spec codes such as `EN6049-007` and annotation words are
   rejected. Junction depth-1 IDs are rebuilt from the corrected sequences; original IDs are kept.
5. Nodes are matched A to B by identity (terminal names, junction label multisets with
   `UNLABELLED` as a half-credit wildcard) and then by **neighbour connectivity**. A junction
   whose own labels changed is still recognised when its neighbours correspond. Junctions are
   also compared on their other directions only, so a changed label of the segment itself
   never hides its endpoints.
6. Segment pairs are scored on endpoint compatibility (both directions), content similarity
   (edit distance, forward and reversed) and wire-number similarity. They are matched one-to-one,
   best first. Before anything is called removed or added, route merges and splits (a junction
   removed or added in the middle of a route) are recovered as review items.
7. Classification:

| Category | Meaning | Image colour |
|---|---|---|
| `same` (same exact) | endpoints and ordered contents identical | not drawn |
| `probable_same` | contents identical; endpoints differ slightly (junction signature, UNLABELLED, OPEN index, uncertain terminal) | not drawn |
| `changed` | same logical segment, meaningful wire number / component / terminal identity changed | orange |
| `review` | annotation-only difference, extraction uncertainty, or route merge / split | yellow |
| `removed` | segment of A with no compatible segment in B | red (PDF A image) |
| `added` | segment of B with no compatible segment in A | green (PDF B image) |

Every finding has an ID (`CHG-001`, `REV-002`, `DEL-001`, `ADD-001`, ...). The same ID is
printed next to the highlighted polyline in the images. Changed items carry the edit script
(`WIRE:459 changed to WIRE:460`), confidence, endpoint confidence, content similarity,
content source and the content-reversed flag.

## Verification

```bash
python tests/verify_pipeline.py
```

This builds synthetic harness PDFs that reproduce the known false positives and runs Scripts 2
and 3 end to end. It checks:

* parallel branches 85 / 87 / 88 do not borrow each other's numbers
* one visible 66 next to two branches is used once
* labels above or left of a wire are never used
* `EN6049-007` is never a wire number
* 105 is kept although it is part of `DOT1[51|60|105]`
* the 32 / 41 bent branch is resolved correctly
* reversed traversal and a shifted drawing do not create changes
* 51 -> 57 between two junctions whose IDs both change is reported as a change, not removed + added
* a copy with a wire-width frame, zone ticks and numbers, a title-block table, and a crop cutting a
  terminal name produces no findings at all
* presentation variants (scaled + skewed with a heavier pen, larger font, bigger dots and split
  wires; turned 90 deg with a lighter pen and smaller font) give no findings against the
  original, and the same findings as the original pair
* multi-page PDFs: the page pairing rules; a single page compared with a 3-page PDF finds the
  identical page as best match and the edited page's changes, and skips a page without a harness

It also covers compilation, the regexes, the side rules, global text and terminal ownership,
output files, and the absence of HTML markup in the sources.

### Checking your own drawings against style variations

```bash
python tests/run_variations.py --pdf-a old.pdf --pdf-b new.pdf --out variation_runs
```

This redraws both PDFs in 18 presentation variants (`tests/make_variation.py`):

* page scale 0.5 / 1.8, wire pen x2 / x0.5, and font x0.7 / x1.4;
* labels moved further out, mirrored to the other side, rotated a few degrees, or read top-down;
* dots x2 and circles x0.7;
* wire ends moved, and wires split into overlapping or duplicated pieces;
* skewed 7 deg, turned 90 deg, and a combination of these.

For each variant it checks that A versus V(A) reports nothing and that A versus V(B) reports
the same findings as A versus B. On the supplied drawing all 18 variants behave. Under the
label-moving variants, label 87 (planted 8 pt out of range) sits within that drawing's own
label range and is reported `NOTE`, as described in the script.

On the supplied drawing (`6866VB`), a modified copy with eight known edits was used during
development. The edits were: 459 -> 460, two branches deleted, one branch and junction added,
a CLAMP deleted, 1990VC-A renamed, label 87 moved out of range, and label 66 nudged. Script 3
reported exactly those edits: 2 changed, 2 removed, 1 added, plus review items for the
annotation, the uncertain label and the merged / split routes. All other segments came out
same or probable-same, including when the new drawing was shifted and scaled, and when a
foreign-format JSON in unrotated coordinates was supplied. A page carrying 16 copies of the
sheet at 0.774 scale (frames and zone numbers included) gave exactly 16 x those findings.
