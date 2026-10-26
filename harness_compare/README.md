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

Script 2 imports Script 1 (keep both files in the same folder). Script 3 is a single stand-alone
file that only needs the two PDFs and their segment JSON files.

## Install

```bash
pip install pymupdf opencv-python-headless numpy
```

Tested with Python 3.13, PyMuPDF 2.0, OpenCV 5.0 and NumPy 2.5.

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

Common options: `--page` (zero-based, default 0), `--zoom` (default 3.0),
`--wire-width` (default `0.539`, or `auto`), `--label-max-side` (default 10.5 pt),
`--projection-margin` (default 2.0 pt), `--mark-same` (Script 3: also draw unchanged segments).

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

## Drawing conventions the rules rely on

* Harness wires are stroked line vectors of one stroke width (0.539 pt in the supplied drawing).
* Junction = small filled dot where 3+ wires leave; large circles (H8P802, P8Q601, ...) are
  circular elements; boxes / ground symbols at wire ends are terminals.
* Horizontal wire: its number is horizontal text **below** the wire (larger y), close to it,
  and its centre projects onto that straight piece.
* Vertical wire: its number is text rotated about 90 degrees on the **right** of the wire.
* Angled pieces never claim a number; the neighbouring horizontal / vertical piece owns it.
* Annotations (CLAMP, MARKER, ROUTE, LABEL, BRAID, SUPPORT) hang on leader lines and are
  treated as review-only content.

All of these are applied in the displayed page frame (page `/Rotate` taken into account).

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

It also covers compilation, the regexes, the side rules, global text and terminal ownership,
output files, and the absence of HTML markup in the sources.

On the supplied drawing (`6866VB`), a modified copy with eight known edits was used during
development. The edits were: 459 -> 460, two branches deleted, one branch and junction added,
a CLAMP deleted, 1990VC-A renamed, label 87 moved out of range, and label 66 nudged. Script 3
reported exactly those edits: 2 changed, 2 removed, 1 added, plus review items for the
annotation, the uncertain label and the merged / split routes. All other segments came out
same or probable-same, including when the new drawing was shifted and scaled, and when a
foreign-format JSON in unrotated coordinates was supplied.
