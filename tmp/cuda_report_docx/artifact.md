# Experiment Analysis template contract for CUDA C report

## Reference

- Retained DOCX: `C:\Users\Administrator\.codex\plugins\cache\openai-curated-remote\openai-templates\0.1.1\skills\artifact-template-experiment-analysis\assets\reference.docx`
- SHA-256: `D823CD0115186B34C01C6E4B4DA3BE28B64EE73CAC849DBD62D6F4BB6385B0FB`
- Package size: retained file at the path above; do not modify.
- Render: 7 pages, inspected from `tmp/cuda_report_docx/template-reference-render/page-1.png` through `page-7.png`.
- Evidence: `tmp/cuda_report_docx/template-style-evidence.json` and section audit output from the packaged documents skill.

## Page system

- One section, US Letter portrait, 8.5 × 11 in.
- Margins: 1.0 in on all sides.
- Section start: new page; no first-page-only or odd/even header variant.
- Header/footer are not linked to a previous section. Footer carries the report name at left and a page-number field at right.
- Footer distance and section geometry must remain inherited from the retained reference.

## Typography and colors

- Body system: Georgia for Latin text. Chinese content may use Microsoft YaHei as the East Asian font while retaining the reference font for ASCII/hAnsi.
- Title style: 40 pt, bold, centered. Retain the reference cover-page hierarchy, whitespace, and central placement. Use black for the final Chinese title per the document quality contract.
- Heading 1: Georgia 18 pt bold, reference green `#1A703A`, 18 pt before and 4 pt after.
- Heading 2: Georgia 16 pt, centered in the reference; use only for the small cover label or a comparable role.
- Heading 3: centered reference subline role on the cover.
- Body paragraphs: approximately 11–12 pt, justified or left aligned, generous paragraph spacing.
- Tables: thin light-gray borders; green header text and a light gray header fill consistent with the reference.

## Components and content flow

- Cover page: small top label, large two-line title, author/team and date, footer report name/page number.
- Subsequent pages: green Heading 1 section titles, body paragraphs, light-grid tables, lists, figures with captions.
- Rewrite the English experiment placeholders into a Chinese technical progress report using this order: executive summary; objective and context; hypothesis and design; implementation completed; validation; outcomes; limitations; interpretation and decision; next actions; PPT mapping; evidence appendix.
- Reuse the title, heading, list, body, table, footer, and page-number patterns from the source. The source's business-experiment field labels are semantic placeholders and may be replaced or removed.

## Slot map

- `word/document.xml`: all body paragraphs and body tables are editable; replace every English placeholder and source sample with the CUDA C report content. Preserve the final `sectPr` and inherited geometry.
- `word/footer1.xml`: replace only the visible `Report Name` text with `CUDA C 算子改写`; preserve the page-number field and layout.
- `word/header1.xml`: preserve unchanged.
- `word/styles.xml`: preserve the reference styles; only add an East Asian font mapping when needed for Chinese readability and set Title color to black.
- `word/numbering.xml`: preserve and reuse existing bullets/numbered-list definitions.
- `word/settings.xml`, theme, font relationships, embedded fonts, content types, and root relationships: preserve unless python-docx must add relationships for the report figures.
- `word/media/image1.png`: preserve; new figure parts may be added for the four report visuals.

## Package preservation

- Baseline package has 15 parts. Preserve all baseline parts and relationships. Expected additions are only figure media plus their `document.xml.rels` entries and content-type declarations.
- Preserve-only parts and baseline SHA-256 values:
  - `word/header1.xml` — `a06e51104c1d1da7889e9d733c189bf2267fff63e556f196bbfd45e4a51079b3`
  - `word/numbering.xml` — `7e32d1fff31b749d1ca67e8033e2f3b294b6336df93dd0061e50ba466fbd5edb`
  - `word/settings.xml` — `fb088d0a3dbf7f5e5a5a399b898acf16aa2a6fc3b2984bc3eb0d4946f73ac005`
  - `word/fontTable.xml` — `f0ecf22349646bb37f6c6e3a3ad0ea4abcfbef6143653cf98e3fb405ad8656ea`
  - `word/_rels/fontTable.xml.rels` — `18b9c91ee35007ae3d239898d32c7287b7eb5574de0f45216e4cb95ee560d2a6`
  - `_rels/.rels` — `77494e8e16bbf29213e494792349e3d49d2397a4007b2816a058337d804e79a1`
  - embedded Noto Sans Symbols font files, theme, and baseline image must remain present.
- Editable baseline parts: `word/document.xml`, `word/footer1.xml`, and `word/styles.xml` under the limits above. `word/_rels/document.xml.rels` and `[Content_Types].xml` may gain only figure relationships/types.

## Fidelity gates

- Retained reference must still match the recorded SHA-256 after authoring.
- Final document keeps one Letter portrait section, 1 in margins, footer structure, page-number field, reference green heading system, and light-grid table treatment.
- No source placeholders or English template sample text remain.
- All figures and captions stay together, fit inside the 6.5 in text width, and do not clip.
- Render every final page through Microsoft Word PDF export and inspect every PNG at 100% for overlap, missing Chinese glyphs, broken tables, and unexpected pagination.
