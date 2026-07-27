---
name: pdf-create
description: Create polished PDF reports and summaries with document_create. Use for any new .pdf artifact, especially research reports, overviews, briefs, and other multi-section documents that must be saved in the Mana-Agent launch directory and visually verified before delivery.
---

# PDF Creation

Create the final PDF directly in the Mana-Agent launch directory. Use a stable,
descriptive basename such as `OpenClaw_Research_Overview.pdf`; never target
`~/.mana/artifacts`, a cache directory, or a temporary directory.

## Workflow

1. Convert the source material into a clear document hierarchy before writing.
2. Call `document_create` with `file_type="pdf"`, `overwrite=false`, and a
   repository-relative basename.
3. Prefer structured content:
   - `title`: concise report title.
   - `subtitle`: scope or report type.
   - `paragraphs`: introductory paragraphs.
   - `sections`: objects containing `heading`, `paragraphs`, and `bullets`.
   - `tables`: optional row arrays for compact comparisons.
4. Keep claims grounded in the supplied evidence. Preserve citations as readable
   text or URLs and disclose incomplete source material instead of reconstructing it.
5. Read the created PDF and render every page to images. Inspect wrapping,
   margins, page breaks, headings, tables, footers, and glyphs.
6. Do not report completion until the PDF is legible and has no clipped,
   overlapping, truncated, or placeholder content.

## Formatting

- Use a restrained professional palette, strong title hierarchy, consistent
  margins, comfortable leading, and page numbers.
- Use short paragraphs and meaningful section headings.
- Use ASCII hyphens for bullets and punctuation that the embedded fonts support.
- Allow content to paginate; never truncate input to fit a page.
- Never invent citations, product capabilities, or missing conclusions.

## Failure handling

Stop without creating a fallback artifact when the skill cannot be loaded, the
target escapes the launch directory, the target already exists, required content
is missing, PDF generation fails, or verification cannot establish readability.
