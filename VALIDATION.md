# Validation

Validated on 2026-09-07 with a fresh Python 3.14 virtual environment and the
pinned dependencies.

- `make setup`: passed.
- `make test`: all five contour contract tests passed.
- `make smoke`: passed using 12 synthetic grayscale/color frames at 24 fps.
- Synthetic lossless master: every decoded BGR frame matched the reference.
- Synthetic soundtrack: 72,000 decoded PCM bytes matched exactly.
- Review media generation: passed.
- Isolated render: passed with no reference video in the staging project;
  the resulting MP4 hash matched the first render.
- Source distribution: ASCII-only filenames and text; no embedded media,
  source-derived geometry, private workspace paths, or later branding edits.

The 60-second source-media extraction was not rerun for this source-only
release. Full-film validation is performed locally with `make all` after
supplying the reference. This file records the packaging validation, not a
claim that an arbitrary reference will match a previously rendered film.
