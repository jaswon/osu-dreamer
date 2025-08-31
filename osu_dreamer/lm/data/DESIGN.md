# Beatmap Tokenization and Vocabulary

### Overview
We convert an `IntermediateBeatmap` into a flat token sequence:
- Tokens represent timing, event types, positions, flags, and slider geometry.
- Map-level settings are emitted once as a context prelude and prepended to every training chunk.

### Token Categories
- Event types: `HIT_CIRCLE`, `SLIDER`, `SPINNER`, `BREAK`
- Flags: `NEW_COMBO`, `WHISTLE`, `FINISH`, `CLAP`
- Time: `TIME_SHIFT_S`, `TIME_SHIFT_MS`, `RELEASE`
- Position: `POS_COARSE(x_bin,y_bin)`, `POS_FINE(x_bin,y_bin)`
- Slider geometry: `SLIDES(count)`, `PERFECT` or `BEZIER`, then `LINE` or `CUBIC`
- Control point geometry: `DEVIATION(bin)`, `MAGNITUDE(bin)`
- Map prelude: `HP_DRAIN_RATE(v)`, `CIRCLE_SIZE(v)`, `OVERALL_DIFFICULTY(v)`, `APPROACH_RATE(v)`, `SLIDER_TICK_RATE(v)`
- Special (in vocab, not currently used by tokenizer): `BOS`, `EOS`, `PAD`

### Encoding Rules
- Sequence: Iterate `timed` events; flatten each event to tokens.
- Time:
  - Emit optional `TIME_SHIFT_S` for whole seconds, then `TIME_SHIFT_MS` for remaining ms.
  - Duration-bearing events end with `RELEASE` (no standalone `DURATION` token).
  - Zero time gaps: emit no time-shift tokens.
- Events:
  - Emit flags first (if present), then one of: `HIT_CIRCLE`, `SLIDER`, `SPINNER`, `BREAK`.
- Positions:
  - Emit `POS_COARSE` then `POS_FINE` (factorized bins), not absolute X/Y tokens.
- Sliders:
  - Emit `SLIDER`, duration via time-shift + `RELEASE`, `SLIDES(count)` (clipped to max bin), head position.
  - `PERFECT`: emit tail position and `DEVIATION`.
  - `BEZIER`: emit one or more segments:
    - `LINE`: tail position.
    - `CUBIC`: tail position, `DEVIATION`, `DEVIATION`, `MAGNITUDE`, `MAGNITUDE`.
- Map prelude:
  - Tokenizer emits HP, CS, OD, AR (rounded to 0.1), then slider tick rate (1–8). Data pipeline prepends this prelude to each chunk.

### Vocabulary Configuration
- Coordinates:
  - Bounds: `x_min/x_max`, `y_min/y_max` (with margin beyond 512x384 playfield).
  - Bins: `coarse_x_bins/coarse_y_bins`, `fine_x_bins/fine_y_bins`.
  - Bin sizes: coarse bin sizes divide evenly; fine bins subdivide coarse bins evenly.
- Time:
  - `TIME_SHIFT_S` supports up to `TIME_SHIFT_SECONDS` (default 60).
  - Each gap encodes at most one `TIME_SHIFT_S` and one `TIME_SHIFT_MS`.
- Sliders:
  - `SLIDES_BINS` for repeats; values clipped to max bin.
  - `DEVIATION_BINS` for signed angular deviation; bins are 1..N with sign.
  - `MAGNITUDE_BINS` with `MIN_MAGNITUDE`/`MAX_MAGNITUDE` using log scaling.
- Difficulty:
  - `DIFFICULTY_BINS` define 0.0–10.0 in 0.1 steps for HP/CS/OD/AR.
  - `SLIDER_TICK_RATE_BINS` cover integer 1–8.

### Constraints and Decoding
- Assertions:
  - Coordinates must lie within bounds; encoding asserts on out-of-range.
  - Coarse/fine bin divisibility must hold.
- Determinism:
  - Decoding is deterministic; unexpected tokens raise errors.
  - Durations derive from time-shift amount preceding `RELEASE`.
  - Positions reconstruct from `POS_COARSE` then `POS_FINE` bin centers.

### Slider Details
- Perfect sliders:
  - Encoded as `PERFECT` with head and tail positions plus one `DEVIATION` token.
  - The `DEVIATION` represents the signed angular offset of the circle center relative to the chord (head→tail). The magnitude is binned into `DEVIATION_BINS`; sign encodes direction.
- Bezier sliders:
  - Arbitrary Bezier curves are approximated as a poly-Bezier sequence of `LINE` and `CUBIC` segments.
  - `LINE` segments carry only the tail position.
  - `CUBIC` segments carry tail position and four parameters: two `DEVIATION` tokens and two `MAGNITUDE` tokens.
- Cubic segment parameters:
  - The two `DEVIATION` tokens encode signed angular offsets (in the head→tail local frame) for the incoming and outgoing control directions (`pc_dev`, `qc_dev`).
  - The two `MAGNITUDE` tokens encode control vector scales (`pc_scale`, `qc_scale`) relative to the chord length, log-binned between `MIN_MAGNITUDE` and `MAX_MAGNITUDE` into `MAGNITUDE_BINS`.

### Versioning
- Changes to `VocabConfig` affect token IDs and decoding; keep a version tag alongside the config when producing datasets/models. 