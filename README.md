# isophote

Flow-guided brush stroke painter. Standalone: Python + OpenCV, no editor needed.

Strokes follow the image's contours (structure-tensor flow field), long and
thin, layered with jitter over an edge-preserving Kuwahara base. Canvas weave
relief optional.

## Usage

```bash
.venv/bin/python isophote.py photo.jpg -o painted.png \
    --len 45 --width 3 --layers 4 --relief 0.1
```

## Knobs

| flag | default | meaning |
|---|---|---|
| `--len`        | 26  | brush length, px (long = 40-70) |
| `--width`      | 7   | brush thickness, px (thin = 3-5) |
| `--layers`     | 2   | stroke passes (more = denser detail) |
| `--density`    | auto | stroke count (auto ~ area/140) |
| `--base-radius`| 6   | Kuwahara base flattening |
| `--flow-smooth`| 8   | flow field coherence (higher = straighter strokes) |
| `--relief`     | 0.35| canvas weave shading (0 = off) |
| `--no-canvas`  |     | disable canvas texture |
| `--seed`       | 42  | stroke randomness |

## How it works

1. **Kuwahara base** — quadrant-min-variance flatten, edge preserving.
2. **Flow field** — structure tensor of smoothed gray; orientation =
   isophote direction, coherence = anisotropy.
3. **Coherent smoothing** — orientation-averaged so flat regions inherit
   nearby directions.
4. **Strokes** — polylines walking along the flow with direction inertia,
   color sampled from the original photo under the stroke midpoint,
   luminance jitter per stroke (loaded-brush feel), gradient-first ordering.
5. **Canvas relief** — procedural weave heightmap → normal map → lambert.

## Tuning notes

- long+thin+art: `--len 45 --width 3 --layers 4`
- chunky impasto: `--len 24 --width 9 --layers 2 --relief 0.3`
- soft pastel: `--len 30 --width 5 --relief 0 --base-radius 8`
