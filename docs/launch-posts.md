# Launch posts — copy/paste, tweak the voice to yours

## Reddit — r/InternetIsBeautiful (best fit)

**Title:** I made a free website that turns your photos into oil paintings — the brush strokes actually follow the shapes in the picture

**Body:**
Most "oil effect" tools just smudge your photo. This one computes the direction of edges and contours in the image (structure-tensor flow field), then paints hundreds of small brush strokes that follow those lines — so petals curve around petals, eyes get outlined properly. There's a canvas-texture option too.

- Works on photos you drop into the page
- Live sliders (stroke size, layers, canvas relief…) re-render as you tweak
- 100% in the browser (WebAssembly) — nothing is uploaded anywhere, it's fast even on phones
- Free, no account

Link: https://jordandubu.github.io/isophote/

Source code + technical details (Python/OpenCV reference + Rust/wasm port, seeded determinism): https://github.com/jordandubu/isophote

Tips: portraits and landscapes with clear shapes work best. Drop your results in the comments!

*(subreddit rules: comment with the link as well as putting it in the text — some subs remove text-only posts with links. r/InternetIsBeautiful allows it, but engaging in comments in the first hour matters a lot.)*

## Hacker News — Show HN

**Title:** Show HN: Isophote – oil-paint filter whose strokes follow the image contours (Rust→wasm, fully client-side)

**Body:**

Hi HN! I built a browser tool that turns photos into painterly renders.

The interesting part isn't the "oil look" — plenty of filters do a smudge. What's different is **stroke placement**: it derives a structure-tensor flow field from the image (orientation of the isophotes — contours of equal brightness), smooths that field coherently, then walks polylines along it with direction inertia. Strokes are colored from the source photo and jittered per stroke, layered over an edge-preserving Kuwahara base, with an optional canvas-weave normal-map relief.

The whole pipeline runs client-side: Python/OpenCV reference first, then a Rust port compiled to WebAssembly (~200 KB). The seeded RNG is bit-exact against NumPy's PCG64, so a seed + settings reproduces the identical painting in the browser and in the CLI. A 720p render takes ~0.6 s on a phone.

Try it (no upload, no account): https://jordandubu.github.io/isophote/
Code: https://github.com/jordandubu/isophote

Happy to talk about the flow-field math, the cv2 parity work, or the wasm port.

## Tweet/X

> I built a free oil-paint filter where the brush strokes actually follow your photo's contours 🎨
> 
> Runs 100% in the browser (wasm), nothing uploaded, works on phones.
> 
> test it → https://jordandubu.github.io/isophote
> 
> before/after 👇 [attach docs/example_photo.jpg + docs/example_art.png side by side]

## Product Hunt (optional, later — needs an account + maker profile)

**Tagline:** Oil-paint photos in your browser — strokes that follow the image
**Description:** isophote turns any photo into a hand-painted-looking render. Unlike smudge filters, it computes the image's contour directions and paints real brush strokes along them — layered, jittered, with optional canvas texture. Fully client-side (Rust/WebAssembly), free, no upload. Also a Python CLI.
