# docs/ — the live demo

`index.html` replays a recorded self-heal run (`timeline.json`). It is published
via GitHub Pages (see `.github/workflows/pages.yml`) at:

**https://maxmilliamokafor.github.io/driftguard/**

- `index.html` — the interactive demo (no backend, no install)
- `timeline.json` — the recorded run the demo reads (synced from `../results/`)
- `preview.svg` — the static preview shown in the root README (links to the demo)

To preview locally: `python -m http.server -d docs` then open
<http://localhost:8000>.
