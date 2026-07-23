# GitHub upload checklist

## Upload now

Upload these files/folders:

- `README.md`
- `requirements.txt`
- `.gitignore`
- `src/`
- `results/`
- `data/README.md`
- `checkpoints/README.md`
- `examples/README.md`
- `LICENSE_INSTRUCTIONS.md`

## Before publishing publicly

Replace all `TODO` fields in `README.md`:
- exact final training command,
- Python version,
- OS,
- CPU/GPU,
- CuPy/CUDA version if used,
- training time,
- official HAM10000 source,
- official HAM10000 citation.

## Optional

- Add `checkpoints/best.pkl`.
- Add a confusion matrix.
- Add a software `LICENSE`.

## Do not upload by default

- full HAM10000 image dataset,
- `.venv`,
- cache folders,
- all intermediate checkpoints,
- temporary files.
