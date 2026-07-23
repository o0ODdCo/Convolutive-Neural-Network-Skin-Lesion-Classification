# Checkpoints

The training script generates `best.pkl` and `last.pkl` in the checkpoint directory.

Recommended for GitHub:
- Upload `best.pkl` only if its size is reasonable and you want users to run inference immediately.
- Do not upload every intermediate checkpoint.
- If the model file is large, use a GitHub Release, Git LFS, or external storage instead.

Before uploading a checkpoint, record in the main README:
- exact training command,
- best validation metric,
- epoch,
- preprocessing/normalization,
- architecture configuration.
