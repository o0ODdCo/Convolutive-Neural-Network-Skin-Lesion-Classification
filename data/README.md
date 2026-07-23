# Dataset folder

Do not upload the complete HAM10000 image dataset here by default.

Locally, you may use a structure such as:

```text
data/HAM10000/
├── HAM10000_metadata.csv
├── HAM10000_images_part_1/
└── HAM10000_images_part_2/
```

The main training script recursively searches for image files.

Before publishing:
1. Add the official dataset download source to the main README.
2. Add the official dataset citation.
3. Verify the dataset license before redistributing any images.
