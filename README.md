# HAM10000 Skin Lesion Classification — CNN from Scratch

## Overview

This repository implements a complete convolutional neural network (CNN) **from scratch**, using **NumPy** for CPU execution and optional **CuPy/CUDA** acceleration for NVIDIA GPUs.

The project targets multiclass skin-lesion classification on the **HAM10000** dataset. The neural-network building blocks, forward propagation, backward propagation, loss function, optimizer, checkpoint system, metrics, data augmentation, batching, and inference pipeline are implemented manually rather than through a deep-learning framework such as PyTorch, TensorFlow, or Keras.

The main goals of the implementation are:

- to expose the internal mechanics of a CNN instead of relying on high-level framework abstractions;
- to implement convolution efficiently through `im2col` / matrix multiplication / `col2im`;
- to support Batch Normalization, ReLU, pooling, Global Average Pooling, Dropout, Softmax Cross-Entropy, and Adam;
- to handle the strong class imbalance of HAM10000 using both class weighting and class-balanced stochastic augmentation;
- to provide reproducible train/validation splitting grouped by `lesion_id`;
- to save complete training checkpoints and resume training;
- to log detailed batch-level and epoch-level metrics;
- to run on either CPU or CUDA without changing the model code.

---

## Table of contents

1. [HAM10000 classes](#ham10000-classes)
2. [Architecture](#architecture)
3. [Tensor convention](#tensor-convention)
4. [How Conv2D is implemented](#how-conv2d-is-implemented)
5. [Neural-network layers](#neural-network-layers)
6. [Weight initialization](#weight-initialization)
7. [Loss function](#loss-function)
8. [Adam optimizer](#adam-optimizer)
9. [Forward and backward propagation](#forward-and-backward-propagation)
10. [Dataset pipeline](#dataset-pipeline)
11. [Data augmentation](#data-augmentation)
12. [Class imbalance handling](#class-imbalance-handling)
13. [Train/validation split](#trainvalidation-split)
14. [Random seeds and reproducibility](#random-seeds-and-reproducibility)
15. [Metrics and diagnostics](#metrics-and-diagnostics)
16. [Training logs](#training-logs)
17. [Checkpoints and training resume](#checkpoints-and-training-resume)
18. [CPU/GPU backend](#cpugpu-backend)
19. [Main classes](#main-classes)
20. [Main utility functions](#main-utility-functions)
21. [Repository structure](#repository-structure)
22. [Installation](#installation)
23. [Dataset setup](#dataset-setup)
24. [Training](#training)
25. [Command-line options](#command-line-options)
26. [Prediction](#prediction)
27. [Results — model 1](#results--model-1)
28. [Reproducibility record](#reproducibility-record)

---

## HAM10000 classes

The model handles the seven diagnostic classes used by HAM10000:

| Code | Class |
|---|---|
| `akiec` | Actinic Keratoses and Intraepithelial Carcinoma |
| `bcc` | Basal Cell Carcinoma |
| `bkl` | Benign Keratosis-like Lesions |
| `df` | Dermatofibroma |
| `mel` | Melanoma |
| `nv` | Melanocytic Nevi |
| `vasc` | Vascular Lesions |

The code keeps the canonical HAM10000 order whenever all detected labels belong to this set.

---

## Architecture

The default model is identified internally as:

```text
v7_two_conv_bn_gap
```

Its architecture is:

```text
Input RGB image: (N, 3, H, W)
        │
        ▼
Conv2D(3 → 32, 3×3, stride=1, padding=1)
BatchNorm2D(32)
ReLU
Conv2D(32 → 32, 3×3, stride=1, padding=1)
BatchNorm2D(32)
ReLU
MaxPool2D(2×2, stride=2)
        │
        ▼
Conv2D(32 → 64, 3×3, stride=1, padding=1)
BatchNorm2D(64)
ReLU
Conv2D(64 → 64, 3×3, stride=1, padding=1)
BatchNorm2D(64)
ReLU
MaxPool2D(2×2, stride=2)
        │
        ▼
Conv2D(64 → 128, 3×3, stride=1, padding=1)
BatchNorm2D(128)
ReLU
Conv2D(128 → 128, 3×3, stride=1, padding=1)
BatchNorm2D(128)
ReLU
MaxPool2D(2×2, stride=2)
        │
        ▼
GlobalAveragePool2D
        │
        ▼
Dense(128 → 256)
ReLU
Dropout(p=0.3)
        │
        ▼
Dense(256 → 7)
        │
        ▼
Logits → Softmax Cross-Entropy
```

Default hyperparameters:

| Parameter | Default |
|---|---:|
| Input image size | `64 × 64` |
| Channels | `32, 64, 128` |
| Dense hidden dimension | `256` |
| Output classes | `7` |
| Dropout rate | `0.3` |
| Conv kernel | `3 × 3` |
| Conv stride | `1` |
| Conv padding | `1` |
| MaxPool kernel | `2 × 2` |
| MaxPool stride | `2` |

For a `64 × 64` image, the spatial resolution evolves as:

```text
64 × 64
  ↓ MaxPool
32 × 32
  ↓ MaxPool
16 × 16
  ↓ MaxPool
 8 × 8
  ↓ Global Average Pooling
1 value per channel
```

With the default configuration (`32,64,128`, dense dimension `256`, seven classes), the network contains **322,727 trainable parameters**. BatchNorm running statistics are state variables but are not counted as trainable parameters.

### Why Global Average Pooling is used

The model does not flatten the final feature map. Instead, `GlobalAveragePool2D` averages each feature channel over its spatial dimensions:

```text
(N, C, H, W) → (N, C)
```

For channel `c`:

```text
y[n,c] = (1 / (H·W)) Σ_h Σ_w x[n,c,h,w]
```

This strongly reduces the number of parameters in the classifier head compared with a full flattening operation.

---

## Tensor convention

Images and intermediate convolutional feature maps use the **NCHW** convention:

```text
N = batch size
C = number of channels
H = height
W = width
```

Therefore an RGB batch has shape:

```text
(N, 3, H, W)
```

Images are loaded as RGB, resized with bilinear interpolation, converted to `float32`, scaled to `[0,1]`, transposed to NCHW, optionally augmented, then normalized.

---

## How Conv2D is implemented

### Principle

The convolution is not implemented with deeply nested Python loops. Instead, the code converts local image patches into columns using `im2col`, then performs the convolution as a **matrix multiplication**.

For an input:

```text
x.shape = (N, Cin, H, W)
```

and filters:

```text
W.shape = (Cout, Cin, KH, KW)
```

`im2col` rearranges all receptive fields into a matrix:

```text
cols.shape = (Cin · KH · KW, N · Hout · Wout)
```

The filters are reshaped as:

```text
W_matrix.shape = (Cout, Cin · KH · KW)
```

The forward convolution then becomes:

```text
out_matrix = W_matrix @ cols + bias
```

or mathematically:

```text
Y = W_col X_col + b
```

The result is finally reshaped back to:

```text
(N, Cout, Hout, Wout)
```

### Output dimensions

The implementation uses:

```text
Hout = floor((H + 2P - KH) / S) + 1
Wout = floor((W + 2P - KW) / S) + 1
```

where:

- `P` is padding;
- `S` is stride;
- `KH × KW` is the kernel size.

For the default convolutions (`3×3`, stride `1`, padding `1`), height and width are preserved.

### Backward convolution

During the forward pass, the layer caches:

- the original input shape;
- the `im2col` matrix.

Given the upstream gradient `dout`, the gradients are computed using matrix products:

```text
db = sum(dout)
dW = dout_col @ cols.T
dcols = W.T @ dout_col
```

Then `col2im` reconstructs the gradient with respect to the input:

```text
dx = col2im(dcols)
```

Overlapping receptive fields are correctly accumulated with indexed addition.

### Cached indexing

The index tensors required by `im2col` are cached in `_IM2COL_INDEX_CACHE` using the backend, tensor dimensions, kernel size, padding, and stride as the cache key. This avoids regenerating identical indexing structures for every batch.

---

## Neural-network layers

### `Layer`

Base interface shared by the manually implemented layers.

Main methods:

```python
forward(x, training=True)
backward(dout)
params()
grads()
state_dict()
load_state_dict(state)
```

Layers without parameters return empty dictionaries from `params()` and `grads()`.

---

### `Conv2D`

Implements a trainable 2D convolution.

Parameters:

```text
W: (Cout, Cin, K, K)
b: (Cout,)
```

Features:

- He initialization;
- configurable kernel, stride, and padding;
- `im2col` vectorization;
- matrix-multiplication forward pass;
- full manual backward pass;
- gradients for weights, bias, and input.

---

### `BatchNorm2D`

Implements channel-wise Batch Normalization for NCHW image tensors.

During training, mean and variance are computed independently for each channel across:

```text
(batch, height, width)
```

that is, axes `(0, 2, 3)`.

For one channel:

```text
μB = mean(x)
σ²B = var(x)
x_hat = (x - μB) / sqrt(σ²B + ε)
y = γ x_hat + β
```

Default values:

```text
ε = 1e-5
momentum = 0.9
```

Trainable parameters:

```text
γ = scale
β = shift
```

Running statistics are updated as:

```text
running_mean = momentum · running_mean + (1 - momentum) · batch_mean
running_var  = momentum · running_var  + (1 - momentum) · batch_var
```

During inference, batch statistics are no longer used. The layer uses:

```text
running_mean
running_var
```

The backward derivative with respect to the input, `γ`, and `β` is implemented manually.

---

### `ReLU`

Activation function:

```text
ReLU(x) = max(0, x)
```

Forward pass:

```text
x > 0  → x
x ≤ 0  → 0
```

Backward pass:

```text
dx = dout · 1[x > 0]
```

A Boolean activation mask is stored during training and reused during backpropagation.

---

### `MaxPool2D`

Implements max pooling, by default:

```text
kernel = 2 × 2
stride = 2
```

The implementation also uses `im2col` internally.

During the forward pass, the index of the maximum element of each pooling region is cached. During backpropagation, each upstream gradient is routed only to the element that produced the maximum value.

---

### `GlobalAveragePool2D`

Averages each feature channel over the complete spatial map:

```text
(N, C, H, W) → (N, C)
```

Backward propagation distributes the upstream gradient uniformly over the `H × W` spatial positions:

```text
dx = dout / (H·W)
```

for each location of the corresponding channel.

---

### `Dense`

Fully connected affine layer:

```text
y = xW + b
```

Parameters:

```text
W: (input_dim, output_dim)
b: (output_dim,)
```

Backward pass:

```text
dW = x.T @ dout
db = sum(dout, axis=0)
dx = dout @ W.T
```

Weights are initialized with Glorot/Xavier-style normal initialization.

---

### `Dropout`

Implements **inverted dropout**.

During training, each activation is kept with probability:

```text
1 - dropout_rate
```

The binary mask is scaled by:

```text
1 / (1 - dropout_rate)
```

so that no additional rescaling is required during inference.

With the default rate:

```text
p = 0.3
```

approximately 30% of hidden activations are randomly dropped during training.

During inference, Dropout is disabled and the layer returns its input unchanged.

---

### `SoftmaxCrossEntropy`

Combines stable Softmax computation with multiclass cross-entropy.

To improve numerical stability, the maximum logit is subtracted before exponentiation:

```text
z = logits - max(logits)
softmax(z_i) = exp(z_i) / Σ_j exp(z_j)
```

The unweighted loss is:

```text
L = -(1/N) Σ_i log(p_i,y_i)
```

When class weights are enabled, each sample uses the weight of its target class, and the batch loss is normalized by the sum of the selected sample weights.

The backward pass directly computes the standard Softmax/Cross-Entropy gradient:

```text
dlogits = probabilities - one_hot(target)
```

followed by the appropriate normalization and optional class weighting.

---

## Weight initialization

Two manual initialization strategies are used.

### He initialization for convolutions

`Conv2D` weights use:

```text
W ~ Normal(0, sqrt(2 / fan_in))
```

with:

```text
fan_in = Cin · KH · KW
```

This initialization is well suited to layers followed by ReLU activations.

### Glorot/Xavier-style initialization for dense layers

`Dense` weights use:

```text
W ~ Normal(0, sqrt(2 / (fan_in + fan_out)))
```

Biases are initialized to zero.

BatchNorm parameters are initialized as:

```text
γ = 1
β = 0
running_mean = 0
running_var = 1
```

---

## Loss function

The model uses **Softmax Cross-Entropy**, with optional inverse-frequency class weights.

When class weighting is enabled, the weight for class `c` is computed from the training split as:

```text
w_c = N / (K · n_c)
```

where:

- `N` is the total number of training images;
- `K` is the number of classes;
- `n_c` is the number of training images in class `c`.

Therefore rare classes receive larger weights and frequent classes receive smaller weights.

Class weighting is enabled by default and can be disabled with:

```bash
--no-class-weights
```

---

## Adam optimizer

The project contains its own `Adam` implementation.

Default optimizer parameters are:

```text
learning rate = 1e-3
β1 = 0.9
β2 = 0.999
ε = 1e-8
weight_decay = configurable, default CLI value 1e-4
```

For each trainable parameter `θ`, Adam keeps:

- `m`: exponential moving average of gradients;
- `v`: exponential moving average of squared gradients;
- `t`: optimization step counter.

The updates are:

```text
m_t = β1 m_(t-1) + (1 - β1) g_t
v_t = β2 v_(t-1) + (1 - β2) g_t²
```

Bias correction:

```text
m_hat = m_t / (1 - β1^t)
v_hat = v_t / (1 - β2^t)
```

Parameter update:

```text
θ ← θ - lr · m_hat / (sqrt(v_hat) + ε)
```

### Weight decay implementation

If `weight_decay > 0`, the code modifies the gradient before the Adam moment update:

```text
g ← g + weight_decay · θ
```

Therefore the current implementation corresponds to **L2 regularization coupled to Adam**, not decoupled AdamW weight decay.

### Optimizer state in checkpoints

The checkpoint stores:

```text
t
learning rate
β1
β2
ε
weight_decay
m for every parameter
v for every parameter
```

This allows the optimizer dynamics to be restored when resuming training.

---

## Forward and backward propagation

### Forward pass

`CNNModel.forward()` iterates through all layers in their defined order:

```python
for layer in self.layers:
    x = layer.forward(x, training=training)
```

Each trainable layer stores only the intermediate values required by its backward pass.

### Backward pass

`CNNModel.backward()` traverses the layers in reverse order:

```python
for layer in reversed(self.layers):
    dout = layer.backward(dout)
```

The training sequence for one batch is:

```text
1. Move batch to CPU/GPU backend
2. Forward pass through CNN
3. Compute Softmax Cross-Entropy loss
4. Compute dLoss/dLogits
5. Backpropagate through all layers
6. Optionally compute global gradient norm
7. Adam optimizer updates all trainable parameters
8. Update metrics and logs
```

No automatic differentiation engine is used.

---

## Dataset pipeline

### Image discovery

`index_images()` recursively searches one or more directories for:

```text
.jpg
.jpeg
.png
```

and creates a mapping:

```text
image_id → file path
```

The image ID is obtained from the filename stem.

### Metadata discovery

The code can:

- receive an explicit metadata CSV with `--metadata-csv`;
- receive the CSV itself as `--data-dir`;
- search recursively for `HAM10000_metadata.csv` below `--data-dir`.

### Metadata fields used

Only the following fields are required by the training pipeline:

```text
image_id
dx
lesion_id
```

If `lesion_id` is missing or empty, `image_id` is used as a fallback group identifier.

### Missing images

Rows whose `image_id` has no matching indexed image file are removed before splitting.

### `HAM10000Dataset`

The dataset class:

- stores metadata rows;
- maps labels to integer class indices;
- loads images lazily;
- converts images to RGB;
- resizes them with bilinear interpolation;
- converts data to contiguous `float32` NCHW arrays;
- applies stochastic augmentation during training;
- applies the selected normalization;
- returns batches of images and integer labels.

### Optional RAM cache

With:

```bash
--cache
```

resized raw images are cached in RAM.

The cache stores the resized image **before stochastic augmentation**, so augmentation is still recomputed when the image is accessed again.

---

## Data augmentation

Augmentation is enabled by default for the training set and disabled for validation.

Disable it with:

```bash
--no-augment
```

### Geometric transformations

The training transform can apply:

- horizontal flip with probability `0.5`;
- vertical flip with probability `0.5`;
- random rotation by `0°`, `90°`, `180°`, or `270°`;
- random translation with probability `0.7` when translation strength is non-zero;
- random cutout with probability `0.45` when cutout strength is non-zero.

### Photometric transformations

Depending on the preset and random draw:

- brightness modification, probability `0.8`;
- contrast modification, probability `0.8`;
- saturation modification, probability `0.6`;
- additive Gaussian noise, probability `0.7`.

The final augmented image is clipped to:

```text
[0, 1]
```

before normalization.

### Augmentation presets

| Parameter | `light` | `medium` | `strong` |
|---|---:|---:|---:|
| Gaussian noise std | `0.010` | `0.020` | `0.035` |
| Brightness amplitude | `0.10` | `0.18` | `0.28` |
| Contrast amplitude | `0.10` | `0.18` | `0.28` |
| Saturation amplitude | `0.06` | `0.10` | `0.18` |
| Max translation fraction | `0.04` | `0.06` | `0.10` |
| Max cutout fraction | `0.08` | `0.12` | `0.18` |

The default preset is:

```text
medium
```

Individual values can override the preset using:

```text
--aug-noise-std
--aug-brightness
--aug-contrast
--aug-saturation
--aug-translate
--aug-cutout
```

### Normalization modes

The code defines three normalization modes.

#### `imagenet` — default

```text
mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]
```

#### `half`

```text
mean = [0.5, 0.5, 0.5]
std  = [0.5, 0.5, 0.5]
```

This approximately maps `[0,1]` to `[-1,1]`.

#### `none`

```text
mean = [0, 0, 0]
std  = [1, 1, 1]
```

No effective normalization is applied beyond scaling the image to `[0,1]`.

---

## Class imbalance handling

HAM10000 is strongly imbalanced. The code includes two independent mechanisms.

### 1. Class-weighted loss

Enabled by default.

Rare classes receive larger Softmax Cross-Entropy weights according to:

```text
w_c = N / (K · n_c)
```

Disable with:

```bash
--no-class-weights
```

### 2. Class-balanced augmentation / oversampling

Enabled by default when augmentation is enabled.

The function `expand_rows_by_class_augmentation()` repeats metadata rows for underrepresented classes. Images are **not duplicated on disk**.

Because transformations are stochastic, repeated rows can produce different augmented versions over the course of training.

The target effective number of samples per class can be:

```text
max
median
mean
an explicit integer
```

Default:

```text
--augment-target max
```

The maximum expansion is capped by:

```text
--max-augment-repeat 8
```

so that a class with `n` original samples cannot exceed approximately `8n` effective repeated entries under the default setting.

Disable class-balanced oversampling with:

```bash
--no-balance-augment
```

---

## Train/validation split

The split is both **grouped** and **stratified by diagnosis**.

### Grouping by `lesion_id`

All images with the same `lesion_id` are kept in the same split.

This prevents images of the same physical lesion from appearing in both training and validation sets.

### Stratification logic

Lesion groups are first organized by diagnosis class. Within each class:

1. lesion IDs are shuffled with a deterministic `RandomState(seed)`;
2. approximately `val_frac` of lesion groups are assigned to validation;
3. when a class contains at least two lesion groups, the code guarantees at least one validation group and at least one training group;
4. a class represented by only one lesion group keeps that group in training.

Default validation fraction:

```text
0.15
```

---

## Random seeds and reproducibility

Default seed:

```text
42
```

The seed affects several independent components.

### NumPy global random generator

At the beginning of training:

```python
np.random.seed(args.seed)
```

This controls, among other things:

- NumPy-based parameter initialization when the CPU backend is used;
- stochastic image augmentation;
- Dropout when the NumPy backend is used.

### CuPy random generator

When CuPy/CUDA is selected, `set_backend()` applies:

```python
cp.random.seed(seed)
```

This controls GPU-side random operations such as parameter initialization and Dropout.

### Grouped train/validation split

The split uses an independent generator:

```python
np.random.RandomState(seed)
```

Therefore the split is deterministic for identical metadata and the same seed.

### Class-balanced row expansion

Oversampling also uses:

```python
np.random.RandomState(seed)
```

for deterministic row selection and ordering at expansion time.

### Batch shuffling

`BatchIterator` uses:

```text
seed + internal_epoch_index
```

for the training shuffle. Therefore successive epochs obtain deterministic but different orderings during one uninterrupted run.

### Important limitation when resuming

The checkpoint currently stores model state, optimizer state, history, configuration, BatchNorm running statistics, and class mapping, but **does not store NumPy or CuPy RNG states**.

In addition, a newly constructed `BatchIterator` starts its own internal epoch counter from zero.

Consequently, a run resumed with `--resume` is designed to continue training correctly, but it is **not guaranteed to be bit-for-bit identical to an uninterrupted run**, because the exact future shuffling, augmentation, and Dropout random sequences are not restored from the checkpoint.

---

## Metrics and diagnostics

The code derives metrics from the confusion matrix.

### Per-class metrics

For each class:

```text
precision = TP / (TP + FP)
recall    = TP / (TP + FN)
F1        = 2 · precision · recall / (precision + recall)
support   = number of true samples
```

### Global metrics

The following are computed:

- accuracy;
- error rate;
- balanced accuracy;
- macro precision;
- macro recall;
- macro F1;
- weighted precision;
- weighted recall;
- weighted F1;
- micro precision;
- micro recall;
- micro F1.

The implementation defines:

```text
balanced_accuracy = mean(per-class recall)
```

which is equivalent to macro recall.

### Confusion matrix

A complete confusion matrix is accumulated for both training and validation.

### Fit diagnostic

After each epoch, `diagnose_fit_state()` analyzes recent training and validation behavior and labels the state as one of:

```text
sur-apprentissage probable
sous-apprentissage probable
instabilite ou degradation recente
apprentissage globalement equilibre
```

It can print recommendations such as:

- increasing or decreasing Dropout;
- changing weight decay;
- changing the number of channels;
- adjusting the learning rate;
- increasing training duration;
- strengthening augmentation.

This diagnostic is **advisory only**. It does not automatically change hyperparameters, stop training, or modify the optimizer.

---

## Training logs

### Console output

The console displays:

- configuration;
- selected backend;
- dataset statistics;
- train/validation class distribution;
- class weights;
- effective oversampling ratios;
- model layers and parameter counts;
- live progress bars;
- loss;
- accuracy;
- macro F1;
- optional gradient norm;
- epoch duration;
- estimated remaining training time;
- periodic classification reports;
- confusion matrix;
- fit diagnostic.

### Batch-level CSV logging

By default, the training creates:

```text
checkpoints_numpy/batch_metrics.csv
```

or a custom path specified with:

```bash
--batch-log-csv PATH
```

Each logged batch includes identifiers and timing information such as:

```text
phase
epoch
batch
num_batches
global_batch
samples_batch
samples_seen_epoch
samples_seen_total
loss
loss_mean_epoch
grad_norm
elapsed_epoch_s
elapsed_total_s
```

For both the current batch and the running epoch totals, the CSV also stores:

- accuracy;
- error rate;
- macro metrics;
- weighted metrics;
- micro metrics;
- balanced accuracy;
- per-class precision;
- per-class recall;
- per-class F1;
- per-class support.

The file is flushed after each row, reducing the amount of log information lost if training is interrupted.

When resuming, the CSV is opened in append mode if it already exists.

Disable all batch CSV logging with:

```bash
--no-batch-log
```

Validation batches are logged by default. Disable validation batch logging with:

```bash
--no-batch-log-val
```

---

## Checkpoints and training resume

The training script creates:

```text
checkpoints_numpy/
├── best.pkl
├── last.pkl
└── batch_metrics.csv
```

### `last.pkl`

Saved at the end of **every epoch**.

It represents the latest completed training state.

### `best.pkl`

Saved only when the validation **macro-F1** strictly improves.

The model-selection metric is:

```text
val_macro_f1
```

This is particularly relevant for an imbalanced multiclass dataset because macro-F1 gives equal importance to each class.

### Checkpoint content

A checkpoint contains:

```text
epoch
best metric name
best validation macro-F1
best validation accuracy
class_to_idx / label mapping
training history
training configuration
architecture configuration
all model parameters
layer-specific states
optimizer state
```

Layer-specific states include BatchNorm:

```text
running_mean
running_var
```

Optimizer state includes the complete Adam moments.

### Atomic checkpoint writing

Checkpoint saving uses a temporary file:

```text
checkpoint.pkl.tmp
```

The temporary file is written first and then replaced atomically with:

```python
os.replace(tmp, path)
```

This reduces the risk of leaving a partially written checkpoint at the final path if the process is interrupted during serialization.

### Resume training

Example:

```bash
python src/ham10000_cnn.py train \
    --data-dir /path/to/HAM10000 \
    --epochs 50 \
    --resume checkpoints_numpy/last.pkl
```

When resuming, the code restores:

- model parameters;
- BatchNorm running statistics;
- Adam optimizer state;
- last saved epoch;
- previous best macro-F1;
- training history;
- stored configuration metadata.

Training restarts at:

```text
saved_epoch + 1
```

The code checks that the current `class_to_idx` mapping matches the checkpoint mapping.

### Important resume requirement

During `train --resume`, the model is first constructed from the **current command-line arguments**, then checkpoint parameters are loaded into that model.

Therefore the safest procedure is to resume with the same architecture-defining settings as the original run, especially:

```text
--image-size
--channels
--dense-dim
number/order of classes
```

The checkpoint configuration is stored and displayed, but the training command does not automatically overwrite all current CLI arguments with those saved values.

---

## CPU/GPU backend

The same model code supports both NumPy and CuPy through a common global numerical backend named `xp`.

### Supported backend values

```text
auto
cupy
cuda
gpu
cpu
numpy
```

### `auto`

Default behavior:

1. check whether CuPy is installed;
2. check whether a usable CUDA device exists;
3. use CuPy/CUDA if available;
4. otherwise fall back to NumPy/CPU.

### Explicit GPU mode

If `cupy`, `cuda`, or `gpu` is explicitly requested but no usable CUDA device is available, the program raises an error instead of silently switching to CPU.

### Device conversion helpers

The code provides:

```text
to_device()
to_cpu()
scalar_to_float()
state_to_cpu()
synchronize_backend()
```

Checkpoints are converted to CPU/NumPy arrays before serialization, which makes saved state independent of active CuPy device arrays.

---

## Main classes

### Console and logging classes

| Class | Role |
|---|---|
| `C` | ANSI terminal color constants, enabled only for TTY output. |
| `ProgressBar` | Displays batch progress, rate, ETA, loss, accuracy, F1, and optional gradient norm. |
| `BatchMetricLogger` | Writes detailed batch and cumulative metrics to CSV and flushes each row. |

### Neural-network classes

| Class | Role |
|---|---|
| `Layer` | Base interface for forward/backward/state methods. |
| `Conv2D` | Trainable 2D convolution using `im2col` + matrix multiplication. |
| `BatchNorm2D` | Per-channel BatchNorm with trainable `γ`,`β` and running statistics. |
| `MaxPool2D` | Max pooling with argmax-based backward routing. |
| `ReLU` | Rectified Linear Unit activation. |
| `GlobalAveragePool2D` | Spatial mean pooling over each channel. |
| `Dense` | Fully connected affine layer. |
| `Dropout` | Inverted Dropout during training. |
| `SoftmaxCrossEntropy` | Stable Softmax plus weighted/unweighted cross-entropy and gradient. |
| `CNNModel` | Composes all layers, exposes parameters, gradients, states, and architecture metadata. |
| `Adam` | Manual Adam optimizer with bias correction and optional coupled L2 weight decay. |

### Data classes

| Class | Role |
|---|---|
| `ImageTransform` | Image augmentation and normalization pipeline. |
| `HAM10000Dataset` | Lazy image loading, resizing, caching, transformation, and label conversion. |
| `BatchIterator` | Batch construction with deterministic epoch-dependent shuffling. |

---

## Main utility functions

### Backend utilities

| Function | Role |
|---|---|
| `_cupy_has_usable_device()` | Checks whether CuPy and at least one CUDA device are usable. |
| `set_backend()` | Selects NumPy or CuPy and optionally seeds CuPy RNG. |
| `using_gpu()` | Returns whether CuPy is active. |
| `backend_info()` | Returns a readable backend/device description. |
| `to_device()` | Converts arrays to the active backend. |
| `to_cpu()` | Converts CuPy arrays back to NumPy. |
| `scalar_to_float()` | Safely converts backend scalar values to Python `float`. |
| `state_to_cpu()` | Recursively converts checkpoint state to CPU arrays. |
| `synchronize_backend()` | Synchronizes the CUDA stream when using GPU. |

### Metric and reporting utilities

| Function | Role |
|---|---|
| `per_class_metrics()` | Computes per-class and aggregate metrics from a confusion matrix. |
| `fmt_classification_report()` | Formats per-class precision/recall/F1/support. |
| `fmt_global_metrics()` | Formats global metrics. |
| `fmt_confusion_matrix()` | Formats the confusion matrix. |
| `flatten_metrics()` | Flattens metrics into CSV-compatible fields. |
| `batch_metric_fieldnames()` | Generates complete batch log column names. |
| `fmt_epoch_metrics_table()` | Compares train and validation metrics. |
| `diagnose_fit_state()` | Heuristic over/underfitting or instability diagnostic. |
| `fmt_fit_diagnostic()` | Formats the diagnostic and recommendations. |
| `make_empty_history()` | Creates metric-history containers. |
| `history_append()` | Appends one history value. |
| `history_append_metrics()` | Appends all defined global metrics. |

### Convolution utilities

| Function | Role |
|---|---|
| `_im2col_indices()` | Generates and caches patch indexing tensors. |
| `im2col()` | Converts receptive fields into columns. |
| `col2im()` | Reconstructs tensor layout and accumulates overlapping gradients. |

### Initialization utilities

| Function | Role |
|---|---|
| `he_init()` | He normal initialization for convolution weights. |
| `glorot_init()` | Glorot/Xavier-style normal initialization for dense weights. |

### Checkpoint utilities

| Function | Role |
|---|---|
| `save_checkpoint()` | Saves model, optimizer, history, metadata, and architecture atomically. |
| `load_checkpoint()` | Restores parameters, layer states, optimizer, and metadata. |
| `read_checkpoint_metadata()` | Reads architecture/configuration information before building an inference model. |

### Dataset utilities

| Function | Role |
|---|---|
| `index_images()` | Recursively indexes supported image files by filename stem. |
| `find_metadata_csv()` | Locates `HAM10000_metadata.csv`. |
| `read_metadata_csv()` | Reads image ID, diagnosis, and lesion ID. |
| `filter_rows_with_images()` | Removes metadata rows whose image is unavailable. |
| `make_class_mapping()` | Builds deterministic class-to-index mapping. |
| `split_train_val_grouped_stratified()` | Creates lesion-grouped, class-stratified train/validation sets. |
| `compute_class_weights()` | Computes inverse-frequency class weights. |
| `expand_rows_by_class_augmentation()` | Oversamples rare classes by metadata repetition. |

### Training and inference utilities

| Function | Role |
|---|---|
| `grad_norm()` | Computes global L2 norm of all gradients. |
| `evaluate()` | Runs validation without Dropout or batch-stat BatchNorm updates. |
| `preprocess_single_image()` | Applies inference resize and normalization to one image. |
| `train()` | Complete training pipeline. |
| `predict()` | Loads a checkpoint and performs top-k single-image inference. |
| `build_parser()` | Defines the command-line interface. |
| `main()` | Dispatches to `train` or `predict`. |

---

## Repository structure

```text
ham10000-cnn/
│
├── README.md
├── requirements.txt
├── .gitignore
├── LICENSE_INSTRUCTIONS.md
│
├── src/
│   └── ham10000_cnn.py
│
├── data/
│   └── README.md
│
├── results/
│   ├── README.md
│   ├── epoch_metrics_model1.csv
│   ├── figures/
│   │   ├── loss_curve.png
│   │   └── macro_f1_curve.png
│   └── raw/
│       └── README.md
│
├── checkpoints/
│   └── README.md
│
└── examples/
    └── README.md
```

Runtime training output is typically written to:

```text
checkpoints_numpy/
├── best.pkl
├── last.pkl
└── batch_metrics.csv
```

---

## Installation

Clone the repository:

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd ham10000-cnn
```

Create a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

On Windows:

```bash
.venv\Scripts\activate
```

Install the required packages:

```bash
pip install -r requirements.txt
```

For NVIDIA GPU acceleration, install a CuPy build compatible with the installed CUDA runtime.

CuPy is optional. In `--backend auto` mode, the program falls back to NumPy/CPU when no usable CUDA backend is available.

---

## Dataset setup

The HAM10000 dataset is **not included in this repository**.

Example local layout:

```text
data/HAM10000/
├── HAM10000_metadata.csv
└── HAM10000_images/
    ├── ISIC_....jpg
    └── ...
```

The image folder may also contain subdirectories because image indexing is recursive.

The training code requires metadata containing at least:

```text
image_id
dx
lesion_id
```

Example command:

```bash
python src/ham10000_cnn.py train \
    --data-dir data/HAM10000
```

Alternative explicit paths:

```bash
python src/ham10000_cnn.py train \
    --data-dir data/HAM10000 \
    --metadata-csv data/HAM10000/HAM10000_metadata.csv \
    --images-dir data/HAM10000/HAM10000_images
```

---

## Training

### Minimal example

```bash
python src/ham10000_cnn.py train \
    --data-dir /path/to/HAM10000
```

### Explicit default-style configuration

```bash
python src/ham10000_cnn.py train \
    --data-dir /path/to/HAM10000 \
    --image-size 64 \
    --epochs 30 \
    --batch-size 32 \
    --lr 0.001 \
    --weight-decay 0.0001 \
    --dropout 0.3 \
    --channels 32,64,128 \
    --dense-dim 256 \
    --val-frac 0.15 \
    --augment-strength medium \
    --norm imagenet \
    --seed 42 \
    --backend auto
```

### GPU example

```bash
python src/ham10000_cnn.py train \
    --data-dir /path/to/HAM10000 \
    --backend cupy
```

### CPU-only example

```bash
python src/ham10000_cnn.py train \
    --data-dir /path/to/HAM10000 \
    --backend cpu
```

### Resume example

```bash
python src/ham10000_cnn.py train \
    --data-dir /path/to/HAM10000 \
    --epochs 50 \
    --channels 32,64,128 \
    --dense-dim 256 \
    --resume checkpoints_numpy/last.pkl
```

---

## Command-line options

### Training options

| Option | Default | Description |
|---|---:|---|
| `--data-dir` | required | Dataset directory or direct metadata CSV path. |
| `--metadata-csv` | `None` | Explicit metadata CSV path. |
| `--images-dir` | `None` | Explicit image directory. |
| `--image-size` | `64` | Square resize dimension. |
| `--batch-size` | `32` | Mini-batch size. |
| `--epochs` | `30` | Total target number of epochs. |
| `--lr` | `1e-3` | Adam learning rate. |
| `--weight-decay` | `1e-4` | Coupled L2 weight decay added to gradients. |
| `--dropout` | `0.3` | Dropout rate in the classifier head. |
| `--channels` | `32,64,128` | Channel count for the three convolutional blocks. |
| `--dense-dim` | `256` | Hidden dense layer dimension. |
| `--val-frac` | `0.15` | Validation fraction at lesion-group level. |
| `--checkpoint-dir` | `checkpoints_numpy` | Output directory for checkpoints/logs. |
| `--resume` | `None` | Checkpoint from which to resume. |
| `--cache` | off | Cache resized raw images in RAM. |
| `--no-augment` | off | Disable stochastic training augmentation. |
| `--balance-augment` | on | Enable class-balanced row oversampling. |
| `--no-balance-augment` | off | Disable class-balanced oversampling. |
| `--augment-target` | `max` | Oversampling target: `max`, `median`, `mean`, or integer. |
| `--max-augment-repeat` | `8` | Maximum repetition factor constraint. |
| `--augment-strength` | `medium` | `light`, `medium`, or `strong`. |
| `--aug-noise-std` | preset | Override Gaussian-noise standard deviation. |
| `--aug-brightness` | preset | Override brightness amplitude. |
| `--aug-contrast` | preset | Override contrast amplitude. |
| `--aug-saturation` | preset | Override saturation amplitude. |
| `--aug-translate` | preset | Override translation fraction. |
| `--aug-cutout` | preset | Override cutout fraction. |
| `--no-class-weights` | off | Disable inverse-frequency class-weighted loss. |
| `--norm` | `imagenet` | Normalization key: `imagenet`, `half`, or `none`. |
| `--report-every` | `5` | Epoch interval for detailed validation report. |
| `--grad-norm-every` | `0` | Gradient-norm frequency; `0` disables it. |
| `--batch-log-csv` | automatic | Custom batch CSV path. |
| `--no-batch-log` | off | Disable batch-level CSV logging. |
| `--batch-log-val` | on | Include validation batches in CSV log. |
| `--no-batch-log-val` | off | Exclude validation batches from CSV log. |
| `--seed` | `42` | Main random seed. |
| `--backend` | `auto` | `auto`, `cupy`, `cuda`, `gpu`, `cpu`, or `numpy`. |

### Prediction options

| Option | Default | Description |
|---|---:|---|
| `--checkpoint` | required | Checkpoint to load. |
| `--image` | required | Image to classify. |
| `--image-size` | checkpoint value | Optional override. |
| `--norm` | checkpoint value | Optional normalization override. |
| `--top-k` | `3` | Number of highest-probability classes to display. |
| `--backend` | `auto` | CPU/GPU backend selection. |

---

## Prediction

After training:

```bash
python src/ham10000_cnn.py predict \
    --checkpoint checkpoints_numpy/best.pkl \
    --image /path/to/image.jpg \
    --top-k 3
```

### Inference loading procedure

Before creating the model, the prediction code reads checkpoint metadata to recover, when available:

```text
image size
channel configuration
dense hidden dimension
normalization mode
class mapping
```

It then:

1. rebuilds the model architecture;
2. loads model weights;
3. loads BatchNorm running statistics;
4. preprocesses the image;
5. runs the network with `training=False`;
6. computes Softmax probabilities;
7. sorts probabilities and prints the top-k classes.

### Inference behavior

During `training=False`:

- BatchNorm uses saved running statistics;
- Dropout is disabled;
- no backward graph/cache is required for optimization.

The display also assigns a textual confidence category:

```text
> 0.70  → HAUTE
> 0.40  → MOYENNE
otherwise → FAIBLE
```

These thresholds are only display rules implemented in the script. The code does not perform a separate probability-calibration procedure.

---

## Results — model 1

The included `results/epoch_metrics_model1.csv` was generated from the supplied training log.

Best validation macro-F1 observed in this run:

| Metric | Value |
|---|---:|
| Best epoch | 29 |
| Validation accuracy | 0.5517 |
| Validation balanced accuracy | 0.6885 |
| Validation macro precision | 0.4818 |
| Validation macro recall | 0.6885 |
| Validation macro-F1 | 0.5170 |
| Validation weighted-F1 | 0.5938 |
| Validation loss | 1.4150 |

### Training curves

![Loss curve](results/figures/loss_curve.png)

![Macro-F1 curve](results/figures/macro_f1_curve.png)

---

## Reproducibility record

Default random seed:

```text
42
```

Example command for a final experiment:

```bash
python src/ham10000_cnn.py train \
    --data-dir ../HAM10000 \
    --backend cupy \
    --seed 42
```

Recorded environment for the referenced run:

| Item | Value |
|---|---|
| Python | `3.12` |
| Operating system | `Windows 11` |
| GPU | `NVIDIA 1060 Ti` |
| CuPy / CUDA environment | `12.8` as recorded for the run |
| Total training time | approximately `3 h` |

For rigorous experiment tracking, also record:

- exact Git commit;
- exact command line;
- dataset version and local preprocessing assumptions;
- dependency versions;
- CUDA driver/runtime version when using GPU;
- seed;
- checkpoint used for evaluation;
- whether class weighting was enabled;
- augmentation preset and overrides;
- batch size, learning rate, weight decay, Dropout, channels, and dense dimension.

---

## Summary of implemented functionality

This project implements the complete training stack required for a small CNN without a deep-learning framework:

```text
Data discovery
    ↓
Metadata loading
    ↓
Grouped stratified split by lesion_id
    ↓
Class weights + optional class-balanced oversampling
    ↓
Image loading / resize / augmentation / normalization
    ↓
Deterministic batch iterator
    ↓
Conv2D via im2col + matrix products
    ↓
BatchNorm + ReLU + MaxPool
    ↓
Global Average Pooling
    ↓
Dense + ReLU + Dropout + Dense
    ↓
Stable Softmax Cross-Entropy
    ↓
Manual backpropagation
    ↓
Adam optimization
    ↓
Detailed metrics and confusion matrix
    ↓
CSV batch logging
    ↓
Atomic best/last checkpoints
    ↓
Training resume
    ↓
Single-image top-k inference
```

The implementation is intentionally explicit: the main mathematical operations normally hidden by deep-learning libraries are represented directly in the source code.
