# ISLR101-LSTM

Lightweight LSTM-based recognition of isolated Iranian Sign Language using body and hand keypoints from the **ISLR101** dataset.

This repository contains the implementation and experimental notebook developed as part of a Bachelor's project in Computer Engineering at **Isfahan University of Technology**.

---

## Overview

The goal of this project is to recognize isolated Iranian Sign Language signs using body and hand keypoint sequences instead of directly processing raw RGB video frames.

The project focuses on relatively lightweight recurrent neural networks and investigates how preprocessing, variable-length sequence handling, temporal aggregation, and recurrent architecture affect recognition performance.

The main experiments are based on Long Short-Term Memory (LSTM) networks, including bidirectional and unidirectional configurations.

---

## ISLR101 Dataset

This project uses the **ISLR101** dataset introduced by H. Ranjbar and A. Taheri.

The full RGB dataset contains:

- **101** isolated Iranian Sign Language classes
- **4,614** videos
- **10** signers
- Video resolution of **800 × 600**
- Frame rate of **25 FPS**

### Original paper

> H. Ranjbar and A. Taheri,  
> **“ISLR101: an Iranian Word-Level Sign Language Recognition Dataset,”**  
> arXiv:2503.12451, 2025.

Paper:

https://arxiv.org/abs/2503.12451

Official ISLR101 repository:

https://github.com/HoseinRanjbar/ISLR101

---

## Pose / Keypoint Representation

The publicly released pose representation contains **67 keypoints** for each frame:

- 25 body keypoints
- 21 left-hand keypoints
- 21 right-hand keypoints

Each keypoint contains three values:

- x coordinate
- y coordinate
- confidence score

Therefore, each frame is represented by:

```text
67 × 3 = 201 features
```

The released pose tensors use the general layout:

```text
(C, T, V, M)
```

where:

- `C = 3` represents x, y, and confidence
- `T` is the temporal dimension
- `V = 67` is the number of keypoints
- `M = 1` is the number of persons in each sample

The maximum released sequence length is 112 frames.

---

## Important Dataset Split Note

The complete RGB version of ISLR101 contains **4,614 videos**, while the publicly released pose data contain **4,512 samples**.

The released pose files are provided in two containers containing:

```text
3,609 samples
903 samples
```

These containers do **not** directly correspond to the official train/validation/test split distributed with the RGB dataset.

Using the released pose containers directly as train and test sets can therefore introduce data leakage relative to the official split.

To avoid this issue, the sample identifiers were matched against the official ISLR101 split files and the official split was reconstructed on the subset for which released pose data are available.

The resulting split used for the main experiments is:

| Split | Official RGB Samples | Available Pose Samples | Missing Pose Samples |
|---|---:|---:|---:|
| Train | 3,229 | 3,158 | 71 |
| Validation | 462 | 452 | 10 |
| Test | 923 | 902 | 21 |
| **Total** | **4,614** | **4,512** | **102** |

Therefore, all main experiments in this repository use:

> **the official ISLR101 split restricted to the 4,512 released pose samples**

The ISLR101 dataset itself is **not redistributed in this repository**.

Users should obtain it from the original ISLR101 sources.

---

## Preprocessing Pipeline

The preprocessing pipeline used in the main experiments includes:

1. Loading the released pose tensors.
2. Converting the samples to a temporal representation.
3. Detecting and removing trailing all-zero padded frames.
4. Recording the actual sequence length.
5. Detecting valid keypoints.
6. Spatially normalizing valid x-y coordinates.
7. Keeping confidence scores unchanged.
8. Flattening each frame into a 201-dimensional feature vector.
9. Padding sequences only at the batch level.
10. Preserving the original valid sequence lengths for temporal processing.

After preprocessing, each sample is represented as:

```text
T × 201
```

where `T` is the actual number of valid frames.

---

## Spatial Normalization

Keypoint coordinates depend on the location and scale of the signer inside the video frame.

To reduce this dependency, spatial normalization is performed independently for each sequence.

The valid x-y coordinates are:

1. centered using the mean position of the valid keypoints;
2. scaled using an RMS-based spatial scale.

Confidence scores are not spatially normalized.

Invalid keypoints remain zero after normalization.

---

## Models

Several recurrent configurations and ablation experiments are evaluated.

### 1. Main BiLSTM

The main model consists of:

- Input size: `201`
- Bidirectional LSTM
- Hidden size: `128` per direction
- Temporal mean pooling
- Layer normalization
- Dropout: `0.3`
- Fully connected classification layer
- Number of classes: `101`

Total trainable parameters:

```text
365,413
```

---

### 2. BiLSTM without Packed-Sequence Processing

This model uses the same architecture as the main BiLSTM.

The difference is that the recurrent network directly receives the batch-padded tensor instead of using packed sequences.

Padding is still excluded from the final temporal mean pooling using the actual sequence lengths.

This configuration is used to investigate whether packed-sequence processing is necessary for the ISLR101 pose sequences.

Total trainable parameters:

```text
365,413
```

---

### 3. UniLSTM

A lighter unidirectional LSTM configuration is also evaluated.

The model uses:

- Input size: `201`
- Hidden size: `128`
- Temporal mean pooling
- Layer normalization
- Dropout
- 101-class output layer

Total trainable parameters:

```text
182,757
```

This is approximately **50% fewer parameters** than the bidirectional model.

---

### 4. Motion-Aware UniLSTM

An additional representation explicitly includes temporal coordinate differences:

```text
[x, y, confidence, dx, dy]
```

This results in:

```text
67 × 5 = 335 features per frame
```

The experiment investigates whether explicitly providing short-term motion information improves recognition compared with learning temporal changes directly through the recurrent network.

---

## Ablation Studies

The notebook also investigates several components of the pipeline, including:

- removing spatial normalization;
- removing packed-sequence processing;
- replacing the bidirectional LSTM with a unidirectional LSTM;
- using the last hidden state instead of temporal mean pooling;
- explicitly adding motion features.

Some ablation experiments are single-run diagnostic experiments and should not be interpreted as statistically conclusive comparisons.

---

## Training Configuration

The main training configuration uses:

| Parameter | Value |
|---|---:|
| Optimizer | AdamW |
| Initial learning rate | 1e-3 |
| Weight decay | 1e-4 |
| Batch size | 32 |
| Maximum epochs | 40 |
| Dropout | 0.3 |
| LR reduction factor | 0.5 |
| Scheduler patience | 3 |
| Early stopping patience | 8 |

The best model checkpoint is selected according to validation performance.

---

## Multi-Seed Evaluation

To reduce dependence on a single random initialization, the main model comparison is performed using three independent random seeds:

```python
SEEDS = [42, 123, 2026]
```

For every independent run, the following components are reinitialized from scratch:

- model;
- optimizer;
- learning-rate scheduler;
- data-order random generator.

This avoids carrying model or optimizer state from previous experiments.

---

## Main Results

The following values are reported as **mean ± standard deviation over three independent runs**.

| Model | Test Accuracy | Macro-F1 | Parameters |
|---|---:|---:|---:|
| Main BiLSTM | 93.87 ± 0.80% | 0.9368 ± 0.0090 | 365,413 |
| BiLSTM without packing | **94.38 ± 0.23%** | **0.9430 ± 0.0021** | 365,413 |
| UniLSTM | 92.87 ± 0.80% | 0.9243 ± 0.0090 | 182,757 |

Among the evaluated main configurations, the BiLSTM without packed-sequence processing produced the highest mean test accuracy and the lowest variation across the three runs.

However, only three random seeds were evaluated. Therefore, this result should **not** be interpreted as evidence that packed-sequence processing is generally harmful.

The supported conclusion is that, under the experimental conditions used in this project, removing packed-sequence processing did not reduce performance.

The UniLSTM contains approximately **50% fewer trainable parameters** than the BiLSTM while remaining close in recognition accuracy.

---

## Error Analysis

Error analysis is performed using predictions from all three runs of the BiLSTM without packed-sequence processing.

Among the **902 test samples**:

| Prediction Behavior | Samples | Percentage |
|---|---:|---:|
| Correct in all three runs | 830 | 92.02% |
| Incorrect in all three runs | 27 | 2.99% |
| Seed-dependent | 45 | 4.99% |

This shows that most test samples are classified consistently across independent initializations.

The notebook also includes:

- normalized confusion matrix;
- per-class recognition accuracy;
- hardest classes;
- frequent confusion pairs;
- analysis of stable and seed-dependent errors.

---

## Reference Model Sanity Check

The publicly released **ST-TR-1s** checkpoint associated with ISLR101 was also evaluated on the publicly released 903-sample pose test file.

The reproduced values were:

```text
Top-1 Accuracy: 89.81%
Top-5 Accuracy: 96.57%
Correct Top-1 Predictions: 811 / 903
```

The original ISLR101 paper reports a Top-1 accuracy of **91.58%** for ST-TR-1s.

The experiment in this repository is used only as a **sanity check**.

It should not be considered an exact reproduction of the published result because differences may exist in:

- preprocessing;
- software and library versions;
- evaluation details;
- the exact experimental protocol.

---

## Comparison with Published ISLR101 Results

The original ISLR101 work reports several appearance-based and skeleton-based baselines.

Examples include:

| Model | Input | Reported Accuracy |
|---|---|---:|
| T-TR | Keypoints | 85.49% |
| ST-TR-1s | Keypoints | 91.58% |
| S-TR | Keypoints | 93.58% |
| ST-TR | Keypoints | 94.02% |
| Appearance-based framework | RGB video | 97.01% |

These results should **not** be interpreted as directly equivalent to the results of this repository.

The experiments in this project are performed on the official ISLR101 split restricted to the **4,512 released pose samples**, while the exact protocol used by the original publication may differ.

Therefore, comparisons are provided for context rather than as claims of direct statistical superiority.

---

## Evaluation Metrics

### Accuracy

Accuracy measures the percentage of test samples whose predicted class matches the ground-truth class.

```text
Accuracy = Correct Predictions / Total Predictions
```

---

### Macro-F1

F1-score is first computed independently for every class.

The final Macro-F1 score is the unweighted mean of the F1 scores of all 101 classes.

This metric provides a more class-balanced view of model performance than overall accuracy alone.

---

### Top-5 Accuracy

Top-5 accuracy is used only for the reference-model sanity check.

A sample is considered correct if its true class appears among the five highest-scoring model predictions.

---

## Repository Structure

```text
ISLR101-LSTM/
│
├── README.md
├── requirements.txt
└── sign_language_LSTM_shalchi.ipynb
```

The notebook contains the complete experimental workflow, including:

- dataset inspection;
- released pose-data analysis;
- reconstruction of the official data split;
- preprocessing;
- variable-length sequence handling;
- model definitions;
- training;
- multi-seed evaluation;
- ablation experiments;
- reference-model evaluation;
- error analysis;
- confusion matrix generation;
- per-class analysis;
- training curves;
- final result tables.

---

## Installation

Clone the repository:

```bash
git clone https://github.com/shokufa/ISLR101-LSTM.git
cd ISLR101-LSTM
```

Install the required Python packages:

```bash
pip install -r requirements.txt
```

The main dependencies are:

```text
numpy
pandas
torch
scikit-learn
matplotlib
opencv-python
```

A CUDA-enabled PyTorch environment is recommended for training.

---

## Running the Notebook

Open:

```text
sign_language_LSTM_shalchi.ipynb
```

using one of the following environments:

- Kaggle
- Jupyter Notebook
- JupyterLab
- Google Colab

The experiments for this project were primarily executed in **Kaggle** with GPU acceleration.

Before running the notebook, update the dataset paths according to the location of the ISLR101 files in your environment.

The original ISLR101 dataset and pose files must be obtained separately.

---

## Reproducibility Notes

The notebook explicitly controls random seeds for the main multi-run experiments.

The main reported results use:

```python
SEEDS = [42, 123, 2026]
```

Each run starts from independently initialized:

- model weights;
- optimizer state;
- scheduler state;
- batch-order random generator.

The train, validation, and test subsets reconstructed from the released pose samples are mutually disjoint.

---

## Important Notes

1. The complete RGB dataset contains **4,614 videos**.

2. The publicly released pose representation contains **4,512 samples**.

3. A total of **102 RGB samples do not have corresponding released pose data**.

4. The released pose containers should not be assumed to represent the official train/test split.

5. The main experiments use the reconstructed official train/validation/test split restricted to available pose samples.

6. Numerical comparison with the original ISLR101 publication should be interpreted cautiously because the evaluation protocols are not guaranteed to be identical.

7. The dataset itself is not redistributed in this repository.

---

## Project Information

**Project Title:**  
Isolated Iranian Sign Language Recognition using Deep Learning Models Based on Body and Hand Keypoints

**Student:**  
Shokoofa Shalchi

**Supervisor:**  
Dr. Mina Amiri

**Department:**  
Department of Electrical and Computer Engineering

**University:**  
Isfahan University of Technology

**Degree:**  
Bachelor's Project in Computer Engineering

---

## Citation

If you use the ISLR101 dataset, please cite the original dataset paper:

> H. Ranjbar and A. Taheri,  
> “ISLR101: an Iranian Word-Level Sign Language Recognition Dataset,”  
> arXiv:2503.12451, 2025.

```bibtex
@article{ranjbar2025islr101,
  title   = {ISLR101: an Iranian Word-Level Sign Language Recognition Dataset},
  author  = {Ranjbar, Hosein and Taheri, Alireza},
  journal = {arXiv preprint arXiv:2503.12451},
  year    = {2025}
}
```

Paper:

https://arxiv.org/abs/2503.12451

Official repository:

https://github.com/HoseinRanjbar/ISLR101

---

## Related References

The project additionally builds on concepts from the following works:

1. H. Ranjbar and A. Taheri,  
   **ISLR101: an Iranian Word-Level Sign Language Recognition Dataset**, 2025.

2. S. Hochreiter and J. Schmidhuber,  
   **Long Short-Term Memory**, Neural Computation, 1997.

3. C. Plizzari, M. Cannici, and M. Matteucci,  
   **Skeleton-Based Action Recognition via Spatial and Temporal Transformer Networks**, Computer Vision and Image Understanding, 2021.

4. D. Li, C. Rodriguez, X. Yu, and H. Li,  
   **Word-Level Deep Sign Language Recognition from Video: A New Large-Scale Dataset and Methods Comparison**, WACV, 2020.

5. Z. Cao, T. Simon, S.-E. Wei, and Y. Sheikh,  
   **Realtime Multi-Person 2D Pose Estimation Using Part Affinity Fields**, CVPR, 2017.

6. A. Vaswani et al.,  
   **Attention Is All You Need**, NeurIPS, 2017.

7. M. Sandler et al.,  
   **MobileNetV2: Inverted Residuals and Linear Bottlenecks**, CVPR, 2018.

8. I. Loshchilov and F. Hutter,  
   **Decoupled Weight Decay Regularization**, ICLR, 2019.

---

## Author

**Shokufa Shalchi**

Bachelor's Project in Computer Engineering  
Isfahan University of Technology

GitHub repository:

https://github.com/shokufa/ISLR101-LSTM
