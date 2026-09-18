# Iranian Sign Language Recognition using LSTM

Bachelor's project on isolated Iranian Sign Language Recognition using
the ISLR101 dataset and pose/keypoint sequences.

## Dataset

The project uses ISLR101, containing 101 isolated Iranian Sign Language
classes.

The released pose representation contains 67 keypoints:
- 25 body keypoints
- 21 left-hand keypoints
- 21 right-hand keypoints

The official train/validation/test split was reconstructed on the
4,512 publicly available pose samples.

## Models

The project evaluates several lightweight recurrent models, including:

- BiLSTM with temporal mean pooling
- BiLSTM without packed-sequence processing
- UniLSTM
- Motion-aware UniLSTM
- Several ablation experiments

## Main Results

| Model | Test Accuracy | Macro-F1 | Parameters |
|---|---:|---:|---:|
| Main BiLSTM | 93.87 ± 0.80% | 0.9368 ± 0.0090 | 365,413 |
| BiLSTM without packing | 94.38 ± 0.23% | 0.9430 ± 0.0021 | 365,413 |
| UniLSTM | 92.87 ± 0.80% | 0.9243 ± 0.0090 | 182,757 |

Results are reported over three random seeds: 42, 123, and 2026.

## Notes

The ISLR101 dataset itself is not redistributed in this repository.
Users should obtain the dataset from its original source.

## Author

Shokoofa Shalchi  
Bachelor's Project — Isfahan University of Technology
