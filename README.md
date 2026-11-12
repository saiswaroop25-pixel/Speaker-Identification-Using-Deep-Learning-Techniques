# Speaker Identification Using Deep Learning Techniques

Identifies who is speaking from an audio clip by extracting MFCC features and comparing classical machine-learning models with neural networks.

## Approach

1. **Feature extraction** – 13 MFCCs per clip via `librosa` (averaged over time for classical models; kept as sequences for the CNN).
2. **Classical baselines** – SVM, Random Forest, XGBoost, k-NN and Logistic Regression.
3. **Deep models** – a dense ANN and a 1D-CNN built with TensorFlow/Keras, trained with early stopping and best-model checkpointing.
4. **Evaluation** – accuracy, classification reports and confusion matrices.

## Repository contents

| File | Description |
|------|-------------|
| `Untitled-1.ipynb` | Main notebook (feature extraction, models, evaluation) |
| `Untitled-1 (2).ipynb` | Second version of the notebook |
| `Speech_Processing_Report (2).pdf` | Project report |
| `Speech_end (2).pptx` | Presentation slides |

## Setup

```bash
pip install librosa soundfile numpy matplotlib seaborn scikit-learn xgboost tensorflow tqdm
```

Place your speaker audio dataset (one folder per speaker) where the notebook expects it, adjust the path in the first cells, then run the notebook top to bottom. The dataset is not included.

## Tech stack

Python, librosa, scikit-learn, XGBoost, TensorFlow/Keras

## Author

Sai Swaroop Talasu – [@saiswaroop25-pixel](https://github.com/saiswaroop25-pixel)
