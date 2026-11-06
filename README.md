# E-Commerce Emotion Classifier using DistilBERT

A PyTorch-based **multi-label emotion classification** project that identifies emotions expressed in written customer-style feedback. It uses **DistilBERT** and the **Google GoEmotions (simplified)** dataset, which contains **28 labels** (27 emotions and neutral).

> **Important:** GoEmotions consists of annotated Reddit comments, **not verified e-commerce reviews**. This project is a general-purpose emotion classifier that can be *applied* to shopping reviews. Evaluate on a separately human-labeled e-commerce dataset before claiming reliable performance in that domain.

## Project objective

Given a text review, predict one or more emotions. For example, the sentence *“The product works well, but delivery was disappointing”* might contain approval and disappointment. These are illustrative possibilities, **not guaranteed model predictions**.

## Dataset

- **Dataset:** [Google GoEmotions — simplified configuration](https://huggingface.co/datasets/google-research-datasets/go_emotions)
- **Task:** Multi-label text classification
- **Total examples:** 54,263
- **Labels:** 28 (27 emotions + `neutral`)
- **Source:** Human-annotated online comments (Reddit)

### Dataset splits

| Split | Rows | Purpose |
| --- | ---: | --- |
| Original training | 43,410 | Train model |
| Original validation | 5,426 | Tune settings and prediction threshold |
| Original test | 5,427 | Final evaluation |
| **Total** | **54,263** | |

**Downloaded 80/20 CSV edition:** If you used the companion export code, `train.csv` contains **43,410** rows and `test.csv` contains **10,853** rows (the original validation and test sets combined). For honest evaluation, **do not tune hyperparameters or prediction thresholds on `test.csv`**. Instead, split off a validation portion from `train.csv` before fitting, then use `test.csv` only for your final evaluation. The Colab notebook's official three-way Hugging Face split is an alternative workflow; the two split schemes should not be mixed when reporting scores.

### Dataset file format

- `train.csv`: Training comments and their emotion labels.
- `test.csv`: Held-out comments and their emotion labels (for the 80/20 export).
- `labels.json`: Maps each numeric label ID (0–27) to its emotion name.

The exported CSV files contain a `text` column, a `labels` column with JSON-style lists such as `"[2, 3]"`, and **28 binary emotion columns** (`0` = absent, `1` = present).

The emotion classes are:

`admiration`, `amusement`, `anger`, `annoyance`, `approval`, `caring`, `confusion`, `curiosity`, `desire`, `disappointment`, `disapproval`, `disgust`, `embarrassment`, `excitement`, `fear`, `gratitude`, `grief`, `joy`, `love`, `nervousness`, `optimism`, `pride`, `realization`, `relief`, `remorse`, `sadness`, `surprise`, `neutral`.

## Technologies

- Python
- PyTorch
- Hugging Face Transformers and Datasets
- DistilBERT (`distilbert-base-uncased`)
- Pandas, NumPy and scikit-learn
- Google Colab (GPU recommended)

## Suggested repository structure

```text
ecommerce-emotion-classifier/
├── README.md
├── GoEmotions_Ecommerce_Colab.ipynb   # Training + evaluation + predictions
├── data/
│   ├── train.csv                      # Optional exported training data
│   ├── test.csv                       # Optional exported testing data
│   └── labels.json
├── predict.py                         # Optional standalone prediction script
├── requirements.txt                   # Optional environment file
└── emotion_model/                     # Generated after training (not bundled)
```

Only include files that are actually in your repository. You can omit the large CSV files and link to the source dataset instead.

## Run in Google Colab

1. Open [Google Colab](https://colab.research.google.com/) and upload `GoEmotions_Ecommerce_Colab.ipynb` (if you have the companion notebook).
2. Select **Runtime → Change runtime type → T4 GPU** if available.
3. Run the notebook cells in order: install dependencies, load data, tokenize, train, evaluate, predict and export.
4. Download the trained model folder as a ZIP from the final notebook cell.

If running code manually instead of the notebook, install the core libraries in a Colab cell:

```python
%pip install "transformers>=4.45,<5" "datasets>=3,<5" "accelerate>=1,<2" torch scikit-learn pandas numpy
```

Load the original dataset with the **official training / validation / testing splits**:

```python
from datasets import load_dataset

dataset = load_dataset("google-research-datasets/go_emotions", "simplified")
print({split: len(rows) for split, rows in dataset.items()})
```

### Load your locally exported CSVs instead

In Colab, upload or extract the CSVs into `data/`, then run:

```python
import json
import pandas as pd

train_df = pd.read_csv("data/train.csv")
test_df = pd.read_csv("data/test.csv")
train_df["labels"] = train_df["labels"].apply(json.loads)
test_df["labels"] = test_df["labels"].apply(json.loads)

with open("data/labels.json", encoding="utf-8") as f:
    label_names = json.load(f)

print("Training examples:", len(train_df))
print("Testing examples:", len(test_df))
print("Emotion labels:", len(label_names))
```

> Reading CSVs is not enough to train DistilBERT: you must tokenize `text`, build **28-element multi-hot vectors**, and convert multi-label targets to **`torch.float32`**. The classifier uses a sigmoid output and binary cross-entropy loss, not softmax single-label classification.

## Model training

- **Base model:** `distilbert-base-uncased`
- **Task:** Multi-label sequence classification
- **Loss:** Binary cross-entropy with logits (automatically selected when `problem_type="multi_label_classification"` and floating-point labels are supplied)
- **Suggested starting settings:** 3 epochs, batch size 16, learning rate `2e-5`, max sequence length 128
- **Training hardware:** Google Colab GPU recommended

These settings are a starting point, not an accuracy guarantee. Use the validation set to adjust thresholds and choose the best model. New classifier-head weights at initialization are normal and require training.

## Model testing and evaluation

Evaluate on **unseen labeled examples** using:

- Micro-F1 (overall performance across all emotion predictions)
- Macro-F1 (average performance across emotion categories)
- Precision and recall per emotion
- Exact-match accuracy (all 28 labels must match exactly)

**Results:** Add your actual results here after you run training and evaluation. No accuracy or F1 score is claimed in advance.

| Metric | Result |
| --- | --- |
| Micro-F1 | _Add after evaluation_ |
| Macro-F1 | _Add after evaluation_ |
| Exact-match accuracy | _Add after evaluation_ |

## Predict emotions

After training, load the saved tokenizer and model from `emotion_model/`. Predictions can be obtained by applying **sigmoid** to the output logits and selecting labels above a chosen threshold.

An example input:

```text
The headphones sound great, but shipping was extremely late.
```

The model may assign multiple emotions to the same sentence. Scores are model outputs, **not calibrated probabilities** unless calibration has been performed. Prediction quality on real customer reviews may differ from evaluation on GoEmotions.

## Download and GitHub

- Download `train.csv`, `test.csv` and `labels.json` using the 80/20 export cell in Colab if you want an offline copy.
- Download the trained model separately as a ZIP after training.
- Upload source code, notebook, README and small metadata files to GitHub. If the data or model is large, link to its source or use an appropriate hosting service rather than committing large binaries.
- Do not upload private customer reviews or personal customer information without permission.

## Limitations and future improvements

1. GoEmotions is **not an e-commerce-specific dataset**; Reddit-language performance may not transfer directly to product reviews.
2. Some emotion labels are rare, so per-class recall and macro-F1 may be lower than micro-F1.
3. A sentence can contain several emotions; a single positive/negative sentiment label cannot replace the 28-label objective.
4. For a stronger e-commerce project, collect properly licensed product reviews, have them human-annotated with the same label scheme, and evaluate on an untouched e-commerce test set.

## References and attribution

- [GoEmotions dataset on Hugging Face](https://huggingface.co/datasets/google-research-datasets/go_emotions)
- [GoEmotions project / paper](https://github.com/google-research/google-research/tree/master/goemotions)
- [DistilBERT model](https://huggingface.co/distilbert/distilbert-base-uncased)

Review the original dataset's license, terms and attribution requirements before redistributing its contents.
