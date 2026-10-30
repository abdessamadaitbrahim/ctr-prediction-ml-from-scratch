# Dataset

**Name:** Click-Through Rate Prediction (`advertising.csv`)
**Source:** [Kaggle - Click-Through Rate Prediction](https://www.kaggle.com/datasets/swekerr/click-through-rate-prediction)
**Target variable:** `Clicked on Ad` (1 = the user clicked on the ad, 0 = did not click)

## Contents

- 10,000 rows, 10 columns, no missing values.
- 215 exact duplicate rows (removed in the notebook, leaving 9,785 rows).
- Classes are nearly balanced: 5,083 rows with `0` and 4,917 rows with `1` (before duplicate removal).

| Column | Type |
|---|---|
| `Daily Time Spent on Site` | numeric |
| `Age` | numeric |
| `Area Income` | numeric |
| `Daily Internet Usage` | numeric |
| `Ad Topic Line` | categorical (text) |
| `City` | categorical |
| `Gender` | categorical |
| `Country` | categorical |
| `Timestamp` | datetime (2016-01-01 to 2016-07-23) |
| `Clicked on Ad` | target (0/1) |

## Download and placement

The CSV is already included in this repository (about 1.1 MB). To use a fresh copy:

1. Download the dataset from the Kaggle link above.
2. Place the file at `data/advertising.csv` (this exact name and location).

The notebook reads it with the relative path `../data/advertising.csv`, so it must be run from the `notebooks/` folder.
