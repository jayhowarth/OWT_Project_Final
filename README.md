# Time-Series Analysis and Prediction Framework

This project is a comprehensive time-series analysis and prediction framework built with **PyTorch** and **Optuna** for training, evaluating, and optimizing machine learning models. The primary focus is on predicting mass values from sensor data using **Transformer** and **CNN** architectures, while incorporating **FFT-based features** for enhanced frequency domain insights.

---

## Features

- **Preprocessing**:
  - Handles large time-series datasets with outlier removal (Isolation Forest and rolling z-score methods).
  - Scales features and target variables using `StandardScaler`.
  - Splits data into train, validation, and test sets with temporal and stratified strategies.

- **Models**:
  - Transformer-based model with positional encoding and self-attention mechanisms.
  - CNN-based model for localized feature extraction.
  - **FFT Integration**: Automatically extracts Fast Fourier Transform (FFT) features from acceleration signals to enhance model inputs by incorporating both time-domain and frequency-domain data.

- **Hyperparameter Optimization**:
  - Automated optimization using **Optuna** for key parameters such as learning rate, batch size, and dropout rates.

- **Training**:
  - Mixed-precision training with gradient clipping for stability.
  - Implements checkpointing, early stopping, and learning rate scheduling.

- **Evaluation**:
  - Generates detailed plots for predictions, residuals, and log-transformed residuals.
  - Computes key metrics such as MSE, MAE, and R².

---

## Installation

### Prerequisites

Ensure the following are installed:
- Python 3.8+
- PyTorch 2.0+
- Optuna
- scikit-learn
- pandas
- matplotlib
- tqdm
- joblib

### Setup

1. Extract and copy code folder:
   ```bash
   cd folder_name

2. Install Dependencies
   ```bash
   pip install -r requirements.txt
   
## Running the Framework

## Train a Model

Run the following command to preprocess the data, train the model, and save checkpoints:```bash
  ```bash
  python main.py --mode train
  ```

## Evaluate a Model

To evaluate the latest checkpoint and generate plots for predictions and residuals:
  ```bash
  python main.py --mode evaluate
  ```


## Optimize Hyperparameters

To tune hyperparameters using Optuna:

  ```bash
  python main.py --mode optimize
  ```


Outputs

	1.	Logs: Saved in the logs/ directory.
	2.	Checkpoints: Model checkpoints are saved in checkpoints/.
	3.	Plots: Residuals, predictions, and performance plots are stored in plots/.

Dataset Requirements

The input dataset must have the following columns:

| Column	        | Description                          |
|----------------|--------------------------------------|
| Timestamp	     | Datetime of the sample               |
| ax1(g), az1(g) | 	Acceleration readings from sensor 1 |
| ax2(g), az2(g) | 	Acceleration readings from sensor 2 |
| ax3(g), az3(g) | 	Acceleration readings from sensor 3 |
| Temp(C)	       | Temperature readings                 |
| Rot_Speed(rpm) | Rotational speed                     |
| Mass(g)	       | Target variable                      |

Data is expected in .txt format, matching the pattern specified in FILE_PATTERN.