import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import logging
import joblib
import argparse
import math
import json
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
from datetime import datetime

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.ensemble import IsolationForest

from plots import plot_log_residuals, plot_predictions, plot_predictions_stratified, plot_residuals

# -----------------------------
# 1. Set Random Seed for Reproducibility
# -----------------------------
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed()

# -----------------------------
# 2. Device Configuration
# -----------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# -----------------------------
# 3. Hyperparameters and Configurations
# -----------------------------
CONFIG = {
    'DATA_DIR': './data_downsample',
    'FILE_PATTERN': '*.txt',
    'BATCH_SIZE': 32,
    'EPOCHS': 50,
    'PATIENCE': 5,
    'CHECKPOINT_DIR': 'checkpoints/',
    'CHECKPOINT_FILENAME_TEMPLATE': 'checkpoint_fold_{fold}.pth',
    'PLOT_DIR': 'plots_exp',
    'MODEL_TYPE': 'transformer',  # Using transformer model
    'WINDOW_SIZE': 60,            # Increased window size
    'SAMPLING_RATE': 30,
    'LEARNING_RATE': 1e-4,
    'WEIGHT_DECAY': 1e-5,
    'NUM_WORKERS': 4,
    'TEST_RATIO': 0.10,
    'FOLD_AMOUNT': 5,
    'TARGET_BINS': 50,
}

FOLD_PROGRESS_FILE = os.path.join(CONFIG['CHECKPOINT_DIR'], 'fold_progress.json')
os.makedirs(CONFIG['CHECKPOINT_DIR'], exist_ok=True)
# -----------------------------
# 4. Configure Logging
# -----------------------------
date_time = datetime.now().strftime("%Y%m%d")
logging.basicConfig(
    filename=f"training_exp_{CONFIG['BATCH_SIZE']}.log",
    filemode='a',
    format='%(asctime)s %(levelname)s %(message)s',
    datefmt='%H:%M:%S',
    level=logging.INFO
)
logging.info('Logging started.')

# -----------------------------
# 5. Data Loading and Preprocessing
# -----------------------------

def remove_outliers_isolation_forest(df, features, contamination=0.005):
    """
    Remove outliers using Isolation Forest.

    Parameters:
        df (pd.DataFrame): DataFrame containing the data.
        features (list): List of feature column names to use for outlier detection.
        contamination (float): Proportion of outliers in the data set.

    Returns:
        pd.DataFrame: DataFrame with outliers removed.
    """
    # Ensure all features are numeric
    numeric_features = df[features].select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_features) < len(features):
        non_numeric = set(features) - set(numeric_features)
        logging.warning(f"Excluding non-numeric features from outlier detection: {non_numeric}")

    iso = IsolationForest(contamination=contamination, random_state=42, n_jobs=-1)
    preds = iso.fit_predict(df[numeric_features])
    return df[preds == 1]

def load_data(config):
    """
    Load data from multiple text files.

    Parameters:
        config (dict): Configuration dictionary.

    Returns:
        pd.DataFrame: Concatenated DataFrame containing all data.
    """
    data_dir = config['DATA_DIR']
    file_pattern = config['FILE_PATTERN']
    
    # Get list of all files matching the pattern
    file_list = glob.glob(os.path.join(data_dir, file_pattern))
    if not file_list:
        logging.error(f"No files found in {data_dir} with pattern {file_pattern}")
        raise ValueError(f"No files found in {data_dir} with pattern {file_pattern}")
    
    all_data = []
    for file_path in tqdm(file_list, desc='Loading Files'):
        try:
            df = pd.read_csv(file_path)
            required_columns = ['Timestamp', 'ax1(g)', 'az1(g)', 'ax2(g)', 'az2(g)', 
                                'ax3(g)', 'az3(g)', 'Temp(C)', 'Rot_Speed(rpm)', 'Mass(g)']
            if not all(col in df.columns for col in required_columns):
                logging.warning(f"Missing columns in {file_path}. Skipping this file.")
                continue
            df = df[required_columns]
            all_data.append(df)
        except Exception as e:
            logging.error(f"Error reading {file_path}: {e}")
            continue
    
    combined_df = pd.concat(all_data, ignore_index=True)
    logging.info(f"Total samples before cleaning: {len(combined_df)}")
    
    return combined_df

def split_train_test(df, test_size=0.10):
    """
    Split the DataFrame into training and test sets based on time.
    
    Parameters:
        df (pd.DataFrame): The sorted DataFrame.
        test_size (float): Proportion of data to include in the test set.
    
    Returns:
        pd.DataFrame, pd.DataFrame: Training and Test DataFrames.
    """
    n_total = len(df)
    n_test = int(n_total * test_size)
    train_df = df.iloc[:-n_test].copy()
    test_df = df.iloc[-n_test:].copy()
    logging.info(f"Training samples: {len(train_df)}")
    logging.info(f"Test samples: {len(test_df)}")
    return train_df, test_df

def create_lag_features(df, lag_features, n_lags=3):
    """
    Create lag features for specified columns.

    Parameters:
        df (pd.DataFrame): DataFrame containing the data.
        lag_features (list): List of columns to create lag features for.
        n_lags (int): Number of lag periods.

    Returns:
        pd.DataFrame: DataFrame with lag features added.
    """
    for feature in lag_features:
        for lag in range(1, n_lags + 1):
            df[f'{feature}_lag_{lag}'] = df[feature].shift(lag)
    df = df.dropna().reset_index(drop=True)
    return df

def create_rolling_features(df, rolling_features, window_size=5):
    """
    Create rolling mean features for specified columns.

    Parameters:
        df (pd.DataFrame): DataFrame containing the data.
        rolling_features (list): List of columns to create rolling features for.
        window_size (int): Window size for rolling calculations.

    Returns:
        pd.DataFrame: DataFrame with rolling features added.
    """
    for feature in rolling_features:
        df[f'{feature}_rolling_mean'] = df[feature].rolling(window=window_size).mean()
    df = df.dropna().reset_index(drop=True)
    return df

def load_and_preprocess_data(config):
    """
    Load, clean, sort by Timestamp, split into train and test, and preprocess the data.

    Parameters:
        config (dict): Configuration dictionary.

    Returns:
        pd.DataFrame, pd.DataFrame, StandardScaler, StandardScaler: 
            Training and Test DataFrames, FFT scaler, target scaler.
    """
    # Load data
    combined_df = load_data(config)
    
    # Convert 'Timestamp' to datetime if not already
    combined_df['Timestamp'] = pd.to_datetime(combined_df['Timestamp'])
    
    # Sort DataFrame by 'Timestamp' in ascending order
    combined_df = combined_df.sort_values('Timestamp').reset_index(drop=True)
    logging.info("Data sorted by Timestamp.")
    
    # Split data into Train and Test
    train_df, test_df = split_train_test(combined_df, test_size=config['TEST_RATIO'])
    
    # Restrict feature columns to acceleration and linear features
    feature_cols = ['ax1(g)', 'az1(g)', 'ax2(g)', 'az2(g)', 'ax3(g)', 'az3(g)', 'Temp(C)', 'Rot_Speed(rpm)']
    fft_feature_cols = ['ax1(g)', 'az1(g)', 'ax2(g)', 'az2(g)', 'ax3(g)', 'az3(g)']
    
    # Remove outliers using Isolation Forest (if necessary)
    train_df = remove_outliers_isolation_forest(train_df, features=feature_cols, contamination=0.005)
    logging.info(f"Training samples after outlier removal: {len(train_df)}")
    
    # Reset index after outlier removal
    train_df = train_df.reset_index(drop=True)
    
    # Scale features
    scaler = StandardScaler()
    train_df[feature_cols] = scaler.fit_transform(train_df[feature_cols])
    test_df[feature_cols] = scaler.transform(test_df[feature_cols])
    
    # Scale target
    target_scaler = StandardScaler()
    train_df['Mass(g)'] = target_scaler.fit_transform(train_df['Mass(g)'].values.reshape(-1, 1)).flatten()
    test_df['Mass(g)'] = target_scaler.transform(test_df['Mass(g)'].values.reshape(-1, 1)).flatten()
    
    # Save scalers
    joblib.dump(scaler, 'feature_scaler_exp.joblib')
    joblib.dump(target_scaler, 'target_scaler.joblib')
    logging.info('Scalers saved using joblib.')
    print("Scalers saved successfully.")
    
    return train_df, test_df, scaler, target_scaler

# -----------------------------
# 6. Expanding Window Split
# -----------------------------
def expanding_window_split_by_days(data, n_splits=5):
    """
    Perform expanding window split for time series data based on unique days.
    
    Parameters:
        data (pd.DataFrame): The dataset to split, must include 'Timestamp' column.
        n_splits (int): Number of splits for validation.
    
    Yields:
        train_idx, val_idx: Indices for training and validation data.
    """
    # Ensure 'Timestamp' is in datetime format
    if not pd.api.types.is_datetime64_any_dtype(data['Timestamp']):
        data['Timestamp'] = pd.to_datetime(data['Timestamp'])
    
    # Get unique sorted days
    unique_days = sorted(data['Timestamp'].dt.date.unique())
    n_days = len(unique_days)
    
    # Calculate the size of each fold in terms of days
    fold_size = n_days // (n_splits + 1)
    remainder = n_days % (n_splits + 1)
    
    # Adjust fold sizes for the remainder
    fold_sizes = [fold_size] * (n_splits + 1)
    for i in range(remainder):
        fold_sizes[i] += 1
    
    current = 0
    for i in range(n_splits):
        train_end = current + sum(fold_sizes[:i + 1])
        val_end = train_end + fold_sizes[i + 1]
        
        train_days = unique_days[:train_end]
        val_days = unique_days[train_end:val_end]
        
        train_idx = data[data['Timestamp'].dt.date.isin(train_days)].index
        val_idx = data[data['Timestamp'].dt.date.isin(val_days)].index
        
        yield train_idx, val_idx
# -----------------------------
# 7. Custom Dataset and Model Definitions
# -----------------------------
class TimeSeriesDataset(Dataset):
    """
    Custom Dataset for Time Series Data with Sliding Window and FFT Integration.
    """
    def __init__(self, df, window_size=60, feature_cols=None, target_col='Mass(g)', 
                 use_fft=True, fft_feature_cols=None):
        """
        Initialize the dataset.

        Parameters:
            df (pd.DataFrame): DataFrame containing the data.
            window_size (int): Number of time steps in each input window.
            feature_cols (list): List of all feature column names.
            target_col (str): Name of the target column.
            use_fft (bool): Whether to include FFT features.
            fft_feature_cols (list): List of feature columns to apply FFT.
        """
        self.window_size = window_size
        self.use_fft = use_fft
        self.feature_cols = feature_cols if feature_cols else df.columns.drop(['Timestamp', target_col]).tolist()
        self.target_col = target_col
        self.fft_feature_cols = fft_feature_cols if fft_feature_cols else []
    
        # Extract features, target, and timestamps
        self.X = df[self.feature_cols].values
        self.y = df[self.target_col].values
        self.timestamps = df['Timestamp'].values
        self.num_samples = len(self.y) - self.window_size
    
        # Store the day (or equivalent grouping) for each sample
        self.days = df['Timestamp'].dt.date.values[self.window_size:]

        # Identify indices of FFT and non-FFT features
        self.fft_indices = [self.feature_cols.index(col) for col in self.fft_feature_cols]
        self.non_fft_indices = [i for i in range(len(self.feature_cols)) if i not in self.fft_indices]
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        """
        Retrieve a single sample using sliding window and integrate FFT features.
        """
        start = idx
        end = idx + self.window_size
        X_window = self.X[start:end]  # Shape: (window_size, num_features)
        y_target = self.y[end]        # Scalar

        if self.use_fft and self.fft_feature_cols:
            # Apply FFT on specified columns
            fft_data = X_window[:, self.fft_indices]  # Shape: (window_size, num_fft_features)
            fft_features = self.extract_fft_features(fft_data)  # Shape: (num_fft_features * 2,)
            # Tile FFT features across the window size
            fft_features_tiled = np.tile(fft_features, (self.window_size, 1))  # Shape: (window_size, num_fft_features * 2)
            # Extract non-FFT features
            non_fft_data = X_window[:, self.non_fft_indices]  # Shape: (window_size, num_non_fft_features)
            # Concatenate non-FFT and FFT features
            X_combined = np.hstack((non_fft_data, fft_features_tiled))  # Shape: (window_size, total_features)
        else:
            X_combined = X_window  # Shape: (window_size, num_features)

        # Convert to tensors
        X_tensor = torch.tensor(X_combined, dtype=torch.float32)
        y_tensor = torch.tensor(y_target, dtype=torch.float32)

        return X_tensor, y_tensor
    
    def extract_fft_features(self, fft_data):
        """
        Extract FFT-based features from the specified FFT data.

        Parameters:
            fft_data (np.ndarray): FFT data of shape (window_size, num_fft_features)

        Returns:
            np.ndarray: FFT features flattened into a single array.
        """
        fft_features = []
        for feature in range(fft_data.shape[1]):
            signal = fft_data[:, feature]
            fft_vals = np.fft.fft(signal)
            fft_mag = np.abs(fft_vals)[:fft_data.shape[0] // 2]  # Magnitude of FFT
            mean_mag = np.mean(fft_mag)
            dominant_freq_idx = np.argmax(fft_mag)
            dominant_freq = (dominant_freq_idx * CONFIG['SAMPLING_RATE']) / fft_data.shape[0]
            fft_features.extend([mean_mag, dominant_freq])
        return np.array(fft_features)

class PositionalEncoding(nn.Module):
    """
    Positional Encoding module injects information about the relative or absolute position
    of the tokens in the sequence. The positional encodings have the same dimension as the
    embeddings so that the two can be summed.
    """
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create constant 'pe' matrix with values dependent on position and i
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)  # Apply sin to even indices in the array
        pe[:, 1::2] = torch.cos(position * div_term)  # Apply cos to odd indices in the array
        pe = pe.unsqueeze(0)  # Add batch dimension
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Inputs:
            x: Tensor of shape (batch_size, seq_len, d_model)

        Outputs:
            x: Tensor with positional encodings added, same shape as input
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class TransformerModel(nn.Module):
    """
    Transformer-based model with positional encoding
    """
    def __init__(self, input_dim, window_size, model_dim=128, num_heads=4, num_layers=2, dropout=0.1):
        super(TransformerModel, self).__init__()
        self.input_dim = input_dim
        self.window_size = window_size
        self.model_dim = model_dim
        self.num_heads = num_heads

        # Positional Encoding
        self.positional_encoding = PositionalEncoding(d_model=self.model_dim, max_len=self.window_size, dropout=dropout)

        # Linear projection to match model_dim
        self.input_projection = nn.Linear(input_dim, self.model_dim)

        # Transformer Encoder Layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.model_dim,
            nhead=self.num_heads,
            dim_feedforward=self.model_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output layer
        self.fc_out = nn.Linear(self.model_dim, 1)
        self.dropout = nn.Dropout(dropout)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """
        Initialize weights using Xavier uniform initialization for linear layers
        and Kaiming normal initialization for transformer layers.
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        Forward pass of the model.

        Parameters:
            x (Tensor): Input tensor of shape (batch_size, window_size, input_dim)

        Returns:
            out (Tensor): Output tensor of shape (batch_size)
        """
        x = self.input_projection(x)  # Shape: (batch_size, window_size, model_dim)
        x = self.positional_encoding(x)  # Shape: (batch_size, window_size, model_dim)
        x = self.transformer_encoder(x)  # Shape: (batch_size, window_size, model_dim)
        x = torch.mean(x, dim=1)        # Shape: (batch_size, model_dim)
        x = self.dropout(x)
        out = self.fc_out(x).squeeze(1) # Shape: (batch_size)
        return out

class CNNModel(nn.Module):
    """
    Simple Convolutional Neural Network for Time Series Prediction.
    """
    def __init__(self, input_dim, window_size, num_classes=1):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        # Calculate the size after pooling
        pooled_size = window_size // 4  # Two pooling layers with kernel_size=2
        self.fc1 = nn.Linear(128 * pooled_size, 256)
        self.fc2 = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = x.permute(0, 2, 1)          # (batch_size, num_features, window_size)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x)                # (batch_size, 64, window_size/2)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.pool2(x)               # (batch_size, 128, window_size/4)
        x = x.view(x.size(0), -1)       # Flatten
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        out = self.fc2(x).squeeze(1)    # (batch_size)
        return out

# -----------------------------
# 8. Training Utilities and Helper Functions
# -----------------------------
def save_checkpoint(state, fold, config):
    filename = f"{config['CHECKPOINT_DIR']}/{config['CHECKPOINT_FILENAME_TEMPLATE'].format(fold=fold)}"
    torch.save(state, filename)
    logging.info(f"Checkpoint saved to {filename}")

def load_checkpoint(filename, model, optimizer=None, scheduler=None):
    if os.path.isfile(filename):
        checkpoint = torch.load(filename, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        if optimizer:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if scheduler and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        epoch = checkpoint.get('epoch', 0)
        best_loss = checkpoint.get('best_loss', np.inf)
        patience_counter = checkpoint.get('patience_counter', 0)
        fold = checkpoint.get('fold', None)  # Retrieve the fold number
        logging.info(f"Checkpoint loaded from {filename} for fold {fold}")
        return epoch, best_loss, patience_counter, fold
    else:
        logging.info(f"No checkpoint found at {filename}")
        return 1, np.inf, 0, None  # Start from epoch 1

def save_fold_progress(fold_number):
    progress = {"last_completed_fold": fold_number}
    with open(FOLD_PROGRESS_FILE, 'w') as f:
        json.dump(progress, f)

def load_fold_progress():
    if os.path.exists(FOLD_PROGRESS_FILE):
        with open(FOLD_PROGRESS_FILE, 'r') as f:
            progress = json.load(f)
        return progress.get("last_completed_fold", 0)
    return 0

# -----------------------------
# 9. Training Utilities and Helper Functions
# -----------------------------
def initialize_model(config, input_dim):
    if config['MODEL_TYPE'] == 'transformer':
        model = TransformerModel(
            input_dim=input_dim,
            window_size=config['WINDOW_SIZE'],
            model_dim=128,
            num_heads=4,
            num_layers=2,
            dropout=0.1
        ).to(device)
    elif config['MODEL_TYPE'] == 'cnn':
        model = CNNModel(
            input_dim=input_dim,
            window_size=config['WINDOW_SIZE'],
            num_classes=1
        ).to(device)
    else:
        raise ValueError("Invalid MODEL_TYPE. Choose from 'transformer', 'cnn'.")
    
    logging.info(f"{config['MODEL_TYPE'].capitalize()} Model initialized with input_dim={input_dim}.")
    print(f"{config['MODEL_TYPE'].capitalize()} Model initialized with input_dim={input_dim}.")
    return model

def train_model(model, train_loader, val_loader, criterion, optimizer, config, scheduler=None, 
                start_epoch=1, best_loss=np.inf, patience_counter=0, fold=0):
    scaler = GradScaler()
    history = {'train_loss': [], 'val_loss': [], 'val_mae': [], 'val_r2': []}

    for epoch in range(start_epoch, config['EPOCHS'] + 1):
        model.train()
        train_losses = []
        loop = tqdm(train_loader, desc=f'Epoch {epoch}/{config["EPOCHS"]} - Training', leave=False)
        for X_batch, y_batch in loop:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            if device.type == 'cuda':
                with autocast():
                    outputs = model(X_batch)
                    loss = criterion(outputs, y_batch)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            train_losses.append(loss.item())
            loop.set_postfix(loss=loss.item())

        avg_train_loss = np.mean(train_losses)

        # Validation Phase
        model.eval()
        val_losses = []
        all_preds = []
        all_targets = []
        with torch.no_grad():
            loop = tqdm(val_loader, desc=f'Epoch {epoch}/{config["EPOCHS"]} - Validation', leave=False)
            for X_batch, y_batch in loop:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                if device.type == 'cuda':
                    with autocast():
                        outputs = model(X_batch)
                        loss = criterion(outputs, y_batch)
                else:
                    outputs = model(X_batch)
                    loss = criterion(outputs, y_batch)
                val_losses.append(loss.item())
                all_preds.extend(outputs.cpu().numpy())
                all_targets.extend(y_batch.cpu().numpy())
                loop.set_postfix(loss=loss.item())

        avg_val_loss = np.mean(val_losses)
        mae = mean_absolute_error(all_targets, all_preds)
        r2 = r2_score(all_targets, all_preds)

        # Update history
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['val_mae'].append(mae)
        history['val_r2'].append(r2)

        # Logging
        logging.info(f'Epoch {epoch}: Train Loss={avg_train_loss:.4f}, Val Loss={avg_val_loss:.4f}, MAE={mae:.4f}, R²={r2:.4f}')
        print(f'Epoch {epoch}: Train Loss={avg_train_loss:.4f}, Val Loss={avg_val_loss:.4f}, MAE={mae:.4f}, R²={r2:.4f}')

        # Scheduler step
        if scheduler:
            scheduler.step(avg_val_loss)

        # Checkpointing and Early Stopping
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            patience_counter = 0
            # Include the fold number in the state dictionary
            state = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'best_loss': best_loss,
                'patience_counter': patience_counter,
                'fold': fold  # Save the fold number in the checkpoint
            }
            save_checkpoint(state, fold, config)
            logging.info(f'Validation loss decreased. Saving model.')
            print(f"Checkpoint saved at epoch {epoch}, validation loss improved to {avg_val_loss:.4f}")
        else:
            # Handle case where validation loss did not improve
            patience_counter += 1
            logging.info(f'Validation loss did not improve. Patience: {patience_counter}/{config["PATIENCE"]}')
            print(f'Validation loss did not improve. Patience: {patience_counter}/{config["PATIENCE"]}')
            if patience_counter >= config['PATIENCE']:
                logging.info('Early stopping triggered.')
                print('Early stopping triggered.')
                break

    return history

def evaluate_last_checkpoint(config):
    print("Loading and preprocessing data for evaluation...")
    train_df, test_df, scaler, target_scaler = load_and_preprocess_data(config)

    # Set WINDOW_SIZE to match training
    config['WINDOW_SIZE'] = 60

    # Define feature columns used during training
    # Assume these were the feature columns during training
    feature_cols = ['ax1(g)', 'az1(g)', 'ax2(g)', 'az2(g)', 'ax3(g)', 'az3(g)', 
                    'Temp(C)', 'Rot_Speed(rpm)']
    fft_feature_cols = ['ax1(g)', 'az1(g)', 'ax2(g)', 'az2(g)', 'ax3(g)', 'az3(g)']

    # Recalculate input_dim
    input_dim = len(feature_cols) - len(fft_feature_cols) + len(fft_feature_cols) * 2  # Should be 14

    # Initialize Dataset and DataLoader with correct features
    test_dataset = TimeSeriesDataset(
        df=test_df,
        window_size=config['WINDOW_SIZE'],
        feature_cols=feature_cols,
        target_col='Mass(g)',
        use_fft=True,
        fft_feature_cols=fft_feature_cols
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['BATCH_SIZE'],
        shuffle=False,
        num_workers=config['NUM_WORKERS'],
        pin_memory=True
    )

    # Initialize Model
    model = initialize_model(config, input_dim=input_dim)

    # Load the last checkpoint
    checkpoints = glob.glob(os.path.join(config['CHECKPOINT_DIR'], config['CHECKPOINT_FILENAME_TEMPLATE'].format(fold='*')))
    if not checkpoints:
        logging.error("No checkpoints found for evaluation.")
        raise FileNotFoundError("No checkpoints found for evaluation.")

    # Find the checkpoint with the lowest validation loss
    best_checkpoint = None
    best_loss = float('inf')
    for checkpoint_path in checkpoints:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if 'best_loss' in checkpoint and checkpoint['best_loss'] < best_loss:
            best_loss = checkpoint['best_loss']
            best_checkpoint = checkpoint

    if best_checkpoint is None:
        logging.error("No valid checkpoints found for evaluation.")
        raise FileNotFoundError("No valid checkpoints found for evaluation.")

    # Load model state from the best checkpoint
    model.load_state_dict(best_checkpoint['model_state_dict'])
    logging.info(f"Loaded model from checkpoint with validation loss: {best_loss:.4f}")
    print(f"Loaded model from checkpoint with validation loss: {best_loss:.4f}")

    # Evaluate on Test Set
    evaluate_model(model, test_loader, config, target_scaler, dataset_type='Test')

def evaluate_model(model, test_loader, device, feature_scaler, target_scaler, plot_dir, dataset_type='Test'):
    """
    Evaluate the model on a specified dataset and generate plots.
    
    Parameters:
        model (nn.Module): Trained model.
        data_loader (DataLoader): DataLoader for the dataset.
        config (dict): Configuration dictionary.
        target_scaler (StandardScaler): Target scaler.
        dataset_type (str): Type of dataset (e.g., 'Test', 'Validation_Fold_1').
    """
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        loop = tqdm(test_loader, desc=f'Evaluating on {dataset_type} Set')
        for X_batch, y_batch in loop:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            all_preds.extend(outputs.cpu().numpy())
            all_targets.extend(y_batch.cpu().numpy())
    
    # Inverse transform the predictions and targets
    all_preds = target_scaler.inverse_transform(np.array(all_preds).reshape(-1, 1)).flatten()
    all_targets = target_scaler.inverse_transform(np.array(all_targets).reshape(-1, 1)).flatten()
    
    # Calculate Metrics
    mae = mean_absolute_error(all_targets, all_preds)
    r2 = r2_score(all_targets, all_preds)
    logging.info(f'{dataset_type} MAE: {mae:.4f}')
    logging.info(f'{dataset_type} R²: {r2:.4f}')
    print(f'{dataset_type} MAE: {mae:.4f}')
    print(f'{dataset_type} R²: {r2:.4f}')
    
    # Generate plots
    plot_predictions(model, test_loader, device, feature_scaler, target_scaler, plot_dir)
    plot_predictions_stratified(model, test_loader, device, feature_scaler, target_scaler, plot_dir)
    plot_residuals(model, test_loader, device, feature_scaler, target_scaler, plot_dir)
    plot_log_residuals(model, test_loader, device, feature_scaler, target_scaler, plot_dir)


# -----------------------------
# 9. Main Training Script with Expanding Window Validation
# -----------------------------
def main(config, resume_training=False):
    # Load last completed fold
    last_completed_fold = load_fold_progress()
    print(f"Last completed fold: {last_completed_fold}")

    # Load and preprocess data
    print("Loading and preprocessing data...")
    train_df, test_df, scaler, target_scaler = load_and_preprocess_data(config)

    # Define feature columns
    feature_cols = [col for col in train_df.columns if col not in ['Timestamp', 'Mass(g)']]
    fft_feature_cols = ['ax1(g)', 'az1(g)', 'ax2(g)', 'az2(g)', 'ax3(g)', 'az3(g)']  # FFT features

    # Expanding Window Validation with Time Bins
    n_splits = CONFIG['FOLD_AMOUNT']  # Number of expanding windows
    splits = expanding_window_split_by_days(train_df, n_splits=n_splits)

    # To store metrics for each fold
    fold_metrics = {'train_loss': [], 'val_loss': [], 'val_mae': [], 'val_r2': []}

    fold_number = 1
    for train_index, val_index in splits:
        # Skip already completed folds
        if resume_training and fold_number <= last_completed_fold:
            print(f"Skipping already completed Fold {fold_number}")
            fold_number += 1
            continue

        print(f"\n--- Starting Fold {fold_number}/{n_splits} ---")

        # Split data
        fold_train_df = train_df.iloc[train_index].copy()
        fold_val_df = train_df.iloc[val_index].copy()
        logging.info(f"Fold {fold_number}: Train samples: {len(fold_train_df)}, Validation samples: {len(fold_val_df)}")
        print(f"Fold {fold_number}: Train samples: {len(fold_train_df)}, Validation samples: {len(fold_val_df)}")

        # Initialize Datasets
        train_dataset = TimeSeriesDataset(
            df=fold_train_df,
            window_size=config['WINDOW_SIZE'],
            feature_cols=feature_cols,
            target_col='Mass(g)',
            use_fft=True,
            fft_feature_cols=fft_feature_cols
        )
        val_dataset = TimeSeriesDataset(
            df=fold_val_df,
            window_size=config['WINDOW_SIZE'],
            feature_cols=feature_cols,
            target_col='Mass(g)',
            use_fft=True,
            fft_feature_cols=fft_feature_cols
        )

        # Create DataLoaders
        train_loader = DataLoader(train_dataset, batch_size=config['BATCH_SIZE'], shuffle=True, num_workers=config['NUM_WORKERS'], pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=config['BATCH_SIZE'], shuffle=False, num_workers=config['NUM_WORKERS'], pin_memory=True)

        # Initialize Model
        # Calculate input_dim based on features
        input_dim = len(feature_cols) - len(fft_feature_cols) + len(fft_feature_cols) * 2
        print(f"Initializing model with input_dim={input_dim}")
        model = initialize_model(config, input_dim=input_dim)

        # Define Loss and Optimizer
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=config['LEARNING_RATE'], weight_decay=config['WEIGHT_DECAY'])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)

        # Load checkpoint if resuming
        start_epoch, best_loss, patience_counter, loaded_fold = 1, np.inf, 0, None
        checkpoint_filename = config['CHECKPOINT_FILENAME_TEMPLATE'].format(fold=fold_number)
        checkpoint_path = os.path.join(config['CHECKPOINT_DIR'], checkpoint_filename)

        if resume_training and os.path.exists(checkpoint_path):
            start_epoch, best_loss, patience_counter, loaded_fold = load_checkpoint(
                checkpoint_path, model, optimizer, scheduler
            )
            if loaded_fold == fold_number:
                print(f"Resuming training from epoch {start_epoch} for fold {loaded_fold}")
            else:
                print(f"Loaded checkpoint fold ({loaded_fold}) does not match current fold ({fold_number}). Starting from scratch.")
                start_epoch, best_loss, patience_counter = 1, np.inf, 0
        else:
            print(f"No checkpoint found at {checkpoint_path}. Starting from scratch.")

        # Train the model
        print(f"Starting training for Fold {fold_number}...")
        history = train_model(
            model,
            train_loader,
            val_loader,
            criterion,
            optimizer,
            config,
            scheduler=scheduler,
            start_epoch=start_epoch,
            best_loss=best_loss,
            patience_counter=patience_counter,
            fold=fold_number
        )

        # Save fold progress
        save_fold_progress(fold_number)

        # Evaluate the model on the validation set
        print(f"Evaluating on Validation Set for Fold {fold_number}...")
        evaluate_model(
            model,
            val_loader,
            device,
            feature_scaler=scaler,
            target_scaler=target_scaler,
            plot_dir=config['PLOT_DIR'],
            dataset_type=f'Validation_Fold_{fold_number}'
        )
        # Append metrics
        fold_metrics['train_loss'].append(history['train_loss'][-1])
        fold_metrics['val_loss'].append(history['val_loss'][-1])
        fold_metrics['val_mae'].append(history['val_mae'][-1])
        fold_metrics['val_r2'].append(history['val_r2'][-1])

        # Increment fold number
        fold_number += 1

    # Average metrics
    avg_metrics = {metric: np.mean(values) for metric, values in fold_metrics.items()}
    logging.info(f"Average Metrics across all folds: {avg_metrics}")
    print(f"\n=== Average Metrics across all folds ===")
    print(f"Train Loss: {avg_metrics['train_loss']:.4f}")
    print(f"Validation Loss: {avg_metrics['val_loss']:.4f}")
    print(f"Validation MAE: {avg_metrics['val_mae']:.4f}")
    print(f"Validation R²: {avg_metrics['val_r2']:.4f}")

# -----------------------------
# 11. Argument Parsing
# -----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and Evaluate the Model with Expanding Window Validation")
    parser.add_argument('--mode', type=str, choices=['train', 'evaluate'], default='train',
                        help="Mode to run the script in: 'train' to train the model, 'evaluate' to evaluate the latest checkpoint.")
    parser.add_argument('--checkpoint', type=str, default='checkpoint_exp.pth',
                        help="Path to the checkpoint file. Default is 'checkpoint_exp.pth'.")
    parser.add_argument('--resume', action='store_true',
                        help="Resume training from the last checkpoint.")
    args = parser.parse_args()
    
    if args.mode == 'train':
        main(CONFIG, resume_training=args.resume)
    elif args.mode == 'evaluate':
        # Evaluation mode implementation
        print("Evaluation mode initiated.")
        logging.info("Evaluation mode initiated.")
        evaluate_last_checkpoint(CONFIG)
    else:
        print("Invalid mode selected. Choose 'train' or 'evaluate'.")
        logging.error("Invalid mode selected. Choose 'train' or 'evaluate'.")