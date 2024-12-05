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
import random
import optuna
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.amp import GradScaler, autocast
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split

from plots import (plot_log_residuals,
                   plot_predictions,
                   plot_predictions_stratified,
                   plot_residuals,
                   plot_loss_curves,
                   plot_validation_metrics)

import warnings

warnings.filterwarnings("ignore", message="enable_nested_tensor is True, but self.use_nested_tensor"
                                          " is False because encoder_layer.norm_first was True")

# -----------------------------
# 1. Set Random Seed for Reproducibility
# -----------------------------
def set_seed(seed=42):
    """

    :param seed:
    """
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
# print(f'Using device: {device}')

# -----------------------------
# 3. Hyperparameters and Configurations
# -----------------------------
CONFIG = {
    'DATA_DIR': './sample_data',  # Path to the directory containing input data files.
    'FILE_PATTERN': '*.txt',  # File pattern to match data files in the DATA_DIR (e.g., all .txt files).
    'BATCH_SIZE': 64,  # Number of samples per batch during training and evaluation.
    'EPOCHS': 2,  # Total number of complete passes through the dataset for training.
    'PATIENCE': 5,  # Number of epochs to wait for improvement before stopping early (early stopping).
    'CHECKPOINT_PATH': './checkpoints/checkpoint.pth',  # Path to save the model checkpoint during training.
    'PLOT_DIR': 'plots',  # Directory where training/validation plots (e.g., loss, accuracy) will be saved.
    'MODEL_TYPE': 'transformer',  # Specifies the type of model to be used (e.g., 'transformer').
    'WINDOW_SIZE': 80,  # Number of consecutive data points considered as a single input window for time-series data.
    'SAMPLING_RATE': 20,  # Frequency of sampling data points for preprocessing or windowing.
    'LEARNING_RATE': 1e-5,  # Learning rate for the optimizer, controlling the step size during gradient descent.
    'WEIGHT_DECAY': 1e-4,  # Regularization term to prevent overfitting by penalizing large weights in the model.
    'NUM_WORKERS': 2,  # Number of worker threads for loading data in parallel (used in DataLoader).
    'TRAIN_RATIO': 0.75,  # Proportion of the dataset used for training.
    'VAL_RATIO': 0.15,  # Proportion of the dataset used for validation.
    'TEST_RATIO': 0.10,  # Proportion of the dataset used for testing (sum of ratios should be 1.0).
    'DROPOUT': 0.2,  # Dropout rate for regularization in the model, dropping 20% of connections during training.
}

# -----------------------------
# 4. Configure Logging
# -----------------------------

os.makedirs("./logs", exist_ok=True)
date_time = datetime.now()
formatted_time = date_time.strftime("%d-%m-%y")
logging.basicConfig(
    filename=f"./logs/{formatted_time}_Model_{CONFIG['MODEL_TYPE']}_Batch_{CONFIG['BATCH_SIZE']}.log",
    filemode='a',
    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
    datefmt='%H:%M:%S',
    level=logging.INFO
)
logging.info(f'Using device: {device}')

# -----------------------------
# 5. Data Loading and Preprocessing
# -----------------------------

def remove_outliers_isolation_forest(df, features, contamination=0.01):
    """
    Remove outliers using Isolation Forest.
    """
    iso = IsolationForest(contamination=contamination, random_state=42)
    preds = iso.fit_predict(df[features])
    return df[preds == 1]

def remove_outliers_local_z_score(df, feature, window=100, threshold=3):
    """
    Remove outliers using a rolling z-score method.
    """
    rolling_mean = df[feature].rolling(window=window, center=True).mean()
    rolling_std = df[feature].rolling(window=window, center=True).std()
    z_scores = np.abs((df[feature] - rolling_mean) / rolling_std)
    return df[z_scores < threshold]


def stratified_train_val_test_split(df, test_size=0.1, val_size=0.15, random_state=42):
    """
    Perform stratified splitting of the dataset into train, validation, and test sets.

    Parameters:
        df (pd.DataFrame): The complete dataset with a 'day' column for stratification.
        test_size (float): Proportion of the dataset to reserve as the test set.
        val_size (float): Proportion of the training set to reserve as the validation set.
        random_state (int): Seed for reproducibility.

    Returns:
        train_df, val_df, test_df (pd.DataFrames): Split datasets.
    """
    # Step 1: Stratify by 'day' to split into Train+Val and Test
    unique_days = df['day'].unique()
    train_val_days, test_days = train_test_split(
        unique_days, test_size=test_size, random_state=random_state
    )
    
    train_val_df = df[df['day'].isin(train_val_days)].reset_index(drop=True)
    test_df = df[df['day'].isin(test_days)].reset_index(drop=True)
    
    # Step 2: Further stratify Train+Val into Train and Validation
    unique_train_val_days = train_val_df['day'].unique()
    train_days, val_days = train_test_split(
        unique_train_val_days, test_size=val_size, random_state=random_state
    )
    
    train_df = train_val_df[train_val_df['day'].isin(train_days)].reset_index(drop=True)
    val_df = train_val_df[train_val_df['day'].isin(val_days)].reset_index(drop=True)
    
    print(f"Train samples: {len(train_df)}, Validation samples: {len(val_df)}, Test samples: {len(test_df)}")
    return train_df, val_df, test_df

def split_days_chronological(unique_days, train_ratio=0.8):
    """
    Split days into train and validation sets chronologically.
    """
    split_index = int(len(unique_days) * train_ratio)
    train_days = unique_days[:split_index]
    val_days = unique_days[split_index:]
    return train_days, val_days

def split_data_per_day(df, config):
    """
    Split the DataFrame into train, validation, and test sets per day.

    Parameters:
        df (pd.DataFrame): The complete dataset with a 'day' column.
        config (dict): Configuration dictionary containing split ratios.

    Returns:
        train_df, val_df, test_df (pd.DataFrame): The split datasets.
    """
    train_list = []
    val_list = []
    test_list = []
    
    unique_days = df['day'].unique()
    # print(unique_days)
    # print(f"Total unique days in dataset: {len(unique_days)}")
    
    train_ratio = config['TRAIN_RATIO']
    val_ratio = config['VAL_RATIO']
    test_ratio = config['TEST_RATIO']
    
    for day in unique_days:
        day_df = df[df['day'] == day]
        
        # Shuffle the day's data
        day_df = day_df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        n = len(day_df)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        
        train_data = day_df.iloc[:n_train]
        val_data = day_df.iloc[n_train:n_train + n_val]
        test_data = day_df.iloc[n_train + n_val:]
        
        train_list.append(train_data)
        val_list.append(val_data)
        test_list.append(test_data)
    
    # Concatenate data from all days
    train_df = pd.concat(train_list, ignore_index=True)
    val_df = pd.concat(val_list, ignore_index=True)
    test_df = pd.concat(test_list, ignore_index=True)
    
    return train_df, val_df, test_df

def load_and_preprocess_data(config):
    """
    Load data from multiple text files in a directory, perform initial preprocessing,
    and split into training, validation, and test sets per day.
    """
    data_dir = config['DATA_DIR']
    file_pattern = config['FILE_PATTERN']
    
    # Get list of all files matching the pattern
    file_list = glob.glob(os.path.join(data_dir, file_pattern))
    if not file_list:
        raise ValueError(f"No files found in {data_dir} with pattern {file_pattern}")
    
    all_data = []
    for file_idx, file_path in enumerate(tqdm(file_list, desc='Loading Files')):
        # Read the file
        try:
            df = pd.read_csv(file_path)
        except Exception as e:
            logging.error(f"Error reading {file_path}: {e}")
            continue
        
        # Ensure required columns exist
        required_columns = ['Timestamp', 'ax1(g)', 'az1(g)', 'ax2(g)', 'az2(g)',
                            'ax3(g)', 'az3(g)', 'Temp(C)', 'Rot_Speed(rpm)', 'Mass(g)']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"Missing columns in {file_path}: {missing_columns}")
            logging.error(f"Missing columns in {file_path}. Required columns missing: {missing_columns}")
            continue
        
        # Parse 'Timestamp' column to datetime
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        
        # Extract 'day' from 'Timestamp'
        df['day'] = df['Timestamp'].dt.date.astype(str)
        
        all_data.append(df)
    
    # Check if any data was loaded
    if not all_data:
        raise ValueError("No data was loaded. Check if the required columns exist in your files.")
    
    # Concatenate all data
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Drop any remaining NaN values
    combined_df = combined_df.dropna().reset_index(drop=True)
    
    # Remove outliers
    features_to_check = ['ax1(g)', 'az1(g)', 'ax2(g)', 'az2(g)', 'ax3(g)', 'az3(g)',
                         'Temp(C)', 'Rot_Speed(rpm)', 'Mass(g)']
    
    # Remove outliers using Isolation Forest
    combined_df = remove_outliers_isolation_forest(combined_df, features_to_check, contamination=0.01)
    
    # Split data per day
    train_df, val_df, test_df = split_data_per_day(combined_df, config)
    
    unique_days = combined_df['day'].unique()
    # print(f"Unique days in dataset: {unique_days}")
    print(f"Total unique days in dataset: {len(unique_days)}")
    
    logging.info(f'Total files processed: {len(file_list)}')
    logging.info(f'Samples after outlier removal: {len(combined_df)}')
    logging.info(f'Training samples: {len(train_df)}')
    logging.info(f'Validation samples: {len(val_df)}')
    logging.info(f'Test samples: {len(test_df)}')
    
    print(f"Train samples: {len(train_df)}, Validation samples: {len(val_df)}, Test samples: {len(test_df)}")
    
    return train_df, val_df, test_df

# -----------------------------
# 6. Custom Dataset Class
# -----------------------------
class TimeSeriesDataset(Dataset):
    """
    Custom Dataset for Time Series Data with Sliding Window, Feature Engineering, and FFT Integration.
    Processes data on the fly to manage memory efficiently.
    """
    def __init__(self, df, window_size=30, feature_scaler=None, target_scaler=None, fit_scalers=False, use_fft=True):
        """
        Initialize the dataset.
        
        Parameters:
            df (pd.DataFrame): DataFrame containing the data.
            window_size (int): Number of time steps in each input window.
            feature_scaler (StandardScaler): Scaler for features.
            target_scaler (StandardScaler): Scaler for target variable.
            fit_scalers (bool): Whether to fit the scalers on this dataset.
            use_fft (bool): Whether to include FFT features.
        """
        self.window_size = window_size
        self.use_fft = use_fft
        self.time_features = ['ax1(g)', 'az1(g)', 'ax2(g)', 'az2(g)', 'ax3(g)', 'az3(g)', 'Temp(C)', 'Rot_Speed(rpm)']
        self.target = 'Mass(g)'
        self.acceleration_features = ['ax1(g)', 'az1(g)', 'ax2(g)', 'az2(g)', 'ax3(g)', 'az3(g)']
        self.accel_feature_indices = [self.time_features.index(f) for f in self.acceleration_features]
        
        # Extract features and target
        self.X_time = df[self.time_features].values
        self.y = df[self.target].values
        self.original_days = df['day'].values
        
        # Total number of samples after sliding window
        self.num_samples = len(self.y) - self.window_size
        
        # Initialize scalers
        if fit_scalers:
            self.feature_scaler = StandardScaler()
            self.X_time = self.feature_scaler.fit_transform(self.X_time)
            
            self.target_scaler = StandardScaler()
            self.y = self.target_scaler.fit_transform(self.y.reshape(-1, 1)).flatten()
        else:
            self.feature_scaler = feature_scaler
            self.target_scaler = target_scaler
            
            # Apply feature scaling if scaler is provided
            if self.feature_scaler is not None:
                self.X_time = self.feature_scaler.transform(self.X_time)
            
            # Apply target scaling if scaler is provided
            if self.target_scaler is not None:
                self.y = self.target_scaler.transform(self.y.reshape(-1, 1)).flatten()
        
        # Store 'day' information for each sample
        self.days = self.original_days[self.window_size:]
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        """
        Retrieve a single sample using sliding window and integrate FFT features.
        """
        start = idx
        end = idx + self.window_size
        X_window_time = self.X_time[start:end]  # Shape: (window_size, num_time_features=8)
        y_target = self.y[end]                  # Scalar
        
        # Perform FFT on acceleration features within the window
        if self.use_fft:
            X_window_acc = X_window_time[:, self.accel_feature_indices]  # Shape: (window_size, 6)
            fft_features = self.extract_fft_features(X_window_acc)
            # fft_features shape: (6*2,) = (12,)
            
            # Repeat FFT features across the window to match time steps
            fft_features_repeated = np.tile(fft_features, (self.window_size, 1))  # Shape: (window_size, 12)
            
            # Combine time-domain and FFT features
            X_combined = np.concatenate([X_window_time, fft_features_repeated], axis=1)  # Shape: (window_size, 8 + 12)
        else:
            X_combined = X_window_time  # Shape: (window_size, 8)
        
        # Convert to tensors
        X_tensor = torch.tensor(X_combined, dtype=torch.float32)
        y_tensor = torch.tensor(y_target, dtype=torch.float32)
        
        return X_tensor, y_tensor
    
    def extract_fft_features(self, acc_window):
        """
        Extract FFT-based features from the acceleration window.
        
        Parameters:
            acc_window (np.ndarray): Acceleration data of shape (window_size, 6)
        
        Returns:
            np.ndarray: Extracted FFT features of shape (12,)
        """
        N = acc_window.shape[0]
        fs = CONFIG['SAMPLING_RATE']
        fft_features = []
        for i in range(acc_window.shape[1]):
            signal = acc_window[:, i]
            fft_vals = np.fft.fft(signal)
            fft_mag = np.abs(fft_vals)[:N//2]  # Positive frequencies
            
            mean_mag = np.mean(fft_mag)
            dominant_freq_index = np.argmax(fft_mag)
            dominant_freq = (dominant_freq_index * fs) / N  # Convert index to frequency in Hz
            
            fft_features.extend([mean_mag, dominant_freq])
        
        fft_features = np.array(fft_features)
        return fft_features

# -----------------------------
# 7. Model Definitions
# -----------------------------

class PositionalEncoding(nn.Module):
    """
    Positional Encoding module injects information about the relative or absolute position
    of the tokens in the sequence. The positional encodings have the same dimension as the
    embeddings so that the two can be summed.
    """
    def __init__(self, d_model, max_len=5000, dropout=0.0):
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
    def __init__(self, input_dim, window_size, model_dim=128, num_heads=2, num_layers=1, dropout=0.2):
        super(TransformerModel, self).__init__()
        self.input_dim = input_dim
        self.window_size = window_size
        self.model_dim = model_dim
        self.num_heads = num_heads

        # Convolutional layers with LeakyReLU activation
        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(16)
        self.activation = nn.LeakyReLU(negative_slope=0.01)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(32)

        # Positional Encoding
        self.positional_encoding = PositionalEncoding(d_model=self.model_dim, max_len=self.window_size)

        # Linear projection to match model_dim
        self.input_projection = nn.Linear(32, self.model_dim)

        # Transformer Encoder Layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.model_dim,
            nhead=self.num_heads,
            dim_feedforward=self.model_dim * 4,
            dropout=dropout,
            activation='gelu',  # Use GELU activation for transformer
            batch_first=True,   # Set batch_first=True for consistency
            norm_first=True     # Pre-Layer Normalization for stability
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
        and Kaiming normal initialization for convolutional layers.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
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
            x (Tensor): Input tensor of shape (batch_size, window_size, input_dim=20)

        Returns:
            out (Tensor): Output tensor of shape (batch_size)
        """
        # Input shape: (batch_size, window_size, input_dim=20)
        # Permute to (batch_size, input_dim=20, window_size=30) for Conv1d
        x = x.permute(0, 2, 1)

        # Convolutional layers with BatchNorm and LeakyReLU activation
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.activation(x)

        # Permute back to (batch_size, window_size, features=32)
        x = x.permute(0, 2, 1)

        # Linear projection to match model_dim
        x = self.input_projection(x)

        # Add positional encoding
        x = self.positional_encoding(x)

        # Transformer Encoder
        x = self.transformer_encoder(x)

        # Pooling: Take the mean across the time dimension
        x = torch.mean(x, dim=1)

        x = self.dropout(x)

        # Output layer
        out = self.fc_out(x).squeeze(1)

        return out

class CNNModel(nn.Module):
    """
    Simple Convolutional Neural Network for Time Series Prediction.
    """

    def __init__(self, input_dim, window_size, num_classes=1):
        super(CNNModel, self).__init__()
        # Single convolutional layer
        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=16, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2)
        # Calculate the size after pooling
        pooled_size = window_size // 2  # One pooling layer with kernel_size=2
        self.fc1 = nn.Linear(16 * pooled_size, 32)
        self.fc2 = nn.Linear(32, num_classes)
    
    def forward(self, x):
        """

        :param x:
        :return:
        """
        x = x.permute(0, 2, 1)  # (batch_size, input_dim, window_size)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)  # (batch_size, 16, window_size/2)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc1(x)
        x = self.relu(x)
        out = self.fc2(x)
        return out.squeeze(1)  # Output shape: (batch_size,) 

# -----------------------------
# 8. Training Utilities
# -----------------------------
def initialize_weights(model):
    """
    Initialize model weights using Kaiming Normal initialization.
    """
    for m in model.modules():
        if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

def initialize_model(config, input_dim):
    """
    Initialize the model based on the specified type.
    """
    if config['MODEL_TYPE'] == 'transformer':
        model = TransformerModel(input_dim=input_dim, window_size=config['WINDOW_SIZE']).to(device)
    elif config['MODEL_TYPE'] == 'cnn':
        model = CNNModel(input_dim=input_dim, window_size=config['WINDOW_SIZE']).to(device)
    else:
        raise ValueError("Invalid MODEL_TYPE. Choose from 'transformer', 'cnn'.")
    
    initialize_weights(model)
    logging.info(f'{config["MODEL_TYPE"].capitalize()} Model initialized and weights initialized.')
    print(f'{config["MODEL_TYPE"].capitalize()} Model initialized and weights initialized.')
    return model

def train_model(model, train_loader, val_loader, criterion, optimizer, config, scheduler=None, checkpoint_path=CONFIG['CHECKPOINT_PATH']):
    """
    Train the model with gradient clipping and mixed precision, and save checkpoints.
    """
    set_seed()  # Ensure reproducibility
    scaler = GradScaler()
    best_loss = np.inf
    patience_counter = 0
    start_epoch = 1
    history = {'train_loss': [], 'val_loss': [], 'val_mae': [], 'val_r2': []}

    # Check if a checkpoint exists
    if os.path.exists(checkpoint_path):
        print("Checkpoint found! Loading...")
        checkpoint = torch.load(checkpoint_path)
        
        # Load model state
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print("Model state loaded.")
        else:
            print("Warning: 'model_state_dict' not found in checkpoint.")
        
        # Load optimizer state
        if 'optimizer_state_dict' in checkpoint and optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print("Optimizer state loaded.")
        else:
            print("Warning: 'optimizer_state_dict' not found in checkpoint.")
        
        # Load scheduler state
        if scheduler and 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict'] is not None:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            print("Scheduler state loaded.")
        else:
            if scheduler:
                print("Warning: 'scheduler_state_dict' not found in checkpoint.")
        
        # Load scaler state
        if 'scaler_state_dict' in checkpoint:
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
            print("Scaler state loaded.")
        else:
            print("Warning: 'scaler_state_dict' not found in checkpoint.")
        
        # Load training state
        if 'epoch' in checkpoint:
            start_epoch = checkpoint['epoch'] + 1
            print(f"Resuming from epoch {start_epoch}.")
        if 'best_loss' in checkpoint:
            best_loss = checkpoint['best_loss']
            print(f"Best loss so far: {best_loss:.4f}")
        if 'patience_counter' in checkpoint:
            patience_counter = checkpoint['patience_counter']
            print(f"Patience counter: {patience_counter}/{config['PATIENCE']}")

    use_autocast = torch.cuda.is_available()
    
    for epoch in range(start_epoch, config['EPOCHS'] + 1):
        model.train()
        train_losses = []
        loop = tqdm(train_loader, desc=f'Epoch {epoch}/{config["EPOCHS"]} - Training', leave=False)
        for X_batch, y_batch in loop:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            if use_autocast:
                with autocast():
                    outputs = model(X_batch)
                    loss = criterion(outputs.squeeze(), y_batch)
            else:
                outputs = model(X_batch)
                loss = criterion(outputs.squeeze(), y_batch)
            
            if use_autocast:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
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
                if use_autocast:
                    with autocast():
                        outputs = model(X_batch)
                        loss = criterion(outputs.squeeze(), y_batch)
                else:
                    outputs = model(X_batch)
                    loss = criterion(outputs.squeeze(), y_batch)
                val_losses.append(loss.item())
                all_preds.extend(outputs.cpu().numpy().flatten())
                all_targets.extend(y_batch.cpu().numpy().flatten())
                loop.set_postfix(loss=loss.item())
        
        avg_val_loss = np.mean(val_losses)
        mae = mean_absolute_error(all_targets, all_preds)
        r2 = r2_score(all_targets, all_preds)

        # Update history with metrics
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['val_mae'].append(mae)
        history['val_r2'].append(r2)
        
        # Log and print metrics
        logging.info(f'Epoch {epoch}: Train Loss={avg_train_loss:.4f}, Val Loss={avg_val_loss:.4f}, MAE={mae:.4f}, R²={r2:.4f}')
        print(f'Epoch {epoch}: Train Loss={avg_train_loss:.4f}, Val Loss={avg_val_loss:.4f}, MAE={mae:.4f}, R²={r2:.4f}')
        
        # Scheduler step
        if scheduler:
            scheduler.step(avg_val_loss)
        
        # Early Stopping and Checkpointing
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'scaler_state_dict': scaler.state_dict(),
                'epoch': epoch,
                'best_loss': best_loss,
                'patience_counter': patience_counter
            }
            torch.save(checkpoint, checkpoint_path)
            logging.info(f'Validation loss decreased. Saving model.')
            patience_counter = 0
        else:
            patience_counter += 1
            logging.info(f'Validation loss did not improve. Patience: {patience_counter}/{config["PATIENCE"]}')
            print(f'Validation loss did not improve. Patience: {patience_counter}/{config["PATIENCE"]}')
            if patience_counter >= config['PATIENCE']:
                logging.info('Early stopping triggered.')
                print('Early stopping triggered.')
                break

        # Save first checkpoint after each epoch
        if epoch <= 1:
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'scaler_state_dict': scaler.state_dict(),
                'epoch': epoch,
                'best_loss': best_loss,
                'patience_counter': patience_counter
            }
            torch.save(checkpoint, checkpoint_path)
            print(f"First checkpoint saved at epoch {epoch}.")

    return model, history

# -----------------------------
# 9. Evaluation and Plotting Functions
# -----------------------------

def evaluate_last_checkpoint(config):
    """
    Loads the latest checkpoint, evaluates the model on the validation set,
    and generates evaluation plots.
    """
    # -----------------------------
    # 1. Load and Preprocess Data
    # -----------------------------
    print("Loading and preprocessing data...")
    train_df, val_df, test_df = load_and_preprocess_data(config)

    # plot_mass_distribution(train_df, 'Training Data Mass Distribution', CONFIG['PLOT_DIR'])
    # plot_mass_distribution(val_df, 'Validation Data Mass Distribution', CONFIG['PLOT_DIR'])
    
    # -----------------------------
    # 2. Initialize Datasets
    # -----------------------------
    print("Initializing datasets...")
    # Load scalers

    if os.path.exists(f"./scalers/feature_scaler_{config['MODEL_TYPE']}_{config['BATCH_SIZE']}.joblib") and os.path.exists(f"feature_scaler_{config['MODEL_TYPE']}_{config['BATCH_SIZE']}.joblib"):
        feature_scaler = joblib.load(f"./scalers/feature_scaler_{config['MODEL_TYPE']}_{config['BATCH_SIZE']}.joblib")
        target_scaler = joblib.load(f"./scalers/target_scaler_{config['MODEL_TYPE']}_{config['BATCH_SIZE']}.joblib")
        print("Scalers loaded from joblib files.")
    else:
        raise FileNotFoundError(f"Scaler files './scalers/feature_scaler_{config['MODEL_TYPE']}_{config['BATCH_SIZE']}.joblib'",
                                f"and/or './scalers/target_scaler_{config['MODEL_TYPE']}_{config['BATCH_SIZE']}.joblib not found."
                                "Ensure that you have trained the model and saved the scalers.")
    
    # Initialize validation dataset
    test_dataset = TimeSeriesDataset(
        df=test_df,
        window_size=config['WINDOW_SIZE'],
        feature_scaler=feature_scaler,
        target_scaler=target_scaler,
        fit_scalers=False,
        use_fft=True
    )
    
    # Create DataLoader for validation
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['BATCH_SIZE'],
        shuffle=False,
        num_workers=config['NUM_WORKERS']
    )
    
    # -----------------------------
    # 3. Initialize and Load Model
    # -----------------------------
    print("Initializing model...")
    input_dim = test_dataset.X_time.shape[1] + 12 if test_dataset.use_fft else test_dataset.X_time.shape[1]
    model = initialize_model(config, input_dim=input_dim)
    
    # Load the latest checkpoint
    checkpoint_path = config.get('CHECKPOINT_PATH', config['CHECKPOINT_PATH'])
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print("Checkpoint loaded successfully.")
        else:
            raise KeyError("'model_state_dict' not found in the checkpoint.")
        model.to(device)
        model.eval()
    else:
        raise FileNotFoundError(f"Checkpoint file '{checkpoint_path}' not found.")
    
    # -----------------------------
    # 4. Evaluate the Model
    # -----------------------------
    print("Evaluating the model...")

    evaluate_model(model, test_loader, device, feature_scaler, target_scaler, plot_dir=config['PLOT_DIR'])
    
    print("Evaluation and plotting completed. Check the 'plots' directory for visualizations.")


# -----------------------------
# 9. Optuna Objective Function
# -----------------------------
def objective(trial):
    """
    Objective function for Optuna optimization.
    """
    # Suggest hyperparameters
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-3)
    weight_decay = trial.suggest_loguniform('weight_decay', 1e-6, 1e-3)
    dropout = trial.suggest_uniform('dropout', 0.1, 0.5)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
    window_size = trial.suggest_int('window_size', 20, 60, step=10)

    # Update CONFIG with suggested hyperparameters
    CONFIG['LEARNING_RATE'] = learning_rate
    CONFIG['WEIGHT_DECAY'] = weight_decay
    CONFIG['DROPOUT'] = dropout
    CONFIG['BATCH_SIZE'] = batch_size
    CONFIG['WINDOW_SIZE'] = window_size

    # Load and preprocess data
    train_df, val_df, test_df = load_and_preprocess_data(CONFIG)

    # Initialize datasets and dataloaders
    train_dataset = TimeSeriesDataset(
        df=train_df,
        window_size=CONFIG['WINDOW_SIZE'],
        fit_scalers=True,
        use_fft=True
    )
    val_dataset = TimeSeriesDataset(
        df=val_df,
        window_size=CONFIG['WINDOW_SIZE'],
        feature_scaler=train_dataset.feature_scaler,
        target_scaler=train_dataset.target_scaler,
        fit_scalers=False,
        use_fft=True
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG['BATCH_SIZE'],
        shuffle=True,
        num_workers=CONFIG['NUM_WORKERS']
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=CONFIG['BATCH_SIZE'],
        shuffle=False,
        num_workers=CONFIG['NUM_WORKERS']
    )

    # Initialize model
    input_dim = train_dataset.X_time.shape[1] + 12 if train_dataset.use_fft else train_dataset.X_time.shape[1]
    model = initialize_model(CONFIG, input_dim=input_dim)

    # Define loss, optimizer, and scheduler
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG['LEARNING_RATE'], weight_decay=CONFIG['WEIGHT_DECAY'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    # Train the model
    _, history = train_model(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        CONFIG,
        scheduler=scheduler
    )

    # Return the final validation loss
    final_val_loss = history['val_loss'][-1]
    return final_val_loss

# -----------------------------
# 10. Main Training Script
# -----------------------------
def main(config):
    # Load and preprocess data
    print("Loading and preprocessing data...")
    train_df, val_df, test_df = load_and_preprocess_data(config)
    
    # Initialize Datasets
    train_dataset = TimeSeriesDataset(
        df=train_df,
        window_size=config['WINDOW_SIZE'],
        fit_scalers=True,  # Fit scalers on training data
        use_fft=True
    )
    val_dataset = TimeSeriesDataset(
        df=val_df,
        window_size=config['WINDOW_SIZE'],
        feature_scaler=train_dataset.feature_scaler,
        target_scaler=train_dataset.target_scaler,
        fit_scalers=False,
        use_fft=True
    )
    test_dataset = TimeSeriesDataset(
        df=test_df,
        window_size=config['WINDOW_SIZE'],
        feature_scaler=train_dataset.feature_scaler,
        target_scaler=train_dataset.target_scaler,
        fit_scalers=False,
        use_fft=True
    )

    # Save the scalers using joblib
    os.makedirs("./scalers", exist_ok=True)
    joblib.dump(train_dataset.feature_scaler, f'./scalers/feature_scaler_{config["MODEL_TYPE"]}_{config["BATCH_SIZE"]}.joblib')
    joblib.dump(train_dataset.target_scaler, f'./scalers/target_scaler_{config["MODEL_TYPE"]}_{config["BATCH_SIZE"]}.joblib')
    logging.info('Feature and target scalers saved using joblib.')
    print("Scalers saved successfully.")
    
    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['BATCH_SIZE'], 
        shuffle=True, 
        num_workers=config['NUM_WORKERS']
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config['BATCH_SIZE'], 
        shuffle=False, 
        num_workers=config['NUM_WORKERS']
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['BATCH_SIZE'],
        shuffle=False,
        num_workers=config['NUM_WORKERS']
    )
    
    # Initialize Model
    input_dim = train_dataset.X_time.shape[1] + 12 if train_dataset.use_fft else train_dataset.X_time.shape[1]
    model = initialize_model(config, input_dim=input_dim)
    
    # Define Loss and Optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['LEARNING_RATE'], weight_decay=config['WEIGHT_DECAY'])
    
    # Learning Rate Scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )
    
    # Train the model
    model, history = train_model(
        model, 
        train_loader, 
        val_loader, 
        criterion, 
        optimizer, 
        config, 
        scheduler=scheduler,
        checkpoint_path=config['CHECKPOINT_PATH']
    )
    
    # Evaluate the model on the test set
    evaluate_model(model, test_loader, device, train_dataset.feature_scaler, train_dataset.target_scaler, plot_dir=config['PLOT_DIR'])
    
    # Plot training history
    plot_loss_curves(history, config['PLOT_DIR'])
    plot_validation_metrics(history, config['PLOT_DIR'])
    
    print("Training and evaluation completed. Check the 'plots' directory for visualizations.")

# -----------------------------
# 10. Evaluation Function
# -----------------------------
def evaluate_model(model, test_loader, device, feature_scaler, target_scaler, plot_dir):
    """
    Evaluate the model on the test set and generate plots.
    """
    model.eval()
    all_preds = []
    all_targets = []
    all_days = []

    with torch.no_grad():
        for X_batch, y_batch in tqdm(test_loader, desc='Evaluating on Test Set'):
            X_batch = X_batch.to(device)
            preds = model(X_batch)
            all_preds.extend(preds.cpu().numpy().flatten())
            all_targets.extend(y_batch.numpy().flatten())
            # Map days
            start_idx = len(all_preds) - len(X_batch)
            end_idx = start_idx + len(X_batch)
            all_days.extend(test_loader.dataset.days[start_idx:end_idx])

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    all_days = np.array(all_days)

    # Inverse transform
    all_preds = target_scaler.inverse_transform(all_preds.reshape(-1, 1)).flatten()
    all_targets = target_scaler.inverse_transform(all_targets.reshape(-1, 1)).flatten()

    # Calculate metrics
    mse = mean_squared_error(all_targets, all_preds)
    mae = mean_absolute_error(all_targets, all_preds)
    r2 = r2_score(all_targets, all_preds)
    print(f"Test MSE: {mse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")

    # Generate plots
    plot_predictions(model, all_preds, all_targets, all_days, device, feature_scaler, target_scaler, plot_dir)
    plot_predictions_stratified(model, all_preds, all_targets, all_days, device, feature_scaler, target_scaler, plot_dir)
    plot_residuals(model, all_preds, all_targets, all_days, device, feature_scaler, target_scaler, plot_dir)
    plot_log_residuals(model, all_preds, all_targets, all_days, device, feature_scaler, target_scaler, plot_dir)

# -----------------------------
# 11. Argument Parsing
# -----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train, Evaluate, or Optimize the Model")
    parser.add_argument('--mode', type=str, choices=['train', 'evaluate', 'optimize'], default='train',
                        help="Mode to run the script: 'train', 'evaluate', or 'optimize'.")
    parser.add_argument('--checkpoint', type=str, default='checkpoint.pth',
                        help="Path to the checkpoint file.")
    args = parser.parse_args()
    
    # Update the CONFIG with the checkpoint path if provided
    os.makedirs("./checkpoints", exist_ok=True)
    CONFIG['CHECKPOINT_PATH'] = f"./checkpoints/{args.checkpoint}"
    
    if args.mode == 'optimize':
        # Create Optuna study
        study = optuna.create_study(
            direction="minimize",
            study_name="hyperparameter_optimization",
            storage="sqlite:///optuna_study.db",  # Save progress to SQLite database
            load_if_exists=True
        )
        
        # Run optimization
        study.optimize(objective, n_trials=50)

        # Print and save best hyperparameters
        print("Best hyperparameters:", study.best_params)
        print("Best validation loss:", study.best_value)

        # Save study results to a CSV file
        study.trials_dataframe().to_csv("optuna_results.csv")
    
    elif args.mode == 'train':
        main(CONFIG)
    
    elif args.mode == 'evaluate':
        evaluate_last_checkpoint(CONFIG)