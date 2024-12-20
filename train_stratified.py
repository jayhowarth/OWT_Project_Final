import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from tqdm import tqdm
import logging
import joblib
import argparse
import math
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
from datetime import datetime

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split

from plots import plot_predictions_stratified, plot_predictions, plot_hexbin, plot_distributions, plot_log_residuals, plot_residuals

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
    'FILE_PATTERN': '*.txt',  # Adjust if your files have a different extension
    'BATCH_SIZE': 32,         # Adjust based on GPU memory
    'EPOCHS': 50,
    'PATIENCE': 5,            # Increased patience for early stopping
    'CHECKPOINT_PATH': 'checkpoint.pth',
    'PLOT_DIR': 'plots',
    'MODEL_TYPE': 'transformer',  # Options: 'transformer', 'cnn'
    'WINDOW_SIZE': 30, 
    'SAMPLING_RATE': 30,      # Adjust based on data characteristics
    'LEARNING_RATE': 1e-5,    # Increased learning rate for better convergence
    'WEIGHT_DECAY': 1e-4,     # Increased weight decay for better regularization
    'NUM_WORKERS': 2,         # Adjust based on system
}

# -----------------------------
# 4. Configure Logging
# -----------------------------
date_time = datetime.now()
logging.basicConfig(
    filename=f"training_{CONFIG['BATCH_SIZE']}_{CONFIG['LEARNING_RATE']}_{CONFIG['WEIGHT_DECAY']}.log",
    filemode='a',
    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
    datefmt='%H:%M:%S',
    level=logging.INFO
)

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

def load_and_preprocess_data(config):
    """
    Load data from multiple text files in a directory and perform initial preprocessing.
    """
    data_dir = config['DATA_DIR']
    file_pattern = config['FILE_PATTERN']
    window_size = config['WINDOW_SIZE']
    
    # Get list of all files matching the pattern
    file_list = glob.glob(os.path.join(data_dir, file_pattern))
    if not file_list:
        raise ValueError(f"No files found in {data_dir} with pattern {file_pattern}")
    
    all_days = []
    for file_idx, file_path in enumerate(tqdm(file_list, desc='Loading Files')):
        # Extract 'day' from filename or assign a unique identifier
        filename = os.path.basename(file_path)
        day_id = os.path.splitext(filename)[0]  # Assuming filename without extension is unique day identifier
        
        # Read the file
        try:
            df = pd.read_csv(file_path)
        except Exception as e:
            logging.error(f"Error reading {file_path}: {e}")
            continue
        
        # Ensure required columns exist
        required_columns = ['ax1(g)', 'az1(g)', 'ax2(g)', 'az2(g)', 'ax3(g)', 'az3(g)', 'Temp(C)', 'Rot_Speed(rpm)', 'Mass(g)']
        if not all(col in df.columns for col in required_columns):
            logging.error(f"Missing columns in {file_path}. Required columns: {required_columns}")
            continue
        
        # Add 'day' identifier
        df['day'] = day_id
        all_days.append(df)
    
    # Concatenate all data
    combined_df = pd.concat(all_days, ignore_index=True)
    
    # Drop any remaining NaN values
    combined_df = combined_df.dropna().reset_index(drop=True)
    
    # Remove outliers
    features_to_check = ['ax1(g)', 'az1(g)', 'ax2(g)', 'az2(g)', 'ax3(g)', 'az3(g)', 'Temp(C)', 'Rot_Speed(rpm)', 'Mass(g)']
    
    # Remove outliers using Isolation Forest
    combined_df = remove_outliers_isolation_forest(combined_df, features_to_check, contamination=0.01)
    
    # Now perform stratified splitting into train, validation, and test sets
    train_df, val_df, test_df = stratified_train_val_test_split(combined_df)
    
    logging.info(f'Total files processed: {len(file_list)}')
    logging.info(f'Samples after outlier removal: {len(combined_df)}')
    logging.info(f'Training samples: {len(train_df)}')
    logging.info(f'Validation samples: {len(val_df)}')
    logging.info(f'Test samples: {len(test_df)}')
    
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
    def __init__(self, input_dim, window_size, model_dim=128, num_heads=2, num_layers=1, dropout=0.3):
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
        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(16)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(32)
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        # Calculate the size after pooling
        pooled_size = window_size // 4  # Two pooling layers with kernel_size=2
        self.fc1 = nn.Linear(32 * pooled_size, 64)
        self.fc2 = nn.Linear(64, num_classes)
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        x = x.permute(0, 2, 1)  # (batch_size, num_features=20, window_size=30)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x)     # (batch_size, 16, window_size/2)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.pool2(x)    # (batch_size, 32, window_size/4)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
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
            torch.save(model.state_dict(), config['CHECKPOINT_PATH'])
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

        # Save checkpoint after each epoch
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
        print(f"Checkpoint saved at epoch {epoch}.")

    return model, history

    # -----------------------------
    # 9. Evaluation and Plotting Functions
    # -----------------------------

# def stratify_by_day_and_mass(predictions, targets, days, samples_per_day, mass_bins=5):
#     """
#     Stratify predictions and targets by day and mass bins, then sample a fixed number of readings per combination.
    
#     Parameters:
#         predictions (np.ndarray): Array of predicted values.
#         targets (np.ndarray): Array of actual target values.
#         days (np.ndarray): Array indicating the day for each sample.
#         samples_per_day (int): Number of samples to select for each day-mass bin combination.
#         mass_bins (int): Number of mass bins to divide the mass range into.
    
#     Returns:
#         np.ndarray, np.ndarray, np.ndarray: Arrays of selected predictions, targets, and days.
#     """
#     selected_preds = []
#     selected_targets = []
#     selected_days = []
    
#     unique_days = np.unique(days)
#     # Define mass bins based on the overall mass range
#     mass_min = targets.min()
#     mass_max = targets.max()
#     bins = np.linspace(mass_min, mass_max, mass_bins + 1)
    
#     for day in unique_days:
#         day_indices = np.where(days == day)[0]
#         day_targets = targets[day_indices]
        
#         # Assign each target to a mass bin
#         bin_indices = np.digitize(day_targets, bins) - 1  # bins are 1-indexed
#         unique_bins = np.unique(bin_indices)
        
#         for bin_idx in unique_bins:
#             bin_mask = bin_indices == bin_idx
#             bin_indices_subset = day_indices[bin_mask]
#             if len(bin_indices_subset) >= samples_per_day:
#                 sampled_indices = np.random.choice(bin_indices_subset, samples_per_day, replace=False)
#             else:
#                 sampled_indices = bin_indices_subset
#                 print(f"Warning: Day '{day}', Mass Bin '{bin_idx}' has only {len(bin_indices_subset)} samples, less than {samples_per_day}.")
            
#             selected_preds.extend(predictions[sampled_indices])
#             selected_targets.extend(targets[sampled_indices])
#             selected_days.extend(days[sampled_indices])
    
#     return np.array(selected_preds), np.array(selected_targets), np.array(selected_days)

# def plot_distributions(targets, preds, plot_dir):
#     """
#     Plots histograms of actual and predicted mass values.
#     """
#     plt.figure(figsize=(12, 6))
    
#     # Plot Actual Mass Distribution
#     plt.subplot(1, 2, 1)
#     plt.hist(targets, bins=50, color='blue', alpha=0.7)
#     plt.title('Actual Mass Distribution')
#     plt.xlabel('Mass (g)')
#     plt.ylabel('Frequency')
    
#     # Plot Predicted Mass Distribution
#     plt.subplot(1, 2, 2)
#     plt.hist(preds, bins=50, color='green', alpha=0.7)
#     plt.title('Predicted Mass Distribution')
#     plt.xlabel('Mass (g)')
#     plt.ylabel('Frequency')
    
#     plt.tight_layout()
#     os.makedirs(plot_dir, exist_ok=True)
#     plt.savefig(os.path.join(plot_dir, 'mass_distributions.png'))
#     plt.close()
#     print("Mass distributions plot saved.")

# def plot_hexbin(targets, preds, plot_dir):
#     """
#     Creates a hexbin plot for Actual vs Predicted Mass.
#     """
#     plt.figure(figsize=(10, 6))
#     plt.hexbin(targets, preds, gridsize=50, cmap='Blues', mincnt=1)
#     plt.colorbar(label='Counts')
#     plt.xlabel('Actual Mass (g)')
#     plt.ylabel('Predicted Mass (g)')
#     plt.title('Hexbin Plot of Actual vs Predicted Mass')
#     plt.plot([targets.min(), targets.max()], [targets.min(), targets.max()], 'r--', label='Perfect Prediction')
#     plt.legend()
#     plt.tight_layout()
#     plt.savefig(os.path.join(plot_dir, 'hexbin_actual_vs_predicted.png'))
#     plt.close()
#     print("Hexbin Actual vs Predicted plot saved.")

# def plot_interactive_actual_vs_predicted(targets, preds, days, plot_dir):
#     """
#     Creates an interactive scatter plot for Actual vs Predicted Mass.
#     """

#     # Create a DataFrame for Plotly
#     df_plot = pd.DataFrame({
#         'Actual Mass (g)': targets,
#         'Predicted Mass (g)': preds,
#         'Day': days
#     })
    
#     fig = px.scatter(
#         df_plot, 
#         x='Actual Mass (g)', 
#         y='Predicted Mass (g)', 
#         color='Day',
#         title='Interactive Actual vs Predicted Mass',
#         labels={'Actual Mass (g)': 'Actual Mass (g)', 'Predicted Mass (g)': 'Predicted Mass (g)'},
#         hover_data=['Day']
#     )
    
#     # Add Perfect Prediction Line
#     fig.add_shape(
#         type='line',
#         x0=df_plot['Actual Mass (g)'].min(),
#         y0=df_plot['Actual Mass (g)'].min(),
#         x1=df_plot['Actual Mass (g)'].max(),
#         y1=df_plot['Actual Mass (g)'].max(),
#         line=dict(color='Red', dash='dash'),
#         name='Perfect Prediction'
#     )
    
#     # Save as HTML
#     os.makedirs(plot_dir, exist_ok=True)
#     fig.write_html(os.path.join(plot_dir, 'interactive_actual_vs_predicted.html'))
#     print("Interactive Actual vs Predicted plot saved as HTML.")

# def plot_predictions(model, val_loader, device, feature_scaler, target_scaler, plot_dir):
#     """
#     Generates Actual vs Predicted Mass plot without color map or legend.
#     """
#     model.eval()
#     all_preds = []
#     all_targets = []
#     all_days = []
    
#     for X_batch, y_batch in tqdm(val_loader, desc='Predicting'):
#         X_batch = X_batch.to(device)
#         with torch.no_grad():
#             preds = model(X_batch)
#         all_preds.extend(preds.cpu().numpy().flatten())
#         all_targets.extend(y_batch.numpy().flatten())
#         # To correctly map days, ensure the indices align
#         start_idx = len(all_preds) - len(X_batch)
#         end_idx = start_idx + len(X_batch)
#         all_days.extend(val_loader.dataset.days[start_idx:end_idx])
    
#     all_preds = np.array(all_preds)
#     all_targets = np.array(all_targets)
    
#     # Apply Inverse Transform
#     all_preds = target_scaler.inverse_transform(all_preds.reshape(-1, 1)).flatten()
#     all_targets = target_scaler.inverse_transform(all_targets.reshape(-1, 1)).flatten()
    
#     # Debugging: Print min and max values
#     print(f"Actual Mass - Min: {all_targets.min()}, Max: {all_targets.max()}")
#     print(f"Predicted Mass - Min: {all_preds.min()}, Max: {all_preds.max()}")
    
#     # Plot Actual vs Predicted
#     plt.figure(figsize=(10, 6))
#     plt.scatter(
#         all_targets, 
#         all_preds, 
#         alpha=0.5, 
#         s=10, 
#         color="green"  # Single color for points
#     )
    
#     # Plot Perfect Prediction Line
#     min_val = min(all_targets.min(), all_preds.min())
#     max_val = max(all_targets.max(), all_preds.max())
#     plt.plot([min_val, max_val], [min_val, max_val], 'r--')  # Perfect prediction line
    
#     plt.xlabel('Actual Mass (g)')
#     plt.ylabel('Predicted Mass (g)')
#     plt.title('Actual vs Predicted Mass')
    
#     os.makedirs(plot_dir, exist_ok=True)
#     plt.savefig(os.path.join(plot_dir, 'actual_vs_predicted.png'))
#     plt.close()
#     print("Actual vs Predicted plot saved.")
    
#     # Plot Hexbin
#     plot_hexbin(all_targets, all_preds, plot_dir)

# def plot_predictions_stratified(model, val_loader, device, feature_scaler, target_scaler, plot_dir):
#     """
#     Generates Actual vs Predicted Mass plot with stratified sampling.
#     """
#     model.eval()
#     all_preds = []
#     all_targets = []
#     all_days = []
    
#     for X_batch, y_batch in tqdm(val_loader, desc='Predicting'):
#         X_batch = X_batch.to(device)
#         with torch.no_grad():
#             preds = model(X_batch)
#         all_preds.extend(preds.cpu().numpy().flatten())
#         all_targets.extend(y_batch.numpy().flatten())
#         # To correctly map days, ensure the indices align
#         start_idx = len(all_preds) - len(X_batch)
#         end_idx = start_idx + len(X_batch)
#         all_days.extend(val_loader.dataset.days[start_idx:end_idx])
    
#     all_preds = np.array(all_preds)
#     all_targets = np.array(all_targets)
#     all_days = np.array(all_days)
    
#     # Apply Inverse Transform
#     all_preds = target_scaler.inverse_transform(all_preds.reshape(-1, 1)).flatten()
#     all_targets = target_scaler.inverse_transform(all_targets.reshape(-1, 1)).flatten()
    
#     # Debugging: Print min and max values
#     print(f"Actual Mass - Min: {all_targets.min()}, Max: {all_targets.max()}")
#     print(f"Predicted Mass - Min: {all_preds.min()}, Max: {all_preds.max()}")
    
#     # Plot distributions
#     plot_distributions(all_targets, all_preds, plot_dir)
    
#     # Stratified Sampling: Fixed number of samples per day and mass bin
#     print("Stratifying samples by day and mass bins...")
#     preds_strat, targets_strat, days_strat = stratify_by_day_and_mass(
#         predictions=all_preds, 
#         targets=all_targets, 
#         days=all_days, 
#         samples_per_day=20,  # Adjust to control downsampling
#         mass_bins=10  # Number of bins for stratification
#     )
    
#     # Mapping days to numbers for coloring
#     unique_days_sorted = np.unique(days_strat)
#     day_to_num = {day: idx for idx, day in enumerate(unique_days_sorted)}
#     numeric_days = np.array([day_to_num[day] for day in days_strat])
    
#     # Plot Actual vs Predicted
#     plt.figure(figsize=(10, 6))
#     plt.scatter(
#         all_targets, 
#         all_preds, 
#         alpha=0.5, 
#         s=10, 
#         color="blue"  # Single color for points
#     )
    
#     # Create colorbar with correct labels
#     # cbar = plt.colorbar(scatter, ticks=np.arange(len(unique_days_sorted)))
#     # cbar.ax.set_yticklabels(unique_days_sorted)
#     # cbar.set_label('Day')
    
#     plt.xlabel('Actual Mass (g)')
#     plt.ylabel('Predicted Mass (g)')
#     plt.title('Actual vs Predicted Mass (Downsampled)')
    
#     # Plot Perfect Prediction Line
#     min_val = min(targets_strat.min(), preds_strat.min())
#     max_val = max(targets_strat.max(), preds_strat.max())
#     plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')
    
#     plt.legend()
#     os.makedirs(plot_dir, exist_ok=True)
#     plt.savefig(os.path.join(plot_dir, 'actual_vs_predicted_downsampled.png'))
#     plt.close()
#     print("Actual vs Predicted plot saved.")

# def plot_log_residuals(model, val_loader, device, feature_scaler, target_scaler, plot_dir):
#     """
#     Generates Residual Plot with natural logarithm transformation and downsampling.
#     """
#     model.eval()
#     all_preds = []
#     all_targets = []
#     all_days = []

#     for X_batch, y_batch in tqdm(val_loader, desc='Predicting for Residuals'):
#         X_batch = X_batch.to(device)
#         with torch.no_grad():
#             preds = model(X_batch)
#         all_preds.extend(preds.cpu().numpy().flatten())
#         all_targets.extend(y_batch.numpy().flatten())
#         # To correctly map days, ensure the indices align
#         start_idx = len(all_preds) - len(X_batch)
#         end_idx = start_idx + len(X_batch)
#         all_days.extend(val_loader.dataset.days[start_idx:end_idx])

#     all_preds = np.array(all_preds)
#     all_targets = np.array(all_targets)
#     all_days = np.array(all_days)

#     # Apply Inverse Transform
#     all_preds = target_scaler.inverse_transform(all_preds.reshape(-1, 1)).flatten()
#     all_targets = target_scaler.inverse_transform(all_targets.reshape(-1, 1)).flatten()

#     # Calculate Residuals
#     residuals = all_targets - all_preds

#     # Apply Natural Log Transformation to Residuals
#     residuals_ln = np.sign(residuals) * np.log1p(np.abs(residuals))

#     # Stratified Sampling: Downsample residuals
#     print("Stratifying residuals by day and mass bins...")
#     preds_strat, targets_strat, days_strat = stratify_by_day_and_mass(
#         predictions=all_preds,
#         targets=all_targets,
#         days=all_days,
#         samples_per_day=20,  # Adjust for clarity
#         mass_bins=10
#     )
#     residuals_strat = targets_strat - preds_strat
#     residuals_ln_strat = np.sign(residuals_strat) * np.log1p(np.abs(residuals_strat))

#     # Plot Residuals Distribution (Natural Log Transformation)
#     plt.figure(figsize=(12, 6))

#     # Histogram of residuals (log-transformed)
#     plt.subplot(1, 2, 1)
#     plt.hist(residuals_ln_strat, bins=50, color='purple', alpha=0.7)
#     plt.title('Log-Transformed Residuals Distribution')
#     plt.xlabel('Log Residual (g)')
#     plt.ylabel('Frequency')

#     # Scatter plot of residuals vs actual (log-transformed residuals)
#     plt.subplot(1, 2, 2)
#     plt.scatter(targets_strat, residuals_ln_strat, alpha=0.5, s=10, color='purple')
#     plt.axhline(0, color='red', linestyle='--')
#     plt.title('Log-Transformed Residuals vs Actual Mass')
#     plt.xlabel('Actual Mass (g)')
#     plt.ylabel('Log Residual (g)')

#     plt.tight_layout()
#     os.makedirs(plot_dir, exist_ok=True)
#     plt.savefig(os.path.join(plot_dir, 'log_transformed_residuals_analysis.png'))
#     plt.close()
#     print("Log-transformed residuals analysis plots saved.")

# def plot_residuals(model, val_loader, device, feature_scaler, target_scaler, plot_dir):
    """
    Generates Residual Plot.
    """
    model.eval()
    all_preds = []
    all_targets = []
    all_days = []
    
    for X_batch, y_batch in tqdm(val_loader, desc='Predicting for Residuals'):
        X_batch = X_batch.to(device)
        with torch.no_grad():
            preds = model(X_batch)
        all_preds.extend(preds.cpu().numpy().flatten())
        all_targets.extend(y_batch.numpy().flatten())
        # To correctly map days, ensure the indices align
        start_idx = len(all_preds) - len(X_batch)
        end_idx = start_idx + len(X_batch)
        all_days.extend(val_loader.dataset.days[start_idx:end_idx])
    
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    all_days = np.array(all_days)
    
    # Apply Inverse Transform
    all_preds = target_scaler.inverse_transform(all_preds.reshape(-1, 1)).flatten()
    all_targets = target_scaler.inverse_transform(all_targets.reshape(-1, 1)).flatten()
    
    # Calculate Residuals
    residuals = all_targets - all_preds
    
    # Plot Residuals Distribution
    plt.figure(figsize=(12, 6))
    
    # Histogram of residuals
    plt.subplot(1, 2, 1)
    plt.hist(residuals, bins=50, color='purple', alpha=0.7)
    plt.title('Residuals Distribution')
    plt.xlabel('Residual (g)')
    plt.ylabel('Frequency')
    
    # Scatter plot of residuals vs actual
    plt.subplot(1, 2, 2)
    plt.scatter(all_targets, residuals, alpha=0.5, s=10, color='purple')
    plt.axhline(0, color='red', linestyle='--')
    plt.title('Residuals vs Actual Mass')
    plt.xlabel('Actual Mass (g)')
    plt.ylabel('Residual (g)')
    
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'residuals_analysis.png'))
    plt.close()
    print("Residuals analysis plots saved.")

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
    if os.path.exists(f"feature_scaler_{config['MODEL_TYPE']}_{config['BATCH_SIZE']}.joblib") and os.path.exists(f"feature_scaler_{config['MODEL_TYPE']}_{config['BATCH_SIZE']}.joblib"):
        feature_scaler = joblib.load(f"feature_scaler_{config['MODEL_TYPE']}_{config['BATCH_SIZE']}.joblib")
        target_scaler = joblib.load(f"target_scaler_{config['MODEL_TYPE']}_{config['BATCH_SIZE']}.joblib")
        print("Scalers loaded from joblib files.")
    else:
        raise FileNotFoundError(f"Scaler files 'feature_scaler_{config['MODEL_TYPE']}_{config['BATCH_SIZE']}.joblib'",
                                f"and/or 'target_scaler_{config['MODEL_TYPE']}_{config['BATCH_SIZE']}.joblib not found."
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

# def plot_loss_curves(history, plot_dir):
    """
    Plots training and validation loss curves.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Curves')
    plt.legend()
    os.makedirs(plot_dir, exist_ok=True)
    plt.savefig(os.path.join(plot_dir, 'loss_curves.png'))
    plt.close()
    print("Loss curves plot saved.")

def plot_validation_metrics(history, plot_dir):
    """
    Plots MAE and R² over epochs.
    """
    plt.figure(figsize=(12, 5))
    
    # MAE
    plt.subplot(1, 2, 1)
    plt.plot(history['val_mae'], label='Validation MAE', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.title('Validation MAE Over Epochs')
    plt.legend()
    
    # R² Score
    plt.subplot(1, 2, 2)
    plt.plot(history['val_r2'], label='Validation R²', color='green')
    plt.xlabel('Epoch')
    plt.ylabel('R² Score')
    plt.title('Validation R² Over Epochs')
    plt.legend()
    
    plt.tight_layout()
    os.makedirs(plot_dir, exist_ok=True)
    plt.savefig(os.path.join(plot_dir, 'validation_metrics.png'))
    plt.close()
    print("Validation metrics plot saved.")

# def plot_mass_distribution(df, title, plot_dir):
#     plt.figure(figsize=(8, 6))
#     plt.hist(df['Mass(g)'], bins=100, color='skyblue', edgecolor='black')
#     plt.title(title)
#     plt.xlabel('Mass (g)')
#     plt.ylabel('Frequency')
#     plt.grid(True)
#     plt.tight_layout()
#     os.makedirs(plot_dir, exist_ok=True)
#     plt.savefig(os.path.join(plot_dir, f'{title.replace(" ", "_").lower()}.png'))
#     plt.close()
#     print(f"{title} plot saved.")

class IdentityScaler:
    """
    A scaler that performs no scaling. It returns the data as-is.
    Useful for bypassing scaling operations.
    """
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        return X
    
    def inverse_transform(self, X, y=None):
        return X

# -----------------------------
# 9. Main Training Script
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
    joblib.dump(train_dataset.feature_scaler, 'feature_scaler.joblib')
    joblib.dump(train_dataset.target_scaler, 'target_scaler.joblib')
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
    parser = argparse.ArgumentParser(description="Train, Evaluate, or Plot the Model")
    parser.add_argument('--mode', type=str, choices=['train', 'evaluate', 'plot'], default='train',
                        help="Mode to run the script in: 'train' to train the model, 'evaluate' to evaluate the latest checkpoint, 'plot' to plot from a checkpoint.")
    parser.add_argument('--checkpoint', type=str, default='checkpoint.pth',
                        help="Path to the checkpoint file. Default is 'checkpoint.pth'.")
    args = parser.parse_args()
    
    # Update the CONFIG with the checkpoint path if provided
    CONFIG['CHECKPOINT_PATH'] = args.checkpoint
    
    if args.mode == 'train':
        main(CONFIG)
    elif args.mode == 'evaluate':
        evaluate_last_checkpoint(CONFIG)
    elif args.mode == 'plot':
        evaluate_last_checkpoint(CONFIG)  # Reuse evaluation for plotting