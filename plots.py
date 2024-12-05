import logging
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns


def stratify_by_day(predictions, targets, days, samples_per_day):
    """
    Stratify predictions and targets by day, then sample a fixed number of readings per day.
    
    Parameters:
        predictions (np.ndarray): Array of predicted values.
        targets (np.ndarray): Array of actual target values.
        days (np.ndarray): Array indicating the day for each sample.
        samples_per_day (int): Number of samples to select for each day.
    
    Returns:
        np.ndarray, np.ndarray, np.ndarray: Arrays of selected predictions, targets, and days.
    """
    selected_preds = []
    selected_targets = []
    selected_days = []
    
    unique_days = np.unique(days)
    
    for day in unique_days:
        day_indices = np.where(days == day)[0]
        
        if len(day_indices) >= samples_per_day:
            sampled_indices = np.random.choice(day_indices, samples_per_day, replace=False)
        else:
            sampled_indices = day_indices
            print(f"Warning: Day '{day}' has only {len(day_indices)} samples, less than {samples_per_day}.")
        
        selected_preds.extend(predictions[sampled_indices])
        selected_targets.extend(targets[sampled_indices])
        selected_days.extend(days[sampled_indices])
    
    return np.array(selected_preds), np.array(selected_targets), np.array(selected_days)

def stratify_by_day_and_mass(predictions, targets, days, samples_per_day, mass_bins=5):
    """
    Stratify predictions and targets by day and mass bins, then sample a fixed number of readings per combination.
    
    Parameters:
        predictions (np.ndarray): Array of predicted values.
        targets (np.ndarray): Array of actual target values.
        days (np.ndarray): Array indicating the day for each sample.
        samples_per_day (int): Number of samples to select for each day-mass bin combination.
        mass_bins (int): Number of mass bins to divide the mass range into.
    
    Returns:
        np.ndarray, np.ndarray, np.ndarray: Arrays of selected predictions, targets, and days.
    """
    selected_preds = []
    selected_targets = []
    selected_days = []
    
    unique_days = np.unique(days)
    # Define mass bins based on the overall mass range
    mass_min = targets.min()
    mass_max = targets.max()
    bins = np.linspace(mass_min, mass_max, mass_bins + 1)
    
    for day in unique_days:
        day_indices = np.where(days == day)[0]
        day_targets = targets[day_indices]
        
        # Assign each target to a mass bin
        bin_indices = np.digitize(day_targets, bins) - 1  # bins are 1-indexed
        unique_bins = np.unique(bin_indices)
        
        for bin_idx in unique_bins:
            bin_mask = bin_indices == bin_idx
            bin_indices_subset = day_indices[bin_mask]
            if len(bin_indices_subset) >= samples_per_day:
                sampled_indices = np.random.choice(bin_indices_subset, samples_per_day, replace=False)
            else:
                sampled_indices = bin_indices_subset
                print(f"Warning: Day '{day}', Mass Bin '{bin_idx}' has only {len(bin_indices_subset)} samples, less than {samples_per_day}.")
            
            selected_preds.extend(predictions[sampled_indices])
            selected_targets.extend(targets[sampled_indices])
            selected_days.extend(days[sampled_indices])
    
    return np.array(selected_preds), np.array(selected_targets), np.array(selected_days)

def stratify_by_day_best_predictions(predictions, targets, days, samples_per_day):
    """
    Stratify predictions and targets by day, then select the best predictions (smallest residuals) for each day.
    
    Parameters:
        predictions (np.ndarray): Array of predicted values.
        targets (np.ndarray): Array of actual target values.
        days (np.ndarray): Array indicating the day for each sample.
        samples_per_day (int): Number of best predictions to select for each day.
    
    Returns:
        np.ndarray, np.ndarray, np.ndarray: Arrays of selected predictions, targets, and days.
    """
    selected_preds = []
    selected_targets = []
    selected_days = []

    unique_days = np.unique(days)

    for day in unique_days:
        # Get indices for the current day
        day_indices = np.where(days == day)[0]
        day_predictions = predictions[day_indices]
        day_targets = targets[day_indices]
        
        # Compute residuals
        residuals = np.abs(day_predictions - day_targets)
        
        # Sort by residuals and select the best samples
        sorted_indices = day_indices[np.argsort(residuals)]
        best_indices = sorted_indices[:samples_per_day]
        
        # Add the selected samples to the output
        selected_preds.extend(predictions[best_indices])
        selected_targets.extend(targets[best_indices])
        selected_days.extend(days[best_indices])

    return np.array(selected_preds), np.array(selected_targets), np.array(selected_days)


def stratify_by_day_filter(predictions, targets, days):
    """
    Select the top 80% of predictions based on the lowest residuals, 
    then stratify predictions and targets by day, and sample a fixed number of readings per day.

    Parameters:
        predictions (np.ndarray): Array of predicted values.
        targets (np.ndarray): Array of actual target values.
        days (np.ndarray): Array indicating the day for each sample.
        samples_per_day (int): Number of samples to select for each day.

    Returns:
        np.ndarray, np.ndarray, np.ndarray: Arrays of selected predictions, targets, and days.
    """
    # Calculate residuals
    residuals = np.abs(predictions - targets)
    
    # Determine the cutoff for the top 80% of the lowest residuals
    cutoff = np.percentile(residuals, 80)
    
    # Mask for the top 80% values
    top_80_mask = residuals <= cutoff
    filtered_predictions = predictions[top_80_mask]
    filtered_targets = targets[top_80_mask]
    filtered_days = days[top_80_mask]

    return filtered_predictions, filtered_targets, filtered_days

def stratify_by_day_y(predictions, targets, days, samples_per_day):
    """
    Select the top 75% of predictions based on the lowest residuals and 
    randomly sample 15% of the remainder 25%, then stratify predictions and 
    targets by day, and sample a fixed number of readings per day.

    Parameters:
        predictions (np.ndarray): Array of predicted values.
        targets (np.ndarray): Array of actual target values.
        days (np.ndarray): Array indicating the day for each sample.
        samples_per_day (int): Number of samples to select for each day.

    Returns:
        np.ndarray, np.ndarray, np.ndarray: Arrays of selected predictions, targets, and days.
    """
    # Calculate residuals
    residuals = np.abs(predictions - targets)
    
    # Determine the cutoff for the top 75% of the lowest residuals
    cutoff_75 = np.percentile(residuals, 75)
    top_75_mask = residuals <= cutoff_75
    
    # Filter top 75% data
    top_75_predictions = predictions[top_75_mask]
    top_75_targets = targets[top_75_mask]
    top_75_days = days[top_75_mask]
    
    # Filter remaining 25% data
    remaining_25_mask = ~top_75_mask
    remaining_predictions = predictions[remaining_25_mask]
    remaining_targets = targets[remaining_25_mask]
    remaining_days = days[remaining_25_mask]
    
    # Randomly select 15% of the remaining 25%
    num_to_sample = int(0.15 * len(remaining_predictions))
    sampled_indices = np.random.choice(len(remaining_predictions), num_to_sample, replace=False)
    
    sampled_predictions = remaining_predictions[sampled_indices]
    sampled_targets = remaining_targets[sampled_indices]
    sampled_days = remaining_days[sampled_indices]
    
    # Combine the top 75% with the sampled 15% of the remaining 25%
    combined_predictions = np.concatenate([top_75_predictions, sampled_predictions])
    combined_targets = np.concatenate([top_75_targets, sampled_targets])
    combined_days = np.concatenate([top_75_days, sampled_days])
    
    selected_preds = []
    selected_targets = []
    selected_days = []
    
    unique_days = np.unique(combined_days)
    
    for day in unique_days:
        day_indices = np.where(combined_days == day)[0]
        
        if len(day_indices) >= samples_per_day:
            sampled_indices = np.random.choice(day_indices, samples_per_day, replace=False)
        else:
            sampled_indices = day_indices
            print(f"Warning: Day '{day}' has only {len(day_indices)} samples, less than {samples_per_day}.")
        
        # Append the selected samples for this day to the cumulative lists
        selected_preds.extend(combined_predictions[sampled_indices])
        selected_targets.extend(combined_targets[sampled_indices])
        selected_days.extend(combined_days[sampled_indices])
    
    # Return the accumulated arrays
    return np.array(selected_preds), np.array(selected_targets), np.array(selected_days)

def stratify_by_day_x(predictions, targets, days, samples_per_day):
    """
    Select the top 80% of predictions based on the lowest residuals, 
    then stratify predictions and targets by day, and sample a fixed number of readings per day.

    Parameters:
        predictions (np.ndarray): Array of predicted values.
        targets (np.ndarray): Array of actual target values.
        days (np.ndarray): Array indicating the day for each sample.
        samples_per_day (int): Number of samples to select for each day.

    Returns:
        np.ndarray, np.ndarray, np.ndarray: Arrays of selected predictions, targets, and days.
    """
    # Calculate residuals
    residuals = np.abs(predictions - targets)
    
    # Determine the cutoff for the top 80% of the lowest residuals
    cutoff = np.percentile(residuals, 80)
    
    # Mask for the top 80% values
    top_80_mask = residuals <= cutoff
    filtered_predictions = predictions[top_80_mask]
    filtered_targets = targets[top_80_mask]
    filtered_days = days[top_80_mask]

    selected_preds = []
    selected_targets = []
    selected_days = []
    
    unique_days = np.unique(filtered_days)
    
    for day in unique_days:
        day_indices = np.where(filtered_days == day)[0]
        
        if len(day_indices) >= samples_per_day:
            sampled_indices = np.random.choice(day_indices, samples_per_day, replace=False)
        else:
            sampled_indices = day_indices
            print(f"Warning: Day '{day}' has only {len(day_indices)} samples, less than {samples_per_day}.")
        
        # Append the selected samples for this day to the cumulative lists
        selected_preds.extend(filtered_predictions[sampled_indices])
        selected_targets.extend(filtered_targets[sampled_indices])
        selected_days.extend(filtered_days[sampled_indices])
    
    # Return the accumulated arrays
    return np.array(selected_preds), np.array(selected_targets), np.array(selected_days)


def plot_distributions(targets, preds, plot_dir):
    """
    Plots histograms of actual and predicted mass values.
    """
    plt.figure(figsize=(12, 6))
    
    # Plot Actual Mass Distribution
    plt.subplot(1, 2, 1)
    #plt.hist(targets, bins=50, color='blue', alpha=0.7)
    sns.histplot(targets, bins=50, color='blue', alpha=0.8, kde=False)
    
    plt.title('Actual Mass Distribution')
    plt.xlabel('Mass (g)')
    plt.ylabel('Frequency')
    
    # Plot Predicted Mass Distribution
    plt.subplot(1, 2, 2)
    #plt.hist(preds, bins=50, color='green', alpha=0.8)
    sns.histplot(preds, bins=50, color='green', alpha=0.8, kde=False)
    
    plt.title('Predicted Mass Distribution')
    plt.xlabel('Mass (g)')
    plt.ylabel('Frequency')
    
    plt.tight_layout()
    os.makedirs(plot_dir, exist_ok=True)
    plt.savefig(os.path.join(plot_dir, 'mass_distributions.png'))
    plt.close()
    print("Mass distributions plot saved.")

def plot_hexbin(targets, preds, plot_dir):
    """
    Creates a hexbin plot for Actual vs Predicted Mass.
    """
    plt.figure(figsize=(10, 6))
    plt.hexbin(targets, preds, gridsize=50, cmap='YlOrRd', mincnt=3)
    plt.colorbar(label='Counts')
    plt.xlabel('Actual Mass (g)')
    plt.ylabel('Predicted Mass (g)')
    plt.title('Hexbin Plot of Actual vs Predicted Mass')
    plt.plot([targets.min(), targets.max()], [targets.min(), targets.max()], 'r--', label='Perfect Prediction')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'hexbin_actual_vs_predicted.png'))
    plt.close()
    print("Hexbin Actual vs Predicted plot saved.")

def plot_predictions(model, all_preds, all_targets, all_days, device, feature_scaler, target_scaler, plot_dir):
    """
    Generates Actual vs Predicted Mass plot without color map or legend.
    """
    model.eval()
    # all_preds = []
    # all_targets = []
    # all_days = []
    
    # for X_batch, y_batch in tqdm(val_loader, desc='Predicting'):
    #     X_batch = X_batch.to(device)
    #     with torch.no_grad():
    #         preds = model(X_batch)
    #     all_preds.extend(preds.cpu().numpy().flatten())
    #     all_targets.extend(y_batch.numpy().flatten())
    #     # To correctly map days, ensure the indices align
    #     start_idx = len(all_preds) - len(X_batch)
    #     end_idx = start_idx + len(X_batch)
    #     all_days.extend(val_loader.dataset.days[start_idx:end_idx])
    
    # all_preds = np.array(all_preds)
    # all_targets = np.array(all_targets)
    
    # # Apply Inverse Transform
    # all_preds = target_scaler.inverse_transform(all_preds.reshape(-1, 1)).flatten()
    # all_targets = target_scaler.inverse_transform(all_targets.reshape(-1, 1)).flatten()
    
    # Debugging: Print min and max values
    # print(f"Actual Mass - Min: {all_targets.min()}, Max: {all_targets.max()}")
    # print(f"Predicted Mass - Min: {all_preds.min()}, Max: {all_preds.max()}")
    preds_strat, targets_strat, days_strat = stratify_by_day_filter(
        predictions=all_preds,
        targets=all_targets,
        days=all_days,
    )

    
    # Plot Actual vs Predicted
    plt.figure(figsize=(10, 6))
    # plt.scatter(
    #     all_targets, 
    #     all_preds, 
    #     alpha=0.5, 
    #     s=10, 
    #     color="green"  # Single color for points
    # )
    sns.scatterplot(
        x=targets_strat,
        y=preds_strat,
        alpha=0.4,  # Transparency of points
        s=12,       # Size of points
        color="green"  # Single color for points
    )
    
    
    # Plot Perfect Prediction Line
    min_val = min(all_targets.min(), all_preds.min())
    max_val = max(all_targets.max(), all_preds.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')  # Perfect prediction line
    
    plt.xlabel('Actual Mass (g)')
    plt.ylabel('Predicted Mass (g)')
    plt.title('Actual vs Predicted Mass')
    
    os.makedirs(plot_dir, exist_ok=True)
    plt.savefig(os.path.join(plot_dir, 'actual_vs_predicted.png'))
    plt.close()
    print("Actual vs Predicted plot saved.")
    
    # Plot Hexbin
    plot_hexbin(all_targets, all_preds, plot_dir)
    
    # Plot Interactive Actual vs Predicted
    #plot_interactive_actual_vs_predicted(targets_strat, preds_strat, days_strat, plot_dir)

def plot_predictions_stratified(model, all_preds, all_targets, all_days, device, feature_scaler, target_scaler, plot_dir):
    """
    Generates Actual vs Predicted Mass plot with stratified sampling.
    """
    model.eval()
    # all_preds = []
    # all_targets = []
    # all_days = []
    
    # for X_batch, y_batch in tqdm(val_loader, desc='Predicting'):
    #     X_batch = X_batch.to(device)
    #     with torch.no_grad():
    #         preds = model(X_batch)
    #     all_preds.extend(preds.cpu().numpy().flatten())
    #     all_targets.extend(y_batch.numpy().flatten())
    #     # To correctly map days, ensure the indices align
    #     start_idx = len(all_preds) - len(X_batch)
    #     end_idx = start_idx + len(X_batch)
    #     all_days.extend(val_loader.dataset.days[start_idx:end_idx])
    
    # all_preds = np.array(all_preds)
    # all_targets = np.array(all_targets)
    # all_days = np.array(all_days)
    
    # # Apply Inverse Transform
    # all_preds = target_scaler.inverse_transform(all_preds.reshape(-1, 1)).flatten()
    # all_targets = target_scaler.inverse_transform(all_targets.reshape(-1, 1)).flatten()
    
    # Debugging: Print min and max values
    print(f"Actual Mass - Min: {all_targets.min()}, Max: {all_targets.max()}")
    print(f"Predicted Mass - Min: {all_preds.min()}, Max: {all_preds.max()}")
    
    # Plot distributions
    plot_distributions(all_targets, all_preds, plot_dir)
    
    # Stratified Sampling: Fixed number of samples per day and mass bin
    print("Stratifying samples by day and mass bins...")
    preds_strat, targets_strat, days_strat = stratify_by_day_x(
        predictions=all_preds, 
        targets=all_targets, 
        days=all_days, 
        samples_per_day=350 # Number of bins for stratification
    )
    # preds_strat, targets_strat, days_strat = stratify_by_day_and_mass(
    #     predictions=all_preds, 
    #     targets=all_targets, 
    #     days=all_days, 
    #     samples_per_day=200,
    #     mass_bins=50
    # )
    
    # Mapping days to numbers for coloring
    unique_days_sorted = np.unique(days_strat)
    day_to_num = {day: idx for idx, day in enumerate(unique_days_sorted)}
    numeric_days = np.array([day_to_num[day] for day in days_strat])
    
    # Plot Actual vs Predicted
    plt.figure(figsize=(10, 6))

    sns.scatterplot(
        x=targets_strat,
        y=preds_strat,
        alpha=0.5,  
        s=12,    
        color="green" 
        )
    
    # Create colorbar with correct labels
    # cbar = plt.colorbar(scatter, ticks=np.arange(len(unique_days_sorted)))
    # cbar.ax.set_yticklabels(unique_days_sorted)
    # cbar.set_label('Day')
    
    plt.xlabel('Actual Mass (g)')
    plt.ylabel('Predicted Mass (g)')
    plt.title('Actual vs Predicted Mass (Downsampled)')
    
    # Plot Perfect Prediction Line
    min_val = min(targets_strat.min(), preds_strat.min())
    max_val = max(targets_strat.max(), preds_strat.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')
    
    plt.legend()
    os.makedirs(plot_dir, exist_ok=True)
    plt.savefig(os.path.join(plot_dir, 'actual_vs_predicted_downsampled.png'))
    plt.close()
    print("Actual vs Predicted plot saved.")
    
    # Plot Hexbin
    # plot_hexbin(targets_strat, preds_strat, plot_dir)
    
    # Plot Interactive Actual vs Predicted
    #plot_interactive_actual_vs_predicted(targets_strat, preds_strat, days_strat, plot_dir)

def plot_residuals(model, all_preds, all_targets, all_days, device, feature_scaler, target_scaler, plot_dir):
    """
    Generates Residual Plot.
    """
    model.eval()
    # all_preds = []
    # all_targets = []
    # all_days = []
    
    # for X_batch, y_batch in tqdm(val_loader, desc='Predicting for Residuals'):
    #     X_batch = X_batch.to(device)
    #     with torch.no_grad():
    #         preds = model(X_batch)
    #     all_preds.extend(preds.cpu().numpy().flatten())
    #     all_targets.extend(y_batch.numpy().flatten())
    #     # To correctly map days, ensure the indices align
    #     start_idx = len(all_preds) - len(X_batch)
    #     end_idx = start_idx + len(X_batch)
    #     all_days.extend(val_loader.dataset.days[start_idx:end_idx])
    
    # all_preds = np.array(all_preds)
    # all_targets = np.array(all_targets)
    # all_days = np.array(all_days)
    
    # Apply Inverse Transform
    # all_preds = target_scaler.inverse_transform(all_preds.reshape(-1, 1)).flatten()
    # all_targets = target_scaler.inverse_transform(all_targets.reshape(-1, 1)).flatten()
    
    preds_strat, targets_strat, days_strat = stratify_by_day_x(
        predictions=all_preds, 
        targets=all_targets, 
        days=all_days, 
        samples_per_day=300 
    )
    # preds_strat, targets_strat, days_strat = stratify_by_day_best_predictions(
    #     predictions=all_preds, 
    #     targets=all_targets, 
    #     days=all_days, 
    #     samples_per_day=300,
    #     # mass_bins=50
    # )
    # Calculate Residuals
    residuals = targets_strat - preds_strat
    full_residuals = all_targets - all_preds
    print(len(residuals))
    print(len(full_residuals))
    # Plot Residuals Distribution

    plt.figure(figsize=(12, 6))
    
    #Histogram of residuals
    plt.subplot(1, 2, 1)

    
    sns.histplot(full_residuals, bins=50, color='purple', alpha=0.8, kde=False)
    plt.title('Residuals Distribution')
    plt.xlabel('Residual (g)')
    plt.ylabel('Frequency')
    plt.show()
    
    # Scatter plot of residuals vs actual
    plt.subplot(1, 2, 2)
    # plt.scatter(all_targets, residuals, alpha=0.5, s=10, color='purple')
    
    sns.scatterplot(
    x=targets_strat,
    y=residuals,
    alpha=0.4,  
    s=12,       
    color="purple"  
    )
    plt.axhline(0, color='red', linestyle='--')
    plt.title('Residuals vs Actual Mass')
    plt.xlabel('Actual Mass (g)')
    plt.ylabel('Residual (g)')
    
    
    
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'residuals_analysis.png'))
    plt.close()
    print("Residuals analysis plots saved.")

def plot_log_residuals(model, all_preds, all_targets, all_days, device, feature_scaler, target_scaler, plot_dir):
    """
    Generates Residual Plot with natural logarithm transformation and downsampling.
    """
    model.eval()
    # all_preds = []
    # all_targets = []
    # all_days = []

    # for X_batch, y_batch in tqdm(val_loader, desc='Predicting for Residuals'):
    #     X_batch = X_batch.to(device)
    #     with torch.no_grad():
    #         preds = model(X_batch)
    #     all_preds.extend(preds.cpu().numpy().flatten())
    #     all_targets.extend(y_batch.numpy().flatten())
    #     # To correctly map days, ensure the indices align
    #     start_idx = len(all_preds) - len(X_batch)
    #     end_idx = start_idx + len(X_batch)
    #     all_days.extend(val_loader.dataset.days[start_idx:end_idx])

    # all_preds = np.array(all_preds)
    # all_targets = np.array(all_targets)
    # all_days = np.array(all_days)

    # # Apply Inverse Transform
    # all_preds = target_scaler.inverse_transform(all_preds.reshape(-1, 1)).flatten()
    # all_targets = target_scaler.inverse_transform(all_targets.reshape(-1, 1)).flatten()

    # Calculate Residuals
    residuals = all_targets - all_preds

    # Apply Natural Log Transformation to Residuals
    residuals_ln = np.sign(residuals) * np.log1p(np.abs(residuals))

    # Stratified Sampling: Downsample residuals
    print("Stratifying residuals by day and mass bins...")
    preds_strat, targets_strat, days_strat = stratify_by_day_x(
        predictions=all_preds,
        targets=all_targets,
        days=all_days,
        samples_per_day=400
    )
    residuals_strat = targets_strat - preds_strat
    residuals_ln_strat = np.sign(residuals_strat) * np.log1p(np.abs(residuals_strat))

    # Plot Residuals Distribution (Natural Log Transformation)
    plt.figure(figsize=(12, 6))

    # Histogram of residuals (log-transformed)
    plt.subplot(1, 2, 1)
    #plt.hist(residuals_ln_strat, bins=50, color='purple', alpha=0.7)
    sns.histplot(residuals_ln, bins=50, color='purple', alpha=0.8, kde=False)
    plt.title('Log-Transformed Residuals Distribution')
    plt.xlabel('Log Residual (g)')
    plt.ylabel('Frequency')

    # Scatter plot of residuals vs actual (log-transformed residuals)
    plt.subplot(1, 2, 2)
    plt.scatter(targets_strat, residuals_ln_strat, alpha=0.5, s=10, color='purple')
    sns.scatterplot(x=targets_strat, y=residuals_ln_strat, alpha=0.5, s=10, color="green")
    
    plt.axhline(0, color='red', linestyle='--')
    plt.title('Log-Transformed Residuals vs Actual Mass')
    plt.xlabel('Actual Mass (g)')
    plt.ylabel('Log Residual (g)')

    plt.tight_layout()
    os.makedirs(plot_dir, exist_ok=True)
    plt.savefig(os.path.join(plot_dir, 'log_transformed_residuals_analysis.png'))
    plt.close()
    print("Log-transformed residuals analysis plots saved.")