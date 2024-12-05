
import os
import numpy as np
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
    # model.eval()
    preds_strat, targets_strat, days_strat = stratify_by_day(
        predictions=all_preds,
        targets=all_targets,
        days=all_days,
        samples_per_day=200,
    )

    
    # Plot Actual vs Predicted
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        x=targets_strat,
        y=preds_strat,
        alpha=0.4,
        s=12,
        color="green"
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

    # Debugging: Print min and max values
    print(f"Actual Mass - Min: {all_targets.min()}, Max: {all_targets.max()}")
    print(f"Predicted Mass - Min: {all_preds.min()}, Max: {all_preds.max()}")
    
    # Plot distributions
    plot_distributions(all_targets, all_preds, plot_dir)
    
    # Stratified Sampling: Fixed number of samples per day and mass bin
    print("Stratifying samples by day and mass bins...")
    preds_strat, targets_strat, days_strat = stratify_by_day(
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
    preds_strat, targets_strat, days_strat = stratify_by_day(
        predictions=all_preds, 
        targets=all_targets, 
        days=all_days, 
        samples_per_day=300 
    )

    # Calculate Residuals
    residuals = targets_strat - preds_strat
    full_residuals = all_targets - all_preds

    # Plot Residuals Distribution

    plt.figure(figsize=(12, 6))
    
    #Histogram of residuals
    plt.subplot(1, 2, 1)

    
    sns.histplot(full_residuals, bins=50, color='purple', alpha=0.8, kde=False)
    plt.title('Residuals Distribution')
    plt.xlabel('Residual (g)')
    plt.ylabel('Frequency')
    
    # Scatter plot of residuals vs actual
    plt.subplot(1, 2, 2)
    # plt.scatter(all_targets, residuals, alpha=0.5, s=10, color='purple')
    
    sns.scatterplot(
    x=all_targets,
    y=full_residuals,
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
    # Calculate Residuals
    residuals = all_targets - all_preds

    # Apply Natural Log Transformation to Residuals
    residuals_ln = np.sign(residuals) * np.log1p(np.abs(residuals))

    # Stratified Sampling: Downsample residuals
    print("Stratifying residuals by day and mass bins...")
    preds_strat, targets_strat, days_strat = stratify_by_day(
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


def plot_loss_curves(history, plot_dir):
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