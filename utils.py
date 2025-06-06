import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score, confusion_matrix, ConfusionMatrixDisplay
def plot_anomaly_detection_results(anomaly_scores: np.ndarray, true_labels: np.ndarray, threshold: float, anomaly_label: int = 1):
    """
    Generates a time series plot to visualize anomaly scores over time (or index),
    with a logarithmic Y-axis, showing the detection threshold.
    This plot style is adapted from the CRPS-ES paper for visualizing statistics over time.

    Args:
        anomaly_scores (np.ndarray): Anomaly scores calculated for the test data.
                                     Assumed to be in chronological order for time series.
        true_labels (np.ndarray): True labels for the test data (e.g., 0 for normal, `anomaly_label` for anomaly).
                                 (Not explicitly plotted as individual points in this style, but can be used for context)
        threshold (float): The calculated anomaly detection threshold.
        anomaly_label (int): The numerical label that represents the anomaly class (default is 1).
    """
    plt.style.use('dark_background') # Using a dark background for better visibility
    plt.rcParams.update({'font.size': 12}) # Adjust font size for better readability

    plt.figure(figsize=(16, 8))

    # Create an index for the X-axis, representing data flow over time
    time_indices = np.arange(len(anomaly_scores))

    # Plot anomaly scores as a line graph (similar to CRPS-ES statistic)
    plt.plot(time_indices, anomaly_scores, color='#32CD32', linewidth=1.5, label='Anomaly Statistic') # Using a specific hex code for lime green

    # Add the threshold line
    plt.axhline(threshold, color='red', linestyle='--', linewidth=2, label=f'Detection Threshold ({threshold:.2f})')

    # Set Y-axis to logarithmic scale
    plt.yscale('log')

    # --- Adjust Y-axis limits for logarithmic scale ---
    # Ensure scores are positive before taking log
    # Replace any zero or negative scores with a small positive number to avoid log(0)
    display_scores = np.copy(anomaly_scores)
    display_scores[display_scores <= 0] = 1e-9 # Adjusted to a smaller number for lower values if needed

    # Determine a suitable y-axis range
    # Calculate percentiles only if there's enough data
    if len(display_scores) > 1:
        p1 = np.percentile(display_scores, 1)   # 1st percentile
        p99 = np.percentile(display_scores, 99) # 99th percentile
    else: # Handle cases with very little data
        p1 = display_scores[0] if len(display_scores) > 0 else 1e-5
        p99 = display_scores[0] if len(display_scores) > 0 else 1e-1

    # Calculate y_min_log and y_max_log to fit the data and threshold well
    # Aim for a bottom limit slightly below the lowest relevant scores or the threshold.
    # Aim for a top limit that captures the main signal and the threshold without excessive empty space.
    y_min_val = min(p1 * 0.1, threshold * 0.05, np.min(display_scores) * 0.1)
    y_max_val = max(p99 * 1.5, threshold * 2.0, np.max(display_scores) * 1.5)

    # Ensure values are positive before logging
    y_min_val = max(1e-9, y_min_val)
    y_max_val = max(1e-8, y_max_val) # Ensure max is greater than min

    plt.ylim(bottom=10**np.floor(np.log10(y_min_val)),
                top=10**np.ceil(np.log10(y_max_val)))
    # --- End of Y-axis limits adjustment ---

    plt.title('Anomaly Scores Over Time with Detection Threshold')
    plt.xlabel('Observation Number (Time)')
    plt.ylabel('Anomaly Statistic')
    plt.legend(loc='upper right')
    plt.grid(axis='y', linestyle='--', alpha=0.4, which='both') # Show grid for both major/minor ticks
    plt.grid(axis='x', linestyle=':', alpha=0.2)
    plt.tight_layout()
    plt.show()
    
def plot_roc_curve(anomaly_scores: np.ndarray, true_labels: np.ndarray, anomaly_label: int = 1):
    """
    Generates and plots the Receiver Operating Characteristic (ROC) curve.

    Args:
        anomaly_scores (np.ndarray): Anomaly scores (probabilities) for the test data.
        true_labels (np.ndarray): True labels for the test data (0 for normal, `anomaly_label` for anomaly).
        anomaly_label (int): The numerical label that represents the anomaly class (default is 1).
    """
    plt.style.use('dark_background')
    plt.rcParams.update({'font.size': 12})

    # Ensure true_labels are binary (0/1) for ROC curve calculation
    binary_true_labels = (true_labels == anomaly_label).astype(int)

    fpr, tpr, thresholds = roc_curve(binary_true_labels, anomaly_scores)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 8))
    plt.plot(fpr, tpr, color='gold', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--') # Random classifier line
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.show()

def plot_precision_recall_curve(anomaly_scores: np.ndarray, true_labels: np.ndarray, anomaly_label: int = 1):
    """
    Generates and plots the Precision-Recall curve.

    Args:
        anomaly_scores (np.ndarray): Anomaly scores (probabilities) for the test data.
        true_labels (np.ndarray): True labels for the test data (0 for normal, `anomaly_label` for anomaly).
        anomaly_label (int): The numerical label that represents the anomaly class (default is 1).
    """
    plt.style.use('dark_background')
    plt.rcParams.update({'font.size': 12})

    # Ensure true_labels are binary (0/1) for PR curve calculation
    binary_true_labels = (true_labels == anomaly_label).astype(int)

    precision, recall, _ = precision_recall_curve(binary_true_labels, anomaly_scores)
    avg_precision = average_precision_score(binary_true_labels, anomaly_scores)

    plt.figure(figsize=(8, 8))
    plt.plot(recall, precision, color='skyblue', lw=2, label=f'Precision-Recall curve (AP = {avg_precision:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.grid(linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.show()
