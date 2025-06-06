from sklearn.metrics import classification_report
from dataloader import load_train_data, load_test_data
from gem import GEMDetector
import utils
import numpy as np
import torch

def main():
    # Load and scale training data (S1 and S2)
    # The scaler is fitted on S1 and also returned to scale test data later.
    S1, S2, scaler = load_train_data(parquet_path='smallCiCDDoS/NTP-testing.parquet')
    print("✅ Scaled Training data (S1) loaded:", S1.shape)
    print("✅ Scaled Baseline data (S2) loaded:", S2.shape)

    # Load test data (X_test is initially unscaled)
    X_test_unscaled, y_test = load_test_data(parquet_path='smallCiCDDoS/NTP-testing.parquet')
    # Scale X_test using the same scaler fitted on the training data (S1)
    X_test = scaler.transform(X_test_unscaled)
    print("✅ Scaled Test data loaded: X_test=", X_test.shape, ", y_test=", y_test.shape)

    # Fit GEM detector on S1 (training data)
    # grid search on a validation set) for optimal results on your specific dataset.
    detector = GEMDetector(k=10, prototype_k=5, prototype_threshold=0.5, use_cuda=torch.cuda.is_available())
    detector.fit(S1) # Use S1 for training
    print("✅ GEM detector trained.")

    # --- Anomaly detection thresholding improvement starts here ---

    # Get anomaly scores for the S2 (benign) data to establish a baseline
    s2_scores = detector.predict(S2)
    print("✅ Scores generated for S2 (baseline) data.")

    # Calculate threshold based on the 95th percentile of S2 scores
    threshold = np.percentile(s2_scores, 95)
    print(f"✅ Anomaly threshold set based on S2 data (95th percentile): {threshold:.4f}")

    # Perform detection on X_test using the S2-derived threshold
    test_scores = detector.predict(X_test)
    y_pred = (test_scores > threshold).astype(int)
    print("✅ Anomaly detection performed on test data.")

    # --- Anomaly detection thresholding improvement ends here ---

    # Evaluate detection
    y_true = np.asarray(y_test).astype(int)

    # Show classification report
    print("\n📊 Classification Report:")
    print(classification_report(y_true, y_pred, target_names=["Benign", "Attack"]))

    print("\n📈 Generating visualization plots...")
    # Pass test_scores as anomaly_scores, y_true as true_labels, and the calculated threshold.
    # Assuming '1' is still your anomaly label.
    utils.plot_anomaly_detection_results(anomaly_scores=test_scores, true_labels=y_true, threshold=threshold, anomaly_label=1)
    utils.plot_roc_curve(anomaly_scores=test_scores, true_labels=y_true, anomaly_label=1)
    utils.plot_precision_recall_curve(anomaly_scores=test_scores, true_labels=y_true, anomaly_label=1)

    print("✅ Visualization plot generated and displayed.")

if __name__ == '__main__':
    main()
