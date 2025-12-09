import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

def detect_anomalies(y_true, y_pred, method="zscore", contamination=0.05, upper_z=2.0, lower_z=-2.0):
    """
    Detect pricing anomalies based on residuals between actual and predicted prices.
    

    anomalies : pandas.DataFrame
        DataFrame containing actual prices, predicted prices, residuals, z-scores, and anomaly labels
    """

    residuals = np.array(y_true) - np.array(y_pred)
    anomalies = pd.DataFrame({
        "Actual Price": y_true,
        "Predicted Price": y_pred,
        "Residual": residuals
    })

    if method == "zscore":
        z_scores = (residuals - np.mean(residuals)) / np.std(residuals)
        anomalies["Z-Score"] = z_scores
        anomalies["Anomaly Type"] = np.where(
            z_scores > upper_z, "Overpriced",
            np.where(z_scores < lower_z, "Underpriced", "Normal")
        )

    elif method == "isolation_forest":
        iso = IsolationForest(contamination=contamination, random_state=42)
        preds = iso.fit_predict(residuals.reshape(-1, 1))
        anomalies["IF_Label"] = np.where(preds == -1, "Anomaly", "Normal")
        anomalies["Anomaly Type"] = anomalies["IF_Label"]

    else:
        raise ValueError("Invalid method. Choose 'zscore' or 'isolation_forest'.")

    return anomalies