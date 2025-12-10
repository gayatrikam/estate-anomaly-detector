import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

def detect_anomalies(y_true, y_pred, method="ensemble", contamination=0.05, 
                    upper_z=2.0, lower_z=-2.0):
    """
    Anomaly detection results based on residuals between actual and predicted prices.
    """
    residuals = np.array(y_true) - np.array(y_pred)
    
    results = pd.DataFrame({
        "Actual_Price": y_true,
        "Predicted_Price": y_pred,
        "Residual": residuals
    })
    
    if method == "ensemble":
        return _ensemble_detection(results, contamination, upper_z, lower_z)
    
    elif method == "zscore":
        z_scores = (residuals - np.mean(residuals)) / np.std(residuals)
        results["Z-Score"] = z_scores
        results["Anomaly Type"] = np.where(
            z_scores > upper_z, "Overpriced",
            np.where(z_scores < lower_z, "Underpriced", "Normal")
        )
    
    elif method == "isolation_forest":
        iso = IsolationForest(contamination=contamination, random_state=42)
        preds = iso.fit_predict(residuals.reshape(-1, 1))
        results["IF_Label"] = np.where(preds == -1, "Anomaly", "Normal")
        results["Anomaly Type"] = results["IF_Label"]
    
    else:
        raise ValueError("Invalid method. Choose 'zscore', 'isolation_forest', or 'ensemble'.")
    
    return results


def _ensemble_detection(results, contamination, z_threshold, lower_z):
    """Ensemble detection using multiple methods."""
    residuals = results["Residual"].values
    actual_prices = results["Actual_Price"].values
    predicted_prices = results["Predicted_Price"].values
    
    features = np.column_stack([
        residuals,
        actual_prices,
        predicted_prices,
        actual_prices / predicted_prices,  
        np.abs(residuals) / actual_prices  
    ])
    
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # z-score method
    z_scores = (residuals - np.mean(residuals)) / np.std(residuals)
    zscore_anomalies = (np.abs(z_scores) > z_threshold).astype(int)
    
    # Isolation forest method
    iso_forest = IsolationForest(contamination=contamination, random_state=42)
    iso_preds = iso_forest.fit_predict(features_scaled)
    iso_anomalies = (iso_preds == -1).astype(int)
    
    # One-class SVM method
    oc_svm = OneClassSVM(gamma='scale', nu=contamination)
    svm_preds = oc_svm.fit_predict(features_scaled)
    svm_anomalies = (svm_preds == -1).astype(int)
    
    # Clustering method
    dbscan = DBSCAN(eps=0.5, min_samples=5)
    dbscan_labels = dbscan.fit_predict(features_scaled)
    dbscan_anomalies = (dbscan_labels == -1).astype(int)
    
    # ensembling with majority vote
    total_votes = zscore_anomalies + iso_anomalies + svm_anomalies + dbscan_anomalies
    ensemble_anomalies = (total_votes >= 2).astype(int) 
    
    confidence = total_votes / 4.0
    
    results["Z_Score_Anomaly"] = zscore_anomalies
    results["Isolation_Forest_Anomaly"] = iso_anomalies
    results["SVM_Anomaly"] = svm_anomalies
    results["DBSCAN_Anomaly"] = dbscan_anomalies
    results["Ensemble_Prediction"] = ensemble_anomalies
    results["Confidence"] = confidence
    results["Z_Score"] = z_scores
    
    results["Ensemble_Label"] = np.where(
        ensemble_anomalies == 0, "Normal",
        np.where(residuals > 0, "Overpriced", "Underpriced")
    )
    
    return results


def plot_ensemble_results(results, save_path=None):
    """Create comprehensive plots of ensemble anomaly detection results."""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Ensemble Anomaly Detection Results', fontsize=16, fontweight='bold')
    
    ax1 = axes[0, 0]
    normal_mask = results['Ensemble_Prediction'] == 0
    anomaly_mask = results['Ensemble_Prediction'] == 1
    
    ax1.scatter(results[normal_mask]['Actual_Price'], 
               results[normal_mask]['Predicted_Price'], 
               alpha=0.6, label='Normal', s=20)
    ax1.scatter(results[anomaly_mask]['Actual_Price'], 
               results[anomaly_mask]['Predicted_Price'], 
               alpha=0.8, label='Anomaly', s=30, c='red')
    
    min_price = min(results['Actual_Price'].min(), results['Predicted_Price'].min())
    max_price = max(results['Actual_Price'].max(), results['Predicted_Price'].max())
    ax1.plot([min_price, max_price], [min_price, max_price], 'k--', alpha=0.5)
    ax1.set_xlabel('Actual Price ($)')
    ax1.set_ylabel('Predicted Price ($)')
    ax1.set_title('Actual vs Predicted Prices')
    ax1.legend()
    
    ax2 = axes[0, 1]
    method_cols = ['Z_Score_Anomaly', 'Isolation_Forest_Anomaly', 
                   'SVM_Anomaly', 'DBSCAN_Anomaly']
    correlation_matrix = results[method_cols].corr()
    
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
               square=True, ax=ax2, fmt='.2f')
    ax2.set_title('Method Agreement Matrix')
    
    ax3 = axes[1, 0]
    ax3.hist(results['Confidence'], bins=20, alpha=0.7, edgecolor='black')
    ax3.axvline(0.5, color='red', linestyle='--', label='Decision Threshold')
    ax3.set_xlabel('Confidence Score')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Ensemble Confidence Distribution')
    ax3.legend()
    
    ax4 = axes[1, 1]
    anomaly_counts = results['Ensemble_Label'].value_counts()
    colors = ['lightgreen', 'red', 'blue']
    ax4.pie(anomaly_counts.values, labels=anomaly_counts.index, autopct='%1.1f%%',
           colors=colors[:len(anomaly_counts)])
    ax4.set_title('Anomaly Type Distribution')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


