"""
Unified module that imports all components for Evil Twin Detection System
"""

# Core libraries
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# Wireless network packet capture
from scapy.all import *
from scapy.layers.dot11 import Dot11, Dot11Beacon, Dot11Elt

# Machine Learning
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, roc_curve

# Deep Learning
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Imbalanced data handling
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Utilities
import time
import pickle
from datetime import datetime
import os


# ============================================================================
# Data Collection Module
# ============================================================================

class WirelessNetworkScanner:
    """Capture and analyze 802.11 wireless packets"""
    
    def __init__(self, interface='wlan0'):
        self.interface = interface
        self.packets = []
        self.networks = {}
        
    def packet_handler(self, pkt):
        """Process each captured packet"""
        if pkt.haslayer(Dot11Beacon):
            # Extract network information
            bssid = pkt[Dot11].addr2
            ssid = pkt[Dot11Elt].info.decode('utf-8', errors='ignore')
            
            try:
                # Extract signal strength
                signal_strength = pkt.dBm_AntSignal
            except:
                signal_strength = -100
                
            # Extract channel
            try:
                channel = int(ord(pkt[Dot11Elt:3].info))
            except:
                channel = 1
            
            # Extract encryption info
            crypto = self.get_crypto_type(pkt)
            
            # Store network info
            if bssid not in self.networks:
                self.networks[bssid] = {
                    'ssid': ssid,
                    'signal_strength': [signal_strength],
                    'channel': channel,
                    'crypto': crypto,
                    'first_seen': time.time(),
                    'beacons': 1
                }
            else:
                self.networks[bssid]['signal_strength'].append(signal_strength)
                self.networks[bssid]['beacons'] += 1
                
    def get_crypto_type(self, pkt):
        """Determine encryption type"""
        cap = pkt.sprintf("{Dot11Beacon:%Dot11Beacon.cap%}")
        
        if 'privacy' in cap:
            if pkt.haslayer(Dot11Elt):
                for layer in pkt[Dot11Elt]:
                    if layer.ID == 48:  # RSN Information
                        return 'WPA2'
                return 'WEP'
        return 'Open'
    
    def start_capture(self, duration=60):
        """Capture packets for specified duration"""
        self.networks = {}  # Reset networks
        print(f"[*] Starting packet capture on {self.interface} for {duration} seconds...")
        try:
            sniff(iface=self.interface, prn=self.packet_handler, timeout=duration)
        except Exception as e:
            print(f"[!] Error during capture: {e}")
        print(f"[+] Captured data from {len(self.networks)} networks")
        return self.networks


# ============================================================================
# Dataset Generation Module
# ============================================================================

def generate_synthetic_dataset(n_samples=1000):
    """Generate synthetic dataset for testing"""
    np.random.seed(42)
    
    # Legitimate networks
    n_legitimate = int(n_samples * 0.7)
    legitimate_data = {
        'avg_signal_strength': np.random.normal(-60, 10, n_legitimate),
        'std_signal_strength': np.random.uniform(2, 8, n_legitimate),
        'min_signal_strength': np.random.normal(-75, 5, n_legitimate),
        'max_signal_strength': np.random.normal(-50, 5, n_legitimate),
        'signal_variance': np.random.uniform(4, 50, n_legitimate),
        'channel': np.random.choice([1, 6, 11], n_legitimate),
        'beacon_count': np.random.randint(50, 200, n_legitimate),
        'beacon_rate': np.random.uniform(8, 12, n_legitimate),
        'mac_vendor': np.random.choice([1, 2, 3], n_legitimate),
        'crypto_type': np.random.choice([1, 2], n_legitimate),  # WPA2, WPA3
        'ssid_length': np.random.randint(5, 20, n_legitimate),
        'ssid_has_spaces': np.random.choice([0, 1], n_legitimate),
        'ssid_special_chars': np.random.randint(0, 2, n_legitimate),
        'uptime': np.random.uniform(3600, 86400, n_legitimate),
        'signal_strength_range': np.random.uniform(10, 25, n_legitimate),
        'label': 0  # Legitimate
    }
    
    # Evil twin networks (malicious)
    n_evil = n_samples - n_legitimate
    evil_data = {
        'avg_signal_strength': np.random.normal(-55, 5, n_evil),  # Typically stronger
        'std_signal_strength': np.random.uniform(8, 15, n_evil),  # More variance
        'min_signal_strength': np.random.normal(-70, 8, n_evil),
        'max_signal_strength': np.random.normal(-45, 3, n_evil),
        'signal_variance': np.random.uniform(60, 150, n_evil),  # Higher variance
        'channel': np.random.choice([1, 6, 11], n_evil),
        'beacon_count': np.random.randint(20, 100, n_evil),  # Lower beacon count
        'beacon_rate': np.random.uniform(5, 9, n_evil),  # Lower rate
        'mac_vendor': np.random.choice([0, 4], n_evil),  # Unknown/suspicious vendors
        'crypto_type': np.random.choice([0, 1], n_evil),  # Often open or weak encryption
        'ssid_length': np.random.randint(3, 25, n_evil),
        'ssid_has_spaces': np.random.choice([0, 1], n_evil),
        'ssid_special_chars': np.random.randint(2, 8, n_evil),  # More special chars
        'uptime': np.random.uniform(60, 7200, n_evil),  # Shorter uptime
        'signal_strength_range': np.random.uniform(20, 40, n_evil),  # Higher range
        'label': 1  # Evil Twin
    }
    
    # Combine datasets
    df_legitimate = pd.DataFrame(legitimate_data)
    df_evil = pd.DataFrame(evil_data)
    df = pd.concat([df_legitimate, df_evil], ignore_index=True)
    
    # Shuffle
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    return df


# ============================================================================
# Feature Engineering Module
# ============================================================================

class FeatureExtractor:
    """Extract features for ML models"""
    
    def __init__(self):
        self.features = []
        
    def extract_features(self, networks_data):
        """Extract relevant features from network data"""
        feature_list = []
        
        for bssid, data in networks_data.items():
            features = {
                # Signal strength statistics
                'avg_signal_strength': np.mean(data['signal_strength']),
                'std_signal_strength': np.std(data['signal_strength']),
                'min_signal_strength': np.min(data['signal_strength']),
                'max_signal_strength': np.max(data['signal_strength']),
                'signal_variance': np.var(data['signal_strength']),
                
                # Channel information
                'channel': data['channel'],
                
                # Beacon statistics
                'beacon_count': data['beacons'],
                'beacon_rate': data['beacons'] / (time.time() - data['first_seen'] + 1),
                
                # MAC address analysis
                'mac_vendor': self.get_mac_vendor(bssid),
                
                # Encryption
                'crypto_type': data['crypto'],
                
                # SSID characteristics
                'ssid_length': len(data['ssid']),
                'ssid_has_spaces': int(' ' in data['ssid']),
                'ssid_special_chars': sum(not c.isalnum() and c != ' ' for c in data['ssid']),
                
                # Timing analysis
                'uptime': time.time() - data['first_seen'],
                
                # Additional features
                'signal_strength_range': np.max(data['signal_strength']) - np.min(data['signal_strength']),
            }
            
            feature_list.append(features)
            
        return pd.DataFrame(feature_list)
    
    def get_mac_vendor(self, mac):
        """Get MAC vendor OUI (simplified)"""
        # In real implementation, use OUI database
        oui = mac[:8].upper()
        vendor_mapping = {
            '00:0C:41': 1,  # Example: Cisco
            '00:50:56': 2,  # VMware
            # Add more vendors
        }
        return vendor_mapping.get(oui, 0)


# ============================================================================
# Data Preprocessing Module
# ============================================================================

class DataPreprocessor:
    """Preprocess data for ML models"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        
    def preprocess(self, df, fit=True):
        """Preprocess the dataset"""
        df_processed = df.copy()
        
        # Separate features and labels
        if 'label' in df_processed.columns:
            X = df_processed.drop('label', axis=1)
            y = df_processed['label']
        else:
            X = df_processed
            y = None
        
        # Encode categorical variables
        categorical_cols = X.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if fit:
                self.label_encoders[col] = LabelEncoder()
                X[col] = self.label_encoders[col].fit_transform(X[col])
            else:
                if col in self.label_encoders:
                    X[col] = self.label_encoders[col].transform(X[col])
        
        # Scale features
        if fit:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = self.scaler.transform(X)
        
        return X_scaled, y
    
    def handle_imbalance(self, X, y, strategy='SMOTE'):
        """Handle class imbalance"""
        if strategy == 'SMOTE':
            smote = SMOTE(random_state=42)
            X_resampled, y_resampled = smote.fit_resample(X, y)
        elif strategy == 'undersample':
            rus = RandomUnderSampler(random_state=42)
            X_resampled, y_resampled = rus.fit_resample(X, y)
        else:
            X_resampled, y_resampled = X, y
            
        return X_resampled, y_resampled


# ============================================================================
# Machine Learning Models
# ============================================================================

class RandomForestDetector:
    """Random Forest for Evil Twin Detection"""
    
    def __init__(self):
        self.model = None
        self.best_params = None
        
    def train(self, X_train, y_train, tune_hyperparameters=True):
        """Train Random Forest model"""
        print("\n[*] Training Random Forest Classifier...")
        
        if tune_hyperparameters:
            # Hyperparameter tuning
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 20, 30, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2']
            }
            
            rf = RandomForestClassifier(random_state=42)
            grid_search = GridSearchCV(
                rf, param_grid, cv=5, scoring='f1', n_jobs=-1, verbose=1
            )
            grid_search.fit(X_train, y_train)
            
            self.model = grid_search.best_estimator_
            self.best_params = grid_search.best_params_
            print(f"[+] Best parameters: {self.best_params}")
        else:
            # Default parameters
            self.model = RandomForestClassifier(
                n_estimators=200,
                max_depth=20,
                min_samples_split=5,
                random_state=42,
                n_jobs=-1
            )
            self.model.fit(X_train, y_train)
        
        # Cross-validation
        cv_scores = cross_val_score(self.model, X_train, y_train, cv=5, scoring='f1')
        print(f"[+] Cross-validation F1 scores: {cv_scores}")
        print(f"[+] Mean F1: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        
    def predict(self, X):
        """Make predictions"""
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """Get prediction probabilities"""
        return self.model.predict_proba(X)
    
    def evaluate(self, X_test, y_test):
        """Evaluate model performance"""
        y_pred = self.predict(X_test)
        y_proba = self.predict_proba(X_test)[:, 1]
        
        print("\n=== Random Forest Performance ===")
        print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
        print(f"\nClassification Report:\n{classification_report(y_test, y_pred)}")
        print(f"\nConfusion Matrix:\n{confusion_matrix(y_test, y_pred)}")
        
        # ROC AUC
        roc_auc = roc_auc_score(y_test, y_proba)
        print(f"\nROC AUC Score: {roc_auc:.4f}")
        
        return {
            'accuracy': accuracy_score(y_test, y_pred),
            'y_pred': y_pred,
            'y_proba': y_proba,
            'roc_auc': roc_auc
        }
    
    def get_feature_importance(self, feature_names):
        """Get feature importance"""
        importance = pd.DataFrame({
            'feature': feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return importance


class SVMDetector:
    """SVM for Evil Twin Detection"""
    
    def __init__(self):
        self.model = None
        self.best_params = None
        
    def train(self, X_train, y_train, tune_hyperparameters=True):
        """Train SVM model"""
        print("\n[*] Training SVM Classifier...")
        
        if tune_hyperparameters:
            # Hyperparameter tuning
            param_grid = {
                'C': [0.1, 1, 10, 100],
                'gamma': ['scale', 'auto', 0.001, 0.01],
                'kernel': ['rbf', 'linear']
            }
            
            svm = SVC(probability=True, random_state=42)
            grid_search = GridSearchCV(
                svm, param_grid, cv=5, scoring='f1', n_jobs=-1, verbose=1
            )
            grid_search.fit(X_train, y_train)
            
            self.model = grid_search.best_estimator_
            self.best_params = grid_search.best_params_
            print(f"[+] Best parameters: {self.best_params}")
        else:
            # Default parameters
            self.model = SVC(
                C=10,
                kernel='rbf',
                gamma='scale',
                probability=True,
                random_state=42
            )
            self.model.fit(X_train, y_train)
        
        # Cross-validation
        cv_scores = cross_val_score(self.model, X_train, y_train, cv=5, scoring='f1')
        print(f"[+] Cross-validation F1 scores: {cv_scores}")
        print(f"[+] Mean F1: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        
    def predict(self, X):
        """Make predictions"""
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """Get prediction probabilities"""
        return self.model.predict_proba(X)
    
    def evaluate(self, X_test, y_test):
        """Evaluate model performance"""
        y_pred = self.predict(X_test)
        y_proba = self.predict_proba(X_test)[:, 1]
        
        print("\n=== SVM Performance ===")
        print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
        print(f"\nClassification Report:\n{classification_report(y_test, y_pred)}")
        print(f"\nConfusion Matrix:\n{confusion_matrix(y_test, y_pred)}")
        
        # ROC AUC
        roc_auc = roc_auc_score(y_test, y_proba)
        print(f"\nROC AUC Score: {roc_auc:.4f}")
        
        return {
            'accuracy': accuracy_score(y_test, y_pred),
            'y_pred': y_pred,
            'y_proba': y_proba,
            'roc_auc': roc_auc
        }


class DeepLearningDetector:
    """Deep Neural Network for Evil Twin Detection"""
    
    def __init__(self, input_dim):
        self.input_dim = input_dim
        self.model = None
        self.history = None
        
    def build_model(self):
        """Build DNN architecture"""
        model = Sequential([
            Dense(128, activation='relu', input_dim=self.input_dim),
            BatchNormalization(),
            Dropout(0.3),
            
            Dense(64, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            
            Dense(32, activation='relu'),
            BatchNormalization(),
            Dropout(0.2),
            
            Dense(16, activation='relu'),
            
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.Precision(), 
                    tf.keras.metrics.Recall(), tf.keras.metrics.AUC()]
        )
        
        self.model = model
        return model
    
    def train(self, X_train, y_train, X_val, y_val, epochs=100, batch_size=32):
        """Train deep learning model"""
        print("\n[*] Training Deep Learning Model...")
        
        # Build model
        self.build_model()
        print(self.model.summary())
        
        # Callbacks
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=1
        )
        
        model_checkpoint = ModelCheckpoint(
            'best_model.h5',
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        )
        
        # Train
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping, model_checkpoint],
            verbose=1
        )
        
        print("[+] Training completed!")
        
    def predict(self, X):
        """Make predictions"""
        predictions = self.model.predict(X)
        return (predictions > 0.5).astype(int).flatten()
    
    def predict_proba(self, X):
        """Get prediction probabilities"""
        return self.model.predict(X).flatten()
    
    def evaluate(self, X_test, y_test):
        """Evaluate model performance"""
        y_pred = self.predict(X_test)
        y_proba = self.predict_proba(X_test)
        
        print("\n=== Deep Learning Performance ===")
        print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
        print(f"\nClassification Report:\n{classification_report(y_test, y_pred)}")
        print(f"\nConfusion Matrix:\n{confusion_matrix(y_test, y_pred)}")
        
        # ROC AUC
        roc_auc = roc_auc_score(y_test, y_proba)
        print(f"\nROC AUC Score: {roc_auc:.4f}")
        
        return {
            'accuracy': accuracy_score(y_test, y_pred),
            'y_pred': y_pred,
            'y_proba': y_proba,
            'roc_auc': roc_auc
        }
    
    def plot_training_history(self):
        """Plot training history"""
        if self.history is None:
            print("[!] No training history available")
            return
            
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss
        axes[0, 0].plot(self.history.history['loss'], label='Train Loss')
        axes[0, 0].plot(self.history.history['val_loss'], label='Val Loss')
        axes[0, 0].set_title('Model Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        
        # Accuracy
        axes[0, 1].plot(self.history.history['accuracy'], label='Train Accuracy')
        axes[0, 1].plot(self.history.history['val_accuracy'], label='Val Accuracy')
        axes[0, 1].set_title('Model Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        
        # Precision
        axes[1, 0].plot(self.history.history['precision'], label='Train Precision')
        axes[1, 0].plot(self.history.history['val_precision'], label='Val Precision')
        axes[1, 0].set_title('Model Precision')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Precision')
        axes[1, 0].legend()
        
        # Recall
        axes[1, 1].plot(self.history.history['recall'], label='Train Recall')
        axes[1, 1].plot(self.history.history['val_recall'], label='Val Recall')
        axes[1, 1].set_title('Model Recall')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Recall')
        axes[1, 1].legend()
        
        plt.tight_layout()
        plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
        plt.close()


class HybridEnsembleDetector:
    """Hybrid ensemble combining RF, SVM, and DL"""
    
    def __init__(self, rf_model, svm_model, dl_model):
        self.rf_model = rf_model
        self.svm_model = svm_model
        self.dl_model = dl_model
        self.weights = [0.35, 0.30, 0.35]  # RF, SVM, DL
        
    def predict_proba(self, X):
        """Weighted ensemble prediction"""
        # Get probabilities from each model
        rf_proba = self.rf_model.predict_proba(X)[:, 1]
        svm_proba = self.svm_model.predict_proba(X)[:, 1]
        dl_proba = self.dl_model.predict_proba(X)
        
        # Weighted average
        ensemble_proba = (
            self.weights[0] * rf_proba +
            self.weights[1] * svm_proba +
            self.weights[2] * dl_proba
        )
        
        return ensemble_proba
    
    def predict(self, X, threshold=0.5):
        """Make final predictions"""
        proba = self.predict_proba(X)
        return (proba > threshold).astype(int)
    
    def evaluate(self, X_test, y_test):
        """Evaluate ensemble performance"""
        y_pred = self.predict(X_test)
        y_proba = self.predict_proba(X_test)
        
        print("\n=== Hybrid Ensemble Performance ===")
        print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
        print(f"\nClassification Report:\n{classification_report(y_test, y_pred)}")
        print(f"\nConfusion Matrix:\n{confusion_matrix(y_test, y_pred)}")
        
        # ROC AUC
        roc_auc = roc_auc_score(y_test, y_proba)
        print(f"\nROC AUC Score: {roc_auc:.4f}")
        
        return {
            'accuracy': accuracy_score(y_test, y_pred),
            'y_pred': y_pred,
            'y_proba': y_proba,
            'roc_auc': roc_auc
        }


# ============================================================================
# Visualization Module
# ============================================================================

class ModelVisualizer:
    """Visualize model performance"""
    
    @staticmethod
    def plot_roc_curves(models_results, y_test):
        """Plot ROC curves for all models"""
        plt.figure(figsize=(10, 8))
        
        for model_name, results in models_results.items():
            fpr, tpr, _ = roc_curve(y_test, results['y_proba'])
            auc_score = results['roc_auc']
            plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc_score:.4f})', linewidth=2)
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curves - Evil Twin Detection Models', fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(alpha=0.3)
        plt.savefig('roc_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    @staticmethod
    def plot_confusion_matrices(models_results, y_test):
        """Plot confusion matrices for all models"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        for idx, (model_name, results) in enumerate(models_results.items()):
            cm = confusion_matrix(y_test, results['y_pred'])
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                       xticklabels=['Legitimate', 'Evil Twin'],
                       yticklabels=['Legitimate', 'Evil Twin'])
            axes[idx].set_title(f'{model_name} - Confusion Matrix', fontsize=12, fontweight='bold')
            axes[idx].set_ylabel('True Label')
            axes[idx].set_xlabel('Predicted Label')
        
        plt.tight_layout()
        plt.savefig('confusion_matrices.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    @staticmethod
    def plot_performance_comparison(models_results, y_test):
        """Compare model performance metrics"""
        metrics_data = []
        
        for model_name, results in models_results.items():
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_test, results['y_pred'], average='binary'
            )
            metrics_data.append({
                'Model': model_name,
                'Accuracy': results['accuracy'],
                'Precision': precision,
                'Recall': recall,
                'F1-Score': f1,
                'ROC AUC': results['roc_auc']
            })
        
        df_metrics = pd.DataFrame(metrics_data)
        
        # Plot
        fig, ax = plt.subplots(figsize=(12, 6))
        x = np.arange(len(df_metrics))
        width = 0.15
        
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC AUC']
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        
        for idx, metric in enumerate(metrics):
            ax.bar(x + idx*width, df_metrics[metric], width, 
                  label=metric, color=colors[idx])
        
        ax.set_xlabel('Models', fontsize=12, fontweight='bold')
        ax.set_ylabel('Score', fontsize=12, fontweight='bold')
        ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
        ax.set_xticks(x + width * 2)
        ax.set_xticklabels(df_metrics['Model'])
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('performance_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        return df_metrics
    
    @staticmethod
    def plot_feature_importance(rf_model, feature_names):
        """Plot feature importance from Random Forest"""
        importance_df = rf_model.get_feature_importance(feature_names)
        
        plt.figure(figsize=(10, 8))
        sns.barplot(data=importance_df.head(15), x='importance', y='feature', palette='viridis')
        plt.title('Top 15 Most Important Features', fontsize=14, fontweight='bold')
        plt.xlabel('Importance Score', fontsize=12)
        plt.ylabel('Features', fontsize=12)
        plt.tight_layout()
        plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
        plt.close()


# ============================================================================
# Model Management Module
# ============================================================================

class ModelManager:
    """Save and load trained models"""
    
    @staticmethod
    def save_models(rf_model, svm_model, dl_model, preprocessor, path='models/'):
        """Save all models"""
        os.makedirs(path, exist_ok=True)
        
        # Save Random Forest
        with open(f'{path}rf_model.pkl', 'wb') as f:
            pickle.dump(rf_model, f)
        print(f"[+] Random Forest saved to {path}rf_model.pkl")
        
        # Save SVM
        with open(f'{path}svm_model.pkl', 'wb') as f:
            pickle.dump(svm_model, f)
        print(f"[+] SVM saved to {path}svm_model.pkl")
        
        # Save Deep Learning model
        dl_model.model.save(f'{path}dl_model.h5')
        print(f"[+] Deep Learning model saved to {path}dl_model.h5")
        
        # Save preprocessor
        with open(f'{path}preprocessor.pkl', 'wb') as f:
            pickle.dump(preprocessor, f)
        print(f"[+] Preprocessor saved to {path}preprocessor.pkl")
    
    @staticmethod
    def load_models(path='models/'):
        """Load all models"""
        # Load Random Forest
        with open(f'{path}rf_model.pkl', 'rb') as f:
            rf_model = pickle.load(f)
        print(f"[+] Random Forest loaded from {path}rf_model.pkl")
        
        # Load SVM
        with open(f'{path}svm_model.pkl', 'rb') as f:
            svm_model = pickle.load(f)
        print(f"[+] SVM loaded from {path}svm_model.pkl")
        
        # Load Deep Learning model
        dl_model_wrapper = type('obj', (object,), {'model': keras.models.load_model(f'{path}dl_model.h5')})()
        print(f"[+] Deep Learning model loaded from {path}dl_model.h5")
        
        # Load preprocessor
        with open(f'{path}preprocessor.pkl', 'rb') as f:
            preprocessor = pickle.load(f)
        print(f"[+] Preprocessor loaded from {path}preprocessor.pkl")
        
        return rf_model, svm_model, dl_model_wrapper, preprocessor


# ============================================================================
# Real-Time Detection Module
# ============================================================================

class RealTimeDetectionSystem:
    """Real-time evil twin detection system"""
    
    def __init__(self, ensemble_model, preprocessor, interface='wlan0'):
        self.ensemble_model = ensemble_model
        self.preprocessor = preprocessor
        self.interface = interface
        self.scanner = WirelessNetworkScanner(interface)
        self.feature_extractor = FeatureExtractor()
        self.alerts = []
        
    def monitor(self, duration=300, check_interval=30):
        """Monitor network in real-time"""
        print(f"\n[*] Starting real-time monitoring for {duration} seconds...")
        print(f"[*] Checking every {check_interval} seconds...\n")
        
        start_time = time.time()
        iteration = 0
        
        while (time.time() - start_time) < duration:
            iteration += 1
            print(f"\n--- Iteration {iteration} ---")
            
            # Capture packets
            networks = self.scanner.start_capture(duration=check_interval)
            
            if not networks:
                print("[!] No networks detected")
                continue
            
            # Extract features
            features_df = self.feature_extractor.extract_features(networks)
            
            # Preprocess
            X_new, _ = self.preprocessor.preprocess(features_df, fit=False)
            
            # Predict
            predictions = self.ensemble_model.predict(X_new)
            probabilities = self.ensemble_model.predict_proba(X_new)
            
            # Check for evil twins
            for idx, (bssid, data) in enumerate(networks.items()):
                if predictions[idx] == 1:  # Evil twin detected
                    alert = {
                        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        'bssid': bssid,
                        'ssid': data['ssid'],
                        'channel': data['channel'],
                        'signal_strength': np.mean(data['signal_strength']),
                        'confidence': probabilities[idx] * 100,
                        'crypto': data['crypto']
                    }
                    
                    self.alerts.append(alert)
                    self.raise_alert(alert)
            
            time.sleep(1)
        
        print(f"\n[+] Monitoring completed. Total alerts: {len(self.alerts)}")
        return self.alerts
    
    def raise_alert(self, alert):
        """Raise alert for detected evil twin"""
        print("\n" + "="*60)
        print("🚨 EVIL TWIN ATTACK DETECTED!")
        print("="*60)
        print(f"Timestamp:       {alert['timestamp']}")
        print(f"BSSID:           {alert['bssid']}")
        print(f"SSID:            {alert['ssid']}")
        print(f"Channel:         {alert['channel']}")
        print(f"Signal Strength: {alert['signal_strength']:.2f} dBm")
        print(f"Confidence:      {alert['confidence']:.2f}%")
        print(f"Encryption:      {alert['crypto']}")
        print("="*60)
        print("⚠️  RECOMMENDED ACTIONS:")
        print("  1. Do not connect to this network")
        print("  2. Alert network administrator")
        print("  3. Monitor for similar attacks")
        print("="*60 + "\n")
    
    def generate_report(self, filename='detection_report.txt'):
        """Generate detection report"""
        with open(filename, 'w') as f:
            f.write("="*60 + "\n")
            f.write("EVIL TWIN DETECTION REPORT\n")
            f.write("="*60 + "\n\n")
            f.write(f"Total Alerts: {len(self.alerts)}\n\n")
            
            for idx, alert in enumerate(self.alerts, 1):
                f.write(f"\nAlert #{idx}\n")
                f.write("-"*40 + "\n")
                for key, value in alert.items():
                    f.write(f"{key.capitalize():20s}: {value}\n")
                f.write("-"*40 + "\n")
        
        print(f"[+] Report saved to {filename}")

