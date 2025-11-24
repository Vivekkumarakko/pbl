# Evil Twin Attack Detection System

A comprehensive machine learning-based framework for detecting evil twin attacks in wireless networks using Random Forest, SVM, Deep Learning, and Hybrid Ensemble models.

## 📋 Table of Contents

- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
  - [Synthetic Data Testing](#synthetic-data-testing)
  - [Real Network Monitoring](#real-network-monitoring)
  - [Custom Dataset](#custom-dataset)
- [Project Structure](#project-structure)
- [System Requirements](#system-requirements)
- [Troubleshooting](#troubleshooting)

## ✨ Features

- **Multiple ML Models**: Random Forest, SVM, and Deep Neural Network
- **Hybrid Ensemble**: Combines all models for improved accuracy
- **Real-time Detection**: Monitor live wireless networks
- **Synthetic Data Generation**: Test models without real network access
- **Comprehensive Visualizations**: ROC curves, confusion matrices, performance comparisons
- **Model Persistence**: Save and load trained models
- **Feature Engineering**: Automatic feature extraction from network data

## 📦 Requirements

### Python Dependencies

All required packages are listed in `requirement.txt`:

```
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
tensorflow>=2.8.0
scapy>=2.4.5
matplotlib>=3.4.0
seaborn>=0.11.0
imbalanced-learn>=0.9.0
xgboost>=1.5.0
joblib>=1.1.0
```

### System Requirements

- **Python**: 3.7 or higher
- **Operating System**: Linux (for real-time monitoring), Windows/Mac/Linux (for synthetic testing)
- **Wireless Interface**: Required for real-time monitoring (Linux)
- **Root/Admin Privileges**: Required for packet capture (Linux)

## 🚀 Installation

### 1. Clone or Download the Project

```bash
# If using git
git clone <repository-url>
cd evil-twin-detection

# Or simply navigate to the project directory
cd "path/to/project"
```

### 2. Install Python Dependencies

```bash
# Using pip
pip install -r requirement.txt

# Or using pip3
pip3 install -r requirement.txt

# For virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirement.txt
```

### 3. Linux-Specific Setup (for Real-time Monitoring)

For real-time network monitoring on Linux, you need:

```bash
# Install aircrack-ng suite (for monitor mode)
sudo apt-get update
sudo apt-get install aircrack-ng

# Or on other distributions:
# sudo yum install aircrack-ng
# sudo pacman -S aircrack-ng
```

## 📖 Usage

### Synthetic Data Testing

This mode generates synthetic data and trains all models. No special privileges required.

```bash
python evil_twin_detection.py
```

**What it does:**
1. Generates synthetic dataset (2000 samples)
2. Preprocesses data and handles class imbalance
3. Trains Random Forest, SVM, and Deep Learning models
4. Creates hybrid ensemble
5. Generates visualizations (ROC curves, confusion matrices, etc.)
6. Saves all models to `models/` directory

**Output Files:**
- `roc_curves.png` - ROC curves for all models
- `confusion_matrices.png` - Confusion matrices comparison
- `performance_comparison.png` - Performance metrics comparison
- `feature_importance.png` - Top 15 important features
- `training_history.png` - Deep learning training history
- `models/` - Directory containing saved models

### Real Network Monitoring

This mode monitors live wireless networks and detects evil twin attacks in real-time.

#### Step 1: Set Wireless Interface to Monitor Mode (Linux)

```bash
# Find your wireless interface
iwconfig

# Set interface to monitor mode
sudo airmon-ng start wlan0

# Note the new interface name (usually wlan0mon)
# If your interface is different, replace wlan0 with your interface name
```

#### Step 2: Run Real-Time Detection

```bash
# Basic usage (default: wlan0, 300 seconds, 30 second intervals)
sudo python real_time_detection.py

# Custom interface
sudo python real_time_detection.py -i wlan0mon

# Custom duration and check interval
sudo python real_time_detection.py -i wlan0mon -d 600 -c 60

# Help
sudo python real_time_detection.py --help
```

**Command-line Arguments:**
- `-i, --interface`: Wireless interface name (default: wlan0)
- `-d, --duration`: Monitoring duration in seconds (default: 300)
- `-c, --check-interval`: Check interval in seconds (default: 30)

**Important Notes:**
- Requires root privileges (use `sudo`)
- Models must be trained first (run `evil_twin_detection.py`)
- Interface must be in monitor mode
- Press `Ctrl+C` to stop monitoring early

**Output:**
- Real-time alerts when evil twin is detected
- `detection_report.txt` - Detailed report of all alerts

#### Step 3: Stop Monitor Mode (After Monitoring)

```bash
sudo airmon-ng stop wlan0mon
```

### Custom Dataset

To use your own dataset:

```python
import pandas as pd
from evil_twin_modules import (
    DataPreprocessor,
    RandomForestDetector,
    SVMDetector,
    DeepLearningDetector,
    HybridEnsembleDetector,
    ModelVisualizer,
    ModelManager,
    train_test_split
)

# Load your dataset
df = pd.read_csv('your_network_data.csv')

# Ensure it has the label column (0=legitimate, 1=evil_twin)
# Required columns:
# - avg_signal_strength, std_signal_strength, min_signal_strength, max_signal_strength
# - signal_variance, channel, beacon_count, beacon_rate
# - mac_vendor, crypto_type, ssid_length, ssid_has_spaces
# - ssid_special_chars, uptime, signal_strength_range
# - label (0 or 1)

# Preprocess
preprocessor = DataPreprocessor()
X, y = preprocessor.preprocess(df, fit=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
X_train_balanced, y_train_balanced = preprocessor.handle_imbalance(
    X_train, y_train, strategy='SMOTE'
)

# Train models (same as in evil_twin_detection.py)
# ... (follow the same training pipeline)
```

## 📁 Project Structure

```
evil-twin-detection/
│
├── evil_twin_modules.py          # Unified module with all classes
├── evil_twin_detection.py        # Main script for synthetic data testing
├── real_time_detection.py        # Main script for real-time monitoring
├── requirement.txt               # Python dependencies
├── README.md                     # This file
│
├── Data Collection.py            # Original data collection module
├── dataset.py                    # Original dataset generation
├── Feature_Engineering.py       # Original feature engineering
├── machine.py                   # Original preprocessing
├── random_forest.py             # Original RF model
├── svm.py                       # Original SVM model
├── dl.py                        # Original DL model
├── hybrid.py                    # Original ensemble
├── vis.py                       # Original visualization
├── modalper.py                  # Original model management
├── realtime.py                  # Original real-time detection
├── compeleteexec.py             # Original complete execution
├── main.py                      # Original main imports
│
├── models/                      # Saved models (created after training)
│   ├── rf_model.pkl
│   ├── svm_model.pkl
│   ├── dl_model.h5
│   └── preprocessor.pkl
│
└── *.png                        # Generated visualization files
```

## 🔧 System Requirements

### For Synthetic Data Testing
- Any OS (Windows, Mac, Linux)
- Python 3.7+
- No special privileges needed

### For Real-Time Monitoring
- Linux OS (recommended)
- Wireless interface with monitor mode support
- Root/Administrator privileges
- aircrack-ng suite installed

## 🐛 Troubleshooting

### Issue: "Models not found" error

**Solution:** Train models first by running:
```bash
python evil_twin_detection.py
```

### Issue: "Permission denied" on Linux

**Solution:** Use sudo:
```bash
sudo python real_time_detection.py
```

### Issue: "No such interface" error

**Solution:** 
1. Check available interfaces: `iwconfig` or `ip link show`
2. Use correct interface name: `sudo python real_time_detection.py -i <interface>`

### Issue: "Monitor mode failed"

**Solution:**
1. Check if interface supports monitor mode
2. Stop NetworkManager: `sudo systemctl stop NetworkManager`
3. Kill interfering processes: `sudo airmon-ng check kill`
4. Try again: `sudo airmon-ng start wlan0`

### Issue: Import errors

**Solution:**
1. Ensure all dependencies are installed: `pip install -r requirement.txt`
2. Check Python version: `python --version` (should be 3.7+)
3. Use virtual environment to avoid conflicts

### Issue: TensorFlow/GPU errors

**Solution:**
- TensorFlow will use CPU by default if GPU is not available
- For GPU support, install: `pip install tensorflow-gpu`
- Ensure CUDA and cuDNN are properly installed

### Issue: Scapy import errors on Windows

**Solution:**
- Install Npcap or WinPcap
- Scapy works better on Linux for real-time monitoring
- For Windows, use synthetic data testing mode

## 📊 Model Performance

The system uses a hybrid ensemble approach combining:
- **Random Forest** (35% weight)
- **SVM** (30% weight)
- **Deep Learning** (35% weight)

Expected performance metrics:
- Accuracy: >90%
- Precision: >85%
- Recall: >85%
- F1-Score: >85%
- ROC AUC: >0.90

## 🔒 Security Note

This tool is for **educational and authorized security testing purposes only**. Only use on networks you own or have explicit permission to test. Unauthorized network monitoring may be illegal in your jurisdiction.

## 📝 License

[Specify your license here]

## 👥 Contributors

[Add contributors if applicable]

## 📧 Support

For issues or questions, please [create an issue] or [contact support].

---

**Last Updated:** 2024

