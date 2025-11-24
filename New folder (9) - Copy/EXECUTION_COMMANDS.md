# Execution Commands Reference

## 📦 Installation Commands

### Install Dependencies
```bash
# Standard installation
pip install -r requirement.txt

# Using pip3
pip3 install -r requirement.txt

# With virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
pip install -r requirement.txt
```

### Linux: Install Additional Tools (for real-time monitoring)
```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install aircrack-ng

# CentOS/RHEL
sudo yum install aircrack-ng

# Arch Linux
sudo pacman -S aircrack-ng
```

---

## 🧪 Synthetic Data Testing

### Basic Execution
```bash
python evil_twin_detection.py
```

### What It Does
1. Generates 2000 synthetic network samples
2. Trains Random Forest, SVM, and Deep Learning models
3. Creates hybrid ensemble
4. Generates performance visualizations
5. Saves all models to `models/` directory

### Expected Output
- Console: Training progress and performance metrics
- Files: 
  - `roc_curves.png`
  - `confusion_matrices.png`
  - `performance_comparison.png`
  - `feature_importance.png`
  - `training_history.png`
  - `models/rf_model.pkl`
  - `models/svm_model.pkl`
  - `models/dl_model.h5`
  - `models/preprocessor.pkl`

### Execution Time
- ~5-10 minutes (depends on CPU/GPU)

---

## 📡 Real-Time Network Monitoring

### Step 1: Prepare Wireless Interface (Linux)

```bash
# Check available interfaces
iwconfig

# Or
ip link show

# Find your wireless interface (usually wlan0, wlp2s0, etc.)
```

### Step 2: Set to Monitor Mode

```bash
# Stop NetworkManager (if needed)
sudo systemctl stop NetworkManager

# Kill interfering processes
sudo airmon-ng check kill

# Start monitor mode
sudo airmon-ng start wlan0

# Note the new interface name (usually wlan0mon)
```

### Step 3: Run Real-Time Detection

```bash
# Basic usage (default: wlan0, 300 seconds, 30 second intervals)
sudo python real_time_detection.py

# Specify interface
sudo python real_time_detection.py -i wlan0mon

# Custom duration (10 minutes)
sudo python real_time_detection.py -i wlan0mon -d 600

# Custom check interval (check every 60 seconds)
sudo python real_time_detection.py -i wlan0mon -c 60

# Full custom
sudo python real_time_detection.py -i wlan0mon -d 600 -c 60

# Help
sudo python real_time_detection.py --help
```

### Step 4: Stop Monitor Mode (After Monitoring)

```bash
# Stop monitor mode
sudo airmon-ng stop wlan0mon

# Restart NetworkManager (if stopped)
sudo systemctl start NetworkManager
```

### Command-Line Arguments

| Argument | Short | Description | Default |
|----------|-------|-------------|---------|
| `--interface` | `-i` | Wireless interface name | `wlan0` |
| `--duration` | `-d` | Monitoring duration (seconds) | `300` |
| `--check-interval` | `-c` | Check interval (seconds) | `30` |

### Expected Output
- Console: Real-time alerts when evil twin detected
- File: `detection_report.txt` (if alerts found)

---

## 📊 Custom Dataset Usage

### Python Script Example

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

# Ensure it has required columns and 'label' column (0=legitimate, 1=evil_twin)
# Required features:
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

# Train models (follow same pattern as evil_twin_detection.py)
rf_detector = RandomForestDetector()
rf_detector.train(X_train_balanced, y_train_balanced, tune_hyperparameters=False)
rf_results = rf_detector.evaluate(X_test, y_test)

# ... continue with other models
```

---

## 🔍 Verification Commands

### Check Python Version
```bash
python --version
# Should be 3.7 or higher
```

### Check Installed Packages
```bash
pip list | grep -E "numpy|pandas|scikit-learn|tensorflow|scapy"
```

### Test Import
```bash
python -c "from evil_twin_modules import generate_synthetic_dataset; print('OK')"
```

### Check Interface Status (Linux)
```bash
iwconfig
# Should show interface in "Monitor" mode
```

---

## 🛠️ Troubleshooting Commands

### Check for Model Files
```bash
ls -la models/
# Should show: rf_model.pkl, svm_model.pkl, dl_model.h5, preprocessor.pkl
```

### Reinstall Dependencies
```bash
pip install --upgrade -r requirement.txt
```

### Check Interface Permissions
```bash
# Linux
sudo iwconfig

# Check if interface exists
ip link show wlan0
```

### Kill Interfering Processes (Linux)
```bash
sudo airmon-ng check kill
```

---

## 📝 Quick Reference

### Most Common Workflow

```bash
# 1. Install dependencies
pip install -r requirement.txt

# 2. Train models (synthetic data)
python evil_twin_detection.py

# 3. Set monitor mode (Linux only, for real-time)
sudo airmon-ng start wlan0

# 4. Run real-time detection
sudo python real_time_detection.py -i wlan0mon

# 5. Stop monitor mode
sudo airmon-ng stop wlan0mon
```

---

## ⚠️ Important Notes

1. **Root Privileges**: Real-time monitoring requires `sudo` on Linux
2. **Models First**: Always train models before real-time detection
3. **Interface Name**: Use the monitor mode interface name (usually ends with `mon`)
4. **Windows**: Real-time monitoring not supported, use synthetic data mode
5. **Interrupt**: Press `Ctrl+C` to stop monitoring early

---

**For detailed documentation, see `README.md`**

