# Quick Start Guide

## 🚀 Fast Setup (5 minutes)

### Step 1: Install Dependencies

```bash
pip install -r requirement.txt
```

### Step 2: Run Synthetic Data Test

```bash
python evil_twin_detection.py
```

This will:
- ✅ Generate synthetic data
- ✅ Train all models
- ✅ Generate visualizations
- ✅ Save models for real-time use

**Time:** ~5-10 minutes (depending on your system)

---

## 📡 Real-Time Monitoring (Linux Only)

### Step 1: Set Interface to Monitor Mode

```bash
# Find your interface
iwconfig

# Start monitor mode
sudo airmon-ng start wlan0

# Note the new interface (usually wlan0mon)
```

### Step 2: Run Detection

```bash
sudo python real_time_detection.py -i wlan0mon
```

### Step 3: Stop Monitor Mode

```bash
sudo airmon-ng stop wlan0mon
```

---

## 📋 Common Commands

### Synthetic Testing
```bash
python evil_twin_detection.py
```

### Real-Time Monitoring
```bash
# Basic
sudo python real_time_detection.py

# Custom interface
sudo python real_time_detection.py -i wlan0mon

# Custom duration (10 minutes, check every 60 seconds)
sudo python real_time_detection.py -i wlan0mon -d 600 -c 60
```

---

## ⚠️ Troubleshooting

**"Models not found"** → Run `python evil_twin_detection.py` first

**"Permission denied"** → Use `sudo` on Linux

**"No interface"** → Check with `iwconfig` and use `-i` flag

---

## 📁 Output Files

After running `evil_twin_detection.py`:
- `roc_curves.png`
- `confusion_matrices.png`
- `performance_comparison.png`
- `feature_importance.png`
- `models/` directory

After running `real_time_detection.py`:
- `detection_report.txt` (if alerts found)

---

**Need more details?** See `README.md`

