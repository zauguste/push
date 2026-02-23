# 👁️ Continuous Eye Health Monitoring System

This cataract detection system has been enhanced with **continuous health monitoring** capabilities. Track your eye health over time and receive alerts when significant changes occur.

## ✨ New Features

### 1. **Health Tracking** 
- Each prediction generates a health score (0-100)
- Stores historical data with timestamps
- Calculates trends and moving averages

### 2. **Automatic Alerts**
- Triggers when health drops **15% from peak**
- Provides medical recommendations
- Includes confidence metrics

### 3. **Continuous Monitoring**
- Single image analysis
- Batch directory scanning
- Continuous watch mode (repeating checks)

### 4. **Health Dashboard**
- Text-based health summary
- Visual trend plots (with matplotlib)
- Detailed statistical reports
- Classification distribution charts

---

## 🚀 Quick Start - Test with "autumn_left.png"

### Step 1: Test Single Image with Health Tracking
```bash
python predict.py --image autumn_left.png --track-health --notes "Initial eye check"
```

**Output shows:**
- ✅ Prediction: Healthy or Severe
- 💚 Health Score: X/100
- 📊 Healthy/Severe probabilities
- 📈 Trend direction (if previous measurements exist)
- 🚨 Alert (if health dropped >15%)

### Step 2: View Your Health Dashboard
```bash
python health_dashboard.py dashboard
```

**Shows:**
- Current health score
- Historical measurements
- Trend analysis
- Classification distribution

### Step 3: Analyze in Continuous Monitor
```bash
python continuous_monitor.py single --image autumn_left.png
```

**Advanced output:**
- Health score with detailed metrics
- Trend comparison
- Sessions report with statistics

---

## 📋 Core Modules

### `health_tracker.py`
**Handles health data persistence and analysis**

Key Functions:
- `record_measurement()` - Store a new health measurement
- `get_health_trend()` - Get trend analysis
- `check_alert_threshold()` - Check if alert should trigger
- `get_statistics()` - Get comprehensive stats
- `export_report()` - Export JSON report

Example:
```python
from health_tracker import HealthTracker

tracker = HealthTracker()
tracker.record_measurement(
    image_path="eye.jpg",
    healthy_prob=0.92,
    severe_prob=0.08,
    predicted_class="Healthy",
    confidence=0.92
)

# Check if alert
alert, msg = tracker.check_alert_threshold(92.0, threshold_percent=15)
if alert:
    print(msg)  # Contains medical recommendation

# View statistics
stats = tracker.get_statistics()
print(f"Total measurements: {stats['total_measurements']}")
```

---

### `continuous_monitor.py`
**Main application for eye health monitoring**

**Three Operating Modes:**

#### Mode 1: Single Image
```bash
python continuous_monitor.py single --image path/to/image.jpg
```
- Tests one image
- Tracks health
- Shows trend
- Saves to `latest_result.json`

#### Mode 2: Directory Scan (One-time batch)
```bash
python continuous_monitor.py directory --directory ./test_images/
```
- Analyzes all images in directory
- Recursively searches subdirs
- Prints batch statistics
- Exports session report

#### Mode 3: Continuous Watch (Repeating)
```bash
python continuous_monitor.py watch --directory ./daily_images/ --interval 300
```
- Checks directory every 5 minutes (300 seconds)
- Processes new images automatically
- Tracks health trends continuously
- **ALERTS when 15% drop detected**
- Export final report on exit

Optional parameters:
- `--interval N` - Check interval in seconds (default 300)
- `--iterations N` - Max checks before stopping (default infinite)
- `--device cuda` - Use GPU (default auto-detect)
- `--model path/to/model.pt` - Custom model checkpoint
- `--notes "text"` - Add notes to measurements

---

### `health_dashboard.py`
**Visualize and analyze health data**

#### Commands:

**1. Text Dashboard**
```bash
python health_dashboard.py dashboard
```
Shows:
```
👁️  EYE HEALTH TRACKING DASHBOARD
═══════════════════════════════════════════════════════════════════════════

📊 OVERVIEW
Total Measurements: 7
Tracking Period: 2026-02-23T10:00:00 to 2026-02-23T16:30:00

💚 HEALTH SCORE
Current:   78.0  ████████████...
Average:   85.7  █████████████...
Peak:      92.0  ███████████████...
Low:       76.0  ████████...

📈 TREND ANALYSIS
Direction:       ↘ declining
Latest Score:    78.0
Change:          -14.0 (-15.2%)
Severity:        significant_decline
```

**2. Quick Summary**
```bash
python health_dashboard.py summary
```

**3. Plot Health Trend** (requires `pip install matplotlib`)
```bash
python health_dashboard.py plot --output my_trend.png
```
Generates visualizations:
- Health score line plot
- Moving average trend
- Classification timeline
- Visual history

**4. Detailed Report**
```bash
python health_dashboard.py report --output detailed_report.txt
```
Exports comprehensive text report with all measurements

---

### Updated `predict.py`
**Now integrates with health tracking**

New parameters:
- `--track-health` - Enable health history tracking
- `--tracking-file path` - Custom history file (default: `eye_health_history.json`)
- `--notes "text"` - Add notes to measurement

Example:
```bash
# Single image with tracking
python predict.py \
  --image eye.jpg \
  --track-health \
  --notes "After clinic visit" \
  --output result.json

# Batch with tracking
python predict.py \
  --image ./images/ \
  --track-health \
  --output batch_results.json
```

---

## 📊 Understanding Health Metrics

### Health Score (0-100)
| Score | Status | Meaning |
|-------|--------|---------|
| 90-100 | ✅ Excellent | Strong healthy classification |
| 70-89 | ✅ Good | Healthy with minor concerns |
| 50-69 | ⚠️ Moderate | Mixed probabilities |
| 30-49 | 🚨 Declining | Nearing severe classification |
| 0-29 | 🚨 Poor | Strong severe classification |

### Alert Mechanism
- Tracks **peak health score** during monitoring period
- Triggers alert when current score drops **15% below peak**
- Example: Peak 85 → Current 72 = 13pt drop (15.3%) → **ALERT**
- Provides medical recommendation to seek eye examination

### Probability Breakdown
- `healthy_prob`: Confidence the eye is healthy (0-1)
- `severe_prob`: Confidence the eye shows cataract severity (0-1)
- Health Score = healthy_prob × 100

---

## 📁 Output Files

### `eye_health_history.json`
Main tracking database with complete measurement history:
```json
[
  {
    "timestamp": "2026-02-23T10:30:00",
    "image": "path/to/autumn_left.png",
    "health_score": 92.0,
    "healthy_prob": 0.92,
    "severe_prob": 0.08,
    "predicted_class": "Healthy",
    "confidence": 0.951,
    "notes": "Initial eye check"
  },
  ...
]
```

### `latest_result.json`
Most recent analysis with full details:
```json
{
  "image": "autumn_left.png",
  "timestamp": "2026-02-23T10:30:00",
  "predicted_class": "Healthy",
  "confidence": 0.951,
  "probabilities": {"Healthy": 0.92, "Severe": 0.08},
  "health_score": 92.0,
  "alert_triggered": false,
  "alert_message": null,
  "trend": {
    "latest_score": 92.0,
    "change": 0.0,
    "trend_direction": "stable"
  }
}
```

### `health_trend.png`
Visual plot of health over time (matplotlib required)

### `eye_health_detailed_report.txt`
Text report with statistics and full measurement history

---

## 🧪 Try the Demo

See the system in action:
```bash
python demo_monitoring.py
```

This:
- ✅ Simulates 7 days of measurements
- 📊 Shows dashboard output
- 🚨 Demonstrates alert trigger
- 💾 Exports sample reports
- 📋 Explains next steps

---

## 🔄 Workflow Examples

### Daily Check
```bash
# Take a photo of your eye daily
python continuous_monitor.py single --image today.jpg

# Check trend
python health_dashboard.py summary
```

### Weekly Monitoring
```bash
# Store weekly images in a folder
mkdir weekly_checks
# ... take photo ...
python continuous_monitor.py directory --directory weekly_checks/

# Generate trend plot
python health_dashboard.py plot
```

### Continuous Surveillance (24/7)
```bash
# Run background monitoring
python continuous_monitor.py watch \
  --directory ./daily_images/ \
  --interval 3600 \
  &
  
# You'll get alerts if health drops >15%
# Retrieve results anytime
python health_dashboard.py dashboard
```

---

## ⚙️ Advanced Configuration

### Custom Model
```bash
python continuous_monitor.py single --image eye.jpg --model my_model.pt
```

### Use GPU
```bash
python continuous_monitor.py watch --directory ./images/ --device cuda
```

### Custom Tracking File
```bash
python predict.py --image eye.jpg --track-health --tracking-file my_health.json
```

### Add Measurement Notes
```bash
python predict.py \
  --image eye.jpg \
  --track-health \
  --notes "After eye drops, morning measurement"
```

---

## 📝 Integration with Existing Scripts

### Training with Health Tracking
The training pipeline automatically tracks test set performance:
```bash
python train.py --epochs 20
# Automatically creates health tracking for test predictions
```

### MLflow Integration
Training results are logged to MLflow along with health metrics:
```bash
python run_pipeline.py
# Training + MLflow tracking + health monitoring
```

---

## ⚠️ Important Notes

### Medical Disclaimer
✅ This system is for **monitoring and surveillance only**  
❌ NOT a medical diagnosis tool  
✅ Always consult ophthalmologist for:
  - Diagnosis
  - Treatment decisions
  - Health concerns
  - Alert interpretation

### Privacy & Security
- ✅ All data stored locally in `eye_health_history.json`
- ✅ NO data sent to external servers
- ⚠️ Keep history file secure - contains health information
- ✅ Delete files if needed for privacy

### Data Continuity
- Health tracking persists across sessions
- Keep `eye_health_history.json` for historical analysis
- Backup regularly for data preservation
- Can export reports for archiving

---

## 🆘 Troubleshooting

**Q: No alerts showing up?**
A: Need at least 2 measurements. Peak must drop 15% from baseline.

**Q: "Image not found" error?**
A: Verify image path and file extension (.jpg, .png, .jpeg)

**Q: Can't plot trends?**
A: Install matplotlib: `pip install matplotlib`

**Q: History not persisting?**
A: Check file permissions on `eye_health_history.json`

**Q: Model loading error?**
A: Ensure model checkpoint exists at `checkpoints/model_best.pt`

---

## 📚 files at a glance

| File | Purpose |
|------|---------|
| `health_tracker.py` | Core health tracking logic |
| `continuous_monitor.py` | Main monitoring application |
| `health_dashboard.py` | Visualization & reporting |
| `predict.py` | **(Updated)** Inference with health tracking |
| `demo_monitoring.py` | Interactive demo |
| `CONTINUOUS_MONITORING_GUIDE.md` | Detailed usage guide |

---

## 🎯 Quick Command Reference

```bash
# Test single image
python continuous_monitor.py single --image autumn_left.png

# Batch analyze
python continuous_monitor.py directory --directory ./images/

# Continuous monitoring
python continuous_monitor.py watch --directory ./images/ --iterations 10

# View dashboard
python health_dashboard.py dashboard

# Plot trends
python health_dashboard.py plot

# Export report
python health_dashboard.py report

# Demo
python demo_monitoring.py
```

---

## 🚀 Getting Started Now

1. **Test with your image:**
   ```bash
   python continuous_monitor.py single --image autumn_left.png
   ```

2. **View the result:**
   ```bash
   cat latest_result.json
   ```

3. **Check dashboard:**
   ```bash
   python health_dashboard.py dashboard
   ```

4. **Explore more:**
   - Read `CONTINUOUS_MONITORING_GUIDE.md`
   - Run `demo_monitoring.py`
   - Try `health_dashboard.py plot`

---

**Happy monitoring! 👁️✨**
