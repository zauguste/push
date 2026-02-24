"""
CONTINUOUS EYE HEALTH MONITORING SYSTEM - USAGE GUIDE

This system tracks eye health changes over time and alerts users to significant
deterioration. It combines the trained cataract detection model with health 
tracking to provide continuous surveillance of eye condition.
"""

# ============================================================================
# 1. QUICK START - TEST A SINGLE IMAGE WITH HEALTH TRACKING
# ============================================================================

# Test one image and track health:
python predict.py --image path/to/autumn_left.png --track-health --notes "Initial measurement"

# Output:
# - Prediction result with confidence
# - Health score (0-100, higher is better)
# - Health probability breakdown
# - Alert if significant change detected
# - Saved to: eye_health_history.json


# ============================================================================
# 2. CONTINUOUS MONITORING MODES
# ============================================================================

# MODE A: Single Image Analysis (One-time check)
# ───────────────────────────────────────────────────────────────────────────
python continuous_monitor.py single --image path/to/eye_image.jpg

# Output:
# - Prediction for that single image
# - Health score tracked
# - Trend comparison
# - Results saved to latest_result.json


# MODE B: Directory Scan (Batch analysis)
# ───────────────────────────────────────────────────────────────────────────
# Analyze all images in a directory once
python continuous_monitor.py directory --directory path/to/images/

# Output:
# - Analyzes all .png, .jpg, .jpeg files
# - Tracks health for each
# - Prints summary report
# - Saves final session report


# MODE C: Continuous Watch (Repeated monitoring)
# ───────────────────────────────────────────────────────────────────────────
# Monitor directory continuously - checks every 5 minutes
python continuous_monitor.py watch --directory path/to/images/ --interval 300

# Optional: Limit iterations (e.g., 10 checks)
python continuous_monitor.py watch --directory path/to/images/ --iterations 10 --interval 60

# Output:
# - Repeatedly scans for new images
# - Tracks health trends
# - ALERTS on 10% drop in eye health
# - Saves comprehensive report on exit
# - Press Ctrl+C to stop


# MODE D: Camera Monitoring (Real-time eye tracking)
# ───────────────────────────────────────────────────────────────────────────
# Monitor your eye health directly from your camera feed
python continuous_monitor.py camera --interval 10

# Save captured frames for later review
python continuous_monitor.py camera --save-frames --frame-save-dir ./my_eye_images/

# Monitor for a limited time (e.g., 50 captures)
python continuous_monitor.py camera --iterations 50 --interval 30

# Output:
# - Continuously captures eye images from camera
# - Analyzes each frame for cataract detection
# - Tracks health score changes over time
# - ALERTS when health drops 10% from baseline
# - Saves comprehensive monitoring report


# ============================================================================
# 3. HEALTH DASHBOARD & VISUALIZATION
# ============================================================================

# View text-based dashboard
python health_dashboard.py dashboard

# View quick summary
python health_dashboard.py summary

# Plot health trend (requires matplotlib)
python health_dashboard.py plot --output my_health_trend.png

# Export detailed report
python health_dashboard.py report --output my_detailed_report.txt


# ============================================================================
# 4. UNDERSTANDING HEALTH METRICS
# ============================================================================

# Health Score (0-100):
# ───────────────────────────────────────────────────────────────────────────
# 90-100: Excellent eye health (strong Healthy classification)
# 70-89:  Good eye health (Healthy classification, slight concerns)
# 50-69:  Moderate concerns (mixed probabilities)
# 30-49:  Eye health declining (approached Severe classification)
# 0-29:   Poor eye health (strong Severe classification)

# Alert Threshold:
# ───────────────────────────────────────────────────────────────────────────
# System alerts when health drops 10% from baseline
# Example: Peak=85 → Current=76.5 (10% drop = 8.5 points) → ALERT!


# ============================================================================
# 5. WORKFLOW EXAMPLES
# ============================================================================

# WORKFLOW 1: Daily Eye Health Monitoring
# ───────────────────────────────────────────────────────────────────────────
# Step 1: Take a selfie or get an eye image
# Step 2: Save to ~/my_eye_images/
# Step 3: Run dashboard check
#   python continuous_monitor.py single --image ~/my_eye_images/today.jpg
# Step 4: View health trend
#   python health_dashboard.py dashboard


# WORKFLOW 2: Weekly Tracking with Alerts
# ───────────────────────────────────────────────────────────────────────────
# Step 1: Schedule a task to run weekly
# Step 2: Test new image with tracking
#   python predict.py --image recent_eye.jpg --track-health --notes "Weekly check"
# Step 3: If alert is triggered, seek medical attention
# Step 4: Review trend
#   python health_dashboard.py plot


# WORKFLOW 3: Continuous Background Monitoring
# ───────────────────────────────────────────────────────────────────────────
# Step 1: Start continuous monitor (can run 24/7 or on a schedule)
#   python continuous_monitor.py watch --directory ./daily_images --interval 3600
# Step 2: Each hour, system checks for new images and updates health
# Step 3: Gets alerted if health drops 15%
# Step 4: Run dashboard to view statistics


# ============================================================================
# 6. OUTPUT FILES GENERATED
# ============================================================================

# eye_health_history.json
# ───────────────────────────────────────────────────────────────────────────
# Main health tracking database
# Contains all measurements with timestamps, scores, and probabilities
# Format: [
#   {
#     "timestamp": "2026-02-23T10:30:00",
#     "image": "path/to/image.jpg",
#     "health_score": 87.5,
#     "healthy_prob": 0.875,
#     "severe_prob": 0.125,
#     "predicted_class": "Healthy",
#     "confidence": 0.951,
#     "notes": "Daily check"
#   },
#   ...
# ]


# latest_result.json
# ───────────────────────────────────────────────────────────────────────────
# Most recent prediction result
# Includes full analysis, trend, and alert status


# eye_health_detailed_report.txt
# ───────────────────────────────────────────────────────────────────────────
# Text report with all measurements and statistics
# Good for archiving and detailed review


# health_trend.png
# ───────────────────────────────────────────────────────────────────────────
# Visual plot of health score over time
# Includes moving average and classification timeline


# ============================================================================
# 7. ADVANCED OPTIONS
# ============================================================================

# Using different model checkpoint:
python continuous_monitor.py single --image eye.jpg --model checkpoints/model_final.pt

# Using GPU instead of CPU:
python continuous_monitor.py watch --directory ./images --device cuda

# Custom alert threshold (10% instead of 15%):
# [Currently 15% is hardcoded, can be modified in health_tracker.py]

# Add notes to measurements:
python predict.py --image eye.jpg --track-health --notes "After eye drops application"

# Batch process with tracking:
python predict.py --image ./images/ --track-health --output results.json


# ============================================================================
# 8. INTEGRATION WITH EXISTING SCRIPTS
# ============================================================================

# Train and automatically track on test images
python train.py --epochs 20 --test-dir datasets/test

# The training script supports health tracking during evaluation


# ============================================================================
# 9. IMPORTANT NOTES
# ============================================================================

# ⚠️  MEDICAL DISCLAIMER
# ───────────────────────────────────────────────────────────────────────────
# This system is for monitoring purposes and NOT a medical diagnosis tool.
# Always consult with an ophthalmologist for actual diagnosis and treatment.
# A 15% health drop should prompt professional eye examination, not panic.


# 📊 DATA PRIVACY
# ───────────────────────────────────────────────────────────────────────────
# All health data is stored locally in eye_health_history.json
# No data is sent to external servers
# Keep this file secure - it contains health information


# 🔄 CONTINUOUS TRAINING / FINE-TUNING
# ───────────────────────────────────────────────────────────────────────────
# The current system uses a frozen pretrained model for prediction
# It tracks changes but does not retrain on new data
# To add retraining:
#   1. Label some user measurements as ground truth
#   2. Create a fine-tuning script
#   3. Retrain on accumulated data
#   (This is a potential future enhancement)


# ============================================================================
# 10. TROUBLESHOOTING
# ============================================================================

# Q: No measurements showing up?
# A: Check that images are in the correct directory and have .jpg/.png extension
#    Verify the model checkpoint exists at checkpoints/model_best.pt

# Q: Matplotlib not found for plotting?
# A: Install with: pip install matplotlib

# Q: Getting different results for same image?
# A: The model prediction is deterministic, but pre/post-processing might vary
#    with different image loads. This is normal.

# Q: How to reset history?
# A: Delete eye_health_history.json (backup first!)
#    Or programmatically: tracker = HealthTracker(); tracker.clear_history()

# Q: Camera not working?
# A: Ensure camera permissions are granted. Try different camera indices (0, 1, 2...)
#    python continuous_monitor.py camera --camera-index 1
#    Close other camera-using applications first.

# Q: Getting "Cannot open camera" error?
# A: Check camera connections and drivers. Try restarting the computer.
#    On Windows, ensure camera privacy settings allow access.


# ============================================================================
# SUMMARY OF COMMANDS
# ============================================================================

print("""
🚀 QUICK REFERENCE - MOST COMMON COMMANDS

1. Test single image:
   python continuous_monitor.py single --image path/to/image.jpg

2. Analyze batch directory:
   python continuous_monitor.py directory --directory ./images/

3. Continuous directory monitoring (every 5 min, 10 times):
   python continuous_monitor.py watch --directory ./images/ --iterations 10

4. Real-time camera monitoring (every 10 seconds):
   python continuous_monitor.py camera --interval 10

5. View health dashboard:
   python health_dashboard.py dashboard

6. Plot health trend:
   python health_dashboard.py plot

7. Export report:
   python health_dashboard.py report

📧 Questions? Check health_tracker.py and continuous_monitor.py docstrings
🔗 Integration with predict.py for quick testing with tracking
💾 All data saved locally in eye_health_history.json
""")
