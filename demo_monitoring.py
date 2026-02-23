#!/usr/bin/env python3
"""
Demo script: Showcase the continuous health monitoring system.
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime, timedelta

# Import our modules
from health_tracker import HealthTracker
from health_dashboard import HealthDashboard


def simulate_measurements():
    """Simulate a series of eye health measurements over time."""
    print("\n" + "=" * 70)
    print("🚀 CONTINUOUS EYE HEALTH MONITORING - DEMO")
    print("=" * 70)
    
    print("\n📝 Creating simulated health measurements...\n")
    
    tracker = HealthTracker("demo_health_history.json")
    
    # Simulate measurements over a week
    measurements = [
        {
            "day": 0,
            "healthy_prob": 0.92,
            "severe_prob": 0.08,
            "predicted_class": "Healthy",
            "notes": "Day 1 - Baseline measurement"
        },
        {
            "day": 1,
            "healthy_prob": 0.91,
            "severe_prob": 0.09,
            "predicted_class": "Healthy",
            "notes": "Day 2 - Slight variation"
        },
        {
            "day": 2,
            "healthy_prob": 0.90,
            "severe_prob": 0.10,
            "predicted_class": "Healthy",
            "notes": "Day 3 - Still healthy"
        },
        {
            "day": 3,
            "healthy_prob": 0.85,
            "severe_prob": 0.15,
            "predicted_class": "Healthy",
            "notes": "Day 4 - Minor decline"
        },
        {
            "day": 4,
            "healthy_prob": 0.78,
            "severe_prob": 0.22,
            "predicted_class": "Severe",
            "notes": "Day 5 - NOTABLE DECLINE (15% drop from baseline)"
        },
        {
            "day": 5,
            "healthy_prob": 0.76,
            "severe_prob": 0.24,
            "predicted_class": "Severe",
            "notes": "Day 6 - Continued decline"
        },
        {
            "day": 6,
            "healthy_prob": 0.77,
            "severe_prob": 0.23,
            "predicted_class": "Severe",
            "notes": "Day 7 - Slight stabilization"
        },
    ]
    
    # Record measurements
    alert_triggered = False
    for m in measurements:
        # Create timestamp
        ts = datetime.now() - timedelta(days=6-m['day'])
        
        # Record
        measurement = tracker.record_measurement(
            image_path=f"demo_image_day{m['day']}.jpg",
            healthy_prob=m['healthy_prob'],
            severe_prob=m['severe_prob'],
            predicted_class=m['predicted_class'],
            confidence=min(m['healthy_prob'], m['severe_prob']) + 0.05,
            notes=m['notes']
        )
        
        # Simulate timestamp (hack for demo)
        if m['day'] > 0:
            tracker.history[-1]['timestamp'] = ts.isoformat()
        
        print(f"  ✓ {m['notes']}")
        print(f"    Health Score: {measurement['health_score']:.1f}/100")
        
        # Check alert
        alert, msg = tracker.check_alert_threshold(
            measurement['health_score'],
            threshold_percent=15.0
        )
        
        if alert and not alert_triggered:
            print(f"    🚨 {msg.split(chr(10))[0]}")
            alert_triggered = True
    
    # Save history
    tracker._save_history()
    
    return tracker


def show_dashboard(tracker):
    """Display the health dashboard."""
    print("\n" + "=" * 70)
    print("📊 HEALTH DASHBOARD")
    print("=" * 70)
    
    dashboard = HealthDashboard("demo_health_history.json")
    
    # Print dashboard
    print()
    stats = tracker.get_statistics()
    
    # Overview
    print("📋 OVERVIEW")
    print("─" * 70)
    print(f"Total Measurements: {stats['total_measurements']}")
    print(f"Tracking Span: {stats['total_measurements']} days")
    
    # Health score
    hs = stats['health_score']
    print(f"\n💚 HEALTH SCORE METRICS")
    print("─" * 70)
    print(f"Current:  {hs['current']:6.1f}  {'█' * int(hs['current']/5)}")
    print(f"Average:  {hs['mean']:6.1f}  {'█' * int(hs['mean']/5)}")
    print(f"Peak:     {hs['max']:6.1f}  {'█' * int(hs['max']/5)}")
    print(f"Low:      {hs['min']:6.1f}  {'█' * int(hs['min']/5)}")
    
    # Classification distribution
    pd = stats['prediction_distribution']
    print(f"\n🔍 CLASSIFICATION DISTRIBUTION")
    print("─" * 70)
    healthy_bar = "█" * int(pd['healthy_percent'] / 2)
    severe_bar = "█" * int(pd['severe_percent'] / 2)
    print(f"Healthy: {pd['healthy_percent']:5.1f}%  {healthy_bar}")
    print(f"Severe:  {pd['severe_percent']:5.1f}%  {severe_bar}")
    
    # Trend
    trend = tracker.get_health_trend()
    print(f"\n📈 TREND ANALYSIS")
    print("─" * 70)
    direction = "↗" if trend['trend_direction'] == 'improving' else "↘"
    print(f"Direction:       {direction} {trend['trend_direction'].replace('_', ' ').title()}")
    print(f"Latest Score:    {trend['latest_score']:.1f}")
    print(f"Overall Change:  {trend['change']:+.1f} ({trend['percent_change']:+.1f}%)")
    print(f"Trend Severity:  {trend['trend_severity'].replace('_', ' ').title()}")
    
    # Recent history
    print(f"\n📅 MEASUREMENT HISTORY")
    print("─" * 70)
    print(f"{'#':<3} {'Day':<5} {'Health':<8} {'Class':<10} {'Notes'}")
    print("─" * 70)
    
    for i, m in enumerate(tracker.history, 1):
        health = m['health_score']
        cls = m['predicted_class']
        notes = m['notes'][:40]
        day = i
        print(f"{i:<3} {day:<5} {health:>6.1f}  {cls:<10} {notes}")


def show_alerts(tracker):
    """Show alert information."""
    print("\n" + "=" * 70)
    print("🚨 ALERT ANALYSIS")
    print("=" * 70)
    
    print("\n📌 Alert Threshold: 15% drop in eye health\n")
    
    history = tracker.get_all_measurements()
    current = history[-1]['health_score']
    peak = max([m['health_score'] for m in history])
    drop = peak - current
    drop_percent = (drop / peak * 100)
    
    print(f"Peak Health Score:     {peak:.1f}")
    print(f"Current Health Score:  {current:.1f}")
    print(f"Total Decline:         {drop:+.1f} ({drop_percent:+.1f}%)")
    
    if drop_percent >= 15:
        print(f"\n⚠️  ALERT TRIGGERED!")
        print(f"Health has declined {drop_percent:.1f}% from peak.")
        print(f"Recommendation: Schedule an eye examination")
    else:
        print(f"\nNo alert. Decline is below 15% threshold.")


def show_export_options(tracker):
    """Show export options."""
    print("\n" + "=" * 70)
    print("💾 DATA EXPORT OPTIONS")
    print("=" * 70)
    
    print("\nGenerated files:")
    
    # Save detailed report
    report_file = tracker.export_report("demo_eye_health_report.json")
    print(f"  ✓ {report_file} (JSON report)")
    
    # Show file sizes
    if os.path.exists("demo_eye_health_report.json"):
        size = os.path.getsize("demo_eye_health_report.json")
        print(f"    Size: {size} bytes")
    
    if os.path.exists("demo_health_history.json"):
        size = os.path.getsize("demo_health_history.json")
        print(f"  ✓ demo_health_history.json ({size} bytes)")


def show_next_steps():
    """Show next steps and real usage."""
    print("\n" + "=" * 70)
    print("🎯 NEXT STEPS - TRY IT YOURSELF")
    print("=" * 70)
    
    print("""
1️⃣  TEST WITH YOUR OWN IMAGE:
    python continuous_monitor.py single --image path/to/your/image.jpg
    
2️⃣  TRACK HEALTH OVER TIME:
    python predict.py --image eye.jpg --track-health
    
3️⃣  VIEW YOUR HEALTH DASHBOARD:
    python health_dashboard.py dashboard
    
4️⃣  PLOT YOUR HEALTH TREND:
    python health_dashboard.py plot
    
5️⃣  CONTINUOUS MONITORING (every 5 minutes):
    python continuous_monitor.py watch --directory ./my_images/ --interval 300

📊 All data is stored in: eye_health_history.json
🔒 Privacy: All data stays on your computer
⚡ Alerts: You'll be notified of 15% health drops
    """)


def cleanup_demo_files():
    """Clean up demo files."""
    demo_files = [
        "demo_health_history.json",
        "demo_eye_health_report.json"
    ]
    
    for f in demo_files:
        if os.path.exists(f):
            os.remove(f)
            print(f"  Cleaned up {f}")


def main():
    """Run the demo."""
    try:
        # Run demo
        tracker = simulate_measurements()
        show_dashboard(tracker)
        show_alerts(tracker)
        show_export_options(tracker)
        show_next_steps()
        
        # Cleanup
        print("\n[Demo completed]")
        response = input("\nClean up demo files? (y/n): ").lower()
        if response == 'y':
            cleanup_demo_files()
            print("✓ Demo files cleaned up")
        else:
            print("✓ Demo files preserved for review")
        
        print("\n" + "=" * 70)
        print("✅ Demo complete! Check out the CONTINUOUS_MONITORING_GUIDE.md for details")
        print("=" * 70 + "\n")
        
        return 0
    
    except Exception as e:
        print(f"\n❌ Error during demo: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
