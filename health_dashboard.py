"""
Eye health dashboard: Visualize trends and health metrics over time.
"""

import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict
import numpy as np

try:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

from health_tracker import HealthTracker


class HealthDashboard:
    """Display and visualize eye health metrics."""
    
    def __init__(self, tracking_file: str = "eye_health_history.json"):
        """Initialize dashboard."""
        self.tracker = HealthTracker(tracking_file)
        self.history = self.tracker.get_all_measurements()
    
    def print_text_dashboard(self):
        """Print a text-based dashboard to console."""
        if not self.history:
            print("No health measurements recorded yet.")
            return
        
        print("\n" + "=" * 80)
        print("👁️  EYE HEALTH TRACKING DASHBOARD")
        print("=" * 80)
        
        # Statistics
        stats = self.tracker.get_statistics()
        print(f"\n📊 OVERVIEW")
        print("─" * 80)
        print(f"Total Measurements: {stats['total_measurements']}")
        print(f"Tracking Period: {stats['first_measurement']} to {stats['latest_measurement']}")
        
        # Health score
        hs = stats['health_score']
        print(f"\n💚 HEALTH SCORE (0-100, higher is better)")
        print("─" * 80)
        print(f"Current:  {hs['current']:6.1f}  {'█' * int(hs['current']/5)} {''}")
        print(f"Average:  {hs['mean']:6.1f}  {'█' * int(hs['mean']/5)}")
        print(f"Peak:     {hs['max']:6.1f}  {'█' * int(hs['max']/5)}")
        print(f"Low:      {hs['min']:6.1f}  {'█' * int(hs['min']/5)}")
        
        # Prediction distribution
        pd = stats['prediction_distribution']
        print(f"\n🔍 CLASSIFICATIONS")
        print("─" * 80)
        healthy_bar = "█" * int(pd['healthy_percent'] / 2)
        severe_bar = "█" * int(pd['severe_percent'] / 2)
        print(f"Healthy: {pd['healthy_percent']:5.1f}%  {healthy_bar}")
        print(f"Severe:  {pd['severe_percent']:5.1f}%  {severe_bar}")
        
        # Trend analysis
        trend = self.tracker.get_health_trend()
        if 'latest_score' in trend:
            print(f"\n📈 TREND ANALYSIS")
            print("─" * 80)
            direction = "↗" if trend['trend_direction'] == 'improving' else "↘"
            print(f"Direction:       {direction} {trend['trend_direction'].replace('_', ' ').title()}")
            print(f"Latest Score:    {trend['latest_score']:.1f}")
            print(f"Previous Score:  {trend['previous_score']:.1f}")
            print(f"Change:          {trend['change']:+.1f} ({trend['percent_change']:+.1f}%)")
            print(f"Trend Severity:  {trend['trend_severity'].replace('_', ' ').title()}")
            print(f"Moving Average:  {trend['moving_average']:.1f}")
        
        # Recent measurements
        print(f"\n📅 RECENT MEASUREMENTS (Last 10)")
        print("─" * 80)
        print(f"{'Date':<20} {'Class':<10} {'Health':<8} {'Prob':<10} {'Confidence':<12}")
        print("─" * 80)
        
        for measurement in self.history[-10:]:
            ts = datetime.fromisoformat(measurement['timestamp']).strftime("%Y-%m-%d %H:%M")
            cls = measurement['predicted_class']
            health = measurement['health_score']
            prob = f"{measurement['healthy_prob']:.1%}"
            conf = f"{measurement['confidence']:.1%}"
            print(f"{ts:<20} {cls:<10} {health:>6.1f}  {prob:>8}  {conf:>10}")
        
        print("\n" + "=" * 80 + "\n")
    
    def print_summary(self):
        """Print a concise health summary."""
        if not self.history:
            print("No measurements recorded.")
            return
        
        latest = self.history[-1]
        stats = self.tracker.get_statistics()
        trend = self.tracker.get_health_trend()
        
        print("\n" + "=" * 60)
        print("👁️  EYE HEALTH SUMMARY")
        print("=" * 60)
        
        print(f"\nCurrent Status: {latest['predicted_class']}")
        print(f"Health Score: {latest['health_score']:.1f}/100")
        print(f"Last Updated: {latest['timestamp']}")
        
        if 'latest_score' in trend:
            print(f"\nTrend: {trend['trend_direction'].replace('_', ' ').title()}")
            print(f"Recent Change: {trend['change']:+.1f} ({trend['percent_change']:+.1f}%)")
        
        print(f"\nTotal Measurements: {stats['total_measurement']}")
        print(f"Healthy: {stats['prediction_distribution']['healthy_percent']:.0f}%")
        print("\n" + "=" * 60 + "\n")
    
    def plot_health_trend(self, output_file: str = "health_trend.png"):
        """
        Plot health trend over time.
        
        Args:
            output_file: Path to save plot
            
        Returns:
            True if successful, False if matplotlib not available
        """
        if not MATPLOTLIB_AVAILABLE:
            print("❌ matplotlib not installed. Install with: pip install matplotlib")
            return False
        
        if len(self.history) < 2:
            print("Need at least 2 measurements to plot trend.")
            return False
        
        # Extract data
        timestamps = [datetime.fromisoformat(m['timestamp']) for m in self.history]
        health_scores = [m['health_score'] for m in self.history]
        classes = [m['predicted_class'] for m in self.history]
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Plot 1: Health score trend
        ax1.plot(timestamps, health_scores, marker='o', linewidth=2, markersize=6, color='#2ecc71')
        ax1.fill_between(timestamps, health_scores, alpha=0.3, color='#2ecc71')
        ax1.axhline(y=np.mean(health_scores), color='#3498db', linestyle='--', label='Average')
        ax1.set_ylabel('Health Score (0-100)', fontsize=11)
        ax1.set_title('Eye Health Score Over Time', fontsize=13, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        ax1.set_ylim([0, 100])
        
        # Plot 2: Classification timeline
        colors = {'Healthy': '#2ecc71', 'Severe': '#e74c3c', 'Mild': '#f39c12', 'Moderate': '#e67e22'}
        class_colors = [colors.get(c, '#95a5a6') for c in classes]
        ax2.scatter(timestamps, [1]*len(timestamps), c=class_colors, s=100, alpha=0.7)
        ax2.set_ylabel('Classification', fontsize=11)
        ax2.set_title('Classification Timeline', fontsize=13, fontweight='bold')
        ax2.set_yticks([])
        
        # Add legend for classifications
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=color, label=label) 
                          for label, color in colors.items() if label in classes]
        ax2.legend(handles=legend_elements, loc='upper right')
        
        # Format x-axis
        for ax in [ax1, ax2]:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            ax.xaxis.set_major_locator(mdates.AutoDateLocator())
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"✅ Plot saved to {output_file}")
        
        return True
    
    def export_detailed_report(self, output_file: str = "eye_health_detailed_report.txt"):
        """Export detailed text report."""
        if not self.history:
            print("No measurements to report.")
            return
        
        with open(output_file, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("EYE HEALTH TRACKING - DETAILED REPORT\n")
            f.write("=" * 80 + "\n\n")
            
            # Summary
            stats = self.tracker.get_statistics()
            f.write("SUMMARY\n")
            f.write("-" * 80 + "\n")
            f.write(f"Generated: {datetime.now().isoformat()}\n")
            f.write(f"Total Measurements: {stats['total_measurements']}\n")
            f.write(f"Tracking Period: {stats['first_measurement']} to {stats['latest_measurement']}\n\n")
            
            # Health metrics
            hs = stats['health_score']
            f.write("HEALTH SCORE METRICS\n")
            f.write("-" * 80 + "\n")
            f.write(f"Current:       {hs['current']:.2f}/100\n")
            f.write(f"Average:       {hs['mean']:.2f}/100\n")
            f.write(f"Peak:          {hs['max']:.2f}/100\n")
            f.write(f"Low:           {hs['min']:.2f}/100\n")
            f.write(f"Std Dev:       {hs['std']:.2f}\n\n")
            
            # Detailed history
            f.write("DETAILED MEASUREMENT HISTORY\n")
            f.write("-" * 80 + "\n")
            f.write(f"{'#':<4} {'Timestamp':<25} {'Class':<10} {'Health':<8} {'Healthy%':<12} {'Severe%':<12} {'Conf':<8}\n")
            f.write("-" * 80 + "\n")
            
            for i, m in enumerate(self.history, 1):
                ts = datetime.fromisoformat(m['timestamp']).strftime("%Y-%m-%d %H:%M:%S")
                f.write(f"{i:<4} {ts:<25} {m['predicted_class']:<10} {m['health_score']:>6.1f}  "
                       f"{m['healthy_prob']:>10.1%}  {m['severe_prob']:>10.1%}  {m['confidence']:>6.1%}\n")
            
            f.write("\n" + "=" * 80 + "\n")
        
        print(f"✅ Detailed report saved to {output_file}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Eye health dashboard and visualization")
    parser.add_argument(
        "--tracking-file",
        default="eye_health_history.json",
        help="Path to health history file"
    )
    parser.add_argument(
        "command",
        choices=["dashboard", "summary", "plot", "report"],
        help="Command to execute"
    )
    parser.add_argument(
        "--output",
        help="Output file path (for plot and report)"
    )
    
    args = parser.parse_args()
    
    # Initialize dashboard
    dashboard = HealthDashboard(args.tracking_file)
    
    # Execute command
    if args.command == "dashboard":
        dashboard.print_text_dashboard()
    elif args.command == "summary":
        dashboard.print_summary()
    elif args.command == "plot":
        output = args.output or "health_trend.png"
        dashboard.plot_health_trend(output)
    elif args.command == "report":
        output = args.output or "eye_health_detailed_report.txt"
        dashboard.export_detailed_report(output)


if __name__ == "__main__":
    main()
