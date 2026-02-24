"""
Health tracking and monitoring module for continuous eye health surveillance.
Tracks probability-based health scores and alerts on significant changes.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np


class HealthTracker:
    """Track eye health over time with alert thresholds."""
    
    def __init__(self, tracking_file: str = "eye_health_history.json"):
        """
        Initialize health tracker.
        
        Args:
            tracking_file: Path to store health history JSON
        """
        self.tracking_file = tracking_file
        self.history = self._load_history()
    
    def _load_history(self) -> List[Dict]:
        """Load health history from file."""
        if os.path.exists(self.tracking_file):
            try:
                with open(self.tracking_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Warning: Could not load history from {self.tracking_file}: {e}")
                return []
        return []
    
    def _save_history(self):
        """Save health history to file."""
        with open(self.tracking_file, 'w') as f:
            json.dump(self.history, f, indent=2)
    
    def record_measurement(
        self,
        image_path: str,
        healthy_prob: float,
        severe_prob: float,
        predicted_class: str,
        confidence: float,
        notes: str = ""
    ) -> Dict:
        """
        Record a health measurement.
        
        Args:
            image_path: Path to the image tested
            healthy_prob: Probability of healthy classification
            severe_prob: Probability of severe classification
            predicted_class: The predicted class label
            confidence: Confidence of the prediction
            notes: Optional notes
            
        Returns:
            Measurement record
        """
        # Calculate health score (0-100, where 100 is perfectly healthy)
        health_score = healthy_prob * 100
        
        record = {
            'timestamp': datetime.now().isoformat(),
            'image': str(image_path),
            'health_score': round(health_score, 2),
            'healthy_prob': round(healthy_prob, 4),
            'severe_prob': round(severe_prob, 4),
            'predicted_class': predicted_class,
            'confidence': round(confidence, 4),
            'notes': notes
        }
        
        self.history.append(record)
        self._save_history()
        
        return record
    
    def get_health_trend(self, window_size: int = 5) -> Dict:
        """
        Get health trend with moving average.
        
        Args:
            window_size: Number of measurements for moving average
            
        Returns:
            Trend analysis
        """
        if len(self.history) < 2:
            return {
                'status': 'insufficient_data',
                'message': 'Need at least 2 measurements'
            }
        
        scores = [m['health_score'] for m in self.history]
        
        # Calculate trend
        current = scores[-1]
        previous = scores[-2] if len(scores) > 1 else scores[-1]
        change = current - previous
        percent_change = (change / previous * 100) if previous > 0 else 0
        
        # Moving average
        window = min(window_size, len(scores))
        moving_avg = np.mean(scores[-window:])
        
        # Long-term average
        overall_avg = np.mean(scores)
        
        trend = {
            'latest_score': round(current, 2),
            'previous_score': round(previous, 2),
            'change': round(change, 2),
            'percent_change': round(percent_change, 2),
            'moving_average': round(moving_avg, 2),
            'overall_average': round(overall_avg, 2),
            'total_measurements': len(self.history),
            'trend_direction': 'improving' if change > 0 else 'declining',
            'trend_severity': self._classify_trend_severity(change),
        }
        
        return trend
    
    def _classify_trend_severity(self, change: float) -> str:
        """Classify severity of health change."""
        if change > 5:
            return "significant_improvement"
        elif change > 1:
            return "mild_improvement"
        elif change > -1:
            return "stable"
        elif change > -5:
            return "mild_decline"
        else:
            return "significant_decline"
    
    def check_alert_threshold(
        self,
        health_score: float,
        threshold_percent: float = 10.0
    ) -> Tuple[bool, Optional[str]]:
        """
        Check if health has dropped by threshold percent.
        
        Args:
            health_score: Latest health score
            threshold_percent: Percent drop to trigger alert (default 15%)
            
        Returns:
            (should_alert, alert_message)
        """
        if len(self.history) < 2:
            return False, None
        
        # Compare to most recent different measurement
        recent_scores = [m['health_score'] for m in self.history[-10:]]
        peak_score = max(recent_scores) if recent_scores else health_score
        
        percent_drop = ((peak_score - health_score) / peak_score * 100)
        
        if percent_drop >= threshold_percent:
            message = (
                f"⚠️  ALERT: Eye health has declined by {percent_drop:.1f}%\n"
                f"   Peak health score: {peak_score:.1f}\n"
                f"   Current health score: {health_score:.1f}\n"
                f"   Recommendation: Consider scheduling an eye examination"
            )
            return True, message
        
        return False, None
    
    def get_statistics(self) -> Dict:
        """Get comprehensive health statistics."""
        if len(self.history) == 0:
            return {'status': 'no_data'}
        
        scores = [m['health_score'] for m in self.history]
        healthy_counts = [m for m in self.history if m['predicted_class'] == 'Healthy']
        severe_counts = [m for m in self.history if m['predicted_class'] == 'Severe']
        
        stats = {
            'total_measurements': len(self.history),
            'first_measurement': self.history[0]['timestamp'],
            'latest_measurement': self.history[-1]['timestamp'],
            'health_score': {
                'current': round(scores[-1], 2),
                'min': round(min(scores), 2),
                'max': round(max(scores), 2),
                'mean': round(np.mean(scores), 2),
                'std': round(np.std(scores), 2),
            },
            'prediction_distribution': {
                'healthy': len(healthy_counts),
                'severe': len(severe_counts),
                'healthy_percent': round(len(healthy_counts) / len(self.history) * 100, 1),
                'severe_percent': round(len(severe_counts) / len(self.history) * 100, 1),
            }
        }
        
        return stats
    
    def export_report(self, output_file: str = "eye_health_report.json") -> str:
        """
        Export comprehensive health report.
        
        Args:
            output_file: Output JSON file path
            
        Returns:
            Path to report file
        """
        report = {
            'generated_at': datetime.now().isoformat(),
            'statistics': self.get_statistics(),
            'trend': self.get_health_trend(),
            'history': self.history
        }
        
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        return output_file
    
    def get_all_measurements(self) -> List[Dict]:
        """Get all recorded measurements."""
        return self.history
    
    def clear_history(self):
        """Clear all history (use with caution)."""
        self.history = []
        self._save_history()


if __name__ == "__main__":
    # Demo usage
    tracker = HealthTracker()
    
    # Simulate measurements
    tracker.record_measurement(
        "test1.jpg",
        healthy_prob=0.95,
        severe_prob=0.05,
        predicted_class="Healthy",
        confidence=0.95,
        notes="First measurement"
    )
    
    tracker.record_measurement(
        "test2.jpg",
        healthy_prob=0.92,
        severe_prob=0.08,
        predicted_class="Healthy",
        confidence=0.92,
        notes="Second measurement"
    )
    
    tracker.record_measurement(
        "test3.jpg",
        healthy_prob=0.78,
        severe_prob=0.22,
        predicted_class="Severe",
        confidence=0.78,
        notes="Third measurement - noticeable decline"
    )
    
    # Print statistics
    print("\n=== Eye Health Tracking Report ===\n")
    print("Statistics:")
    stats = tracker.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print("\nTrend Analysis:")
    trend = tracker.get_health_trend()
    for key, value in trend.items():
        print(f"  {key}: {value}")
    
    # Check alerts
    alert, msg = tracker.check_alert_threshold(78, threshold_percent=15)
    if alert:
        print(f"\n{msg}")
    
    print(f"\nReport exported to {tracker.export_report()}")
