"""
Sophisticated discontinuity detection for transformer v2 predictions
Detects level jumps, direction reversals, cusps, and phase errors
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
import math

class DiscontinuityDetector:
    """
    Sophisticated detector for transformer v2 prediction discontinuities
    """
    
    def __init__(self, 
                 level_jump_threshold_mm: float = 50.0,
                 direction_reversal_min_change_mm: float = 10.0,
                 cusp_detection_window: int = 6,
                 phase_error_threshold_ratio: float = 2.0):
        """
        Initialize discontinuity detector with configurable thresholds
        
        Args:
            level_jump_threshold_mm: Threshold for detecting level jumps
            direction_reversal_min_change_mm: Minimum change to consider direction significant
            cusp_detection_window: Number of points to analyze for cusp detection
            phase_error_threshold_ratio: Ratio threshold for phase error detection
        """
        self.level_jump_threshold = level_jump_threshold_mm
        self.direction_reversal_min_change = direction_reversal_min_change_mm
        self.cusp_window = cusp_detection_window
        self.phase_error_ratio = phase_error_threshold_ratio
        
    def calculate_trend(self, values: List[float], window_size: int = None) -> Dict:
        """
        Calculate trend in a series of values
        
        Args:
            values: List of water level values
            window_size: Number of points to analyze (default: all)
            
        Returns:
            Dictionary with trend metrics
        """
        if not values or len(values) < 2:
            return {
                'direction': 'unknown',
                'slope': 0.0,
                'magnitude': 0.0,
                'linearity': 0.0,
                'valid': False
            }
        
        if window_size:
            values = values[-window_size:] if len(values) > window_size else values
        
        # Remove -999 missing values
        valid_values = [v for v in values if v != -999]
        if len(valid_values) < 2:
            return {
                'direction': 'unknown',
                'slope': 0.0,
                'magnitude': 0.0,
                'linearity': 0.0,
                'valid': False
            }
        
        # Calculate linear trend
        x = np.arange(len(valid_values))
        y = np.array(valid_values)
        
        # Linear regression
        if len(x) > 1:
            slope, intercept = np.polyfit(x, y, 1)
            
            # Calculate R-squared for linearity
            y_pred = slope * x + intercept
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        else:
            slope = 0
            r_squared = 0
        
        # Determine direction
        total_change = valid_values[-1] - valid_values[0]
        direction = 'rising' if total_change > self.direction_reversal_min_change else \
                   'falling' if total_change < -self.direction_reversal_min_change else \
                   'stable'
        
        return {
            'direction': direction,
            'slope': slope,
            'magnitude': abs(total_change),
            'total_change': total_change,
            'linearity': r_squared,
            'valid': True,
            'point_count': len(valid_values)
        }
    
    def detect_level_jump(self, last_actual: float, first_predicted: float) -> Dict:
        """
        Detect simple level jumps between last actual and first predicted value
        
        Returns:
            Detection result with details
        """
        if last_actual == -999 or first_predicted == -999:
            return {
                'detected': False,
                'type': 'level_jump',
                'reason': 'missing_data',
                'jump_size': None
            }
        
        jump_size = abs(first_predicted - last_actual)
        detected = jump_size > self.level_jump_threshold
        
        return {
            'detected': detected,
            'type': 'level_jump',
            'jump_size': jump_size,
            'threshold': self.level_jump_threshold,
            'last_actual': last_actual,
            'first_predicted': first_predicted,
            'severity': 'high' if jump_size > self.level_jump_threshold * 2 else 'medium'
        }
    
    def detect_direction_reversal(self, actual_trend: Dict, predicted_trend: Dict) -> Dict:
        """
        Detect direction reversals where prediction goes opposite direction to actual trend
        
        Returns:
            Detection result with details
        """
        if not actual_trend['valid'] or not predicted_trend['valid']:
            return {
                'detected': False,
                'type': 'direction_reversal',
                'reason': 'invalid_trends'
            }
        
        # Check for significant direction mismatch
        actual_dir = actual_trend['direction']
        predicted_dir = predicted_trend['direction']
        
        # Define opposing directions
        opposites = {
            'rising': 'falling',
            'falling': 'rising'
        }
        
        # Only flag if both trends are significant (not stable)
        if (actual_dir in opposites and 
            predicted_dir == opposites[actual_dir] and
            actual_trend['magnitude'] > self.direction_reversal_min_change and
            predicted_trend['magnitude'] > self.direction_reversal_min_change):
            
            return {
                'detected': True,
                'type': 'direction_reversal',
                'actual_direction': actual_dir,
                'predicted_direction': predicted_dir,
                'actual_magnitude': actual_trend['magnitude'],
                'predicted_magnitude': predicted_trend['magnitude'],
                'actual_slope': actual_trend['slope'],
                'predicted_slope': predicted_trend['slope'],
                'severity': 'high'
            }
        
        return {
            'detected': False,
            'type': 'direction_reversal',
            'actual_direction': actual_dir,
            'predicted_direction': predicted_dir
        }
    
    def detect_cusp(self, actual_values: List[float], predicted_values: List[float]) -> Dict:
        """
        Detect cusps: correct starting level but wrong tidal phase/direction
        
        A cusp occurs when:
        1. First predicted value is close to last actual (good continuity)
        2. But predicted trend diverges significantly from actual trend
        
        Returns:
            Detection result with details
        """
        if not actual_values or not predicted_values:
            return {
                'detected': False,
                'type': 'cusp',
                'reason': 'insufficient_data'
            }
        
        # Get last actual and first predicted for continuity check
        last_actual = actual_values[-1] if actual_values else -999
        first_predicted = predicted_values[0] if predicted_values else -999
        
        if last_actual == -999 or first_predicted == -999:
            return {
                'detected': False,
                'type': 'cusp',
                'reason': 'missing_values'
            }
        
        # Check continuity (small jump at start)
        level_continuity = abs(first_predicted - last_actual)
        good_continuity = level_continuity < self.level_jump_threshold * 0.5  # Half the jump threshold
        
        if not good_continuity:
            return {
                'detected': False,
                'type': 'cusp',
                'reason': 'poor_continuity',
                'level_continuity': level_continuity
            }
        
        # Analyze trends in recent actual vs predicted
        window_size = min(self.cusp_window, len(actual_values), len(predicted_values))
        
        actual_trend = self.calculate_trend(actual_values, window_size)
        predicted_trend = self.calculate_trend(predicted_values, window_size)
        
        # Check for significant trend divergence despite good continuity
        direction_reversal = self.detect_direction_reversal(actual_trend, predicted_trend)
        
        # Additional cusp-specific checks
        if (direction_reversal['detected'] and 
            actual_trend['valid'] and predicted_trend['valid']):
            
            # Calculate trend divergence ratio
            if actual_trend['magnitude'] > 0:
                trend_ratio = predicted_trend['magnitude'] / actual_trend['magnitude']
            else:
                trend_ratio = float('inf') if predicted_trend['magnitude'] > 0 else 1.0
            
            # Check if prediction trend is unreasonably different
            significant_divergence = (trend_ratio > self.phase_error_ratio or 
                                    trend_ratio < 1.0 / self.phase_error_ratio)
            
            if significant_divergence:
                return {
                    'detected': True,
                    'type': 'cusp',
                    'level_continuity': level_continuity,
                    'actual_trend': actual_trend,
                    'predicted_trend': predicted_trend,
                    'trend_ratio': trend_ratio,
                    'direction_reversal': direction_reversal,
                    'severity': 'high' if trend_ratio > self.phase_error_ratio * 2 else 'medium'
                }
        
        return {
            'detected': False,
            'type': 'cusp',
            'level_continuity': level_continuity,
            'actual_trend': actual_trend,
            'predicted_trend': predicted_trend
        }
    
    def detect_phase_error(self, actual_values: List[float], predicted_values: List[float], 
                          timestamps: List[datetime] = None) -> Dict:
        """
        Detect phase errors: predictions that are temporally shifted from reality
        
        Returns:
            Detection result with details
        """
        if len(actual_values) < 4 or len(predicted_values) < 4:
            return {
                'detected': False,
                'type': 'phase_error',
                'reason': 'insufficient_data'
            }
        
        # Remove missing values for analysis
        actual_clean = [v for v in actual_values if v != -999]
        predicted_clean = [v for v in predicted_values if v != -999]
        
        if len(actual_clean) < 4 or len(predicted_clean) < 4:
            return {
                'detected': False,
                'type': 'phase_error',
                'reason': 'insufficient_valid_data'
            }
        
        # Calculate cross-correlation to detect phase shifts
        # Normalize sequences for comparison
        actual_norm = np.array(actual_clean) - np.mean(actual_clean)
        predicted_norm = np.array(predicted_clean) - np.mean(predicted_clean)
        
        # Calculate correlation at different lags
        max_lag = min(len(actual_norm), len(predicted_norm)) // 2
        correlations = []
        
        for lag in range(-max_lag, max_lag + 1):
            try:
                if lag == 0:
                    # Ensure arrays are same length
                    min_len = min(len(actual_norm), len(predicted_norm))
                    corr = np.corrcoef(actual_norm[:min_len], predicted_norm[:min_len])[0, 1]
                elif lag > 0:
                    # Predicted is shifted forward
                    if len(predicted_norm) > lag and len(actual_norm) > lag:
                        end_actual = min(len(actual_norm), len(predicted_norm) - lag)
                        corr = np.corrcoef(actual_norm[:end_actual], predicted_norm[lag:lag+end_actual])[0, 1]
                    else:
                        corr = 0
                else:  # lag < 0
                    # Actual is shifted forward  
                    abs_lag = abs(lag)
                    if len(actual_norm) > abs_lag and len(predicted_norm) > abs_lag:
                        end_predicted = min(len(predicted_norm), len(actual_norm) - abs_lag)
                        corr = np.corrcoef(actual_norm[abs_lag:abs_lag+end_predicted], predicted_norm[:end_predicted])[0, 1]
                    else:
                        corr = 0
                
                correlations.append(corr if not np.isnan(corr) else 0)
            except (ValueError, IndexError):
                correlations.append(0)
        
        # Find best correlation and its lag
        best_corr_idx = np.argmax(np.abs(correlations))
        best_lag = best_corr_idx - max_lag
        best_correlation = correlations[best_corr_idx]
        zero_lag_correlation = correlations[max_lag]  # lag = 0
        
        # Detect phase error if best correlation is significantly better at non-zero lag
        phase_error_detected = (abs(best_lag) > 1 and 
                               abs(best_correlation) > abs(zero_lag_correlation) + 0.3)
        
        return {
            'detected': phase_error_detected,
            'type': 'phase_error',
            'best_lag': best_lag,
            'best_correlation': best_correlation,
            'zero_lag_correlation': zero_lag_correlation,
            'correlation_improvement': abs(best_correlation) - abs(zero_lag_correlation),
            'phase_shift_minutes': best_lag * 10 if timestamps else None,  # 10-minute intervals
            'severity': 'high' if abs(best_lag) > 3 else 'medium'
        }
    
    def analyze_discontinuity(self, 
                            input_sequence: List[float],
                            predicted_sequence: List[float],
                            actual_sequence: List[float],
                            timestamps: List[datetime] = None) -> Dict:
        """
        Comprehensive discontinuity analysis combining all detection methods
        
        Args:
            input_sequence: 432-point input sequence used for prediction
            predicted_sequence: 144-point predicted sequence
            actual_sequence: 144-point actual sequence for comparison
            timestamps: Optional timestamps for each point
            
        Returns:
            Dictionary with all discontinuity detection results
        """
        
        results = {
            'timestamp': timestamps[0] if timestamps else datetime.now(),
            'has_discontinuity': False,
            'discontinuity_types': [],
            'severity': 'none',
            'details': {}
        }
        
        # Ensure we have valid data
        if not input_sequence or not predicted_sequence or not actual_sequence:
            results['error'] = 'insufficient_data'
            return results
        
        # 1. Level jump detection
        last_actual = input_sequence[-1] if input_sequence else -999
        first_predicted = predicted_sequence[0] if predicted_sequence else -999
        level_jump = self.detect_level_jump(last_actual, first_predicted)
        results['details']['level_jump'] = level_jump
        
        if level_jump['detected']:
            results['has_discontinuity'] = True
            results['discontinuity_types'].append('level_jump')
        
        # 2. Direction reversal detection
        # Use last part of input + first part of prediction vs first part of actual
        recent_actual = input_sequence[-self.cusp_window:] if len(input_sequence) >= self.cusp_window else input_sequence
        early_predicted = predicted_sequence[:self.cusp_window]
        early_actual = actual_sequence[:self.cusp_window]
        
        actual_trend = self.calculate_trend(recent_actual)
        predicted_trend = self.calculate_trend(early_predicted)
        actual_future_trend = self.calculate_trend(early_actual)
        
        direction_reversal = self.detect_direction_reversal(actual_future_trend, predicted_trend)
        results['details']['direction_reversal'] = direction_reversal
        
        if direction_reversal['detected']:
            results['has_discontinuity'] = True
            results['discontinuity_types'].append('direction_reversal')
        
        # 3. Cusp detection
        cusp = self.detect_cusp(input_sequence, predicted_sequence)
        results['details']['cusp'] = cusp
        
        if cusp['detected']:
            results['has_discontinuity'] = True
            results['discontinuity_types'].append('cusp')
        
        # 4. Phase error detection
        phase_error = self.detect_phase_error(actual_sequence, predicted_sequence, timestamps)
        results['details']['phase_error'] = phase_error
        
        if phase_error['detected']:
            results['has_discontinuity'] = True
            results['discontinuity_types'].append('phase_error')
        
        # 5. Determine overall severity
        severities = []
        for detail in results['details'].values():
            if isinstance(detail, dict) and detail.get('detected') and 'severity' in detail:
                severities.append(detail['severity'])
        
        if 'high' in severities:
            results['severity'] = 'high'
        elif 'medium' in severities:
            results['severity'] = 'medium'
        elif results['has_discontinuity']:
            results['severity'] = 'low'
        
        # 6. Add summary metrics
        results['summary'] = {
            'level_jump_size': level_jump.get('jump_size', 0),
            'direction_mismatch': direction_reversal['detected'],
            'cusp_detected': cusp['detected'],
            'phase_shift_detected': phase_error['detected'],
            'phase_shift_minutes': phase_error.get('phase_shift_minutes', 0),
            'total_discontinuity_types': len(results['discontinuity_types'])
        }
        
        return results

if __name__ == "__main__":
    # Test the discontinuity detector
    detector = DiscontinuityDetector()
    
    # Create test cases
    print("Testing Discontinuity Detector...")
    
    # Test case 1: Normal continuation (should not detect discontinuity)
    input_seq = list(range(1000, 1432))  # Rising trend
    predicted_seq = list(range(1432, 1576))  # Continues rising
    actual_seq = list(range(1430, 1574))  # Similar to prediction
    
    result1 = detector.analyze_discontinuity(input_seq, predicted_seq, actual_seq)
    print(f"Test 1 (normal): Discontinuity detected: {result1['has_discontinuity']}")
    
    # Test case 2: Level jump (should detect level_jump)
    predicted_seq_jump = [1500] + list(range(1501, 1645))  # Big jump at start
    result2 = detector.analyze_discontinuity(input_seq, predicted_seq_jump, actual_seq)
    print(f"Test 2 (level jump): Discontinuity detected: {result2['has_discontinuity']}, Types: {result2['discontinuity_types']}")
    
    # Test case 3: Direction reversal (should detect direction_reversal)
    predicted_seq_reverse = list(range(1432, 1288, -1))  # Goes down instead of up
    result3 = detector.analyze_discontinuity(input_seq, predicted_seq_reverse, actual_seq)
    print(f"Test 3 (direction reversal): Discontinuity detected: {result3['has_discontinuity']}, Types: {result3['discontinuity_types']}")
    
    print("Discontinuity detector tests completed!")