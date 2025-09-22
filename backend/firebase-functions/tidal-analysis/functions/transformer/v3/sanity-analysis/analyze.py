#!/usr/bin/env python3
"""
Transformer v3 Sanity Analysis Tool
Single entry point for all sanity analysis tasks
"""

import argparse
import sys
import numpy as np
import torch
import json
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Tuple, Optional

# Add paths
sys.path.append(str(Path(__file__).parent / 'core'))
sys.path.append(str(Path(__file__).parent.parent / 'shared'))

from discontinuity_detector import DiscontinuityDetector
from inference import TransformerV2Inference
from data_packaging import DataPackager

class DiscontinuityAnalyzer:
    """Main analysis class with all functionality"""
    
    def __init__(self):
        self.detector = DiscontinuityDetector()
        self.inference_engine = None
        self.data_packager = None
        
    def analyze_training_data(self, analysis_types: List[str]):
        """Analyze training data quality"""
        print("Analyzing Training Data Quality")
        print("=" * 40)
        
        # Load training data
        data_dir = Path(__file__).parent.parent / 'data-preparation' / 'data'
        try:
            X_train = np.load(data_dir / 'X_train.npy')
            y_train = np.load(data_dir / 'y_train.npy')
            
            with open(data_dir / 'normalization_params.json', 'r') as f:
                norm_params = json.load(f)
                
            print(f"Loaded {len(X_train)} training sequences")
            
        except FileNotFoundError as e:
            print(f"Error: Could not load training data: {e}")
            return
            
        # Denormalization function
        def denormalize(data):
            return data * norm_params['std'] + norm_params['mean']
            
        X_train_denorm = denormalize(X_train)
        y_train_denorm = denormalize(y_train)
        
        if 'continuity' in analysis_types:
            self._analyze_continuity(X_train_denorm, y_train_denorm)
            
        if 'direction' in analysis_types:
            self._analyze_direction_consistency(X_train_denorm, y_train_denorm)
            
        if 'statistics' in analysis_types:
            self._analyze_statistics(X_train_denorm, y_train_denorm)
            
        if 'sequences' in analysis_types:
            self._find_suspect_sequences(X_train_denorm, y_train_denorm)
            
    def analyze_inference_data(self, analysis_types: List[str], num_tests: int = 50):
        """Analyze inference using pre-generated sanity test sequences"""
        print("Analyzing Inference Sanity (v3)")
        print("=" * 32)
        
        # Load model
        model_path = Path(__file__).parent.parent / 'shared' / 'model.pth'
        if not model_path.exists():
            print(f"Error: Model not found at {model_path}")
            return
            
        # Load normalization parameters
        norm_params_path = Path(__file__).parent.parent / 'data-preparation' / 'data' / 'normalization_params.json'
        if not norm_params_path.exists():
            print(f"Error: Normalization params not found at {norm_params_path}")
            return
            
        try:
            self.inference_engine = TransformerV2Inference(str(model_path), str(norm_params_path))
            print("Model loaded successfully")
        except Exception as e:
            print(f"Error loading model: {e}")
            return
            
        # Load pre-generated sanity test sequences
        data_dir = Path(__file__).parent.parent / 'data-preparation' / 'data'
        try:
            X_sanity = np.load(data_dir / 'X_sanity.npy')
            with open(data_dir / 'sequence_names_sanity.json', 'r') as f:
                sanity_names = json.load(f)
            print(f"Loaded {len(X_sanity)} sanity test sequences")
            print(f"Date range: {sanity_names[0]} to {sanity_names[-1]}")
        except FileNotFoundError as e:
            print(f"Error: Could not load sanity test data: {e}")
            print("Make sure you have run the v3 data preparation to generate X_sanity.npy")
            return
            
        # Limit test count to available sequences
        sequences_to_test = min(num_tests, len(X_sanity))
        print(f"Testing {sequences_to_test} sequences from sanity dataset")
        
        discontinuities_found = []
        
        for sequence_index in range(sequences_to_test):
            sequence_name = sanity_names[sequence_index]
            input_sequence = X_sanity[sequence_index]  # Already normalized
            
            try:
                # Run inference directly (input is already normalized)
                prediction = self.inference_engine.predict(input_sequence)
                
                # Detect discontinuities
                if 'level_jumps' in analysis_types:
                    last_input_value = input_sequence[-1]  # Last value of input sequence
                    first_predicted_value = prediction[0]  # First value of prediction
                    level_jump_result = self.detector.detect_level_jump(last_input_value, first_predicted_value)
                    if level_jump_result['detected']:
                        discontinuities_found.append({
                            'sequence_name': sequence_name,
                            'sequence_index': sequence_index,
                            'type': 'level_jump',
                            'details': level_jump_result
                        })
                        
                if 'direction_changes' in analysis_types:
                    # Get trend for last 12 points of input and first 12 points of prediction
                    input_trend = self.detector.calculate_trend(input_sequence[-12:].tolist(), 12)
                    prediction_trend = self.detector.calculate_trend(prediction[:12].tolist(), 12)
                    direction_reversal_result = self.detector.detect_direction_reversal(input_trend, prediction_trend)
                    if direction_reversal_result['detected']:
                        discontinuities_found.append({
                            'sequence_name': sequence_name,
                            'sequence_index': sequence_index,
                            'type': 'direction_reversal',
                            'details': direction_reversal_result
                        })
                        
                if 'cusps' in analysis_types:
                    # Use last 12 points of input and first 12 points of prediction
                    cusp_detection_result = self.detector.detect_cusp(input_sequence[-12:].tolist(), prediction[:12].tolist())
                    if cusp_detection_result['detected']:
                        discontinuities_found.append({
                            'sequence_name': sequence_name,
                            'sequence_index': sequence_index,
                            'type': 'cusp',
                            'details': cusp_detection_result
                        })
                        
            except Exception as e:
                print(f"Error processing sequence {sequence_name} (index {sequence_index}): {e}")
                continue
                
            if (sequence_index + 1) % 10 == 0:
                print(f"Processed {sequence_index + 1}/{sequences_to_test} sequences, found {len(discontinuities_found)} discontinuities")
                
        print(f"\nAnalysis complete:")
        print(f"Tests run: {sequences_to_test}")
        print(f"Discontinuities found: {len(discontinuities_found)}")
        print(f"Discontinuity rate: {len(discontinuities_found)/sequences_to_test*100:.1f}%")
        
        if discontinuities_found:
            try:
                self._save_sanity_results(discontinuities_found)
            except Exception as e:
                print(f"Warning: Could not save results to JSON: {e}")
                print("Discontinuities found but results not saved to file")
            
    def _analyze_continuity(self, X_train, y_train):
        """Analyze continuity gaps in training data"""
        print("\n1. Continuity Analysis")
        print("-" * 20)
        
        gaps = []
        for i in range(len(X_train)):
            last_input = X_train[i, -1]
            first_output = y_train[i, 0]
            gap = abs(first_output - last_input)
            gaps.append(gap)
            
        gaps = np.array(gaps)
        
        print(f"Mean gap: {np.mean(gaps):.1f}mm")
        print(f"Median gap: {np.median(gaps):.1f}mm")
        print(f"95th percentile: {np.percentile(gaps, 95):.1f}mm")
        print(f"99th percentile: {np.percentile(gaps, 99):.1f}mm")
        print(f"Max gap: {np.max(gaps):.1f}mm")
        print(f"Gaps > 10mm: {np.sum(gaps > 10)} ({np.sum(gaps > 10)/len(gaps)*100:.2f}%)")
        print(f"Gaps > 20mm: {np.sum(gaps > 20)} ({np.sum(gaps > 20)/len(gaps)*100:.2f}%)")
        print(f"Gaps > 50mm: {np.sum(gaps > 50)} ({np.sum(gaps > 50)/len(gaps)*100:.3f}%)")
        
        # Show worst sequences
        large_gap_indices = np.where(gaps > 20)[0]
        if len(large_gap_indices) > 0:
            print(f"\nSequences with gaps > 20mm:")
            gap_pairs = [(idx, gaps[idx]) for idx in large_gap_indices]
            gap_pairs.sort(key=lambda x: x[1], reverse=True)
            for idx, gap in gap_pairs[:5]:
                print(f"  Sequence {idx}: {gap:.1f}mm gap")
                
    def _analyze_direction_consistency(self, X_train, y_train):
        """Analyze direction consistency"""
        print("\n2. Direction Analysis")
        print("-" * 20)
        
        direction_mismatches = 0
        for i in range(len(X_train)):
            input_trend = X_train[i, -6:]
            output_trend = y_train[i, :6]
            
            input_slope = np.polyfit(range(6), input_trend, 1)[0]
            output_slope = np.polyfit(range(6), output_trend, 1)[0]
            
            if abs(input_slope) > 5 and abs(output_slope) > 5:
                if np.sign(input_slope) != np.sign(output_slope):
                    direction_mismatches += 1
                    
        print(f"Direction mismatches: {direction_mismatches} ({direction_mismatches/len(X_train)*100:.2f}%)")
        print("Note: Some direction changes are normal at tidal turning points")
        
    def _analyze_statistics(self, X_train, y_train):
        """Analyze basic statistics"""
        print("\n3. Statistics Analysis")
        print("-" * 20)
        
        print(f"Training sequences: {len(X_train)}")
        print(f"Input length: {X_train.shape[1]} points (72 hours)")
        print(f"Output length: {y_train.shape[1]} points (24 hours)")
        print(f"Data range: {np.min(X_train):.1f}mm to {np.max(X_train):.1f}mm")
        
    def _find_suspect_sequences(self, X_train, y_train):
        """Find sequences that need manual inspection"""
        print("\n4. Suspect Sequences")
        print("-" * 20)
        
        gaps = []
        for i in range(len(X_train)):
            last_input = X_train[i, -1]
            first_output = y_train[i, 0]
            gap = abs(first_output - last_input)
            gaps.append(gap)
            
        gaps = np.array(gaps)
        
        # Find top problematic sequences
        worst_indices = np.argsort(gaps)[-5:]
        
        print("Top 5 sequences to inspect:")
        for idx in reversed(worst_indices):
            print(f"  Sequence {idx}: {gaps[idx]:.1f}mm gap (type '{idx+1}' in testing interface)")
            
    def _find_validation_end_time(self):
        """Find when validation data ends"""
        # Training data ends on 9/12/2025, so validation likely ends around then
        from datetime import timezone
        validation_end = datetime(2025, 9, 12, tzinfo=timezone.utc)
        return validation_end
        
    def _save_sanity_results(self, results):
        """Save sanity analysis results"""
        output_file = Path(__file__).parent / 'sanity_results.json'
        
        # Convert numpy types to native Python types for JSON serialization
        def convert_for_json(obj):
            if isinstance(obj, np.bool_):
                return bool(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_for_json(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_for_json(item) for item in obj]
            else:
                return obj
        
        # Convert results to JSON-serializable format
        json_safe_results = convert_for_json(results)
        
        with open(output_file, 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'total_tests': len(results),
                'total_discontinuities': len(results),
                'discontinuity_rate': f"{len(results)/100*100:.1f}%" if len(results) > 0 else "0%",
                'discontinuities': json_safe_results
            }, f, indent=2)
        print(f"Results saved to {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Transformer v3 Sanity Analysis')
    
    # Data source
    parser.add_argument('--data', choices=['training', 'inference'], required=True,
                       help='Analyze training data quality or inference sanity')
    
    # Analysis types for training data
    parser.add_argument('--training-analysis', nargs='+', 
                       choices=['continuity', 'direction', 'statistics', 'sequences'],
                       default=['continuity', 'direction', 'statistics', 'sequences'],
                       help='Types of training data analysis to perform')
    
    # Analysis types for inference data  
    parser.add_argument('--inference-analysis', nargs='+',
                       choices=['level_jumps', 'direction_changes', 'cusps', 'phase_errors'],
                       default=['level_jumps', 'direction_changes', 'cusps'],
                       help='Types of discontinuity detection to perform')
    
    # Number of inference tests
    parser.add_argument('--num-tests', type=int, default=50,
                       help='Number of inference tests to run (default: 50)')
    
    args = parser.parse_args()
    
    analyzer = DiscontinuityAnalyzer()
    
    if args.data == 'training':
        analyzer.analyze_training_data(args.training_analysis)
    elif args.data == 'inference':
        analyzer.analyze_inference_data(args.inference_analysis, args.num_tests)

if __name__ == "__main__":
    main()