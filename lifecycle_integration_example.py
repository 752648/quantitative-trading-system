#!/usr/bin/env python3
"""
Feature Lifecycle Integration Example

This example shows how to integrate the FeatureLifecycleManager with the 
existing quantitative trading system to dramatically reduce computational overhead.

Author: Manus AI
"""

import pandas as pd
import numpy as np
from feature_lifecycle_manager import FeatureLifecycleManager
from regression_testing import RegressionTester
import time

def demonstrate_lifecycle_optimization():
    """
    Demonstrate how feature lifecycle management reduces computational overhead
    """
    
    print("=" * 80)
    print("FEATURE LIFECYCLE MANAGEMENT DEMONSTRATION")
    print("=" * 80)
    
    # Initialize managers
    lifecycle_manager = FeatureLifecycleManager()
    regression_tester = RegressionTester()
    
    # Simulate having 200 features (typical for our system)
    all_features = [
        f"sma_{period}" for period in [5, 10, 20, 50, 100, 200]
    ] + [
        f"rsi_{period}" for period in [14, 20, 30, 50]
    ] + [
        f"bb_position_{period}" for period in [20, 50]
    ] + [
        f"volume_ratio_{period}" for period in [10, 20, 50]
    ] + [
        f"volatility_{period}" for period in [10, 20, 50, 100]
    ] + [
        f"momentum_{period}" for period in [5, 10, 20, 50]
    ] + [
        f"mean_reversion_{period}" for period in [10, 20, 50]
    ] + [
        # Add many more features to simulate real scenario
        f"feature_{i}" for i in range(150)  # Simulate 150 additional features
    ]
    
    print(f"Total features available: {len(all_features)}")
    
    # Simulate some regression results to populate the lifecycle database
    print("\\nSimulating historical regression results...")
    
    # Create mock regression results
    mock_results = []
    
    for i in range(100):  # 100 historical tests
        # Simulate different feature performance patterns
        feature_importance = {}
        
        # Some features are consistently good (currently working)
        for feature in ["sma_20", "rsi_14", "bb_position_20", "volume_ratio_20"]:
            if feature in all_features:
                feature_importance[feature] = np.random.normal(0.8, 0.2)
        
        # Some features were good but are decaying (past working)
        for feature in ["sma_200", "momentum_50", "volatility_100"]:
            if feature in all_features:
                # Simulate decay over time
                decay_factor = max(0.1, 1.0 - (i / 100) * 0.8)
                feature_importance[feature] = np.random.normal(0.6 * decay_factor, 0.3)
        
        # Some features are emerging (getting better)
        for feature in ["mean_reversion_10", "rsi_30"]:
            if feature in all_features:
                # Simulate emergence over time
                emergence_factor = min(1.0, 0.2 + (i / 100) * 0.6)
                feature_importance[feature] = np.random.normal(0.4 * emergence_factor, 0.2)
        
        # Most features never work (never working)
        for feature in np.random.choice(all_features, size=20, replace=False):
            if feature not in feature_importance:
                feature_importance[feature] = np.random.normal(0.05, 0.1)
        
        # Create mock result
        r2_score = max(0, np.random.normal(0.02, 0.01))
        
        mock_result = {
            'ticker': f'TICKER_{i % 10}',
            'method': np.random.choice(['lasso', 'ridge', 'random_forest']),
            'return_horizon': np.random.choice([1, 5, 10]),
            'test_r2': r2_score,
            'feature_importance': feature_importance,
            'coefficients': {k: v * 0.1 for k, v in feature_importance.items()},
            'p_values': {k: max(0.001, 1 - v) for k, v in feature_importance.items()},
            'n_observations': np.random.randint(100, 500),
            'analysis_date': f'2024-{(i % 12) + 1:02d}-{(i % 28) + 1:02d}'
        }
        
        mock_results.append(mock_result)
    
    # Convert to DataFrame and record performance
    mock_results_df = pd.DataFrame(mock_results)
    lifecycle_manager.record_feature_performance(mock_results_df)
    
    print(f"✓ Recorded {len(mock_results)} historical regression results")
    
    # Get lifecycle summary
    print("\\nFeature Lifecycle Summary:")
    summary = lifecycle_manager.get_lifecycle_summary()
    
    print(f"Total features tracked: {summary['total_features']}")
    for status_info in summary['by_status']:
        status = status_info['current_status']
        count = status_info['count']
        avg_r2 = status_info['avg_best_r2'] or 0
        print(f"  {status}: {count} features (avg R² = {avg_r2:.4f})")
    
    # Demonstrate optimization
    print("\\n" + "=" * 50)
    print("COMPUTATIONAL OPTIMIZATION DEMONSTRATION")
    print("=" * 50)
    
    # Without optimization (test all features)
    print("\\nWithout Lifecycle Management:")
    print(f"  Features to test: {len(all_features)}")
    print(f"  Estimated regression time: {len(all_features) * 0.5:.1f} seconds")
    print(f"  Expected significant results: ~{len(all_features) * 0.05:.0f} features")
    
    # With optimization
    print("\\nWith Lifecycle Management:")
    optimization_results = lifecycle_manager.optimize_feature_selection(
        all_features, 
        max_features=50  # Limit to 50 most promising features
    )
    
    total_selected = sum(
        len(features) for key, features in optimization_results.items() 
        if key != 'skip' and isinstance(features, list)
    )
    
    print(f"  Features to test: {total_selected}")
    print(f"  Features skipped: {len(optimization_results.get('skip', []))}")
    print(f"  Estimated regression time: {total_selected * 0.5:.1f} seconds")
    print(f"  Expected significant results: ~{total_selected * 0.15:.0f} features")
    
    # Calculate efficiency gains
    time_saved = (len(all_features) - total_selected) * 0.5
    efficiency_gain = (len(all_features) - total_selected) / len(all_features) * 100
    
    print(f"\\n📊 Efficiency Gains:")
    print(f"  Time saved: {time_saved:.1f} seconds ({efficiency_gain:.1f}% reduction)")
    print(f"  Compute resources saved: {efficiency_gain:.1f}%")
    print(f"  Focus on high-value features: {len(optimization_results.get('high_priority', []))} priority features")
    
    # Show feature categories
    print(f"\\n🎯 Feature Selection Breakdown:")
    for category, features in optimization_results.items():
        if isinstance(features, list) and features:
            print(f"  {category}: {len(features)} features")
            if category == 'high_priority':
                print(f"    Examples: {', '.join(features[:5])}")
    
    # Demonstrate exclusion rules
    print(f"\\n🚫 Adding Exclusion Rules:")
    lifecycle_manager.add_exclusion_rule("feature_", "Simulated features for testing")
    lifecycle_manager.add_exclusion_rule("sma_200", "Too slow for current market regime")
    
    # Re-optimize with exclusion rules
    optimized_with_rules = lifecycle_manager.optimize_feature_selection(
        all_features, 
        max_features=50
    )
    
    excluded_count = len(lifecycle_manager.get_excluded_features(all_features))
    print(f"  Features excluded by rules: {excluded_count}")
    
    # Save optimization report
    report_file = lifecycle_manager.save_optimization_report(optimized_with_rules)
    
    print(f"\\n💾 Optimization report saved: {report_file}")
    
    # Demonstrate integration with regression testing
    print(f"\\n" + "=" * 50)
    print("INTEGRATION WITH REGRESSION TESTING")
    print("=" * 50)
    
    # Show how to use optimized features in regression testing
    selected_features = []
    for category in ['high_priority', 'medium_priority', 'new_features']:
        selected_features.extend(optimized_with_rules.get(category, []))
    
    print(f"\\nSelected {len(selected_features)} features for regression testing:")
    print(f"  High priority (currently working): {len(optimized_with_rules.get('high_priority', []))}")
    print(f"  Medium priority (untested): {len(optimized_with_rules.get('medium_priority', []))}")
    print(f"  New features: {len(optimized_with_rules.get('new_features', []))}")
    
    # Example of how this integrates with the main system
    print(f"\\n🔄 Integration Example:")
    print(f"```python")
    print(f"# In your main analysis notebook:")
    print(f"lifecycle_manager = FeatureLifecycleManager()")
    print(f"")
    print(f"# Before running regressions, optimize feature selection")
    print(f"optimization = lifecycle_manager.optimize_feature_selection(")
    print(f"    all_computed_features, max_features=50")
    print(f")")
    print(f"")
    print(f"# Use only selected features for regression testing")
    print(f"selected_features = []")
    print(f"for category in ['high_priority', 'medium_priority', 'new_features']:")
    print(f"    selected_features.extend(optimization.get(category, []))")
    print(f"")
    print(f"# Run regressions only on selected features")
    print(f"filtered_features_dict = {{")
    print(f"    ticker: features[selected_features] ")
    print(f"    for ticker, features in features_dict.items()")
    print(f"}}")
    print(f"")
    print(f"# After regressions, record results for future optimization")
    print(f"lifecycle_manager.record_feature_performance(regression_results)")
    print(f"```")
    
    print(f"\\n✅ Feature Lifecycle Management System Ready!")
    print(f"\\nNext Steps:")
    print(f"1. Integrate with your main notebook")
    print(f"2. Run initial analysis to populate feature history")
    print(f"3. Use optimization before each regression run")
    print(f"4. Monitor feature performance over time")
    print(f"5. Adjust thresholds based on your specific needs")
    
    return optimization_results

if __name__ == "__main__":
    # Run the demonstration
    results = demonstrate_lifecycle_optimization()
    
    print(f"\\n" + "=" * 80)
    print(f"DEMONSTRATION COMPLETE")
    print(f"=" * 80)
