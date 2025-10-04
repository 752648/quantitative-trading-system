#!/usr/bin/env python3
"""
Feature Lifecycle Management System for Quantitative Trading

This module manages the lifecycle of trading features by categorizing them into:
- Never Working: Features that have never shown predictive power
- Past Working: Features that worked before but have decayed
- Currently Working: Features that are currently predictive
- Emerging: Features that are starting to show promise

This dramatically reduces computational overhead by avoiding regressions on 
non-performing features while identifying nascent opportunities.

Author: Manus AI
"""

import pandas as pd
import numpy as np
import sqlite3
import os
import json
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

@dataclass
class FeaturePerformance:
    """Data class to track feature performance over time"""
    feature_name: str
    first_tested: str
    last_tested: str
    total_tests: int
    significant_tests: int  # R² > threshold
    best_r2: float
    recent_r2: float  # Last 5 tests average
    decay_rate: float  # Performance decline rate
    stability_score: float
    current_status: str  # 'never_working', 'past_working', 'currently_working', 'emerging'
    last_significant: Optional[str] = None
    consecutive_failures: int = 0
    emergence_score: float = 0.0

class FeatureLifecycleManager:
    """
    Manages feature lifecycle and performance tracking
    """
    
    def __init__(self, data_path="./quant_data", significance_threshold=0.01):
        self.data_path = data_path
        self.significance_threshold = significance_threshold
        self.db_path = os.path.join(data_path, "feature_lifecycle.db")
        self.performance_cache = {}
        
        # Lifecycle thresholds
        self.thresholds = {
            'min_tests_for_classification': 5,
            'never_working_threshold': 0.0,  # Never exceeded significance
            'decay_threshold': 0.5,  # 50% performance decline
            'emergence_threshold': 0.8,  # 80% recent improvement
            'consecutive_failure_limit': 10,
            'stability_requirement': 0.3,
            'recent_window': 5  # Number of recent tests to consider
        }
        
        self._init_database()
        print(f"FeatureLifecycleManager initialized:")
        print(f"  Database: {self.db_path}")
        print(f"  Significance threshold: {self.significance_threshold}")
        print(f"  Classification thresholds: {self.thresholds}")
    
    def _init_database(self):
        """Initialize SQLite database for feature tracking"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Feature performance history table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS feature_performance_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                feature_name TEXT,
                ticker TEXT,
                timeframe TEXT,
                return_horizon INTEGER,
                method TEXT,
                test_date TEXT,
                r2_score REAL,
                p_value REAL,
                coefficient REAL,
                feature_importance REAL,
                n_observations INTEGER,
                is_significant BOOLEAN,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Feature lifecycle status table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS feature_lifecycle_status (
                feature_name TEXT PRIMARY KEY,
                first_tested TEXT,
                last_tested TEXT,
                total_tests INTEGER DEFAULT 0,
                significant_tests INTEGER DEFAULT 0,
                best_r2 REAL DEFAULT 0.0,
                recent_r2 REAL DEFAULT 0.0,
                decay_rate REAL DEFAULT 0.0,
                stability_score REAL DEFAULT 0.0,
                current_status TEXT DEFAULT 'untested',
                last_significant TEXT,
                consecutive_failures INTEGER DEFAULT 0,
                emergence_score REAL DEFAULT 0.0,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Feature exclusion rules table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS feature_exclusion_rules (
                rule_id INTEGER PRIMARY KEY AUTOINCREMENT,
                feature_pattern TEXT,
                exclusion_reason TEXT,
                created_date TEXT,
                is_active BOOLEAN DEFAULT 1
            )
        ''')
        
        # Create indexes for performance
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_feature_name ON feature_performance_history(feature_name)')\n        cursor.execute('CREATE INDEX IF NOT EXISTS idx_test_date ON feature_performance_history(test_date)')\n        cursor.execute('CREATE INDEX IF NOT EXISTS idx_significance ON feature_performance_history(is_significant)')\n        \n        conn.commit()\n        conn.close()\n    \n    def record_feature_performance(self, regression_results_df):\n        \"\"\"Record feature performance from regression results\"\"\"\n        if regression_results_df.empty:\n            return\n        \n        conn = sqlite3.connect(self.db_path)\n        cursor = conn.cursor()\n        \n        records_added = 0\n        \n        for _, result in regression_results_df.iterrows():\n            if 'feature_importance' not in result or not isinstance(result['feature_importance'], dict):\n                continue\n            \n            test_date = result.get('analysis_date', datetime.now().isoformat())\n            ticker = result.get('ticker', 'unknown')\n            timeframe = result.get('timeframe', '1d')\n            return_horizon = result.get('return_horizon', 1)\n            method = result.get('method', 'unknown')\n            \n            # Record each feature's performance\n            for feature_name, importance in result['feature_importance'].items():\n                r2_score = result.get('test_r2', 0.0)\n                coefficient = result.get('coefficients', {}).get(feature_name, 0.0)\n                p_value = result.get('p_values', {}).get(feature_name, 1.0)\n                n_observations = result.get('n_observations', 0)\n                is_significant = r2_score > self.significance_threshold\n                \n                cursor.execute('''\n                    INSERT INTO feature_performance_history \n                    (feature_name, ticker, timeframe, return_horizon, method, test_date, \n                     r2_score, p_value, coefficient, feature_importance, n_observations, is_significant)\n                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)\n                ''', (\n                    feature_name, ticker, timeframe, return_horizon, method, test_date,\n                    r2_score, p_value, coefficient, importance, n_observations, is_significant\n                ))\n                \n                records_added += 1\n        \n        conn.commit()\n        conn.close()\n        \n        print(f\"✓ Recorded {records_added} feature performance records\")\n        \n        # Update lifecycle status\n        self._update_feature_lifecycle_status()\n    \n    def _update_feature_lifecycle_status(self):\n        \"\"\"Update feature lifecycle status based on performance history\"\"\"\n        conn = sqlite3.connect(self.db_path)\n        \n        # Get all features and their performance statistics\n        query = '''\n            SELECT \n                feature_name,\n                MIN(test_date) as first_tested,\n                MAX(test_date) as last_tested,\n                COUNT(*) as total_tests,\n                SUM(CASE WHEN is_significant THEN 1 ELSE 0 END) as significant_tests,\n                MAX(r2_score) as best_r2,\n                AVG(CASE WHEN test_date >= date('now', '-30 days') THEN r2_score END) as recent_r2\n            FROM feature_performance_history\n            GROUP BY feature_name\n        '''\n        \n        feature_stats = pd.read_sql_query(query, conn)\n        \n        for _, stats in feature_stats.iterrows():\n            feature_name = stats['feature_name']\n            \n            # Calculate additional metrics\n            performance_metrics = self._calculate_feature_metrics(feature_name, conn)\n            \n            # Determine lifecycle status\n            status = self._classify_feature_status(stats, performance_metrics)\n            \n            # Update or insert lifecycle status\n            cursor = conn.cursor()\n            cursor.execute('''\n                INSERT OR REPLACE INTO feature_lifecycle_status\n                (feature_name, first_tested, last_tested, total_tests, significant_tests,\n                 best_r2, recent_r2, decay_rate, stability_score, current_status,\n                 last_significant, consecutive_failures, emergence_score, updated_at)\n                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)\n            ''', (\n                feature_name,\n                stats['first_tested'],\n                stats['last_tested'],\n                stats['total_tests'],\n                stats['significant_tests'],\n                stats['best_r2'],\n                stats['recent_r2'] or 0.0,\n                performance_metrics['decay_rate'],\n                performance_metrics['stability_score'],\n                status,\n                performance_metrics['last_significant'],\n                performance_metrics['consecutive_failures'],\n                performance_metrics['emergence_score'],\n                datetime.now().isoformat()\n            ))\n        \n        conn.commit()\n        conn.close()\n        \n        print(f\"✓ Updated lifecycle status for {len(feature_stats)} features\")\n    \n    def _calculate_feature_metrics(self, feature_name, conn):\n        \"\"\"Calculate detailed metrics for a feature\"\"\"\n        # Get recent performance history\n        recent_query = '''\n            SELECT r2_score, test_date, is_significant\n            FROM feature_performance_history\n            WHERE feature_name = ?\n            ORDER BY test_date DESC\n            LIMIT 20\n        '''\n        \n        recent_data = pd.read_sql_query(recent_query, conn, params=(feature_name,))\n        \n        if recent_data.empty:\n            return {\n                'decay_rate': 0.0,\n                'stability_score': 0.0,\n                'last_significant': None,\n                'consecutive_failures': 0,\n                'emergence_score': 0.0\n            }\n        \n        # Calculate decay rate (performance decline over time)\n        if len(recent_data) >= 10:\n            early_performance = recent_data.tail(10)['r2_score'].mean()\n            recent_performance = recent_data.head(5)['r2_score'].mean()\n            \n            if early_performance > 0:\n                decay_rate = (early_performance - recent_performance) / early_performance\n            else:\n                decay_rate = 0.0\n        else:\n            decay_rate = 0.0\n        \n        # Calculate stability score (consistency of performance)\n        if len(recent_data) > 1:\n            stability_score = 1.0 / (1.0 + recent_data['r2_score'].std())\n        else:\n            stability_score = 0.0\n        \n        # Find last significant performance\n        significant_tests = recent_data[recent_data['is_significant']]\n        last_significant = significant_tests['test_date'].iloc[0] if not significant_tests.empty else None\n        \n        # Count consecutive failures\n        consecutive_failures = 0\n        for _, row in recent_data.iterrows():\n            if not row['is_significant']:\n                consecutive_failures += 1\n            else:\n                break\n        \n        # Calculate emergence score (recent improvement)\n        if len(recent_data) >= 5:\n            old_avg = recent_data.tail(5)['r2_score'].mean()\n            new_avg = recent_data.head(5)['r2_score'].mean()\n            \n            if old_avg > 0:\n                emergence_score = (new_avg - old_avg) / old_avg\n            else:\n                emergence_score = new_avg\n        else:\n            emergence_score = 0.0\n        \n        return {\n            'decay_rate': decay_rate,\n            'stability_score': stability_score,\n            'last_significant': last_significant,\n            'consecutive_failures': consecutive_failures,\n            'emergence_score': emergence_score\n        }\n    \n    def _classify_feature_status(self, stats, metrics):\n        \"\"\"Classify feature into lifecycle category\"\"\"\n        total_tests = stats['total_tests']\n        significant_tests = stats['significant_tests']\n        best_r2 = stats['best_r2']\n        recent_r2 = stats['recent_r2'] or 0.0\n        \n        # Need minimum tests for classification\n        if total_tests < self.thresholds['min_tests_for_classification']:\n            return 'untested'\n        \n        significance_rate = significant_tests / total_tests\n        \n        # Never working: Never exceeded significance threshold\n        if best_r2 <= self.significance_threshold:\n            return 'never_working'\n        \n        # Currently working: Recent performance is good\n        if (recent_r2 > self.significance_threshold and \n            metrics['consecutive_failures'] < 3 and\n            significance_rate > 0.2):\n            return 'currently_working'\n        \n        # Emerging: Showing recent improvement\n        if (metrics['emergence_score'] > self.thresholds['emergence_threshold'] and\n            recent_r2 > best_r2 * 0.5):\n            return 'emerging'\n        \n        # Past working: Was good before but declined\n        if (best_r2 > self.significance_threshold * 2 and\n            (recent_r2 < best_r2 * 0.5 or \n             metrics['consecutive_failures'] > 5)):\n            return 'past_working'\n        \n        # Default to untested if unclear\n        return 'untested'\n    \n    def get_features_by_status(self, status: str) -> List[str]:\n        \"\"\"Get list of features by their lifecycle status\"\"\"\n        conn = sqlite3.connect(self.db_path)\n        \n        query = '''\n            SELECT feature_name \n            FROM feature_lifecycle_status \n            WHERE current_status = ?\n            ORDER BY best_r2 DESC\n        '''\n        \n        result = pd.read_sql_query(query, conn, params=(status,))\n        conn.close()\n        \n        return result['feature_name'].tolist()\n    \n    def get_features_to_test(self, max_features: int = 50) -> Dict[str, List[str]]:\n        \"\"\"Get prioritized list of features to test\"\"\"\n        conn = sqlite3.connect(self.db_path)\n        \n        # Priority order: currently_working > emerging > untested > past_working\n        # Never exclude never_working features entirely\n        \n        priority_query = '''\n            SELECT feature_name, current_status, best_r2, recent_r2, emergence_score\n            FROM feature_lifecycle_status\n            WHERE current_status IN ('currently_working', 'emerging', 'untested', 'past_working')\n            ORDER BY \n                CASE current_status\n                    WHEN 'currently_working' THEN 1\n                    WHEN 'emerging' THEN 2\n                    WHEN 'untested' THEN 3\n                    WHEN 'past_working' THEN 4\n                    ELSE 5\n                END,\n                best_r2 DESC,\n                emergence_score DESC\n        '''\n        \n        priority_features = pd.read_sql_query(priority_query, conn)\n        conn.close()\n        \n        # Categorize features for testing\n        test_allocation = {\n            'high_priority': [],  # Currently working + emerging\n            'medium_priority': [],  # Untested with potential\n            'low_priority': [],  # Past working + never working sample\n            'skip': []  # Never working features to skip\n        }\n        \n        # Allocate features based on status and performance\n        for _, feature in priority_features.iterrows():\n            name = feature['feature_name']\n            status = feature['current_status']\n            \n            if status in ['currently_working', 'emerging']:\n                test_allocation['high_priority'].append(name)\n            elif status == 'untested':\n                test_allocation['medium_priority'].append(name)\n            elif status == 'past_working':\n                test_allocation['low_priority'].append(name)\n        \n        # Get never working features (sample a few for monitoring)\n        never_working = self.get_features_by_status('never_working')\n        \n        # Sample 10% of never working features for periodic retesting\n        sample_size = max(1, len(never_working) // 10)\n        sampled_never_working = np.random.choice(\n            never_working, \n            size=min(sample_size, len(never_working)), \n            replace=False\n        ).tolist() if never_working else []\n        \n        test_allocation['low_priority'].extend(sampled_never_working)\n        test_allocation['skip'] = [f for f in never_working if f not in sampled_never_working]\n        \n        # Limit total features if requested\n        if max_features > 0:\n            total_selected = 0\n            for priority in ['high_priority', 'medium_priority', 'low_priority']:\n                remaining = max_features - total_selected\n                if remaining <= 0:\n                    test_allocation[priority] = []\n                else:\n                    test_allocation[priority] = test_allocation[priority][:remaining]\n                    total_selected += len(test_allocation[priority])\n        \n        return test_allocation\n    \n    def get_lifecycle_summary(self) -> Dict:\n        \"\"\"Get summary of feature lifecycle status\"\"\"\n        conn = sqlite3.connect(self.db_path)\n        \n        summary_query = '''\n            SELECT \n                current_status,\n                COUNT(*) as count,\n                AVG(best_r2) as avg_best_r2,\n                AVG(recent_r2) as avg_recent_r2,\n                AVG(total_tests) as avg_tests\n            FROM feature_lifecycle_status\n            GROUP BY current_status\n        '''\n        \n        summary_df = pd.read_sql_query(summary_query, conn)\n        \n        # Get total feature count\n        total_query = 'SELECT COUNT(*) as total FROM feature_lifecycle_status'\n        total_count = pd.read_sql_query(total_query, conn)['total'].iloc[0]\n        \n        conn.close()\n        \n        summary = {\n            'total_features': total_count,\n            'by_status': summary_df.to_dict('records'),\n            'last_updated': datetime.now().isoformat()\n        }\n        \n        return summary\n    \n    def add_exclusion_rule(self, feature_pattern: str, reason: str):\n        \"\"\"Add rule to exclude certain features from testing\"\"\"\n        conn = sqlite3.connect(self.db_path)\n        cursor = conn.cursor()\n        \n        cursor.execute('''\n            INSERT INTO feature_exclusion_rules (feature_pattern, exclusion_reason, created_date)\n            VALUES (?, ?, ?)\n        ''', (feature_pattern, reason, datetime.now().isoformat()))\n        \n        conn.commit()\n        conn.close()\n        \n        print(f\"✓ Added exclusion rule: {feature_pattern} - {reason}\")\n    \n    def get_excluded_features(self, all_features: List[str]) -> List[str]:\n        \"\"\"Get list of features that should be excluded based on rules\"\"\"\n        conn = sqlite3.connect(self.db_path)\n        \n        rules_query = '''\n            SELECT feature_pattern \n            FROM feature_exclusion_rules \n            WHERE is_active = 1\n        '''\n        \n        rules_df = pd.read_sql_query(rules_query, conn)\n        conn.close()\n        \n        excluded = []\n        \n        for feature in all_features:\n            for _, rule in rules_df.iterrows():\n                pattern = rule['feature_pattern']\n                if pattern in feature or feature.startswith(pattern):\n                    excluded.append(feature)\n                    break\n        \n        return excluded\n    \n    def optimize_feature_selection(self, all_features: List[str], \n                                 max_features: int = 50) -> Dict[str, List[str]]:\n        \"\"\"Optimize feature selection for regression testing\"\"\"\n        print(f\"Optimizing feature selection from {len(all_features)} total features...\")\n        \n        # Get excluded features\n        excluded = self.get_excluded_features(all_features)\n        \n        # Filter out excluded features\n        available_features = [f for f in all_features if f not in excluded]\n        \n        # Get features by lifecycle status\n        test_allocation = self.get_features_to_test(max_features)\n        \n        # Filter to only include available features\n        for priority in test_allocation:\n            test_allocation[priority] = [\n                f for f in test_allocation[priority] \n                if f in available_features\n            ]\n        \n        # Add any new features not in lifecycle database\n        tracked_features = set()\n        for priority_list in test_allocation.values():\n            tracked_features.update(priority_list)\n        \n        new_features = [f for f in available_features if f not in tracked_features]\n        test_allocation['new_features'] = new_features[:max(0, max_features - sum(len(v) for v in test_allocation.values() if isinstance(v, list)))]\n        \n        # Summary\n        total_selected = sum(len(v) for k, v in test_allocation.items() if k != 'skip' and isinstance(v, list))\n        \n        print(f\"Feature selection optimization:\")\n        print(f\"  Total available: {len(available_features)}\")\n        print(f\"  Excluded by rules: {len(excluded)}\")\n        print(f\"  Selected for testing: {total_selected}\")\n        print(f\"  Skipped (never working): {len(test_allocation.get('skip', []))}\")\n        \n        for priority, features in test_allocation.items():\n            if isinstance(features, list) and features:\n                print(f\"  {priority}: {len(features)} features\")\n        \n        return test_allocation\n    \n    def save_optimization_report(self, optimization_results: Dict, filename: str = None):\n        \"\"\"Save feature optimization report\"\"\"\n        if filename is None:\n            filename = os.path.join(self.data_path, f\"feature_optimization_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json\")\n        \n        report = {\n            'timestamp': datetime.now().isoformat(),\n            'optimization_results': optimization_results,\n            'lifecycle_summary': self.get_lifecycle_summary(),\n            'thresholds': self.thresholds\n        }\n        \n        with open(filename, 'w') as f:\n            json.dump(report, f, indent=2)\n        \n        print(f\"✓ Optimization report saved to {filename}\")\n        \n        return filename\n    \n    def reset_feature_status(self, feature_names: List[str] = None):\n        \"\"\"Reset lifecycle status for specific features or all features\"\"\"\n        conn = sqlite3.connect(self.db_path)\n        cursor = conn.cursor()\n        \n        if feature_names:\n            placeholders = ','.join(['?' for _ in feature_names])\n            cursor.execute(f'''\n                DELETE FROM feature_lifecycle_status \n                WHERE feature_name IN ({placeholders})\n            ''', feature_names)\n            print(f\"✓ Reset status for {len(feature_names)} features\")\n        else:\n            cursor.execute('DELETE FROM feature_lifecycle_status')\n            cursor.execute('DELETE FROM feature_performance_history')\n            print(\"✓ Reset all feature lifecycle data\")\n        \n        conn.commit()\n        conn.close()\n    \n    def export_feature_performance(self, filename: str = None) -> str:\n        \"\"\"Export feature performance data to CSV\"\"\"\n        if filename is None:\n            filename = os.path.join(self.data_path, f\"feature_performance_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv\")\n        \n        conn = sqlite3.connect(self.db_path)\n        \n        query = '''\n            SELECT \n                fls.*,\n                COUNT(fph.id) as total_records,\n                AVG(fph.r2_score) as avg_r2,\n                MAX(fph.r2_score) as max_r2,\n                MIN(fph.r2_score) as min_r2\n            FROM feature_lifecycle_status fls\n            LEFT JOIN feature_performance_history fph ON fls.feature_name = fph.feature_name\n            GROUP BY fls.feature_name\n            ORDER BY fls.current_status, fls.best_r2 DESC\n        '''\n        \n        df = pd.read_sql_query(query, conn)\n        conn.close()\n        \n        df.to_csv(filename, index=False)\n        print(f\"✓ Feature performance data exported to {filename}\")\n        \n        return filename
