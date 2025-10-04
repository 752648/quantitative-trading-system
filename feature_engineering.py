#!/usr/bin/env python3
"""
Feature Engineering Module for Quantitative Trading

This module provides comprehensive feature engineering capabilities with:
- 100+ technical indicators
- Multi-core parallel processing
- GPU acceleration support
- Memory-efficient batch processing
- Caching for performance

Author: Manus AI
"""

import pandas as pd
import numpy as np
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from joblib import Parallel, delayed
import multiprocessing as mp
from functools import partial
import warnings
warnings.filterwarnings('ignore')

# GPU support
try:
    import cudf
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

# Technical analysis
try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False

class FeatureEngineer:
    """
    Comprehensive feature engineering for quantitative trading
    """
    
    def __init__(self, use_gpu=False, n_jobs=-1):
        self.use_gpu = use_gpu and GPU_AVAILABLE
        self.n_jobs = n_jobs if n_jobs > 0 else mp.cpu_count()
        self.feature_cache = {}
        
        print(f"FeatureEngineer initialized:")
        print(f"  GPU acceleration: {'✓' if self.use_gpu else '✗'}")
        print(f"  TA-Lib available: {'✓' if TALIB_AVAILABLE else '✗'}")
        print(f"  Parallel jobs: {self.n_jobs}")
    
    def compute_price_features(self, data, periods=[5, 10, 20, 50, 100, 200]):
        """
        Compute price-based technical features
        
        Args:
            data: DataFrame with OHLCV columns
            periods: List of periods for moving averages and indicators
            
        Returns:
            DataFrame with price features
        """
        if self.use_gpu:
            return self._compute_price_features_gpu(data, periods)
        else:
            return self._compute_price_features_cpu(data, periods)
    
    def _compute_price_features_cpu(self, data, periods):
        """CPU implementation of price features"""
        features = pd.DataFrame(index=data.index)
        
        # Basic price features
        features['price_change'] = data['Close'].diff()
        features['price_change_pct'] = data['Close'].pct_change()
        features['high_low_ratio'] = data['High'] / data['Low']
        features['close_open_ratio'] = data['Close'] / data['Open']
        
        # Moving averages and ratios
        for period in periods:
            if len(data) >= period:
                # Simple Moving Average
                sma = data['Close'].rolling(period).mean()
                features[f'sma_{period}'] = sma
                features[f'price_sma_ratio_{period}'] = data['Close'] / sma
                features[f'sma_slope_{period}'] = sma.diff(5) / sma.shift(5)
                
                # Exponential Moving Average
                ema = data['Close'].ewm(span=period).mean()
                features[f'ema_{period}'] = ema
                features[f'price_ema_ratio_{period}'] = data['Close'] / ema
                
                # Price position in range
                high_period = data['High'].rolling(period).max()
                low_period = data['Low'].rolling(period).min()
                features[f'price_position_{period}'] = (data['Close'] - low_period) / (high_period - low_period)
                
                # Returns
                features[f'return_{period}'] = data['Close'].pct_change(period)
                features[f'log_return_{period}'] = np.log(data['Close'] / data['Close'].shift(period))
                
                # Volatility
                features[f'volatility_{period}'] = data['Close'].pct_change().rolling(period).std()
                features[f'realized_vol_{period}'] = np.sqrt(data['Close'].pct_change().rolling(period).var() * 252)
                
                # Price momentum
                features[f'momentum_{period}'] = data['Close'] / data['Close'].shift(period) - 1
                
                # Mean reversion
                mean_price = data['Close'].rolling(period).mean()
                features[f'mean_reversion_{period}'] = (data['Close'] - mean_price) / mean_price
                
                # Bollinger Bands
                bb_std = data['Close'].rolling(period).std()
                bb_upper = sma + (bb_std * 2)
                bb_lower = sma - (bb_std * 2)
                features[f'bb_upper_{period}'] = bb_upper
                features[f'bb_lower_{period}'] = bb_lower
                features[f'bb_position_{period}'] = (data['Close'] - bb_lower) / (bb_upper - bb_lower)
                features[f'bb_width_{period}'] = (bb_upper - bb_lower) / sma
                
                # RSI
                features[f'rsi_{period}'] = self._calculate_rsi(data['Close'], period)
                
                # MACD (for specific periods)\n                if period in [12, 26]:\n                    if period == 12:\n                        ema12 = data['Close'].ewm(span=12).mean()\n                        ema26 = data['Close'].ewm(span=26).mean()\n                        features['macd'] = ema12 - ema26\n                        features['macd_signal'] = features['macd'].ewm(span=9).mean()\n                        features['macd_histogram'] = features['macd'] - features['macd_signal']\n        \n        return features\n    \n    def _compute_price_features_gpu(self, data, periods):\n        \"\"\"GPU implementation of price features using cuDF\"\"\"\n        # Convert to cuDF\n        gpu_data = cudf.from_pandas(data)\n        features = cudf.DataFrame(index=gpu_data.index)\n        \n        # Basic price features\n        features['price_change'] = gpu_data['Close'].diff()\n        features['price_change_pct'] = gpu_data['Close'].pct_change()\n        features['high_low_ratio'] = gpu_data['High'] / gpu_data['Low']\n        features['close_open_ratio'] = gpu_data['Close'] / gpu_data['Open']\n        \n        # Moving averages (GPU accelerated)\n        for period in periods:\n            if len(gpu_data) >= period:\n                # Simple Moving Average\n                sma = gpu_data['Close'].rolling(period).mean()\n                features[f'sma_{period}'] = sma\n                features[f'price_sma_ratio_{period}'] = gpu_data['Close'] / sma\n                \n                # Returns\n                features[f'return_{period}'] = gpu_data['Close'].pct_change(period)\n                \n                # Volatility\n                features[f'volatility_{period}'] = gpu_data['Close'].pct_change().rolling(period).std()\n        \n        # Convert back to pandas\n        return features.to_pandas()\n    \n    def compute_volume_features(self, data, periods=[5, 10, 20, 50]):\n        \"\"\"Compute volume-based features\"\"\"\n        features = pd.DataFrame(index=data.index)\n        \n        # Basic volume features\n        features['volume_change'] = data['Volume'].diff()\n        features['volume_change_pct'] = data['Volume'].pct_change()\n        \n        # Volume moving averages\n        for period in periods:\n            if len(data) >= period:\n                vol_sma = data['Volume'].rolling(period).mean()\n                features[f'volume_sma_{period}'] = vol_sma\n                features[f'volume_ratio_{period}'] = data['Volume'] / vol_sma\n                \n                # Volume volatility\n                features[f'volume_volatility_{period}'] = data['Volume'].rolling(period).std()\n                \n                # Volume momentum\n                features[f'volume_momentum_{period}'] = data['Volume'] / data['Volume'].shift(period) - 1\n        \n        # VWAP (Volume Weighted Average Price)\n        features['vwap'] = (data['Close'] * data['Volume']).cumsum() / data['Volume'].cumsum()\n        features['price_vwap_ratio'] = data['Close'] / features['vwap']\n        \n        # On-Balance Volume (OBV)\n        price_change = data['Close'].diff()\n        volume_direction = np.where(price_change > 0, data['Volume'], \n                                  np.where(price_change < 0, -data['Volume'], 0))\n        features['obv'] = pd.Series(volume_direction, index=data.index).cumsum()\n        \n        # Volume-Price Trend (VPT)\n        features['vpt'] = (data['Volume'] * data['Close'].pct_change()).cumsum()\n        \n        # Accumulation/Distribution Line\n        mfm = ((data['Close'] - data['Low']) - (data['High'] - data['Close'])) / (data['High'] - data['Low'])\n        mfm = mfm.fillna(0)  # Handle division by zero\n        features['ad_line'] = (mfm * data['Volume']).cumsum()\n        \n        return features\n    \n    def compute_microstructure_features(self, data):\n        \"\"\"Compute microstructure and intraday features\"\"\"\n        features = pd.DataFrame(index=data.index)\n        \n        # Spread measures\n        features['hl_spread'] = (data['High'] - data['Low']) / data['Close']\n        features['oc_spread'] = abs(data['Open'] - data['Close']) / data['Close']\n        \n        # Gap analysis\n        features['overnight_gap'] = data['Open'] / data['Close'].shift(1) - 1\n        features['intraday_return'] = data['Close'] / data['Open'] - 1\n        \n        # Price efficiency\n        features['price_efficiency'] = abs(data['Close'] - data['Open']) / (data['High'] - data['Low'])\n        features['price_efficiency'] = features['price_efficiency'].fillna(0)\n        \n        # Volatility measures\n        features['true_range'] = np.maximum(\n            data['High'] - data['Low'],\n            np.maximum(\n                abs(data['High'] - data['Close'].shift(1)),\n                abs(data['Low'] - data['Close'].shift(1))\n            )\n        )\n        \n        # Average True Range\n        for period in [14, 20, 50]:\n            if len(data) >= period:\n                features[f'atr_{period}'] = features['true_range'].rolling(period).mean()\n                features[f'atr_ratio_{period}'] = features['true_range'] / features[f'atr_{period}']\n        \n        # Doji patterns (simplified)\n        body_size = abs(data['Close'] - data['Open'])\n        total_range = data['High'] - data['Low']\n        features['doji_ratio'] = body_size / total_range\n        features['doji_ratio'] = features['doji_ratio'].fillna(0)\n        \n        return features\n    \n    def compute_advanced_features(self, data, periods=[14, 20, 50]):\n        \"\"\"Compute advanced technical indicators\"\"\"\n        features = pd.DataFrame(index=data.index)\n        \n        # Stochastic Oscillator\n        for period in periods:\n            if len(data) >= period:\n                lowest_low = data['Low'].rolling(period).min()\n                highest_high = data['High'].rolling(period).max()\n                features[f'stoch_k_{period}'] = 100 * (data['Close'] - lowest_low) / (highest_high - lowest_low)\n                features[f'stoch_d_{period}'] = features[f'stoch_k_{period}'].rolling(3).mean()\n        \n        # Williams %R\n        for period in periods:\n            if len(data) >= period:\n                highest_high = data['High'].rolling(period).max()\n                lowest_low = data['Low'].rolling(period).min()\n                features[f'williams_r_{period}'] = -100 * (highest_high - data['Close']) / (highest_high - lowest_low)\n        \n        # Commodity Channel Index (CCI)\n        for period in periods:\n            if len(data) >= period:\n                typical_price = (data['High'] + data['Low'] + data['Close']) / 3\n                sma_tp = typical_price.rolling(period).mean()\n                mad = typical_price.rolling(period).apply(lambda x: np.mean(np.abs(x - x.mean())))\n                features[f'cci_{period}'] = (typical_price - sma_tp) / (0.015 * mad)\n        \n        # Money Flow Index (MFI)\n        for period in periods:\n            if len(data) >= period:\n                typical_price = (data['High'] + data['Low'] + data['Close']) / 3\n                money_flow = typical_price * data['Volume']\n                \n                positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0)\n                negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0)\n                \n                positive_mf = positive_flow.rolling(period).sum()\n                negative_mf = negative_flow.rolling(period).sum()\n                \n                mfi = 100 - (100 / (1 + positive_mf / negative_mf))\n                features[f'mfi_{period}'] = mfi.fillna(50)\n        \n        # Parabolic SAR (simplified)\n        features['sar'] = self._calculate_parabolic_sar(data)\n        \n        return features\n    \n    def compute_talib_features(self, data):\n        \"\"\"Compute TA-Lib features if available\"\"\"\n        if not TALIB_AVAILABLE:\n            return pd.DataFrame(index=data.index)\n        \n        features = pd.DataFrame(index=data.index)\n        \n        # Convert to numpy arrays\n        high = data['High'].values\n        low = data['Low'].values\n        close = data['Close'].values\n        volume = data['Volume'].values\n        \n        try:\n            # Overlap Studies\n            features['ta_sma_20'] = talib.SMA(close, timeperiod=20)\n            features['ta_ema_20'] = talib.EMA(close, timeperiod=20)\n            features['ta_wma_20'] = talib.WMA(close, timeperiod=20)\n            \n            # Bollinger Bands\n            bb_upper, bb_middle, bb_lower = talib.BBANDS(close, timeperiod=20)\n            features['ta_bb_upper'] = bb_upper\n            features['ta_bb_middle'] = bb_middle\n            features['ta_bb_lower'] = bb_lower\n            \n            # MACD\n            macd, macd_signal, macd_hist = talib.MACD(close)\n            features['ta_macd'] = macd\n            features['ta_macd_signal'] = macd_signal\n            features['ta_macd_hist'] = macd_hist\n            \n            # Momentum Indicators\n            features['ta_rsi_14'] = talib.RSI(close, timeperiod=14)\n            features['ta_stoch_k'], features['ta_stoch_d'] = talib.STOCH(high, low, close)\n            features['ta_williams_r'] = talib.WILLR(high, low, close)\n            features['ta_cci'] = talib.CCI(high, low, close)\n            features['ta_mfi'] = talib.MFI(high, low, close, volume)\n            \n            # Volume Indicators\n            features['ta_obv'] = talib.OBV(close, volume)\n            features['ta_ad'] = talib.AD(high, low, close, volume)\n            \n            # Volatility Indicators\n            features['ta_atr'] = talib.ATR(high, low, close)\n            features['ta_natr'] = talib.NATR(high, low, close)\n            \n            # Pattern Recognition (selected patterns)\n            features['ta_doji'] = talib.CDLDOJI(data['Open'].values, high, low, close)\n            features['ta_hammer'] = talib.CDLHAMMER(data['Open'].values, high, low, close)\n            features['ta_engulfing'] = talib.CDLENGULFING(data['Open'].values, high, low, close)\n            \n        except Exception as e:\n            print(f\"Warning: TA-Lib feature computation failed: {e}\")\n        \n        return features\n    \n    def compute_all_features(self, data, feature_groups=None):\n        \"\"\"Compute all features for a single ticker\"\"\"\n        if feature_groups is None:\n            feature_groups = ['price', 'volume', 'microstructure', 'advanced', 'talib']\n        \n        all_features = pd.DataFrame(index=data.index)\n        \n        # Price features\n        if 'price' in feature_groups:\n            price_features = self.compute_price_features(data)\n            all_features = pd.concat([all_features, price_features], axis=1)\n        \n        # Volume features\n        if 'volume' in feature_groups:\n            volume_features = self.compute_volume_features(data)\n            all_features = pd.concat([all_features, volume_features], axis=1)\n        \n        # Microstructure features\n        if 'microstructure' in feature_groups:\n            micro_features = self.compute_microstructure_features(data)\n            all_features = pd.concat([all_features, micro_features], axis=1)\n        \n        # Advanced features\n        if 'advanced' in feature_groups:\n            advanced_features = self.compute_advanced_features(data)\n            all_features = pd.concat([all_features, advanced_features], axis=1)\n        \n        # TA-Lib features\n        if 'talib' in feature_groups and TALIB_AVAILABLE:\n            talib_features = self.compute_talib_features(data)\n            all_features = pd.concat([all_features, talib_features], axis=1)\n        \n        return all_features\n    \n    def compute_features_parallel(self, data_dict, feature_groups=None, n_jobs=None):\n        \"\"\"Compute features for multiple tickers in parallel\"\"\"\n        if n_jobs is None:\n            n_jobs = self.n_jobs\n        \n        print(f\"Computing features for {len(data_dict)} tickers using {n_jobs} cores...\")\n        \n        # Prepare function for parallel execution\n        compute_func = partial(self._compute_single_ticker_features, feature_groups=feature_groups)\n        \n        # Execute in parallel\n        results = Parallel(n_jobs=n_jobs, backend='threading')(\n            delayed(compute_func)(ticker, data) \n            for ticker, data in data_dict.items()\n        )\n        \n        # Combine results\n        feature_dict = {}\n        for ticker, features in results:\n            if features is not None:\n                feature_dict[ticker] = features\n        \n        print(f\"✓ Features computed for {len(feature_dict)} tickers\")\n        return feature_dict\n    \n    def _compute_single_ticker_features(self, ticker, data, feature_groups=None):\n        \"\"\"Helper function for parallel feature computation\"\"\"\n        try:\n            features = self.compute_all_features(data, feature_groups)\n            return ticker, features\n        except Exception as e:\n            print(f\"Error computing features for {ticker}: {e}\")\n            return ticker, None\n    \n    def compute_forward_returns(self, data, horizons=[1, 5, 10, 20, 60]):\n        \"\"\"Compute forward returns for different horizons\"\"\"\n        returns = pd.DataFrame(index=data.index)\n        \n        for horizon in horizons:\n            # Simple returns\n            returns[f'forward_return_{horizon}'] = data['Close'].pct_change(horizon).shift(-horizon)\n            \n            # Log returns\n            returns[f'forward_log_return_{horizon}'] = np.log(data['Close'] / data['Close'].shift(-horizon))\n            \n            # Binary returns (up/down)\n            returns[f'forward_binary_{horizon}'] = (returns[f'forward_return_{horizon}'] > 0).astype(int)\n        \n        return returns\n    \n    def _calculate_rsi(self, prices, period=14):\n        \"\"\"Calculate Relative Strength Index\"\"\"\n        delta = prices.diff()\n        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()\n        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()\n        rs = gain / loss\n        rsi = 100 - (100 / (1 + rs))\n        return rsi\n    \n    def _calculate_parabolic_sar(self, data, af_start=0.02, af_increment=0.02, af_max=0.2):\n        \"\"\"Calculate Parabolic SAR (simplified implementation)\"\"\"\n        high = data['High'].values\n        low = data['Low'].values\n        close = data['Close'].values\n        \n        sar = np.zeros(len(data))\n        trend = np.zeros(len(data))\n        af = np.zeros(len(data))\n        ep = np.zeros(len(data))\n        \n        # Initialize\n        sar[0] = low[0]\n        trend[0] = 1  # 1 for uptrend, -1 for downtrend\n        af[0] = af_start\n        ep[0] = high[0]\n        \n        for i in range(1, len(data)):\n            if trend[i-1] == 1:  # Uptrend\n                sar[i] = sar[i-1] + af[i-1] * (ep[i-1] - sar[i-1])\n                \n                if high[i] > ep[i-1]:\n                    ep[i] = high[i]\n                    af[i] = min(af[i-1] + af_increment, af_max)\n                else:\n                    ep[i] = ep[i-1]\n                    af[i] = af[i-1]\n                \n                if low[i] <= sar[i]:\n                    trend[i] = -1\n                    sar[i] = ep[i-1]\n                    af[i] = af_start\n                    ep[i] = low[i]\n                else:\n                    trend[i] = 1\n            \n            else:  # Downtrend\n                sar[i] = sar[i-1] + af[i-1] * (ep[i-1] - sar[i-1])\n                \n                if low[i] < ep[i-1]:\n                    ep[i] = low[i]\n                    af[i] = min(af[i-1] + af_increment, af_max)\n                else:\n                    ep[i] = ep[i-1]\n                    af[i] = af[i-1]\n                \n                if high[i] >= sar[i]:\n                    trend[i] = 1\n                    sar[i] = ep[i-1]\n                    af[i] = af_start\n                    ep[i] = high[i]\n                else:\n                    trend[i] = -1\n        \n        return pd.Series(sar, index=data.index)\n    \n    def get_feature_importance_stats(self, features_df):\n        \"\"\"Get statistics about computed features\"\"\"\n        stats = {\n            'total_features': len(features_df.columns),\n            'feature_categories': {},\n            'missing_data': {},\n            'feature_correlations': None\n        }\n        \n        # Categorize features\n        categories = {\n            'price': ['sma_', 'ema_', 'price_', 'return_', 'momentum_', 'bb_', 'rsi_', 'macd'],\n            'volume': ['volume_', 'vwap', 'obv', 'vpt', 'ad_line', 'mfi_'],\n            'microstructure': ['spread', 'gap', 'efficiency', 'atr_', 'doji'],\n            'advanced': ['stoch_', 'williams_', 'cci_', 'sar'],\n            'talib': ['ta_']\n        }\n        \n        for category, prefixes in categories.items():\n            count = sum(1 for col in features_df.columns \n                       if any(prefix in col for prefix in prefixes))\n            stats['feature_categories'][category] = count\n        \n        # Missing data analysis\n        for col in features_df.columns:\n            missing_pct = features_df[col].isnull().sum() / len(features_df) * 100\n            if missing_pct > 0:\n                stats['missing_data'][col] = missing_pct\n        \n        # Feature correlations (sample)\n        if len(features_df.columns) > 0:\n            sample_features = features_df.select_dtypes(include=[np.number]).iloc[:, :50]  # First 50 numeric features\n            if not sample_features.empty:\n                corr_matrix = sample_features.corr()\n                high_corr_pairs = []\n                \n                for i in range(len(corr_matrix.columns)):\n                    for j in range(i+1, len(corr_matrix.columns)):\n                        corr_val = corr_matrix.iloc[i, j]\n                        if abs(corr_val) > 0.9:  # High correlation threshold\n                            high_corr_pairs.append((\n                                corr_matrix.columns[i], \n                                corr_matrix.columns[j], \n                                corr_val\n                            ))\n                \n                stats['high_correlations'] = high_corr_pairs[:10]  # Top 10\n        \n        return stats
