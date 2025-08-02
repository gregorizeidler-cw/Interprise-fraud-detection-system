"""
Advanced Temporal Features for Fraud Detection

This module implements state-of-the-art time series feature engineering techniques
for capturing temporal patterns, seasonality, trends, and behavioral changes
in financial transaction data.

Key Features:
- Fourier Transform for detecting periodicities
- Wavelet decomposition for multi-resolution analysis
- Change point detection (CUSUM, PELT)
- Seasonal decomposition (STL)
- Rolling window statistics with various time horizons
- Velocity and acceleration features
- Time-based behavioral pattern detection
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Union
import warnings
warnings.filterwarnings('ignore')

# Scientific computing imports
from scipy import signal, stats
from scipy.fft import fft, fftfreq
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Time series analysis imports
try:
    import pywt
    PYWT_AVAILABLE = True
except ImportError:
    PYWT_AVAILABLE = False
    print("PyWavelets not available. Wavelet features will be disabled.")

try:
    from statsmodels.tsa.seasonal import STL
    from statsmodels.tsa.stattools import adfuller
    STL_AVAILABLE = True
except ImportError:
    STL_AVAILABLE = False
    print("Statsmodels not available. Some temporal features will be disabled.")

import logging

logger = logging.getLogger(__name__)


class TemporalFeatureExtractor:
    """
    Advanced temporal feature extraction for fraud detection
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.time_windows = self.config.get('time_windows', [
            '1H', '6H', '24H', '7D', '30D', '90D'
        ])
        self.feature_cache = {}
        
    def extract_all_temporal_features(self, 
                                    transactions_df: pd.DataFrame,
                                    customer_id: str,
                                    reference_time: datetime = None) -> Dict:
        """
        Extract comprehensive temporal features for a customer
        
        Args:
            transactions_df: Transaction data
            customer_id: Customer identifier
            reference_time: Reference time for feature calculation
            
        Returns:
            Dictionary of temporal features
        """
        if reference_time is None:
            reference_time = datetime.now()
            
        # Filter customer transactions
        customer_txns = transactions_df[
            transactions_df['customer_id'] == customer_id
        ].copy()
        
        if len(customer_txns) == 0:
            return self._get_default_temporal_features()
        
        # Ensure timestamp is datetime
        customer_txns['timestamp'] = pd.to_datetime(customer_txns['timestamp'])
        customer_txns = customer_txns.sort_values('timestamp')
        
        features = {}
        
        # Basic temporal features
        features.update(self._extract_basic_temporal_features(customer_txns, reference_time))
        
        # Rolling window features
        features.update(self._extract_rolling_window_features(customer_txns, reference_time))
        
        # Velocity and acceleration features
        features.update(self._extract_velocity_features(customer_txns, reference_time))
        
        # Periodicity features
        features.update(self._extract_periodicity_features(customer_txns))
        
        # Seasonal decomposition features
        if STL_AVAILABLE:
            features.update(self._extract_seasonal_features(customer_txns))
        
        # Change point detection features
        features.update(self._extract_changepoint_features(customer_txns))
        
        # Wavelet features
        if PYWT_AVAILABLE:
            features.update(self._extract_wavelet_features(customer_txns))
        
        # Behavioral pattern features
        features.update(self._extract_behavioral_patterns(customer_txns, reference_time))
        
        return features
    
    def _extract_basic_temporal_features(self, 
                                       transactions_df: pd.DataFrame,
                                       reference_time: datetime) -> Dict:
        """Extract basic temporal statistics"""
        features = {}
        
        # Time since first and last transaction
        first_txn = transactions_df['timestamp'].min()
        last_txn = transactions_df['timestamp'].max()
        
        features['days_since_first_transaction'] = (reference_time - first_txn).days
        features['days_since_last_transaction'] = (reference_time - last_txn).days
        features['account_activity_span_days'] = (last_txn - first_txn).days
        
        # Transaction frequency
        total_days = max((last_txn - first_txn).days, 1)
        features['avg_transactions_per_day'] = len(transactions_df) / total_days
        
        # Inter-transaction times
        if len(transactions_df) > 1:
            inter_times = transactions_df['timestamp'].diff().dt.total_seconds() / 3600  # hours
            inter_times = inter_times.dropna()
            
            features['avg_inter_transaction_hours'] = inter_times.mean()
            features['std_inter_transaction_hours'] = inter_times.std()
            features['min_inter_transaction_hours'] = inter_times.min()
            features['max_inter_transaction_hours'] = inter_times.max()
            features['median_inter_transaction_hours'] = inter_times.median()
        else:
            features.update({
                'avg_inter_transaction_hours': 0,
                'std_inter_transaction_hours': 0,
                'min_inter_transaction_hours': 0,
                'max_inter_transaction_hours': 0,
                'median_inter_transaction_hours': 0
            })
        
        return features
    
    def _extract_rolling_window_features(self, 
                                       transactions_df: pd.DataFrame,
                                       reference_time: datetime) -> Dict:
        """Extract rolling window statistics for various time horizons"""
        features = {}
        
        for window in self.time_windows:
            window_start = reference_time - pd.Timedelta(window)
            window_txns = transactions_df[
                transactions_df['timestamp'] >= window_start
            ]
            
            prefix = f'{window}_'
            
            # Basic counts and amounts
            features[f'{prefix}transaction_count'] = len(window_txns)
            features[f'{prefix}total_amount'] = window_txns['amount'].sum()
            features[f'{prefix}avg_amount'] = window_txns['amount'].mean() if len(window_txns) > 0 else 0
            features[f'{prefix}std_amount'] = window_txns['amount'].std() if len(window_txns) > 0 else 0
            features[f'{prefix}max_amount'] = window_txns['amount'].max() if len(window_txns) > 0 else 0
            features[f'{prefix}min_amount'] = window_txns['amount'].min() if len(window_txns) > 0 else 0
            
            # Product diversity
            features[f'{prefix}unique_products'] = window_txns['product_type'].nunique()
            
            # Channel diversity
            if 'channel' in window_txns.columns:
                features[f'{prefix}unique_channels'] = window_txns['channel'].nunique()
            
            # Beneficiary patterns
            if 'beneficiary_id' in window_txns.columns:
                features[f'{prefix}unique_beneficiaries'] = window_txns['beneficiary_id'].nunique()
                features[f'{prefix}repeat_beneficiary_rate'] = (
                    1 - features[f'{prefix}unique_beneficiaries'] / max(len(window_txns), 1)
                )
        
        return features
    
    def _extract_velocity_features(self, 
                                 transactions_df: pd.DataFrame,
                                 reference_time: datetime) -> Dict:
        """Extract velocity and acceleration features"""
        features = {}
        
        # Define measurement windows
        velocity_windows = ['1H', '6H', '24H', '7D']
        
        for i, window in enumerate(velocity_windows):
            window_start = reference_time - pd.Timedelta(window)
            window_txns = transactions_df[
                transactions_df['timestamp'] >= window_start
            ]
            
            # Transaction velocity (count per hour)
            hours = pd.Timedelta(window).total_seconds() / 3600
            velocity = len(window_txns) / hours
            features[f'velocity_txn_per_hour_{window}'] = velocity
            
            # Amount velocity (amount per hour)
            amount_velocity = window_txns['amount'].sum() / hours
            features[f'velocity_amount_per_hour_{window}'] = amount_velocity
            
            # Calculate acceleration (change in velocity)
            if i > 0:
                prev_window = velocity_windows[i-1]
                prev_velocity_key = f'velocity_txn_per_hour_{prev_window}'
                if prev_velocity_key in features:
                    acceleration = velocity - features[prev_velocity_key]
                    features[f'acceleration_txn_{prev_window}_to_{window}'] = acceleration
        
        return features
    
    def _extract_periodicity_features(self, transactions_df: pd.DataFrame) -> Dict:
        """Extract periodicity features using Fourier Transform"""
        features = {}
        
        if len(transactions_df) < 10:  # Need minimum data for meaningful analysis
            return {'fourier_dominant_period_hours': 0, 'fourier_power_concentration': 0}
        
        try:
            # Create hourly time series
            transactions_df = transactions_df.set_index('timestamp')
            hourly_counts = transactions_df.resample('H').size()
            
            # Fill missing hours with 0
            hourly_counts = hourly_counts.reindex(
                pd.date_range(hourly_counts.index.min(), hourly_counts.index.max(), freq='H'),
                fill_value=0
            )
            
            if len(hourly_counts) < 24:  # Need at least 24 hours for daily patterns
                return {'fourier_dominant_period_hours': 0, 'fourier_power_concentration': 0}
            
            # Apply FFT
            fft_values = fft(hourly_counts.values)
            fft_freqs = fftfreq(len(hourly_counts), d=1)  # d=1 hour
            
            # Get power spectrum
            power = np.abs(fft_values) ** 2
            
            # Find dominant frequency (excluding DC component)
            positive_freqs = fft_freqs[1:len(fft_freqs)//2]
            positive_power = power[1:len(power)//2]
            
            if len(positive_power) > 0:
                dominant_freq_idx = np.argmax(positive_power)
                dominant_freq = positive_freqs[dominant_freq_idx]
                dominant_period_hours = 1 / abs(dominant_freq) if dominant_freq != 0 else 0
                
                # Power concentration (how much power is in the dominant frequency)
                total_power = np.sum(positive_power)
                power_concentration = positive_power[dominant_freq_idx] / total_power if total_power > 0 else 0
            else:
                dominant_period_hours = 0
                power_concentration = 0
            
            features['fourier_dominant_period_hours'] = dominant_period_hours
            features['fourier_power_concentration'] = power_concentration
            
        except Exception as e:
            logger.warning(f"Error in periodicity extraction: {e}")
            features['fourier_dominant_period_hours'] = 0
            features['fourier_power_concentration'] = 0
        
        return features
    
    def _extract_seasonal_features(self, transactions_df: pd.DataFrame) -> Dict:
        """Extract seasonal decomposition features using STL"""
        features = {}
        
        try:
            # Create daily time series
            transactions_df = transactions_df.set_index('timestamp')
            daily_counts = transactions_df.resample('D').size()
            
            if len(daily_counts) < 30:  # Need at least 30 days for seasonal analysis
                return self._get_default_seasonal_features()
            
            # Fill missing days
            daily_counts = daily_counts.reindex(
                pd.date_range(daily_counts.index.min(), daily_counts.index.max(), freq='D'),
                fill_value=0
            )
            
            # Apply STL decomposition
            stl = STL(daily_counts, seasonal=7)  # Weekly seasonality
            decomposition = stl.fit()
            
            # Extract features from components
            features['seasonal_strength'] = np.var(decomposition.seasonal) / np.var(daily_counts)
            features['trend_strength'] = np.var(decomposition.trend) / np.var(daily_counts)
            features['remainder_strength'] = np.var(decomposition.resid) / np.var(daily_counts)
            
            # Trend direction
            trend_slope = np.polyfit(range(len(decomposition.trend)), decomposition.trend, 1)[0]
            features['trend_slope'] = trend_slope
            
            # Seasonal pattern consistency
            seasonal_values = decomposition.seasonal
            features['seasonal_consistency'] = 1 - (np.std(seasonal_values) / (np.mean(np.abs(seasonal_values)) + 1e-8))
            
        except Exception as e:
            logger.warning(f"Error in seasonal decomposition: {e}")
            features.update(self._get_default_seasonal_features())
        
        return features
    
    def _extract_changepoint_features(self, transactions_df: pd.DataFrame) -> Dict:
        """Extract change point detection features"""
        features = {}
        
        if len(transactions_df) < 20:  # Need minimum data for change point detection
            return {'changepoints_detected': 0, 'time_since_last_changepoint_days': 0}
        
        try:
            # Create daily time series
            transactions_df = transactions_df.set_index('timestamp')
            daily_amounts = transactions_df.resample('D')['amount'].sum()
            
            # Simple CUSUM-based change point detection
            changepoints = self._cusum_changepoint_detection(daily_amounts.values)
            
            features['changepoints_detected'] = len(changepoints)
            
            if changepoints:
                last_changepoint_idx = max(changepoints)
                last_changepoint_date = daily_amounts.index[last_changepoint_idx]
                days_since_last = (daily_amounts.index[-1] - last_changepoint_date).days
                features['time_since_last_changepoint_days'] = days_since_last
            else:
                features['time_since_last_changepoint_days'] = len(daily_amounts)
                
        except Exception as e:
            logger.warning(f"Error in change point detection: {e}")
            features['changepoints_detected'] = 0
            features['time_since_last_changepoint_days'] = 0
        
        return features
    
    def _cusum_changepoint_detection(self, data: np.ndarray, threshold: float = 5.0) -> List[int]:
        """Simple CUSUM change point detection"""
        if len(data) < 5:
            return []
        
        # Calculate CUSUM statistics
        mean_data = np.mean(data)
        cusum_pos = np.zeros(len(data))
        cusum_neg = np.zeros(len(data))
        
        for i in range(1, len(data)):
            cusum_pos[i] = max(0, cusum_pos[i-1] + data[i] - mean_data - 0.5)
            cusum_neg[i] = min(0, cusum_neg[i-1] + data[i] - mean_data + 0.5)
        
        # Find change points
        changepoints = []
        for i in range(1, len(data)-1):
            if abs(cusum_pos[i]) > threshold or abs(cusum_neg[i]) > threshold:
                changepoints.append(i)
        
        return changepoints
    
    def _extract_wavelet_features(self, transactions_df: pd.DataFrame) -> Dict:
        """Extract wavelet decomposition features"""
        features = {}
        
        try:
            # Create hourly time series
            transactions_df = transactions_df.set_index('timestamp')
            hourly_counts = transactions_df.resample('H').size()
            
            if len(hourly_counts) < 32:  # Need power of 2 for efficient wavelet transform
                return self._get_default_wavelet_features()
            
            # Pad to nearest power of 2
            next_pow2 = 2 ** int(np.ceil(np.log2(len(hourly_counts))))
            padded_data = np.pad(hourly_counts.values, (0, next_pow2 - len(hourly_counts)), 'constant')
            
            # Perform wavelet decomposition
            coeffs = pywt.wavedec(padded_data, 'db4', level=4)
            
            # Extract features from wavelet coefficients
            for level, coeff in enumerate(coeffs):
                features[f'wavelet_energy_level_{level}'] = np.sum(coeff ** 2)
                features[f'wavelet_std_level_{level}'] = np.std(coeff)
                features[f'wavelet_max_level_{level}'] = np.max(np.abs(coeff))
            
            # Total wavelet energy
            total_energy = sum(features[f'wavelet_energy_level_{i}'] for i in range(len(coeffs)))
            
            # Energy distribution across levels
            for level in range(len(coeffs)):
                features[f'wavelet_energy_ratio_level_{level}'] = (
                    features[f'wavelet_energy_level_{level}'] / (total_energy + 1e-8)
                )
                
        except Exception as e:
            logger.warning(f"Error in wavelet decomposition: {e}")
            features.update(self._get_default_wavelet_features())
        
        return features
    
    def _extract_behavioral_patterns(self, 
                                   transactions_df: pd.DataFrame,
                                   reference_time: datetime) -> Dict:
        """Extract behavioral pattern features"""
        features = {}
        
        # Hour of day patterns
        transactions_df['hour'] = transactions_df['timestamp'].dt.hour
        hour_counts = transactions_df['hour'].value_counts()
        
        # Business vs non-business hours
        business_hours = list(range(9, 18))  # 9 AM to 5 PM
        business_txns = transactions_df[transactions_df['hour'].isin(business_hours)]
        non_business_txns = transactions_df[~transactions_df['hour'].isin(business_hours)]
        
        total_txns = len(transactions_df)
        features['business_hours_ratio'] = len(business_txns) / max(total_txns, 1)
        features['non_business_hours_ratio'] = len(non_business_txns) / max(total_txns, 1)
        
        # Weekend vs weekday patterns
        transactions_df['day_of_week'] = transactions_df['timestamp'].dt.dayofweek
        weekend_txns = transactions_df[transactions_df['day_of_week'].isin([5, 6])]
        weekday_txns = transactions_df[~transactions_df['day_of_week'].isin([5, 6])]
        
        features['weekend_ratio'] = len(weekend_txns) / max(total_txns, 1)
        features['weekday_ratio'] = len(weekday_txns) / max(total_txns, 1)
        
        # Hour diversity (entropy)
        if len(hour_counts) > 0:
            hour_probs = hour_counts / hour_counts.sum()
            hour_entropy = -np.sum(hour_probs * np.log2(hour_probs + 1e-8))
            features['hour_entropy'] = hour_entropy
        else:
            features['hour_entropy'] = 0
        
        # Most active hours
        if len(hour_counts) > 0:
            features['most_active_hour'] = hour_counts.index[0]
            features['most_active_hour_count'] = hour_counts.iloc[0]
        else:
            features['most_active_hour'] = 0
            features['most_active_hour_count'] = 0
        
        return features
    
    def _get_default_temporal_features(self) -> Dict:
        """Return default values for temporal features"""
        features = {}
        
        # Basic temporal features
        features.update({
            'days_since_first_transaction': 0,
            'days_since_last_transaction': 0,
            'account_activity_span_days': 0,
            'avg_transactions_per_day': 0,
            'avg_inter_transaction_hours': 0,
            'std_inter_transaction_hours': 0,
            'min_inter_transaction_hours': 0,
            'max_inter_transaction_hours': 0,
            'median_inter_transaction_hours': 0
        })
        
        # Rolling window features
        for window in self.time_windows:
            prefix = f'{window}_'
            features.update({
                f'{prefix}transaction_count': 0,
                f'{prefix}total_amount': 0,
                f'{prefix}avg_amount': 0,
                f'{prefix}std_amount': 0,
                f'{prefix}max_amount': 0,
                f'{prefix}min_amount': 0,
                f'{prefix}unique_products': 0,
                f'{prefix}unique_channels': 0,
                f'{prefix}unique_beneficiaries': 0,
                f'{prefix}repeat_beneficiary_rate': 0
            })
        
        # Other default features
        features.update(self._get_default_seasonal_features())
        features.update(self._get_default_wavelet_features())
        features.update({
            'fourier_dominant_period_hours': 0,
            'fourier_power_concentration': 0,
            'changepoints_detected': 0,
            'time_since_last_changepoint_days': 0,
            'business_hours_ratio': 0,
            'non_business_hours_ratio': 0,
            'weekend_ratio': 0,
            'weekday_ratio': 0,
            'hour_entropy': 0,
            'most_active_hour': 0,
            'most_active_hour_count': 0
        })
        
        return features
    
    def _get_default_seasonal_features(self) -> Dict:
        """Default seasonal features"""
        return {
            'seasonal_strength': 0,
            'trend_strength': 0,
            'remainder_strength': 0,
            'trend_slope': 0,
            'seasonal_consistency': 0
        }
    
    def _get_default_wavelet_features(self) -> Dict:
        """Default wavelet features"""
        features = {}
        for level in range(5):  # 4 levels + approximation
            features[f'wavelet_energy_level_{level}'] = 0
            features[f'wavelet_std_level_{level}'] = 0
            features[f'wavelet_max_level_{level}'] = 0
            features[f'wavelet_energy_ratio_level_{level}'] = 0
        return features


class RealTimeTemporalProcessor:
    """
    Real-time processor for temporal features with online learning capabilities
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.feature_extractor = TemporalFeatureExtractor(config)
        self.online_stats = {}
        self.feature_history = {}
        
    def update_online_features(self, 
                             customer_id: str,
                             new_transaction: Dict,
                             timestamp: datetime) -> Dict:
        """
        Update temporal features in real-time as new transactions arrive
        
        Args:
            customer_id: Customer identifier
            new_transaction: New transaction data
            timestamp: Transaction timestamp
            
        Returns:
            Updated temporal features
        """
        if customer_id not in self.online_stats:
            self.online_stats[customer_id] = {
                'transaction_count': 0,
                'total_amount': 0,
                'last_transaction_time': None,
                'first_transaction_time': None,
                'amount_sum_squares': 0,
                'inter_transaction_times': []
            }
        
        stats = self.online_stats[customer_id]
        
        # Update basic statistics
        stats['transaction_count'] += 1
        stats['total_amount'] += new_transaction['amount']
        stats['amount_sum_squares'] += new_transaction['amount'] ** 2
        
        # Update timing information
        if stats['first_transaction_time'] is None:
            stats['first_transaction_time'] = timestamp
        
        if stats['last_transaction_time'] is not None:
            inter_time = (timestamp - stats['last_transaction_time']).total_seconds() / 3600
            stats['inter_transaction_times'].append(inter_time)
            
            # Keep only recent inter-transaction times (last 100)
            if len(stats['inter_transaction_times']) > 100:
                stats['inter_transaction_times'] = stats['inter_transaction_times'][-100:]
        
        stats['last_transaction_time'] = timestamp
        
        # Calculate real-time features
        features = self._calculate_realtime_features(customer_id, timestamp)
        
        return features
    
    def _calculate_realtime_features(self, customer_id: str, current_time: datetime) -> Dict:
        """Calculate features from online statistics"""
        stats = self.online_stats[customer_id]
        features = {}
        
        # Basic features
        features['total_transactions'] = stats['transaction_count']
        features['total_amount'] = stats['total_amount']
        features['avg_amount'] = stats['total_amount'] / max(stats['transaction_count'], 1)
        
        # Amount variance (online calculation)
        if stats['transaction_count'] > 1:
            mean_amount = features['avg_amount']
            variance = (stats['amount_sum_squares'] / stats['transaction_count']) - (mean_amount ** 2)
            features['amount_std'] = np.sqrt(max(variance, 0))
        else:
            features['amount_std'] = 0
        
        # Timing features
        if stats['first_transaction_time']:
            features['days_active'] = (current_time - stats['first_transaction_time']).days
            features['transactions_per_day'] = stats['transaction_count'] / max(features['days_active'], 1)
        else:
            features['days_active'] = 0
            features['transactions_per_day'] = 0
        
        if stats['last_transaction_time']:
            features['hours_since_last_transaction'] = (
                current_time - stats['last_transaction_time']
            ).total_seconds() / 3600
        else:
            features['hours_since_last_transaction'] = 0
        
        # Inter-transaction time features
        if stats['inter_transaction_times']:
            inter_times = stats['inter_transaction_times']
            features['avg_inter_transaction_hours'] = np.mean(inter_times)
            features['std_inter_transaction_hours'] = np.std(inter_times)
            features['median_inter_transaction_hours'] = np.median(inter_times)
        else:
            features['avg_inter_transaction_hours'] = 0
            features['std_inter_transaction_hours'] = 0
            features['median_inter_transaction_hours'] = 0
        
        return features


class TemporalAnomalyDetector:
    """
    Detect temporal anomalies in transaction patterns
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.baseline_patterns = {}
        
    def detect_temporal_anomalies(self, 
                                customer_id: str,
                                transactions_df: pd.DataFrame,
                                current_transaction: Dict) -> Dict:
        """
        Detect temporal anomalies in current transaction
        
        Returns anomaly scores and detected anomalies
        """
        anomalies = {}
        
        # Time-of-day anomaly
        anomalies.update(self._detect_time_of_day_anomaly(
            customer_id, transactions_df, current_transaction
        ))
        
        # Frequency anomaly
        anomalies.update(self._detect_frequency_anomaly(
            customer_id, transactions_df, current_transaction
        ))
        
        # Amount timing anomaly
        anomalies.update(self._detect_amount_timing_anomaly(
            customer_id, transactions_df, current_transaction
        ))
        
        return anomalies
    
    def _detect_time_of_day_anomaly(self, 
                                  customer_id: str,
                                  transactions_df: pd.DataFrame,
                                  current_transaction: Dict) -> Dict:
        """Detect if current transaction time is unusual for this customer"""
        
        customer_txns = transactions_df[transactions_df['customer_id'] == customer_id]
        
        if len(customer_txns) < 5:  # Need baseline
            return {'time_of_day_anomaly_score': 0}
        
        # Historical hour distribution
        historical_hours = pd.to_datetime(customer_txns['timestamp']).dt.hour
        hour_counts = historical_hours.value_counts()
        total_txns = len(customer_txns)
        
        # Current transaction hour
        current_hour = pd.to_datetime(current_transaction['timestamp']).hour
        
        # Calculate anomaly score
        if current_hour in hour_counts:
            hour_probability = hour_counts[current_hour] / total_txns
            anomaly_score = 1 - hour_probability  # Higher score = more unusual
        else:
            anomaly_score = 1.0  # Never seen at this hour
        
        return {'time_of_day_anomaly_score': anomaly_score}
    
    def _detect_frequency_anomaly(self, 
                                customer_id: str,
                                transactions_df: pd.DataFrame,
                                current_transaction: Dict) -> Dict:
        """Detect if transaction frequency is unusual"""
        
        customer_txns = transactions_df[transactions_df['customer_id'] == customer_id]
        
        if len(customer_txns) < 10:
            return {'frequency_anomaly_score': 0}
        
        # Calculate recent transaction frequency
        current_time = pd.to_datetime(current_transaction['timestamp'])
        recent_window = current_time - pd.Timedelta(hours=24)
        recent_txns = customer_txns[
            pd.to_datetime(customer_txns['timestamp']) >= recent_window
        ]
        
        recent_count = len(recent_txns) + 1  # Include current transaction
        
        # Historical 24-hour frequencies
        customer_txns['timestamp'] = pd.to_datetime(customer_txns['timestamp'])
        customer_txns = customer_txns.sort_values('timestamp')
        
        historical_frequencies = []
        for i in range(len(customer_txns)):
            window_start = customer_txns.iloc[i]['timestamp'] - pd.Timedelta(hours=24)
            window_txns = customer_txns[customer_txns['timestamp'] >= window_start]
            window_txns = window_txns[window_txns['timestamp'] <= customer_txns.iloc[i]['timestamp']]
            historical_frequencies.append(len(window_txns))
        
        if len(historical_frequencies) > 0:
            mean_freq = np.mean(historical_frequencies)
            std_freq = np.std(historical_frequencies)
            
            if std_freq > 0:
                z_score = abs(recent_count - mean_freq) / std_freq
                anomaly_score = min(z_score / 3, 1.0)  # Normalize to 0-1
            else:
                anomaly_score = 0
        else:
            anomaly_score = 0
        
        return {'frequency_anomaly_score': anomaly_score}
    
    def _detect_amount_timing_anomaly(self, 
                                    customer_id: str,
                                    transactions_df: pd.DataFrame,
                                    current_transaction: Dict) -> Dict:
        """Detect if amount-timing combination is unusual"""
        
        customer_txns = transactions_df[transactions_df['customer_id'] == customer_id]
        
        if len(customer_txns) < 10:
            return {'amount_timing_anomaly_score': 0}
        
        # Get hour and amount for current transaction
        current_hour = pd.to_datetime(current_transaction['timestamp']).hour
        current_amount = current_transaction['amount']
        
        # Historical amount distribution for this hour
        customer_txns['hour'] = pd.to_datetime(customer_txns['timestamp']).dt.hour
        same_hour_txns = customer_txns[customer_txns['hour'] == current_hour]
        
        if len(same_hour_txns) == 0:
            return {'amount_timing_anomaly_score': 1.0}  # Never transacted at this hour
        
        # Calculate amount anomaly for this specific hour
        hour_amounts = same_hour_txns['amount']
        mean_amount = hour_amounts.mean()
        std_amount = hour_amounts.std()
        
        if std_amount > 0:
            z_score = abs(current_amount - mean_amount) / std_amount
            anomaly_score = min(z_score / 3, 1.0)
        else:
            anomaly_score = 0 if current_amount == mean_amount else 1.0
        
        return {'amount_timing_anomaly_score': anomaly_score}