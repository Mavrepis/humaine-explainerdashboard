"""
Weakspot Analysis functionality for ExplainerDashboard.

This module provides the core WeakspotAnalyzer class that identifies
data slices where machine learning models perform significantly worse
than their overall performance.
"""

__all__ = [
    "WeakspotAnalyzer",
    "WeakspotResult",
    "WeakspotValidationError",
]

from dataclasses import dataclass
from typing import List, Dict, Union, Optional, Tuple, Any
import warnings

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    accuracy_score,
    log_loss,
    brier_score_loss,
)


class WeakspotValidationError(Exception):
    """Custom exception for weakspot analysis validation errors."""
    pass


@dataclass
class WeakspotResult:
    """Data structure for weakspot analysis results."""
    slice_features: List[str]
    slice_method: str
    metric: str
    overall_metric: float
    threshold_value: float
    bin_results: List[Dict]  # [{range, performance, sample_count, is_weak}]
    weak_regions: List[Dict]  # [{range, performance, sample_count, severity}]
    summary_stats: Dict
    
    def __post_init__(self):
        """Validate the result structure after initialization."""
        if not self.slice_features:
            raise WeakspotValidationError("slice_features cannot be empty")
        if len(self.slice_features) > 2:
            raise WeakspotValidationError("Maximum 2 slice features supported")


def mape_score(y_true, y_pred):
    """Calculate Mean Absolute Percentage Error."""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    # Avoid division by zero
    mask = y_true != 0
    if not mask.any():
        return np.inf
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


class WeakspotAnalyzer:
    """
    Core class for performing weakspot analysis on machine learning models.
    
    This class provides methods to identify data slices where model performance
    is significantly worse than overall performance using histogram-based or
    tree-based slicing methods.
    """
    
    # Supported metrics for different model types
    REGRESSION_METRICS = {
        'mse': mean_squared_error,
        'mae': mean_absolute_error, 
        'mape': mape_score,
    }
    
    CLASSIFICATION_METRICS = {
        'accuracy': accuracy_score,
        'log_loss': log_loss,
        'brier_score': brier_score_loss,
    }
    
    # Metrics where lower values are better
    LOWER_IS_BETTER = {'mse', 'mae', 'mape', 'log_loss', 'brier_score'}
    
    def __init__(self, is_classifier: bool = False):
        """
        Initialize the WeakspotAnalyzer.
        
        Args:
            is_classifier: Whether the model is a classifier (True) or regressor (False)
        """
        self.is_classifier = is_classifier
        self.supported_metrics = (
            self.CLASSIFICATION_METRICS if is_classifier else self.REGRESSION_METRICS
        )
    
    def validate_inputs(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        y_pred: np.ndarray,
        slice_features: List[str],
        metric: str,
        slice_method: str,
        bins: int,
        min_samples: int,
        threshold: float
    ) -> None:
        """
        Validate all inputs for weakspot analysis.
        
        Args:
            X: Feature dataframe
            y: True target values
            y_pred: Model predictions
            slice_features: Features to slice on
            metric: Performance metric to use
            slice_method: Method for slicing ('histogram' or 'tree')
            bins: Number of bins for histogram method
            min_samples: Minimum samples per slice
            threshold: Threshold for identifying weak regions
            
        Raises:
            WeakspotValidationError: If any validation fails
        """
        # Basic data validation
        if X.empty:
            raise WeakspotValidationError("X cannot be empty")
        if len(y) == 0:
            raise WeakspotValidationError("y cannot be empty")
        if len(y_pred) == 0:
            raise WeakspotValidationError("y_pred cannot be empty")
        if len(X) != len(y) or len(X) != len(y_pred):
            raise WeakspotValidationError("X, y, and y_pred must have same length")
        
        # Feature validation
        if not slice_features:
            raise WeakspotValidationError("slice_features cannot be empty")
        if len(slice_features) > 2:
            raise WeakspotValidationError("Maximum 2 slice features supported")
        
        missing_features = set(slice_features) - set(X.columns)
        if missing_features:
            raise WeakspotValidationError(
                f"Features not found in X: {missing_features}"
            )
        
        # Check for numeric features
        for feature in slice_features:
            if not pd.api.types.is_numeric_dtype(X[feature]):
                raise WeakspotValidationError(
                    f"Feature '{feature}' must be numeric for slicing"
                )
        
        # Metric validation
        if metric not in self.supported_metrics:
            raise WeakspotValidationError(
                f"Metric '{metric}' not supported. "
                f"Supported metrics: {list(self.supported_metrics.keys())}"
            )
        
        # Parameter validation
        if slice_method not in ['histogram', 'tree']:
            raise WeakspotValidationError(
                "slice_method must be 'histogram' or 'tree'"
            )
        if bins < 2:
            raise WeakspotValidationError("bins must be at least 2")
        if min_samples < 1:
            raise WeakspotValidationError("min_samples must be at least 1")
        if threshold <= 0:
            raise WeakspotValidationError("threshold must be positive")
    
    def calculate_metric(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        metric: str
    ) -> float:
        """
        Calculate the specified metric.
        
        Args:
            y_true: True target values
            y_pred: Model predictions
            metric: Metric name
            
        Returns:
            Calculated metric value
        """
        try:
            metric_func = self.supported_metrics[metric]
            
            # Handle special cases for classification metrics
            if metric == 'log_loss':
                # Ensure predictions are probabilities
                if self.is_classifier and len(y_pred.shape) == 1:
                    # Binary classification - convert to probabilities
                    y_pred_proba = np.column_stack([1 - y_pred, y_pred])
                    return metric_func(y_true, y_pred_proba)
                return metric_func(y_true, y_pred)
            elif metric == 'brier_score':
                # Brier score expects probabilities for positive class
                if len(y_pred.shape) > 1:
                    y_pred = y_pred[:, 1]  # Use positive class probability
                return metric_func(y_true, y_pred)
            else:
                return metric_func(y_true, y_pred)
                
        except Exception as e:
            warnings.warn(f"Error calculating {metric}: {e}")
            return np.nan
    
    def create_histogram_slices(
        self,
        X: pd.DataFrame,
        slice_features: List[str],
        bins: int
    ) -> List[Dict]:
        """
        Create data slices using histogram binning.
        
        Args:
            X: Feature dataframe
            slice_features: Features to slice on
            bins: Number of bins
            
        Returns:
            List of slice definitions with conditions and indices
        """
        slices = []
        
        if len(slice_features) == 1:
            # 1D slicing
            feature = slice_features[0]
            values = X[feature].dropna()
            
            if len(values) == 0:
                return slices
            
            # Create bins
            bin_edges = np.histogram_bin_edges(values, bins=bins)
            
            for i in range(len(bin_edges) - 1):
                left, right = bin_edges[i], bin_edges[i + 1]
                
                # Handle edge case for last bin
                if i == len(bin_edges) - 2:
                    mask = (X[feature] >= left) & (X[feature] <= right)
                else:
                    mask = (X[feature] >= left) & (X[feature] < right)
                
                indices = X.index[mask].tolist()
                
                slices.append({
                    'feature_ranges': {feature: (left, right)},
                    'indices': indices,
                    'description': f"{feature}: [{left:.3f}, {right:.3f}]"
                })
        
        elif len(slice_features) == 2:
            # 2D slicing
            feature1, feature2 = slice_features
            values1 = X[feature1].dropna()
            values2 = X[feature2].dropna()
            
            if len(values1) == 0 or len(values2) == 0:
                return slices
            
            # Create bins for both features
            bin_edges1 = np.histogram_bin_edges(values1, bins=bins)
            bin_edges2 = np.histogram_bin_edges(values2, bins=bins)
            
            for i in range(len(bin_edges1) - 1):
                for j in range(len(bin_edges2) - 1):
                    left1, right1 = bin_edges1[i], bin_edges1[i + 1]
                    left2, right2 = bin_edges2[j], bin_edges2[j + 1]
                    
                    # Handle edge cases for last bins
                    mask1 = (X[feature1] >= left1) & (
                        X[feature1] <= right1 if i == len(bin_edges1) - 2 
                        else X[feature1] < right1
                    )
                    mask2 = (X[feature2] >= left2) & (
                        X[feature2] <= right2 if j == len(bin_edges2) - 2 
                        else X[feature2] < right2
                    )
                    
                    mask = mask1 & mask2
                    indices = X.index[mask].tolist()
                    
                    slices.append({
                        'feature_ranges': {
                            feature1: (left1, right1),
                            feature2: (left2, right2)
                        },
                        'indices': indices,
                        'description': (
                            f"{feature1}: [{left1:.3f}, {right1:.3f}], "
                            f"{feature2}: [{left2:.3f}, {right2:.3f}]"
                        )
                    })
        
        return slices
    
    def create_tree_slices(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        slice_features: List[str],
        max_depth: int = 3
    ) -> List[Dict]:
        """
        Create data slices using decision tree-based splitting.
        
        Args:
            X: Feature dataframe
            y: Target values
            slice_features: Features to slice on
            max_depth: Maximum depth of decision tree
            
        Returns:
            List of slice definitions with conditions and indices
        """
        slices = []
        
        # Prepare data for tree
        X_slice = X[slice_features].copy()
        
        # Handle missing values
        X_slice = X_slice.fillna(X_slice.median())
        
        # Create and fit decision tree
        if self.is_classifier:
            tree = DecisionTreeClassifier(
                max_depth=max_depth,
                min_samples_leaf=20,
                random_state=42
            )
        else:
            tree = DecisionTreeRegressor(
                max_depth=max_depth,
                min_samples_leaf=20,
                random_state=42
            )
        
        try:
            tree.fit(X_slice, y)
        except Exception as e:
            warnings.warn(f"Failed to fit decision tree: {e}")
            return slices
        
        # Extract leaf nodes and their conditions
        leaf_indices = tree.apply(X_slice)
        unique_leaves = np.unique(leaf_indices)
        
        for leaf_id in unique_leaves:
            mask = leaf_indices == leaf_id
            indices = X.index[mask].tolist()
            
            # Get the path conditions for this leaf
            # This is a simplified approach - in practice, you'd want to
            # extract the actual decision path from the tree
            slices.append({
                'feature_ranges': self._extract_leaf_conditions(
                    tree, leaf_id, slice_features, X_slice
                ),
                'indices': indices,
                'description': f"Tree leaf {leaf_id}"
            })
        
        return slices
    
    def _extract_leaf_conditions(
        self,
        tree,
        leaf_id: int,
        feature_names: List[str],
        X: pd.DataFrame
    ) -> Dict[str, Tuple[float, float]]:
        """
        Extract feature range conditions for a specific leaf node.
        
        This is a simplified implementation that estimates ranges
        based on the samples in each leaf.
        """
        # Get samples in this leaf
        leaf_mask = tree.apply(X) == leaf_id
        leaf_samples = X[leaf_mask]
        
        conditions = {}
        for feature in feature_names:
            if len(leaf_samples) > 0:
                min_val = leaf_samples[feature].min()
                max_val = leaf_samples[feature].max()
                conditions[feature] = (min_val, max_val)
            else:
                conditions[feature] = (np.nan, np.nan)
        
        return conditions
    
    def analyze_weakspots(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        y_pred: np.ndarray,
        slice_features: List[str],
        slice_method: str = "histogram",
        bins: int = 10,
        metric: str = None,
        threshold: float = 1.1,
        min_samples: int = 20
    ) -> WeakspotResult:
        """
        Perform weakspot analysis on the given data.
        
        Args:
            X: Feature dataframe
            y: True target values
            y_pred: Model predictions
            slice_features: Features to slice on (1-2 features)
            slice_method: 'histogram' or 'tree'
            bins: Number of bins for histogram method
            metric: Performance metric to use (auto-detected if None)
            threshold: Multiplier for identifying weak regions
            min_samples: Minimum samples required per slice
            
        Returns:
            WeakspotResult containing analysis results
        """
        # Auto-detect metric if not provided
        if metric is None:
            if self.is_classifier:
                metric = 'accuracy'
            else:
                metric = 'mse'
        
        # Validate inputs
        self.validate_inputs(
            X, y, y_pred, slice_features, metric, slice_method, 
            bins, min_samples, threshold
        )
        
        # Calculate overall performance
        overall_metric = self.calculate_metric(y, y_pred, metric)
        
        # Create slices
        if slice_method == "histogram":
            slices = self.create_histogram_slices(X, slice_features, bins)
        else:  # tree
            slices = self.create_tree_slices(X, y, slice_features)
        
        # Analyze each slice
        bin_results = []
        weak_regions = []
        
        for slice_info in slices:
            indices = slice_info['indices']
            
            # Skip slices with insufficient samples
            if len(indices) < min_samples:
                continue
            
            # Calculate performance for this slice
            slice_y = y.iloc[indices]
            slice_y_pred = y_pred[indices]
            
            slice_metric = self.calculate_metric(slice_y, slice_y_pred, metric)
            
            # Determine if this is a weak region
            is_weak = self._is_weak_region(
                slice_metric, overall_metric, threshold, metric
            )
            
            bin_result = {
                'range': slice_info['feature_ranges'],
                'description': slice_info['description'],
                'performance': slice_metric,
                'sample_count': len(indices),
                'is_weak': is_weak,
                'indices': indices
            }
            
            bin_results.append(bin_result)
            
            if is_weak:
                severity = self._calculate_severity(
                    slice_metric, overall_metric, metric
                )
                weak_regions.append({
                    'range': slice_info['feature_ranges'],
                    'description': slice_info['description'],
                    'performance': slice_metric,
                    'sample_count': len(indices),
                    'severity': severity
                })
        
        # Calculate summary statistics
        summary_stats = self._calculate_summary_stats(
            bin_results, overall_metric, metric
        )
        
        return WeakspotResult(
            slice_features=slice_features,
            slice_method=slice_method,
            metric=metric,
            overall_metric=overall_metric,
            threshold_value=threshold,
            bin_results=bin_results,
            weak_regions=weak_regions,
            summary_stats=summary_stats
        )
    
    def _is_weak_region(
        self,
        slice_metric: float,
        overall_metric: float,
        threshold: float,
        metric: str
    ) -> bool:
        """Determine if a slice represents a weak region."""
        if np.isnan(slice_metric) or np.isnan(overall_metric):
            return False
        
        if metric in self.LOWER_IS_BETTER:
            # For metrics where lower is better, weak regions have higher values
            return slice_metric > (overall_metric * threshold)
        else:
            # For metrics where higher is better, weak regions have lower values
            return slice_metric < (overall_metric / threshold)
    
    def _calculate_severity(
        self,
        slice_metric: float,
        overall_metric: float,
        metric: str
    ) -> float:
        """Calculate severity score for a weak region."""
        if np.isnan(slice_metric) or np.isnan(overall_metric):
            return 0.0
        
        if overall_metric == 0:
            return 0.0
        
        if metric in self.LOWER_IS_BETTER:
            return slice_metric / overall_metric
        else:
            return overall_metric / slice_metric if slice_metric != 0 else np.inf
    
    def _calculate_summary_stats(
        self,
        bin_results: List[Dict],
        overall_metric: float,
        metric: str
    ) -> Dict:
        """Calculate summary statistics for the analysis."""
        if not bin_results:
            return {
                'total_slices': 0,
                'weak_slices': 0,
                'weak_percentage': 0.0,
                'worst_performance': np.nan,
                'best_performance': np.nan,
                'performance_range': np.nan
            }
        
        performances = [r['performance'] for r in bin_results if not np.isnan(r['performance'])]
        weak_count = sum(1 for r in bin_results if r['is_weak'])
        
        if performances:
            worst_perf = max(performances) if metric in self.LOWER_IS_BETTER else min(performances)
            best_perf = min(performances) if metric in self.LOWER_IS_BETTER else max(performances)
            perf_range = worst_perf - best_perf if metric in self.LOWER_IS_BETTER else best_perf - worst_perf
        else:
            worst_perf = best_perf = perf_range = np.nan
        
        return {
            'total_slices': len(bin_results),
            'weak_slices': weak_count,
            'weak_percentage': (weak_count / len(bin_results)) * 100,
            'worst_performance': worst_perf,
            'best_performance': best_perf,
            'performance_range': perf_range,
            'overall_performance': overall_metric
        }