"""
Unit tests for weakspot analysis functionality.
"""

import pytest
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split

from explainerdashboard.weakspot_analyzer import (
    WeakspotAnalyzer,
    WeakspotResult,
    WeakspotValidationError,
    mape_score
)
from explainerdashboard import ClassifierExplainer, RegressionExplainer


class TestWeakspotAnalyzer:
    """Test the core WeakspotAnalyzer class."""
    
    @pytest.fixture
    def classification_data(self):
        """Create sample classification data."""
        X, y = make_classification(
            n_samples=1000,
            n_features=5,
            n_informative=3,
            n_redundant=1,
            n_clusters_per_class=1,
            random_state=42
        )
        
        feature_names = [f'feature_{i}' for i in range(X.shape[1])]
        X_df = pd.DataFrame(X, columns=feature_names)
        y_series = pd.Series(y, name='target')
        
        # Train a simple model
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X_df, y_series)
        y_pred = model.predict_proba(X_df)[:, 1]
        
        return X_df, y_series, y_pred, model
    
    @pytest.fixture
    def regression_data(self):
        """Create sample regression data."""
        X, y = make_regression(
            n_samples=1000,
            n_features=5,
            n_informative=3,
            noise=0.1,
            random_state=42
        )
        
        feature_names = [f'feature_{i}' for i in range(X.shape[1])]
        X_df = pd.DataFrame(X, columns=feature_names)
        y_series = pd.Series(y, name='target')
        
        # Train a simple model
        model = RandomForestRegressor(n_estimators=10, random_state=42)
        model.fit(X_df, y_series)
        y_pred = model.predict(X_df)
        
        return X_df, y_series, y_pred, model
    
    def test_analyzer_initialization(self):
        """Test WeakspotAnalyzer initialization."""
        # Test classifier initialization
        analyzer_clf = WeakspotAnalyzer(is_classifier=True)
        assert analyzer_clf.is_classifier is True
        assert analyzer_clf.supported_metrics == analyzer_clf.CLASSIFICATION_METRICS
        
        # Test regressor initialization
        analyzer_reg = WeakspotAnalyzer(is_classifier=False)
        assert analyzer_reg.is_classifier is False
        assert analyzer_reg.supported_metrics == analyzer_reg.REGRESSION_METRICS
    
    def test_mape_score(self):
        """Test MAPE score calculation."""
        y_true = np.array([100, 200, 300, 400])
        y_pred = np.array([110, 190, 310, 380])
        
        expected_mape = np.mean([10/100, 10/200, 10/300, 20/400]) * 100
        calculated_mape = mape_score(y_true, y_pred)
        
        assert abs(calculated_mape - expected_mape) < 1e-10
        
        # Test with zeros (should handle gracefully)
        y_true_with_zero = np.array([0, 100, 200])
        y_pred_with_zero = np.array([10, 110, 190])
        mape_with_zero = mape_score(y_true_with_zero, y_pred_with_zero)
        assert not np.isnan(mape_with_zero)
    
    def test_input_validation(self, classification_data):
        """Test input validation."""
        X, y, y_pred, _ = classification_data
        analyzer = WeakspotAnalyzer(is_classifier=True)
        
        # Test valid inputs (should not raise)
        analyzer.validate_inputs(
            X, y, y_pred, ['feature_0'], 'accuracy', 'histogram', 10, 20, 1.1
        )
        
        # Test empty data
        with pytest.raises(WeakspotValidationError, match="X cannot be empty"):
            analyzer.validate_inputs(
                pd.DataFrame(), y, y_pred, ['feature_0'], 'accuracy', 'histogram', 10, 20, 1.1
            )
        
        # Test mismatched lengths
        with pytest.raises(WeakspotValidationError, match="same length"):
            analyzer.validate_inputs(
                X, y[:500], y_pred, ['feature_0'], 'accuracy', 'histogram', 10, 20, 1.1
            )
        
        # Test invalid features
        with pytest.raises(WeakspotValidationError, match="Features not found"):
            analyzer.validate_inputs(
                X, y, y_pred, ['invalid_feature'], 'accuracy', 'histogram', 10, 20, 1.1
            )
        
        # Test too many features
        with pytest.raises(WeakspotValidationError, match="Maximum 2 slice features"):
            analyzer.validate_inputs(
                X, y, y_pred, ['feature_0', 'feature_1', 'feature_2'], 
                'accuracy', 'histogram', 10, 20, 1.1
            )
        
        # Test invalid metric
        with pytest.raises(WeakspotValidationError, match="not supported"):
            analyzer.validate_inputs(
                X, y, y_pred, ['feature_0'], 'invalid_metric', 'histogram', 10, 20, 1.1
            )
        
        # Test invalid slice method
        with pytest.raises(WeakspotValidationError, match="slice_method must be"):
            analyzer.validate_inputs(
                X, y, y_pred, ['feature_0'], 'accuracy', 'invalid_method', 10, 20, 1.1
            )
        
        # Test invalid parameters
        with pytest.raises(WeakspotValidationError, match="bins must be at least"):
            analyzer.validate_inputs(
                X, y, y_pred, ['feature_0'], 'accuracy', 'histogram', 1, 20, 1.1
            )
        
        with pytest.raises(WeakspotValidationError, match="min_samples must be at least"):
            analyzer.validate_inputs(
                X, y, y_pred, ['feature_0'], 'accuracy', 'histogram', 10, 0, 1.1
            )
        
        with pytest.raises(WeakspotValidationError, match="threshold must be positive"):
            analyzer.validate_inputs(
                X, y, y_pred, ['feature_0'], 'accuracy', 'histogram', 10, 20, -1.0
            )
    
    def test_metric_calculation_classification(self, classification_data):
        """Test metric calculation for classification."""
        X, y, y_pred, _ = classification_data
        analyzer = WeakspotAnalyzer(is_classifier=True)
        
        # Test accuracy
        accuracy = analyzer.calculate_metric(y, (y_pred > 0.5).astype(int), 'accuracy')
        assert 0 <= accuracy <= 1
        
        # Test log loss (requires probability predictions)
        log_loss_val = analyzer.calculate_metric(y, y_pred, 'log_loss')
        assert log_loss_val >= 0
        
        # Test brier score
        brier_score = analyzer.calculate_metric(y, y_pred, 'brier_score')
        assert 0 <= brier_score <= 1
    
    def test_metric_calculation_regression(self, regression_data):
        """Test metric calculation for regression."""
        X, y, y_pred, _ = regression_data
        analyzer = WeakspotAnalyzer(is_classifier=False)
        
        # Test MSE
        mse = analyzer.calculate_metric(y, y_pred, 'mse')
        assert mse >= 0
        
        # Test MAE
        mae = analyzer.calculate_metric(y, y_pred, 'mae')
        assert mae >= 0
        
        # Test MAPE
        mape = analyzer.calculate_metric(y, y_pred, 'mape')
        assert mape >= 0
    
    def test_histogram_slicing_1d(self, classification_data):
        """Test 1D histogram slicing."""
        X, y, y_pred, _ = classification_data
        analyzer = WeakspotAnalyzer(is_classifier=True)
        
        slices = analyzer.create_histogram_slices(X, ['feature_0'], bins=5)
        
        assert len(slices) == 5
        for slice_info in slices:
            assert 'feature_ranges' in slice_info
            assert 'indices' in slice_info
            assert 'description' in slice_info
            assert 'feature_0' in slice_info['feature_ranges']
            assert len(slice_info['indices']) > 0
    
    def test_histogram_slicing_2d(self, classification_data):
        """Test 2D histogram slicing."""
        X, y, y_pred, _ = classification_data
        analyzer = WeakspotAnalyzer(is_classifier=True)
        
        slices = analyzer.create_histogram_slices(X, ['feature_0', 'feature_1'], bins=3)
        
        assert len(slices) == 9  # 3x3 grid
        for slice_info in slices:
            assert 'feature_ranges' in slice_info
            assert 'indices' in slice_info
            assert 'description' in slice_info
            assert 'feature_0' in slice_info['feature_ranges']
            assert 'feature_1' in slice_info['feature_ranges']
    
    def test_tree_slicing(self, classification_data):
        """Test tree-based slicing."""
        X, y, y_pred, _ = classification_data
        analyzer = WeakspotAnalyzer(is_classifier=True)
        
        slices = analyzer.create_tree_slices(X, y, ['feature_0', 'feature_1'])
        
        assert len(slices) > 0
        for slice_info in slices:
            assert 'feature_ranges' in slice_info
            assert 'indices' in slice_info
            assert 'description' in slice_info
            assert len(slice_info['indices']) > 0
    
    def test_weakspot_analysis_classification(self, classification_data):
        """Test full weakspot analysis for classification."""
        X, y, y_pred, _ = classification_data
        analyzer = WeakspotAnalyzer(is_classifier=True)
        
        result = analyzer.analyze_weakspots(
            X=X,
            y=y,
            y_pred=(y_pred > 0.5).astype(int),  # Convert to binary predictions
            slice_features=['feature_0'],
            slice_method='histogram',
            bins=5,
            metric='accuracy',
            threshold=1.2,
            min_samples=50
        )
        
        assert isinstance(result, WeakspotResult)
        assert result.slice_features == ['feature_0']
        assert result.slice_method == 'histogram'
        assert result.metric == 'accuracy'
        assert result.overall_metric >= 0
        assert len(result.bin_results) > 0
        assert 'total_slices' in result.summary_stats
    
    def test_weakspot_analysis_regression(self, regression_data):
        """Test full weakspot analysis for regression."""
        X, y, y_pred, _ = regression_data
        analyzer = WeakspotAnalyzer(is_classifier=False)
        
        result = analyzer.analyze_weakspots(
            X=X,
            y=y,
            y_pred=y_pred,
            slice_features=['feature_0'],
            slice_method='histogram',
            bins=5,
            metric='mse',
            threshold=1.5,
            min_samples=50
        )
        
        assert isinstance(result, WeakspotResult)
        assert result.slice_features == ['feature_0']
        assert result.slice_method == 'histogram'
        assert result.metric == 'mse'
        assert result.overall_metric >= 0
        assert len(result.bin_results) > 0
        assert 'total_slices' in result.summary_stats
    
    def test_weak_region_detection(self, classification_data):
        """Test weak region detection logic."""
        X, y, y_pred, _ = classification_data
        analyzer = WeakspotAnalyzer(is_classifier=True)
        
        # Test with lower threshold to ensure some weak regions are found
        result = analyzer.analyze_weakspots(
            X=X,
            y=y,
            y_pred=(y_pred > 0.5).astype(int),
            slice_features=['feature_0'],
            slice_method='histogram',
            bins=10,
            metric='accuracy',
            threshold=1.05,  # Very low threshold
            min_samples=20
        )
        
        # Check that weak regions are properly identified
        weak_count = sum(1 for br in result.bin_results if br['is_weak'])
        assert weak_count == len(result.weak_regions)
        
        # Check severity calculation
        for weak_region in result.weak_regions:
            assert 'severity' in weak_region
            assert weak_region['severity'] > 0
    
    def test_edge_cases(self, classification_data):
        """Test edge cases and error handling."""
        X, y, y_pred, _ = classification_data
        analyzer = WeakspotAnalyzer(is_classifier=True)
        
        # Test with very high threshold (no weak regions expected)
        result = analyzer.analyze_weakspots(
            X=X,
            y=y,
            y_pred=(y_pred > 0.5).astype(int),
            slice_features=['feature_0'],
            slice_method='histogram',
            bins=5,
            metric='accuracy',
            threshold=10.0,  # Very high threshold
            min_samples=50
        )
        
        assert len(result.weak_regions) == 0
        
        # Test with very high min_samples (few slices expected)
        result = analyzer.analyze_weakspots(
            X=X,
            y=y,
            y_pred=(y_pred > 0.5).astype(int),
            slice_features=['feature_0'],
            slice_method='histogram',
            bins=10,
            metric='accuracy',
            threshold=1.1,
            min_samples=500  # Very high min_samples
        )
        
        assert len(result.bin_results) < 10  # Some slices should be filtered out


class TestWeakspotFixtures:
    """Test fixtures for various model types and scenarios."""
    
    @pytest.fixture
    def small_dataset_classifier(self):
        """Create small dataset for edge case testing."""
        X, y = make_classification(
            n_samples=100,
            n_features=3,
            n_informative=2,
            n_redundant=1,
            random_state=42
        )
        
        feature_names = [f'feature_{i}' for i in range(X.shape[1])]
        X_df = pd.DataFrame(X, columns=feature_names)
        y_series = pd.Series(y, name='target')
        
        model = RandomForestClassifier(n_estimators=5, random_state=42)
        model.fit(X_df, y_series)
        y_pred = model.predict_proba(X_df)[:, 1]
        
        return X_df, y_series, y_pred, model
    
    @pytest.fixture
    def large_dataset_classifier(self):
        """Create large dataset for performance testing."""
        X, y = make_classification(
            n_samples=5000,
            n_features=10,
            n_informative=7,
            n_redundant=2,
            random_state=42
        )
        
        feature_names = [f'feature_{i}' for i in range(X.shape[1])]
        X_df = pd.DataFrame(X, columns=feature_names)
        y_series = pd.Series(y, name='target')
        
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X_df, y_series)
        y_pred = model.predict_proba(X_df)[:, 1]
        
        return X_df, y_series, y_pred, model
    
    @pytest.fixture
    def imbalanced_dataset_classifier(self):
        """Create imbalanced dataset for bias testing."""
        X, y = make_classification(
            n_samples=1000,
            n_features=5,
            n_informative=3,
            n_redundant=1,
            weights=[0.9, 0.1],  # Highly imbalanced
            random_state=42
        )
        
        feature_names = [f'feature_{i}' for i in range(X.shape[1])]
        X_df = pd.DataFrame(X, columns=feature_names)
        y_series = pd.Series(y, name='target')
        
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X_df, y_series)
        y_pred = model.predict_proba(X_df)[:, 1]
        
        return X_df, y_series, y_pred, model
    
    @pytest.fixture
    def multiclass_dataset(self):
        """Create multiclass dataset for testing."""
        X, y = make_classification(
            n_samples=1000,
            n_features=5,
            n_informative=3,
            n_redundant=1,
            n_classes=3,
            random_state=42
        )
        
        feature_names = [f'feature_{i}' for i in range(X.shape[1])]
        X_df = pd.DataFrame(X, columns=feature_names)
        y_series = pd.Series(y, name='target')
        
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X_df, y_series)
        y_pred = model.predict(X_df)  # Use class predictions for multiclass
        
        return X_df, y_series, y_pred, model
    
    @pytest.fixture
    def regression_with_outliers(self):
        """Create regression dataset with outliers."""
        X, y = make_regression(
            n_samples=1000,
            n_features=5,
            n_informative=3,
            noise=0.1,
            random_state=42
        )
        
        # Add outliers
        outlier_indices = np.random.choice(len(y), size=50, replace=False)
        y[outlier_indices] += np.random.normal(0, 10, size=50)
        
        feature_names = [f'feature_{i}' for i in range(X.shape[1])]
        X_df = pd.DataFrame(X, columns=feature_names)
        y_series = pd.Series(y, name='target')
        
        model = RandomForestRegressor(n_estimators=10, random_state=42)
        model.fit(X_df, y_series)
        y_pred = model.predict(X_df)
        
        return X_df, y_series, y_pred, model
    
    @pytest.fixture
    def categorical_features_dataset(self):
        """Create dataset with categorical features."""
        n_samples = 1000
        
        # Create mixed data types
        numerical_features = np.random.randn(n_samples, 3)
        categorical_feature_1 = np.random.choice(['A', 'B', 'C'], size=n_samples)
        categorical_feature_2 = np.random.choice(['X', 'Y'], size=n_samples)
        
        # Create target based on features
        y = (
            numerical_features[:, 0] + 
            numerical_features[:, 1] * 0.5 +
            (categorical_feature_1 == 'A').astype(int) * 2 +
            (categorical_feature_2 == 'X').astype(int) * 1.5 +
            np.random.normal(0, 0.1, n_samples)
        )
        
        # Convert to binary classification
        y = (y > np.median(y)).astype(int)
        
        X_df = pd.DataFrame({
            'num_feature_0': numerical_features[:, 0],
            'num_feature_1': numerical_features[:, 1],
            'num_feature_2': numerical_features[:, 2],
            'cat_feature_1': categorical_feature_1,
            'cat_feature_2': categorical_feature_2
        })
        
        # One-hot encode categorical features for model
        X_encoded = pd.get_dummies(X_df, columns=['cat_feature_1', 'cat_feature_2'])
        y_series = pd.Series(y, name='target')
        
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X_encoded, y_series)
        y_pred = model.predict_proba(X_encoded)[:, 1]
        
        return X_encoded, y_series, y_pred, model


class TestExplainerIntegration:
    """Test integration with ClassifierExplainer and RegressionExplainer."""
    
    @pytest.fixture
    def classifier_explainer(self):
        """Create a ClassifierExplainer for testing."""
        X, y = make_classification(
            n_samples=500,
            n_features=4,
            n_informative=3,
            n_redundant=1,
            random_state=42
        )
        
        feature_names = [f'feature_{i}' for i in range(X.shape[1])]
        X_df = pd.DataFrame(X, columns=feature_names)
        y_series = pd.Series(y, name='target')
        
        model = LogisticRegression(random_state=42)
        model.fit(X_df, y_series)
        
        explainer = ClassifierExplainer(model, X_df, y_series)
        return explainer
    
    @pytest.fixture
    def regression_explainer(self):
        """Create a RegressionExplainer for testing."""
        X, y = make_regression(
            n_samples=500,
            n_features=4,
            n_informative=3,
            noise=0.1,
            random_state=42
        )
        
        feature_names = [f'feature_{i}' for i in range(X.shape[1])]
        X_df = pd.DataFrame(X, columns=feature_names)
        y_series = pd.Series(y, name='target')
        
        model = LinearRegression()
        model.fit(X_df, y_series)
        
        explainer = RegressionExplainer(model, X_df, y_series)
        return explainer
    
    def test_classifier_weakspot_analysis(self, classifier_explainer):
        """Test weakspot analysis with ClassifierExplainer."""
        result = classifier_explainer.calculate_weakspot_analysis(
            slice_features=['feature_0'],
            slice_method='histogram',
            bins=5,
            metric='accuracy',
            threshold=1.2,
            min_samples=20
        )
        
        assert isinstance(result, dict)
        assert 'slice_features' in result
        assert 'bin_results' in result
        assert 'weak_regions' in result
        assert 'summary_stats' in result
        assert result['slice_features'] == ['feature_0']
        assert result['metric'] == 'accuracy'
    
    def test_regression_weakspot_analysis(self, regression_explainer):
        """Test weakspot analysis with RegressionExplainer."""
        result = regression_explainer.calculate_weakspot_analysis(
            slice_features=['feature_0'],
            slice_method='histogram',
            bins=5,
            metric='mse',
            threshold=1.5,
            min_samples=20
        )
        
        assert isinstance(result, dict)
        assert 'slice_features' in result
        assert 'bin_results' in result
        assert 'weak_regions' in result
        assert 'summary_stats' in result
        assert result['slice_features'] == ['feature_0']
        assert result['metric'] == 'mse'
    
    def test_get_weakspot_data(self, classifier_explainer):
        """Test get_weakspot_data method."""
        df = classifier_explainer.get_weakspot_data(
            slice_features=['feature_0'],
            slice_method='histogram',
            bins=5,
            threshold=1.1,
            min_samples=20
        )
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert 'description' in df.columns
        assert 'performance' in df.columns
        assert 'sample_count' in df.columns
        assert 'is_weak' in df.columns
        assert 'feature_0_min' in df.columns
        assert 'feature_0_max' in df.columns
        assert 'feature_0_center' in df.columns
        
        # Check metadata
        assert 'metric' in df.attrs
        assert 'overall_metric' in df.attrs
        assert 'threshold_value' in df.attrs
        assert 'slice_features' in df.attrs
    
    def test_weakspot_summary(self, classifier_explainer):
        """Test weakspot_summary method."""
        summary = classifier_explainer.weakspot_summary(
            slice_features=['feature_0'],
            slice_method='histogram',
            bins=5,
            threshold=1.1,
            min_samples=20
        )
        
        assert isinstance(summary, str)
        assert 'Weakspot Analysis Summary' in summary
        assert 'feature_0' in summary
        assert 'histogram' in summary
        assert 'Total slices:' in summary
        assert 'Weak slices:' in summary
    
    def test_auto_metric_detection(self, classifier_explainer, regression_explainer):
        """Test automatic metric detection."""
        # Test classifier (should default to accuracy)
        result_clf = classifier_explainer.calculate_weakspot_analysis(
            slice_features=['feature_0'],
            metric=None  # Should auto-detect
        )
        assert result_clf['metric'] == 'accuracy'
        
        # Test regressor (should default to mse)
        result_reg = regression_explainer.calculate_weakspot_analysis(
            slice_features=['feature_0'],
            metric=None  # Should auto-detect
        )
        assert result_reg['metric'] == 'mse'
    
    def test_two_feature_analysis(self, classifier_explainer):
        """Test analysis with two features."""
        result = classifier_explainer.calculate_weakspot_analysis(
            slice_features=['feature_0', 'feature_1'],
            slice_method='histogram',
            bins=3,  # 3x3 = 9 slices
            threshold=1.2,
            min_samples=10
        )
        
        assert result['slice_features'] == ['feature_0', 'feature_1']
        # Should have some slices (may be filtered by min_samples)
        assert len(result['bin_results']) > 0
        
        # Check that ranges include both features
        for bin_result in result['bin_results']:
            assert 'feature_0' in bin_result['range']
            assert 'feature_1' in bin_result['range']
    
    def test_tree_method(self, classifier_explainer):
        """Test tree slicing method."""
        result = classifier_explainer.calculate_weakspot_analysis(
            slice_features=['feature_0', 'feature_1'],
            slice_method='tree',
            threshold=1.2,
            min_samples=20
        )
        
        assert result['slice_method'] == 'tree'
        assert len(result['bin_results']) > 0
        
        # Tree method should create meaningful slices
        for bin_result in result['bin_results']:
            assert bin_result['sample_count'] >= 20  # Respects min_samples