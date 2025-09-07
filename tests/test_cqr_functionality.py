"""Tests for Split-Conformalized Quantile Regression functionality."""

import pytest
import numpy as np
import pandas as pd
from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, QuantileRegressor
from sklearn.model_selection import train_test_split

from explainerdashboard.explainers import RegressionExplainer
from explainerdashboard.explainer_methods import (
    empirical_quantile,
    conformalized_quantile_regression,
    get_uncertainty_intervals_df
)
from explainerdashboard.explainer_plots import (
    plotly_uncertainty_intervals,
    plotly_uncertainty_width,
    plotly_coverage_diagnostic
)
from explainerdashboard.dashboard_components.regression_components import (
    ConformalizedQuantileRegressionComponent
)


# Test fixtures
@pytest.fixture
def regression_data():
    """Generate synthetic regression data for testing."""
    X, y = make_regression(
        n_samples=1000, 
        n_features=10, 
        noise=10.0, 
        random_state=42
    )
    feature_names = [f"feature_{i}" for i in range(X.shape[1])]
    X = pd.DataFrame(X, columns=feature_names)
    y = pd.Series(y, name="target")
    return X, y


@pytest.fixture
def trained_model(regression_data):
    """Train a model on the regression data."""
    X, y = regression_data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    model = RandomForestRegressor(n_estimators=10, random_state=42)
    model.fit(X_train, y_train)
    
    return model, X_train, X_test, y_train, y_test


@pytest.fixture
def regression_explainer(trained_model, regression_data):
    """Create a RegressionExplainer instance."""
    model, X_train, X_test, y_train, y_test = trained_model
    X, y = regression_data
    
    explainer = RegressionExplainer(model, X_test, y_test)
    return explainer


# Test empirical_quantile function
class TestEmpiricalQuantile:
    
    def test_empirical_quantile_basic(self):
        """Test basic functionality of empirical_quantile."""
        residuals = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        miscoverage_rate = 0.1
        n_calib = len(residuals)
        
        q_hat = empirical_quantile(residuals, miscoverage_rate, n_calib)
        
        # Should return a value from the residuals array
        assert q_hat in residuals
        assert isinstance(q_hat, (int, float, np.integer, np.floating))
    
    def test_empirical_quantile_conservative(self):
        """Test that empirical_quantile provides conservative coverage."""
        residuals = np.random.normal(0, 1, 100)
        miscoverage_rate = 0.1
        n_calib = len(residuals)
        
        q_hat = empirical_quantile(residuals, miscoverage_rate, n_calib)
        
        # Check that at least (1-alpha) proportion of residuals are <= q_hat
        coverage = np.mean(residuals <= q_hat)
        assert coverage >= (1 - miscoverage_rate)
    
    def test_empirical_quantile_edge_cases(self):
        """Test edge cases for empirical_quantile."""
        # Single value
        residuals = np.array([5.0])
        q_hat = empirical_quantile(residuals, 0.1, 1)
        assert q_hat == 5.0
        
        # Small array
        residuals = np.array([1.0, 2.0])
        q_hat = empirical_quantile(residuals, 0.1, 2)
        assert q_hat in [1.0, 2.0]


# Test conformalized_quantile_regression function
class TestConformalized_QuantileRegression:
    
    def test_cqr_basic_functionality(self, regression_data):
        """Test basic CQR functionality."""
        X, y = regression_data
        
        predictor = conformalized_quantile_regression(
            QuantileRegressor,
            X, y,
            miscoverage_rate=0.1,
            test_size=0.3
        )
        
        # Test that predictor is callable
        assert callable(predictor)
        
        # Test prediction intervals shape
        intervals = predictor(X.iloc[:10])
        assert intervals.shape == (10, 2)
        assert np.all(intervals[:, 0] <= intervals[:, 1])  # lower <= upper
    
    def test_cqr_coverage_guarantee(self, regression_data):
        """Test that CQR provides coverage guarantee."""
        X, y = regression_data
        
        # Split data for testing coverage
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        
        predictor = conformalized_quantile_regression(
            QuantileRegressor,
            X_train, y_train,
            miscoverage_rate=0.1,
            test_size=0.3
        )
        
        intervals = predictor(X_test)
        
        # Check empirical coverage
        in_interval = (y_test >= intervals[:, 0]) & (y_test <= intervals[:, 1])
        empirical_coverage = np.mean(in_interval)
        
        # Should be at least 90% coverage (allowing some tolerance for small samples)
        assert empirical_coverage >= 0.85
    
    def test_cqr_different_coverage_levels(self, regression_data):
        """Test CQR with different coverage levels."""
        X, y = regression_data
        
        for miscoverage_rate in [0.05, 0.1, 0.2]:
            predictor = conformalized_quantile_regression(
                QuantileRegressor,
                X, y,
                miscoverage_rate=miscoverage_rate,
                test_size=0.3
            )
            
            intervals = predictor(X.iloc[:50])
            
            # Higher coverage should give wider intervals
            widths = intervals[:, 1] - intervals[:, 0]
            assert np.all(widths > 0)


# Test RegressionExplainer CQR extensions
class TestRegressionExplainerCQR:
    
    def test_cqr_properties(self, regression_explainer):
        """Test CQR properties on RegressionExplainer."""
        explainer = regression_explainer
        
        # Test prediction_intervals property
        intervals = explainer.prediction_intervals
        assert intervals.shape == (len(explainer.X), 2)
        assert np.all(intervals[:, 0] <= intervals[:, 1])
        
        # Test uncertainty_width property
        widths = explainer.uncertainty_width
        assert len(widths) == len(explainer.X)
        assert np.all(widths > 0)
    
    def test_calculate_prediction_intervals(self, regression_explainer):
        """Test calculate_prediction_intervals method."""
        explainer = regression_explainer
        
        # Test with different parameters
        intervals = explainer.calculate_prediction_intervals(
            miscoverage_rate=0.05,
            test_size=0.2
        )
        
        assert intervals.shape == (len(explainer.X), 2)
        assert explainer._cqr_coverage == 0.95
    
    def test_uncertainty_intervals_df(self, regression_explainer):
        """Test uncertainty_intervals_df method."""
        explainer = regression_explainer
        
        df = explainer.uncertainty_intervals_df()
        
        required_cols = ['prediction', 'lower_bound', 'upper_bound', 
                        'uncertainty_width', 'coverage_level']
        for col in required_cols:
            assert col in df.columns
        
        assert len(df) == len(explainer.X)
        assert np.all(df['lower_bound'] <= df['upper_bound'])
        assert np.all(df['uncertainty_width'] > 0)


# Test plotting functions
class TestCQRPlotting:
    
    def test_plotly_uncertainty_intervals(self, regression_explainer):
        """Test uncertainty intervals plotting function."""
        explainer = regression_explainer
        intervals = explainer.prediction_intervals
        
        fig = plotly_uncertainty_intervals(
            y_true=explainer.y.values,
            predictions=explainer.preds,
            prediction_intervals=intervals,
            feature_values=explainer.X.iloc[:, 0].values,
            feature_name="feature_0",
            target=explainer.target,
            units=explainer.units
        )
        
        # Check that figure was created
        assert fig is not None
        assert len(fig.data) >= 2  # Should have at least interval and points
    
    def test_plotly_uncertainty_width(self, regression_explainer):
        """Test uncertainty width plotting function."""
        explainer = regression_explainer
        intervals = explainer.prediction_intervals
        
        fig = plotly_uncertainty_width(
            prediction_intervals=intervals,
            feature_values=explainer.X.iloc[:, 0].values,
            feature_name="feature_0"
        )
        
        assert fig is not None
        assert len(fig.data) >= 1
    
    def test_plotly_uncertainty_intervals_categorical(self):
        """Test uncertainty intervals plotting function with categorical features."""
        # Create test data with categorical features
        np.random.seed(42)
        n_samples = 100
        
        # Create categorical feature values
        categories = ['Category_A', 'Category_B', 'Category_C']
        categorical_values = np.random.choice(categories, size=n_samples)
        
        # Create dummy data
        y_true = np.random.randn(n_samples)
        predictions = y_true + np.random.randn(n_samples) * 0.1
        intervals = np.column_stack([
            predictions - 1,
            predictions + 1
        ])
        
        # This should not raise a formatting error
        fig = plotly_uncertainty_intervals(
            y_true=y_true,
            predictions=predictions,
            prediction_intervals=intervals,
            feature_values=categorical_values,
            feature_name="categorical_feature",
            target="target"
        )
        
        # Check that figure was created
        assert fig is not None
        assert len(fig.data) >= 2  # Should have at least interval and points
    
    def test_plotly_uncertainty_width_categorical(self):
        """Test uncertainty width plotting function with categorical features."""
        # Create test data with categorical features
        np.random.seed(42)
        n_samples = 100
        
        # Create categorical feature values
        categories = ['Category_A', 'Category_B', 'Category_C']
        categorical_values = np.random.choice(categories, size=n_samples)
        
        # Create dummy interval data
        predictions = np.random.randn(n_samples)
        intervals = np.column_stack([
            predictions - 1,
            predictions + 1
        ])
        
        # This should not raise a formatting error
        fig = plotly_uncertainty_width(
            prediction_intervals=intervals,
            feature_values=categorical_values,
            feature_name="categorical_feature"
        )
        
        assert fig is not None
        assert len(fig.data) >= 1


# Test dashboard components
class TestCQRComponents:
    
    def test_cqr_component_initialization(self, regression_explainer):
        """Test ConformalizedQuantileRegressionComponent initialization."""
        component = ConformalizedQuantileRegressionComponent(
            regression_explainer,
            miscoverage_rate=0.1
        )
        
        assert component.explainer == regression_explainer
        assert component.miscoverage_rate == 0.1
        assert component.slice_feature in regression_explainer.merged_cols
    
    def test_component_layouts(self, regression_explainer):
        """Test that component layouts can be generated."""
        cqr_component = ConformalizedQuantileRegressionComponent(regression_explainer)
        
        # Should not raise exceptions
        cqr_layout = cqr_component.layout()
        
        assert cqr_layout is not None




# Integration tests
class TestCQRIntegration:
    
    def test_end_to_end_workflow(self, regression_data):
        """Test complete CQR workflow from data to visualization."""
        X, y = regression_data
        
        # Train model
        model = RandomForestRegressor(n_estimators=10, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        model.fit(X_train, y_train)
        
        # Create explainer
        explainer = RegressionExplainer(model, X_test, y_test)
        
        # Calculate prediction intervals
        intervals = explainer.calculate_prediction_intervals(miscoverage_rate=0.1)
        
        # Create visualization
        fig = plotly_uncertainty_intervals(
            y_true=explainer.y.values,
            predictions=explainer.preds,
            prediction_intervals=intervals,
            feature_values=explainer.X.iloc[:, 0].values,
            feature_name=explainer.X.columns[0]
        )
        
        # Create dashboard component
        component = ConformalizedQuantileRegressionComponent(explainer)
        
        # All steps should complete without errors
        assert intervals is not None
        assert fig is not None
        assert component is not None
    
    def test_error_handling(self, regression_data):
        """Test error handling in CQR functionality."""
        X, y = regression_data
        
        # Test with classifier (should fail)
        from sklearn.ensemble import RandomForestClassifier
        from explainerdashboard.explainers import ClassifierExplainer
        
        # Create binary classification data
        y_binary = (y > y.median()).astype(int)
        clf_model = RandomForestClassifier(n_estimators=10, random_state=42)
        clf_model.fit(X, y_binary)
        
        clf_explainer = ClassifierExplainer(clf_model, X, y_binary)
        
        # Should raise assertion error for non-regression explainer
        with pytest.raises(AssertionError):
            ConformalizedQuantileRegressionComponent(clf_explainer)


if __name__ == "__main__":
    pytest.main([__file__])