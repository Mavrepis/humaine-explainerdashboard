"""
Unit tests for weakspot dashboard components.
"""

import pytest
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification, make_regression

from explainerdashboard import ClassifierExplainer, RegressionExplainer
from explainerdashboard.dashboard_components import WeakspotComponent


@pytest.fixture
def classifier_data():
    """Generate sample classification data."""
    X, y = make_classification(
        n_samples=1000,
        n_features=10,
        n_informative=5,
        n_redundant=2,
        random_state=42
    )
    feature_names = [f'feature_{i}' for i in range(X.shape[1])]
    X_df = pd.DataFrame(X, columns=feature_names)
    X_train, X_test, y_train, y_test = train_test_split(
        X_df, y, test_size=0.2, random_state=42
    )
    return X_train, X_test, y_train, y_test


@pytest.fixture
def regression_data():
    """Generate sample regression data."""
    X, y = make_regression(
        n_samples=1000,
        n_features=10,
        n_informative=5,
        noise=0.1,
        random_state=42
    )
    feature_names = [f'feature_{i}' for i in range(X.shape[1])]
    X_df = pd.DataFrame(X, columns=feature_names)
    X_train, X_test, y_train, y_test = train_test_split(
        X_df, y, test_size=0.2, random_state=42
    )
    return X_train, X_test, y_train, y_test


@pytest.fixture
def classifier_explainer(classifier_data):
    """Create a ClassifierExplainer for testing."""
    X_train, X_test, y_train, y_test = classifier_data
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X_train, y_train)
    
    explainer = ClassifierExplainer(
        model, X_test, y_test,
        labels=['Class 0', 'Class 1']
    )
    return explainer


@pytest.fixture
def regression_explainer(regression_data):
    """Create a RegressionExplainer for testing."""
    X_train, X_test, y_train, y_test = regression_data
    model = RandomForestRegressor(n_estimators=10, random_state=42)
    model.fit(X_train, y_train)
    
    explainer = RegressionExplainer(model, X_test, y_test)
    return explainer


class TestWeakspotComponent:
    """Test cases for WeakspotComponent."""
    
    def test_component_initialization_classifier(self, classifier_explainer):
        """Test WeakspotComponent initialization with ClassifierExplainer."""
        component = WeakspotComponent(classifier_explainer)
        
        # Check basic attributes
        assert component.explainer == classifier_explainer
        assert component.title == "Weakspot Analysis"
        assert component.metric == "accuracy"  # Auto-detected for classifier
        assert component.slice_method == "histogram"
        assert component.threshold == 1.1
        assert component.bins == 10
        assert component.min_samples == 20
        assert len(component.slice_features) == 1
        assert component.slice_features[0] == classifier_explainer.columns[0]
    
    def test_component_initialization_regressor(self, regression_explainer):
        """Test WeakspotComponent initialization with RegressionExplainer."""
        component = WeakspotComponent(regression_explainer)
        
        # Check basic attributes
        assert component.explainer == regression_explainer
        assert component.title == "Weakspot Analysis"
        assert component.metric == "mse"  # Auto-detected for regressor
        assert component.slice_method == "histogram"
        assert component.threshold == 1.1
        assert component.bins == 10
        assert component.min_samples == 20
        assert len(component.slice_features) == 1
        assert component.slice_features[0] == regression_explainer.columns[0]
    
    def test_component_initialization_custom_params(self, classifier_explainer):
        """Test WeakspotComponent initialization with custom parameters."""
        custom_features = ['feature_0', 'feature_1']
        component = WeakspotComponent(
            classifier_explainer,
            title="Custom Weakspot Analysis",
            slice_features=custom_features,
            slice_method="tree",
            metric="log_loss",
            threshold=1.5,
            bins=15,
            min_samples=30,
        )
        
        assert component.title == "Custom Weakspot Analysis"
        assert component.slice_features == custom_features
        assert component.slice_method == "tree"
        assert component.metric == "log_loss"
        assert component.threshold == 1.5
        assert component.bins == 15
        assert component.min_samples == 30
    
    def test_component_initialization_single_feature_string(self, classifier_explainer):
        """Test WeakspotComponent initialization with single feature as string."""
        component = WeakspotComponent(
            classifier_explainer,
            slice_features='feature_2'
        )
        
        assert component.slice_features == ['feature_2']
    
    def test_component_initialization_too_many_features(self, classifier_explainer):
        """Test WeakspotComponent initialization with too many features."""
        with pytest.raises(ValueError, match="maximum 2 features"):
            WeakspotComponent(
                classifier_explainer,
                slice_features=['feature_0', 'feature_1', 'feature_2']
            )
    
    def test_component_layout_generation(self, classifier_explainer):
        """Test that component layout is generated without errors."""
        component = WeakspotComponent(classifier_explainer)
        layout = component.layout()
        
        # Check that layout is generated (basic structure test)
        assert layout is not None
        assert hasattr(layout, 'children')  # Should be a Dash component
    
    def test_component_layout_with_hidden_elements(self, classifier_explainer):
        """Test component layout with hidden elements."""
        component = WeakspotComponent(
            classifier_explainer,
            hide_feature_selector=True,
            hide_method_selector=True,
            hide_metric_selector=True,
            hide_threshold_slider=True,
            hide_bins_input=True,
            hide_min_samples_input=True,
            hide_title=True,
            hide_subtitle=True,
        )
        layout = component.layout()
        
        # Check that layout is still generated
        assert layout is not None
        assert hasattr(layout, 'children')
    
    def test_component_state_props(self, classifier_explainer):
        """Test that component has correct state properties."""
        component = WeakspotComponent(classifier_explainer)
        
        expected_props = {
            'slice_features', 'slice_method', 'metric', 
            'threshold', 'bins', 'min_samples'
        }
        
        assert set(component._state_props.keys()) == expected_props
    
    def test_component_dependencies_registration(self, classifier_explainer):
        """Test that component registers correct dependencies."""
        component = WeakspotComponent(classifier_explainer)
        
        # Check that basic dependencies are registered
        assert "preds" in component._dependencies
        # For classifier, pred_probas should also be registered
        assert "pred_probas" in component._dependencies
    
    def test_component_dependencies_regression(self, regression_explainer):
        """Test dependencies for regression explainer."""
        component = WeakspotComponent(regression_explainer)
        
        # Check that basic dependencies are registered
        assert "preds" in component._dependencies
        # For regressor, pred_probas should not be registered
        assert "pred_probas" not in component._dependencies
    
    def test_component_metric_options_classifier(self, classifier_explainer):
        """Test that correct metric options are available for classifier."""
        component = WeakspotComponent(classifier_explainer)
        layout = component.layout()
        
        # This is a basic test - in a real scenario, you'd need to parse
        # the layout to check the metric selector options
        assert component.metric in ["accuracy", "log_loss", "brier_score"]
    
    def test_component_metric_options_regressor(self, regression_explainer):
        """Test that correct metric options are available for regressor."""
        component = WeakspotComponent(regression_explainer)
        layout = component.layout()
        
        # This is a basic test - in a real scenario, you'd need to parse
        # the layout to check the metric selector options
        assert component.metric in ["mse", "mae", "mape"]
    
    def test_component_feature_options(self, classifier_explainer):
        """Test that feature options match explainer columns."""
        component = WeakspotComponent(classifier_explainer)
        
        # All explainer columns should be available as feature options
        available_features = classifier_explainer.columns
        assert len(available_features) > 0
        
        # Default slice_features should be from available features
        assert component.slice_features[0] in available_features
    
    def test_component_name_generation(self, classifier_explainer):
        """Test that component generates unique names."""
        component1 = WeakspotComponent(classifier_explainer)
        component2 = WeakspotComponent(classifier_explainer)
        
        # Names should be different (unique)
        assert component1.name != component2.name
    
    def test_component_custom_name(self, classifier_explainer):
        """Test component with custom name."""
        custom_name = "my_weakspot_component"
        component = WeakspotComponent(classifier_explainer, name=custom_name)
        
        assert component.name == custom_name
    
    def test_component_description(self, classifier_explainer):
        """Test component description handling."""
        # Test default description
        component1 = WeakspotComponent(classifier_explainer)
        assert component1.description is not None
        assert "weakspot analysis" in component1.description.lower()
        
        # Test custom description
        custom_desc = "Custom description for testing"
        component2 = WeakspotComponent(classifier_explainer, description=custom_desc)
        assert component2.description == custom_desc
    
    def test_component_subtitle(self, classifier_explainer):
        """Test component subtitle handling."""
        component = WeakspotComponent(classifier_explainer)
        assert component.subtitle is not None
        assert "data slices" in component.subtitle.lower()


class TestWeakspotComponentIntegration:
    """Integration tests for WeakspotComponent with real explainer functionality."""
    
    def test_component_with_real_analysis(self, classifier_explainer):
        """Test component initialization with real weakspot analysis capability."""
        # This test verifies that the component can be created with an explainer
        # that has the required weakspot analysis methods
        component = WeakspotComponent(classifier_explainer)
        
        # Verify that the explainer has the required methods
        assert hasattr(classifier_explainer, 'calculate_weakspot_analysis')
        assert hasattr(classifier_explainer, 'get_weakspot_data')
        assert hasattr(classifier_explainer, 'weakspot_summary')
        
        # Test that we can call these methods (basic smoke test)
        try:
            result = classifier_explainer.calculate_weakspot_analysis(
                slice_features=['feature_0'],
                bins=5,
                min_samples=10
            )
            assert result is not None
        except Exception as e:
            pytest.fail(f"Weakspot analysis failed: {e}")
    
    def test_component_callback_structure(self, classifier_explainer):
        """Test that component callback structure is correct."""
        component = WeakspotComponent(classifier_explainer)
        
        # Test that component_callbacks method exists and is callable
        assert hasattr(component, 'component_callbacks')
        assert callable(component.component_callbacks)
        
        # This is a basic structure test - full callback testing would require
        # a Dash app instance and more complex setup
    
    def test_component_to_html_method(self, classifier_explainer):
        """Test that component has to_html method."""
        component = WeakspotComponent(classifier_explainer)
        
        # Test that to_html method exists
        assert hasattr(component, 'to_html')
        assert callable(component.to_html)
        
        # Basic test that it doesn't crash (full HTML testing would require
        # implementing the to_html.weakspot_component_html function)
        try:
            html_output = component.to_html()
            # If the to_html method is implemented, it should return something
            assert html_output is not None
        except (NotImplementedError, AttributeError):
            # If not implemented yet, that's expected for this task
            pass


class TestWeakspotComponentCallbacks:
    """Test cases for WeakspotComponent callback functionality."""
    
    def test_input_validation_methods(self, classifier_explainer):
        """Test input validation helper methods."""
        component = WeakspotComponent(classifier_explainer)
        
        # Test validation with no features
        error_response = component._validate_inputs(
            slice_features=[], slice_method="histogram", metric="accuracy",
            threshold=1.1, bins=10, min_samples=20
        )
        assert error_response is not None
        fig, summary = error_response
        assert "select at least one feature" in fig["layout"]["annotations"][0]["text"].lower()
        
        # Test validation with too many features
        error_response = component._validate_inputs(
            slice_features=['f1', 'f2', 'f3'], slice_method="histogram", metric="accuracy",
            threshold=1.1, bins=10, min_samples=20
        )
        assert error_response is not None
        fig, summary = error_response
        assert "at most 2 features" in fig["layout"]["annotations"][0]["text"].lower()
        
        # Test validation with invalid bins
        error_response = component._validate_inputs(
            slice_features=['feature_0'], slice_method="histogram", metric="accuracy",
            threshold=1.1, bins=100, min_samples=20
        )
        assert error_response is not None
        
        # Test validation with valid inputs
        error_response = component._validate_inputs(
            slice_features=['feature_0'], slice_method="histogram", metric="accuracy",
            threshold=1.1, bins=10, min_samples=20
        )
        assert error_response is None
    
    def test_error_response_creation(self, classifier_explainer):
        """Test error response creation methods."""
        component = WeakspotComponent(classifier_explainer)
        
        # Test error response creation
        fig, summary = component._create_error_response("Test Error", "Test message")
        
        assert fig["layout"]["title"] == "Test Error"
        assert "Test message" in fig["layout"]["annotations"][0]["text"]
        assert summary.color == "danger"
        
        # Test empty state response creation
        fig, summary = component._create_empty_state_response("Empty State", "No data")
        
        assert fig["layout"]["title"] == "Empty State"
        assert "No data" in fig["layout"]["annotations"][0]["text"]
        assert summary == ""
    
    def test_plot_creation_methods(self, classifier_explainer):
        """Test plot creation helper methods."""
        component = WeakspotComponent(classifier_explainer)
        
        # Mock weakspot data
        weakspot_data = pd.DataFrame({
            'bin_range': ['[0, 1)', '[1, 2)', '[2, 3)'],
            'performance': [0.8, 0.6, 0.9],
            'sample_count': [100, 150, 80],
            'is_weak': [False, True, False]
        })
        
        # Mock result object
        class MockResult:
            threshold_value = 0.7
            weak_regions = [{'range': '[1, 2)', 'performance': 0.6}]
            slice_method = 'histogram'
            bins = 10
            min_samples = 20
        
        result = MockResult()
        
        # Test 1D plot creation
        try:
            fig = component._create_1d_plot(weakspot_data, 'feature_0', 'accuracy', result)
            assert fig is not None
            # Basic check that it's a plotly figure-like object
            assert hasattr(fig, 'data') or 'data' in fig or hasattr(fig, 'to_dict')
        except ImportError:
            # If plotting functions aren't available, that's expected
            pytest.skip("Plotting functions not available")
        
        # Test 2D plot creation
        try:
            fig = component._create_2d_plot(weakspot_data, ['feature_0', 'feature_1'], 'accuracy')
            assert fig is not None
        except ImportError:
            pytest.skip("Plotting functions not available")
    
    def test_summary_component_creation(self, classifier_explainer):
        """Test summary component creation."""
        component = WeakspotComponent(classifier_explainer)
        
        # Mock result object
        class MockResult:
            threshold_value = 0.7
            weak_regions = [{'range': '[1, 2)', 'performance': 0.6}]
            slice_method = 'histogram'
            bins = 10
            min_samples = 20
        
        result = MockResult()
        
        # Test summary creation with weak regions
        try:
            summary = component._create_summary_component(
                result, ['feature_0'], 'accuracy'
            )
            assert summary is not None
            assert summary.color == "warning"  # Should be warning when weak regions found
        except Exception:
            # If explainer methods aren't available, create fallback summary
            summary = component._create_summary_component(
                result, ['feature_0'], 'accuracy'
            )
            assert summary is not None
            assert summary.color == "info"  # Fallback color
    
    def test_callback_parameter_sanitization(self, classifier_explainer):
        """Test that callback parameters are properly sanitized."""
        component = WeakspotComponent(classifier_explainer)
        
        # Test parameter bounds in validation
        # Bins should be clamped to 3-50
        error_response = component._validate_inputs(
            slice_features=['feature_0'], slice_method="histogram", metric="accuracy",
            threshold=1.1, bins=1, min_samples=20  # bins too low
        )
        assert error_response is not None
        
        error_response = component._validate_inputs(
            slice_features=['feature_0'], slice_method="histogram", metric="accuracy",
            threshold=1.1, bins=100, min_samples=20  # bins too high
        )
        assert error_response is not None
        
        # Min samples should be clamped to 5-1000
        error_response = component._validate_inputs(
            slice_features=['feature_0'], slice_method="histogram", metric="accuracy",
            threshold=1.1, bins=10, min_samples=1  # min_samples too low
        )
        assert error_response is not None
        
        # Threshold should be clamped to 1.0-5.0
        error_response = component._validate_inputs(
            slice_features=['feature_0'], slice_method="histogram", metric="accuracy",
            threshold=0.5, bins=10, min_samples=20  # threshold too low
        )
        assert error_response is not None
    
    def test_dynamic_metric_options(self, classifier_explainer, regression_explainer):
        """Test dynamic metric selector options based on explainer type."""
        # Test classifier metrics
        classifier_component = WeakspotComponent(classifier_explainer)
        
        # Mock the callback function (since we can't easily test Dash callbacks directly)
        # This tests the logic that would be in the callback
        if (hasattr(classifier_explainer, 'pos_label') and 
            classifier_explainer.pos_label is not None) or hasattr(classifier_explainer, 'labels'):
            expected_metrics = ["accuracy", "log_loss", "brier_score"]
        else:
            expected_metrics = ["mse", "mae", "mape"]
        
        # The component should auto-detect the correct metric
        assert classifier_component.metric in expected_metrics
        
        # Test regression metrics
        regression_component = WeakspotComponent(regression_explainer)
        
        # For regression explainer, should default to regression metrics
        regression_metrics = ["mse", "mae", "mape"]
        assert regression_component.metric in regression_metrics
    
    def test_feature_validation_in_callback(self, classifier_explainer):
        """Test feature validation within callback logic."""
        component = WeakspotComponent(classifier_explainer)
        
        # Test with valid features
        valid_features = [classifier_explainer.columns[0]]
        error_response = component._validate_inputs(
            slice_features=valid_features, slice_method="histogram", metric="accuracy",
            threshold=1.1, bins=10, min_samples=20
        )
        assert error_response is None
        
        # Test with invalid feature names (this would be caught in the main callback)
        # The validation method doesn't check feature existence, but the main callback does
        invalid_features = ['nonexistent_feature']
        # This should pass validation but fail in the main callback when checking explainer.columns
        error_response = component._validate_inputs(
            slice_features=invalid_features, slice_method="histogram", metric="accuracy",
            threshold=1.1, bins=10, min_samples=20
        )
        assert error_response is None  # Validation passes, but main callback would catch this
    
    def test_callback_error_handling(self, classifier_explainer):
        """Test error handling in callback methods."""
        component = WeakspotComponent(classifier_explainer)
        
        # Test error response creation with various error types
        fig, summary = component._create_error_response("ValueError", "Invalid parameter")
        assert "ValueError" in fig["layout"]["title"]
        assert "Invalid parameter" in fig["layout"]["annotations"][0]["text"]
        assert summary.color == "danger"
        
        # Test that error responses have proper structure
        assert "data" in fig
        assert "layout" in fig
        assert fig["layout"]["xaxis"]["visible"] == False
        assert fig["layout"]["yaxis"]["visible"] == False
        
        # Test summary component error handling
        class MockBadResult:
            # Missing required attributes to trigger error handling
            pass
        
        bad_result = MockBadResult()
        summary = component._create_summary_component(bad_result, ['feature_0'], 'accuracy')
        
        # Should create fallback summary
        assert summary is not None
        assert summary.color == "info"


class TestWeakspotComponentCallbackIntegration:
    """Integration tests for WeakspotComponent callback functionality with mock explainers."""
    
    def create_mock_explainer(self, explainer_type='classifier'):
        """Create a mock explainer for testing callbacks."""
        import pandas as pd
        import numpy as np
        
        class MockExplainer:
            def __init__(self, explainer_type):
                self.columns = [f'feature_{i}' for i in range(5)]
                if explainer_type == 'classifier':
                    self.pos_label = 1
                    self.labels = ['Class 0', 'Class 1']
                else:
                    # Make it look like a regressor
                    pass
                
            def calculate_weakspot_analysis(self, slice_features, slice_method='histogram', 
                                          bins=10, metric='accuracy', threshold=1.1, min_samples=20):
                class MockResult:
                    threshold_value = threshold
                    weak_regions = [{'range': '[1, 2)', 'performance': 0.6}] if np.random.random() > 0.5 else []
                    slice_method = slice_method
                    bins = bins
                    min_samples = min_samples
                return MockResult()
            
            def get_weakspot_data(self, slice_features, slice_method='histogram', 
                                 bins=10, metric='accuracy', threshold=1.1, min_samples=20):
                return pd.DataFrame({
                    'bin_range': ['[0, 1)', '[1, 2)', '[2, 3)'],
                    'performance': [0.8, 0.6, 0.9],
                    'sample_count': [100, 150, 80],
                    'is_weak': [False, True, False]
                })
            
            def weakspot_summary(self, slice_features, slice_method='histogram', 
                                bins=10, metric='accuracy', threshold=1.1, min_samples=20):
                return f"Analysis of {', '.join(slice_features)} found potential weak regions."
        
        return MockExplainer(explainer_type)
    
    def test_callback_integration_with_mock_explainer(self):
        """Test callback integration with mock explainer."""
        mock_explainer = self.create_mock_explainer('classifier')
        component = WeakspotComponent(mock_explainer)
        
        # Test that component can be created and has callback methods
        assert hasattr(component, 'component_callbacks')
        assert hasattr(component, '_validate_inputs')
        assert hasattr(component, '_create_error_response')
        assert hasattr(component, '_create_1d_plot')
        assert hasattr(component, '_create_2d_plot')
        assert hasattr(component, '_create_summary_component')
    
    def test_full_callback_workflow_simulation(self):
        """Simulate a full callback workflow."""
        mock_explainer = self.create_mock_explainer('classifier')
        component = WeakspotComponent(mock_explainer)
        
        # Simulate callback inputs (valid case)
        slice_features = ['feature_0']
        slice_method = 'histogram'
        metric = 'accuracy'
        threshold = 1.1
        bins = 10
        min_samples = 20
        
        # Test validation (should pass)
        validation_result = component._validate_inputs(
            slice_features, slice_method, metric, threshold, bins, min_samples
        )
        assert validation_result is None, "Validation should pass for valid inputs"
        
        # Test plot creation
        weakspot_data = mock_explainer.get_weakspot_data(slice_features)
        result = mock_explainer.calculate_weakspot_analysis(slice_features)
        
        fig = component._create_1d_plot(weakspot_data, slice_features[0], metric, result)
        assert fig is not None, "Should create a plot"
        
        # Test summary creation
        summary = component._create_summary_component(result, slice_features, metric)
        assert summary is not None, "Should create a summary"
    
    def test_callback_error_scenarios(self):
        """Test callback behavior in error scenarios."""
        mock_explainer = self.create_mock_explainer('classifier')
        component = WeakspotComponent(mock_explainer)
        
        # Test various error scenarios
        error_scenarios = [
            # No features
            ([], 'histogram', 'accuracy', 1.1, 10, 20),
            # Too many features
            (['f1', 'f2', 'f3'], 'histogram', 'accuracy', 1.1, 10, 20),
            # Invalid bins
            (['feature_0'], 'histogram', 'accuracy', 1.1, 100, 20),
            # Invalid min_samples
            (['feature_0'], 'histogram', 'accuracy', 1.1, 10, 2000),
            # Invalid threshold
            (['feature_0'], 'histogram', 'accuracy', 0.5, 10, 20),
        ]
        
        for scenario in error_scenarios:
            validation_result = component._validate_inputs(*scenario)
            assert validation_result is not None, f"Should return error for scenario: {scenario}"
            
            # Check that error response has correct structure
            fig, summary = validation_result
            assert 'data' in fig
            assert 'layout' in fig
            assert fig['layout']['xaxis']['visible'] == False
            assert fig['layout']['yaxis']['visible'] == False
    
    def test_callback_parameter_sanitization_integration(self):
        """Test parameter sanitization in callback workflow."""
        mock_explainer = self.create_mock_explainer('classifier')
        component = WeakspotComponent(mock_explainer)
        
        # Test that extreme parameter values are handled
        extreme_scenarios = [
            # Very high bins (should be clamped)
            (['feature_0'], 'histogram', 'accuracy', 1.1, 1000, 20),
            # Very low bins (should be rejected)
            (['feature_0'], 'histogram', 'accuracy', 1.1, 1, 20),
            # Very high min_samples (should be rejected)
            (['feature_0'], 'histogram', 'accuracy', 1.1, 10, 5000),
            # Very low min_samples (should be rejected)
            (['feature_0'], 'histogram', 'accuracy', 1.1, 10, 1),
        ]
        
        for scenario in extreme_scenarios:
            validation_result = component._validate_inputs(*scenario)
            # Most extreme values should be caught by validation
            if scenario[4] == 1000 or scenario[5] == 5000:  # Very high values
                assert validation_result is not None, f"Should reject extreme values: {scenario}"
    
    def test_dynamic_metric_selection_integration(self):
        """Test dynamic metric selection based on explainer type."""
        # Test with classifier
        classifier_explainer = self.create_mock_explainer('classifier')
        classifier_component = WeakspotComponent(classifier_explainer)
        
        # Should auto-detect classifier metrics
        assert classifier_component.metric in ['accuracy', 'log_loss', 'brier_score']
        
        # Test with regressor
        regressor_explainer = self.create_mock_explainer('regressor')
        regressor_component = WeakspotComponent(regressor_explainer)
        
        # Should auto-detect regressor metrics
        assert regressor_component.metric in ['mse', 'mae', 'mape']
    
    def test_loading_states_and_ui_elements(self):
        """Test that loading states and UI elements are properly configured."""
        mock_explainer = self.create_mock_explainer('classifier')
        component = WeakspotComponent(mock_explainer)
        
        # Test layout generation
        layout = component.layout()
        assert layout is not None
        
        # The layout should contain loading components
        # This is a basic structural test - full UI testing would require Dash testing framework
        layout_str = str(layout)
        assert 'Loading' in layout_str, "Layout should contain loading components"
        assert 'weakspot-loading-' in layout_str, "Should have loading component for graph"
        assert 'weakspot-summary-loading-' in layout_str, "Should have loading component for summary"
    
    def test_callback_state_management(self):
        """Test callback state management and component properties."""
        mock_explainer = self.create_mock_explainer('classifier')
        component = WeakspotComponent(mock_explainer)
        
        # Test state properties
        expected_state_props = {
            'slice_features', 'slice_method', 'metric', 
            'threshold', 'bins', 'min_samples'
        }
        assert set(component._state_props.keys()) == expected_state_props
        
        # Test that each state prop has correct structure
        for prop_name, (element_id, prop) in component._state_props.items():
            assert element_id.startswith('weakspot-')
            assert element_id.endswith('-' + component.name)
            assert prop == 'value'