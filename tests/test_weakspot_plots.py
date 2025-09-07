"""
Unit tests for weakspot plotting functions.
"""

import pytest
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.datasets import make_classification, make_regression

from explainerdashboard import ClassifierExplainer, RegressionExplainer
from explainerdashboard.explainer_plots import (
    plotly_weakspot_analysis,
    plotly_weakspot_heatmap
)


@pytest.fixture
def sample_weakspot_data_1d():
    """Create sample 1D weakspot data for testing."""
    return pd.DataFrame({
        'description': ['[0.0, 1.0)', '[1.0, 2.0)', '[2.0, 3.0)', '[3.0, 4.0)'],
        'performance': [0.85, 0.65, 0.90, 0.75],
        'sample_count': [100, 150, 80, 120],
        'is_weak': [False, True, False, False],
        'feature_0_min': [0.0, 1.0, 2.0, 3.0],
        'feature_0_max': [1.0, 2.0, 3.0, 4.0],
        'feature_0_center': [0.5, 1.5, 2.5, 3.5]
    })


@pytest.fixture
def sample_weakspot_data_2d():
    """Create sample 2D weakspot data for testing."""
    return pd.DataFrame({
        'description': [
            '[0.0, 1.0) × [0.0, 1.0)', '[0.0, 1.0) × [1.0, 2.0)',
            '[1.0, 2.0) × [0.0, 1.0)', '[1.0, 2.0) × [1.0, 2.0)'
        ],
        'performance': [0.85, 0.65, 0.90, 0.75],
        'sample_count': [100, 150, 80, 120],
        'is_weak': [False, True, False, False],
        'feature_0_min': [0.0, 0.0, 1.0, 1.0],
        'feature_0_max': [1.0, 1.0, 2.0, 2.0],
        'feature_0_center': [0.5, 0.5, 1.5, 1.5],
        'feature_1_min': [0.0, 1.0, 0.0, 1.0],
        'feature_1_max': [1.0, 2.0, 1.0, 2.0],
        'feature_1_center': [0.5, 1.5, 0.5, 1.5]
    })


@pytest.fixture
def sample_weak_regions():
    """Create sample weak regions data."""
    return [
        {
            'range': '[1.0, 2.0)',
            'performance': 0.65,
            'sample_count': 150,
            'severity': 1.31  # (0.85 / 0.65)
        }
    ]


class TestPlotlyWeakspotAnalysis:
    """Test plotly_weakspot_analysis function."""
    
    def test_1d_plot_creation(self, sample_weakspot_data_1d, sample_weak_regions):
        """Test creation of 1D weakspot analysis plot."""
        fig = plotly_weakspot_analysis(
            weakspot_data=sample_weakspot_data_1d,
            slice_feature='feature_0',
            metric='accuracy',
            threshold_value=0.8,
            weak_regions=sample_weak_regions,
            title="Test Weakspot Analysis"
        )
        
        # Check that figure is created
        assert fig is not None
        
        # Check basic structure
        assert 'data' in fig
        assert 'layout' in fig
        
        # Check that we have traces for performance and threshold
        assert len(fig['data']) >= 2  # At least performance bars and threshold line
        
        # Check layout properties
        assert fig['layout']['title']['text'] == "Test Weakspot Analysis"
        assert 'feature_0' in fig['layout']['xaxis']['title']['text'].lower()
        assert 'accuracy' in fig['layout']['yaxis']['title']['text'].lower()
    
    def test_1d_plot_with_no_weak_regions(self, sample_weakspot_data_1d):
        """Test 1D plot creation when no weak regions are found."""
        # Create data with no weak regions
        data_no_weak = sample_weakspot_data_1d.copy()
        data_no_weak['is_weak'] = False
        
        fig = plotly_weakspot_analysis(
            weakspot_data=data_no_weak,
            slice_feature='feature_0',
            metric='accuracy',
            threshold_value=0.5,  # Very low threshold
            weak_regions=[],
            title="No Weak Regions Test"
        )
        
        assert fig is not None
        assert 'data' in fig
        assert len(fig['data']) >= 1  # Should still have performance bars
    
    def test_1d_plot_different_metrics(self, sample_weakspot_data_1d, sample_weak_regions):
        """Test 1D plot with different metrics."""
        metrics_to_test = ['accuracy', 'log_loss', 'mse', 'mae', 'mape']
        
        for metric in metrics_to_test:
            fig = plotly_weakspot_analysis(
                weakspot_data=sample_weakspot_data_1d,
                slice_feature='feature_0',
                metric=metric,
                threshold_value=0.8,
                weak_regions=sample_weak_regions
            )
            
            assert fig is not None
            assert metric in fig['layout']['yaxis']['title']['text'].lower()
    
    def test_1d_plot_hover_information(self, sample_weakspot_data_1d, sample_weak_regions):
        """Test that hover information is properly configured."""
        fig = plotly_weakspot_analysis(
            weakspot_data=sample_weakspot_data_1d,
            slice_feature='feature_0',
            metric='accuracy',
            threshold_value=0.8,
            weak_regions=sample_weak_regions
        )
        
        # Check that performance bars have hover information
        performance_trace = fig['data'][0]  # First trace should be performance bars
        assert 'hovertemplate' in performance_trace
        
        # Check that hover template includes relevant information
        hover_template = performance_trace['hovertemplate']
        assert 'Range:' in hover_template
        assert 'Performance:' in hover_template
        assert 'Sample Count:' in hover_template
    
    def test_1d_plot_color_coding(self, sample_weakspot_data_1d, sample_weak_regions):
        """Test that weak regions are properly color-coded."""
        fig = plotly_weakspot_analysis(
            weakspot_data=sample_weakspot_data_1d,
            slice_feature='feature_0',
            metric='accuracy',
            threshold_value=0.8,
            weak_regions=sample_weak_regions
        )
        
        # Check that performance bars have color information
        performance_trace = fig['data'][0]
        assert 'marker' in performance_trace
        assert 'color' in performance_trace['marker']
        
        # Colors should be different for weak vs normal regions
        colors = performance_trace['marker']['color']
        assert len(set(colors)) > 1  # Should have at least 2 different colors
    
    def test_1d_plot_empty_data(self):
        """Test 1D plot with empty data."""
        empty_data = pd.DataFrame(columns=[
            'description', 'performance', 'sample_count', 'is_weak',
            'feature_0_min', 'feature_0_max', 'feature_0_center'
        ])
        
        fig = plotly_weakspot_analysis(
            weakspot_data=empty_data,
            slice_feature='feature_0',
            metric='accuracy',
            threshold_value=0.8,
            weak_regions=[],
            title="Empty Data Test"
        )
        
        assert fig is not None
        # Should handle empty data gracefully
        assert 'data' in fig
        assert 'layout' in fig


class TestPlotlyWeakspotHeatmap:
    """Test plotly_weakspot_heatmap function."""
    
    def test_2d_heatmap_creation(self, sample_weakspot_data_2d):
        """Test creation of 2D weakspot heatmap."""
        fig = plotly_weakspot_heatmap(
            weakspot_data=sample_weakspot_data_2d,
            slice_features=['feature_0', 'feature_1'],
            metric='accuracy'
        )
        
        # Check that figure is created
        assert fig is not None
        
        # Check basic structure
        assert 'data' in fig
        assert 'layout' in fig
        
        # Should have heatmap trace
        assert len(fig['data']) >= 1
        
        # Check that it's a heatmap
        heatmap_trace = fig['data'][0]
        assert heatmap_trace['type'] == 'heatmap'
        
        # Check layout properties
        assert 'feature_0' in fig['layout']['xaxis']['title']['text'].lower()
        assert 'feature_1' in fig['layout']['yaxis']['title']['text'].lower()
    
    def test_2d_heatmap_data_structure(self, sample_weakspot_data_2d):
        """Test that heatmap data is properly structured."""
        fig = plotly_weakspot_heatmap(
            weakspot_data=sample_weakspot_data_2d,
            slice_features=['feature_0', 'feature_1'],
            metric='accuracy'
        )
        
        heatmap_trace = fig['data'][0]
        
        # Check that z values (performance) are present
        assert 'z' in heatmap_trace
        assert len(heatmap_trace['z']) > 0
        
        # Check that x and y coordinates are present
        assert 'x' in heatmap_trace
        assert 'y' in heatmap_trace
        assert len(heatmap_trace['x']) > 0
        assert len(heatmap_trace['y']) > 0
    
    def test_2d_heatmap_colorscale(self, sample_weakspot_data_2d):
        """Test heatmap colorscale configuration."""
        fig = plotly_weakspot_heatmap(
            weakspot_data=sample_weakspot_data_2d,
            slice_features=['feature_0', 'feature_1'],
            metric='accuracy'
        )
        
        heatmap_trace = fig['data'][0]
        
        # Check colorscale is configured
        assert 'colorscale' in heatmap_trace
        assert 'colorbar' in heatmap_trace
        
        # Check colorbar title
        colorbar_title = heatmap_trace['colorbar']['title']['text']
        assert 'accuracy' in colorbar_title.lower()
    
    def test_2d_heatmap_hover_information(self, sample_weakspot_data_2d):
        """Test heatmap hover information."""
        fig = plotly_weakspot_heatmap(
            weakspot_data=sample_weakspot_data_2d,
            slice_features=['feature_0', 'feature_1'],
            metric='accuracy'
        )
        
        heatmap_trace = fig['data'][0]
        
        # Check hover information
        assert 'hovertemplate' in heatmap_trace
        hover_template = heatmap_trace['hovertemplate']
        assert 'feature_0' in hover_template.lower()
        assert 'feature_1' in hover_template.lower()
        assert 'performance' in hover_template.lower()
    
    def test_2d_heatmap_empty_data(self):
        """Test 2D heatmap with empty data."""
        empty_data = pd.DataFrame(columns=[
            'description', 'performance', 'sample_count', 'is_weak',
            'feature_0_min', 'feature_0_max', 'feature_0_center',
            'feature_1_min', 'feature_1_max', 'feature_1_center'
        ])
        
        fig = plotly_weakspot_heatmap(
            weakspot_data=empty_data,
            slice_features=['feature_0', 'feature_1'],
            metric='accuracy'
        )
        
        assert fig is not None
        # Should handle empty data gracefully
        assert 'data' in fig
        assert 'layout' in fig
    
    def test_2d_heatmap_different_metrics(self, sample_weakspot_data_2d):
        """Test 2D heatmap with different metrics."""
        metrics_to_test = ['accuracy', 'log_loss', 'mse', 'mae', 'mape']
        
        for metric in metrics_to_test:
            fig = plotly_weakspot_heatmap(
                weakspot_data=sample_weakspot_data_2d,
                slice_features=['feature_0', 'feature_1'],
                metric=metric
            )
            
            assert fig is not None
            heatmap_trace = fig['data'][0]
            colorbar_title = heatmap_trace['colorbar']['title']['text']
            assert metric in colorbar_title.lower()


class TestPlotIntegration:
    """Integration tests for plotting functions with real explainers."""
    
    @pytest.fixture
    def classifier_with_weakspot_data(self):
        """Create classifier explainer with weakspot analysis capability."""
        X, y = make_classification(
            n_samples=500,
            n_features=4,
            n_informative=3,
            n_redundant=1,
            random_state=42
        )
        
        feature_names = [f'feature_{i}' for i in range(X.shape[1])]
        X_df = pd.DataFrame(X, columns=feature_names)
        
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X_df, y)
        
        explainer = ClassifierExplainer(model, X_df, y)
        return explainer
    
    @pytest.fixture
    def regression_with_weakspot_data(self):
        """Create regression explainer with weakspot analysis capability."""
        X, y = make_regression(
            n_samples=500,
            n_features=4,
            n_informative=3,
            noise=0.1,
            random_state=42
        )
        
        feature_names = [f'feature_{i}' for i in range(X.shape[1])]
        X_df = pd.DataFrame(X, columns=feature_names)
        
        model = RandomForestRegressor(n_estimators=10, random_state=42)
        model.fit(X_df, y)
        
        explainer = RegressionExplainer(model, X_df, y)
        return explainer
    
    def test_plot_with_real_classifier_data(self, classifier_with_weakspot_data):
        """Test plotting with real classifier weakspot data."""
        explainer = classifier_with_weakspot_data
        
        # Get real weakspot data
        weakspot_data = explainer.get_weakspot_data(
            slice_features=['feature_0'],
            bins=5,
            min_samples=20
        )
        
        # Get analysis result for weak regions
        result = explainer.calculate_weakspot_analysis(
            slice_features=['feature_0'],
            bins=5,
            min_samples=20
        )
        
        # Create plot
        fig = plotly_weakspot_analysis(
            weakspot_data=weakspot_data,
            slice_feature='feature_0',
            metric=weakspot_data.attrs['metric'],
            threshold_value=weakspot_data.attrs['threshold_value'],
            weak_regions=result['weak_regions'],
            title="Real Classifier Data Test"
        )
        
        assert fig is not None
        assert 'data' in fig
        assert len(fig['data']) > 0
    
    def test_plot_with_real_regression_data(self, regression_with_weakspot_data):
        """Test plotting with real regression weakspot data."""
        explainer = regression_with_weakspot_data
        
        # Get real weakspot data
        weakspot_data = explainer.get_weakspot_data(
            slice_features=['feature_0'],
            bins=5,
            min_samples=20
        )
        
        # Get analysis result for weak regions
        result = explainer.calculate_weakspot_analysis(
            slice_features=['feature_0'],
            bins=5,
            min_samples=20
        )
        
        # Create plot
        fig = plotly_weakspot_analysis(
            weakspot_data=weakspot_data,
            slice_feature='feature_0',
            metric=weakspot_data.attrs['metric'],
            threshold_value=weakspot_data.attrs['threshold_value'],
            weak_regions=result['weak_regions'],
            title="Real Regression Data Test"
        )
        
        assert fig is not None
        assert 'data' in fig
        assert len(fig['data']) > 0
    
    def test_2d_plot_with_real_data(self, classifier_with_weakspot_data):
        """Test 2D plotting with real data."""
        explainer = classifier_with_weakspot_data
        
        # Get real 2D weakspot data
        weakspot_data = explainer.get_weakspot_data(
            slice_features=['feature_0', 'feature_1'],
            bins=3,  # 3x3 grid
            min_samples=10
        )
        
        # Create 2D plot
        fig = plotly_weakspot_heatmap(
            weakspot_data=weakspot_data,
            slice_features=['feature_0', 'feature_1'],
            metric=weakspot_data.attrs['metric']
        )
        
        assert fig is not None
        assert 'data' in fig
        assert len(fig['data']) > 0
        assert fig['data'][0]['type'] == 'heatmap'


class TestPlotErrorHandling:
    """Test error handling in plotting functions."""
    
    def test_1d_plot_invalid_feature(self, sample_weakspot_data_1d, sample_weak_regions):
        """Test 1D plot with invalid feature name."""
        # This should handle gracefully or raise appropriate error
        try:
            fig = plotly_weakspot_analysis(
                weakspot_data=sample_weakspot_data_1d,
                slice_feature='nonexistent_feature',
                metric='accuracy',
                threshold_value=0.8,
                weak_regions=sample_weak_regions
            )
            # If it doesn't raise an error, it should still create a figure
            assert fig is not None
        except (KeyError, ValueError) as e:
            # Expected error for invalid feature
            assert 'feature' in str(e).lower()
    
    def test_2d_plot_invalid_features(self, sample_weakspot_data_2d):
        """Test 2D plot with invalid feature names."""
        try:
            fig = plotly_weakspot_heatmap(
                weakspot_data=sample_weakspot_data_2d,
                slice_features=['nonexistent_feature_0', 'nonexistent_feature_1'],
                metric='accuracy'
            )
            # If it doesn't raise an error, it should still create a figure
            assert fig is not None
        except (KeyError, ValueError) as e:
            # Expected error for invalid features
            assert 'feature' in str(e).lower()
    
    def test_plot_with_nan_values(self, sample_weakspot_data_1d, sample_weak_regions):
        """Test plotting with NaN values in data."""
        # Introduce NaN values
        data_with_nan = sample_weakspot_data_1d.copy()
        data_with_nan.loc[1, 'performance'] = np.nan
        
        fig = plotly_weakspot_analysis(
            weakspot_data=data_with_nan,
            slice_feature='feature_0',
            metric='accuracy',
            threshold_value=0.8,
            weak_regions=sample_weak_regions
        )
        
        # Should handle NaN values gracefully
        assert fig is not None
        assert 'data' in fig
    
    def test_plot_with_extreme_values(self, sample_weakspot_data_1d, sample_weak_regions):
        """Test plotting with extreme values."""
        # Create data with extreme values
        extreme_data = sample_weakspot_data_1d.copy()
        extreme_data.loc[0, 'performance'] = 1e6  # Very large value
        extreme_data.loc[1, 'performance'] = -1e6  # Very small value
        
        fig = plotly_weakspot_analysis(
            weakspot_data=extreme_data,
            slice_feature='feature_0',
            metric='accuracy',
            threshold_value=0.8,
            weak_regions=sample_weak_regions
        )
        
        # Should handle extreme values gracefully
        assert fig is not None
        assert 'data' in fig