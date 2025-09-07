"""Integration tests for WeakspotComponent dashboard integration."""

import pytest

from explainerdashboard import ExplainerDashboard
from explainerdashboard.custom import *

import dash_bootstrap_components as dbc
from dash import html

pytestmark = pytest.mark.selenium


class WeakspotCustomDashboard(ExplainerComponent):
    """Custom dashboard demonstrating WeakspotComponent integration."""
    
    def __init__(self, explainer, title="Weakspot Analysis Dashboard", name=None):
        super().__init__(explainer, title, name)
        
        # Create weakspot component
        self.weakspot = WeakspotComponent(
            explainer,
            name=self.name + "_weakspot",
            hide_title=True,
            slice_features=explainer.columns[:1],  # Start with first feature
        )
        
        # Add a performance overview for comparison
        if hasattr(explainer, 'pos_label'):  # Classifier
            self.performance = ClassificationStatsComponent(
                explainer,
                name=self.name + "_stats",
                hide_title=True,
            )
        else:  # Regression
            self.performance = RegressionStatsComponent(
                explainer,
                name=self.name + "_stats", 
                hide_title=True,
            )

    def layout(self):
        return dbc.Container([
            dbc.Row([
                dbc.Col([
                    html.H1("Weakspot Analysis Integration"),
                    html.P("This dashboard demonstrates WeakspotComponent integration with other components."),
                ])
            ]),
            dbc.Row([
                dbc.Col([
                    html.H3("Model Performance Overview"),
                    self.performance.layout(),
                ], width=6),
                dbc.Col([
                    html.H3("Weakspot Analysis"),
                    self.weakspot.layout(),
                ], width=6),
            ]),
        ])


def test_weakspot_component_in_custom_dashboard_classifier(
    dash_duo, precalculated_rf_classifier_explainer
):
    """Test WeakspotComponent integration in custom dashboard with classifier."""
    custom_dashboard = WeakspotCustomDashboard(
        precalculated_rf_classifier_explainer, 
        name="weakspot_custom"
    )
    
    db = ExplainerDashboard(
        precalculated_rf_classifier_explainer,
        [custom_dashboard],
        title="Weakspot Integration Test",
        responsive=False,
    )
    
    # Test HTML generation
    html = db.to_html()
    assert html.startswith("\n<!DOCTYPE html>\n<html"), (
        "failed to generate dashboard to_html with WeakspotComponent"
    )
    
    # Test dashboard startup
    dash_duo.start_server(db.app)
    dash_duo.wait_for_text_to_equal("h1", "Weakspot Integration Test", timeout=30)
    
    # Check that weakspot component elements are present
    dash_duo.wait_for_element("#weakspot-feature-selector-weakspot_custom_weakspot", timeout=10)
    dash_duo.wait_for_element("#weakspot-graph-weakspot_custom_weakspot", timeout=10)
    
    assert dash_duo.get_logs() == [], "browser console should contain no error"


def test_weakspot_component_in_custom_dashboard_regression(
    dash_duo, precalculated_rf_regression_explainer
):
    """Test WeakspotComponent integration in custom dashboard with regression."""
    custom_dashboard = WeakspotCustomDashboard(
        precalculated_rf_regression_explainer,
        name="weakspot_regression"
    )
    
    db = ExplainerDashboard(
        precalculated_rf_regression_explainer,
        [custom_dashboard],
        title="Weakspot Regression Test",
        responsive=False,
    )
    
    # Test HTML generation
    html = db.to_html()
    assert html.startswith("\n<!DOCTYPE html>\n<html"), (
        "failed to generate dashboard to_html with WeakspotComponent for regression"
    )
    
    # Test dashboard startup
    dash_duo.start_server(db.app)
    dash_duo.wait_for_text_to_equal("h1", "Weakspot Regression Test", timeout=30)
    
    # Check that weakspot component elements are present
    dash_duo.wait_for_element("#weakspot-feature-selector-weakspot_regression_weakspot", timeout=10)
    dash_duo.wait_for_element("#weakspot-graph-weakspot_regression_weakspot", timeout=10)
    
    assert dash_duo.get_logs() == [], "browser console should contain no error"


def test_weakspot_component_standalone_classifier(
    dash_duo, precalculated_rf_classifier_explainer
):
    """Test standalone WeakspotComponent with classifier."""
    weakspot_component = WeakspotComponent(
        precalculated_rf_classifier_explainer,
        name="standalone_weakspot",
        title="Standalone Weakspot Analysis"
    )
    
    db = ExplainerDashboard(
        precalculated_rf_classifier_explainer,
        [weakspot_component],
        title="Standalone Weakspot Test",
        responsive=False,
    )
    
    # Test HTML generation
    html = db.to_html()
    assert html.startswith("\n<!DOCTYPE html>\n<html"), (
        "failed to generate dashboard to_html with standalone WeakspotComponent"
    )
    
    # Test dashboard startup
    dash_duo.start_server(db.app)
    dash_duo.wait_for_text_to_equal("h1", "Standalone Weakspot Test", timeout=30)
    
    # Verify component title is displayed
    dash_duo.wait_for_text_to_equal("#weakspot-title-standalone_weakspot", "Standalone Weakspot Analysis", timeout=10)
    
    assert dash_duo.get_logs() == [], "browser console should contain no error"


def test_weakspot_component_standalone_regression(
    dash_duo, precalculated_rf_regression_explainer
):
    """Test standalone WeakspotComponent with regression."""
    weakspot_component = WeakspotComponent(
        precalculated_rf_regression_explainer,
        name="standalone_regression",
        title="Regression Weakspot Analysis"
    )
    
    db = ExplainerDashboard(
        precalculated_rf_regression_explainer,
        [weakspot_component],
        title="Standalone Regression Weakspot Test",
        responsive=False,
    )
    
    # Test HTML generation
    html = db.to_html()
    assert html.startswith("\n<!DOCTYPE html>\n<html"), (
        "failed to generate dashboard to_html with standalone regression WeakspotComponent"
    )
    
    # Test dashboard startup
    dash_duo.start_server(db.app)
    dash_duo.wait_for_text_to_equal("h1", "Standalone Regression Weakspot Test", timeout=30)
    
    # Verify component title is displayed
    dash_duo.wait_for_text_to_equal("#weakspot-title-standalone_regression", "Regression Weakspot Analysis", timeout=10)
    
    assert dash_duo.get_logs() == [], "browser console should contain no error"


def test_weakspot_component_with_existing_components(
    dash_duo, precalculated_rf_classifier_explainer
):
    """Test WeakspotComponent integration alongside existing dashboard components."""
    
    # Create a mix of existing and new components
    components = [
        ShapSummaryComponent(precalculated_rf_classifier_explainer, name="shap_summary"),
        WeakspotComponent(precalculated_rf_classifier_explainer, name="weakspot_mixed"),
        ConfusionMatrixComponent(precalculated_rf_classifier_explainer, name="confusion"),
    ]
    
    db = ExplainerDashboard(
        precalculated_rf_classifier_explainer,
        components,
        title="Mixed Components Test",
        responsive=False,
    )
    
    # Test HTML generation
    html = db.to_html()
    assert html.startswith("\n<!DOCTYPE html>\n<html"), (
        "failed to generate dashboard to_html with mixed components including WeakspotComponent"
    )
    
    # Test dashboard startup
    dash_duo.start_server(db.app)
    dash_duo.wait_for_text_to_equal("h1", "Mixed Components Test", timeout=30)
    
    # Check that all component elements are present
    dash_duo.wait_for_element("#shap-summary-graph-shap_summary", timeout=10)
    dash_duo.wait_for_element("#weakspot-graph-weakspot_mixed", timeout=10)
    dash_duo.wait_for_element("#confusion-matrix-graph-confusion", timeout=10)
    
    assert dash_duo.get_logs() == [], "browser console should contain no error"