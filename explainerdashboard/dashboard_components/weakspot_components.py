"""
Weakspot Analysis Dashboard Components.

This module provides dashboard components for interactive weakspot analysis,
allowing users to identify and visualize data slices where machine learning
models perform significantly worse than their overall performance.
"""

__all__ = [
    "WeakspotComponent",
]

from typing import List, Dict, Union, Optional, Tuple, Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from dash import html, dcc, Input, Output, State
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc

from ..dashboard_methods import *
from .. import to_html


class WeakspotComponent(ExplainerComponent):
    """Interactive weakspot analysis component.
    
    This component provides an interactive interface for performing weakspot
    analysis on machine learning models. Users can select features to analyze,
    configure analysis parameters, and visualize results showing data slices
    where the model performs poorly.
    """
    
    _state_props = dict(
        slice_features=("weakspot-feature-selector-", "value"),
        slice_method=("weakspot-method-selector-", "value"),
        metric=("weakspot-metric-selector-", "value"),
        threshold=("weakspot-threshold-slider-", "value"),
        bins=("weakspot-bins-input-", "value"),
        min_samples=("weakspot-min-samples-input-", "value"),
    )

    def __init__(
        self,
        explainer,
        title="Weakspot Analysis",
        name=None,
        hide_feature_selector=False,
        hide_method_selector=False,
        hide_metric_selector=False,
        hide_threshold_slider=False,
        hide_bins_input=False,
        hide_min_samples_input=False,
        hide_title=False,
        hide_subtitle=False,
        slice_features=None,
        slice_method="histogram",
        bins=10,
        metric=None,
        threshold=1.1,
        min_samples=20,
        description=None,
        **kwargs,
    ):
        """Initialize WeakspotComponent.

        Args:
            explainer (Explainer): explainer object constructed with either
                        ClassifierExplainer() or RegressionExplainer()
            title (str, optional): Title of component. Defaults to "Weakspot Analysis".
            name (str, optional): unique name to add to Component elements.
                        If None then random uuid is generated to make sure
                        it's unique. Defaults to None.
            hide_feature_selector (bool, optional): hide feature selector dropdown. 
                        Defaults to False.
            hide_method_selector (bool, optional): hide slicing method selector. 
                        Defaults to False.
            hide_metric_selector (bool, optional): hide metric selector dropdown. 
                        Defaults to False.
            hide_threshold_slider (bool, optional): hide threshold slider. 
                        Defaults to False.
            hide_bins_input (bool, optional): hide bins input field. 
                        Defaults to False.
            hide_min_samples_input (bool, optional): hide minimum samples input field. 
                        Defaults to False.
            hide_title (bool, optional): hide title. Defaults to False.
            hide_subtitle (bool, optional): hide subtitle. Defaults to False.
            slice_features (list, optional): initial features to analyze. 
                        Defaults to None (first feature).
            slice_method (str, optional): slicing method ('histogram' or 'tree'). 
                        Defaults to 'histogram'.
            bins (int, optional): number of bins for histogram method. 
                        Defaults to 10.
            metric (str, optional): performance metric to use. If None, 
                        auto-detected based on explainer type. Defaults to None.
            threshold (float, optional): threshold multiplier for weakness detection. 
                        Defaults to 1.1.
            min_samples (int, optional): minimum samples required per slice. 
                        Defaults to 20.
            description (str, optional): description to show in tooltip. 
                        Defaults to None.
        """
        super().__init__(explainer, title, name)

        # Set default slice features if not provided
        if slice_features is None:
            slice_features = [self.explainer.columns[0]]
        elif isinstance(slice_features, str):
            slice_features = [slice_features]
        
        # Validate slice features
        if len(slice_features) > 2:
            raise ValueError("Weakspot analysis supports maximum 2 features")
        
        # Auto-detect metric if not provided
        if metric is None:
            # Check if it's a classifier by looking for classification-specific attributes
            if (hasattr(self.explainer, 'pos_label') and 
                self.explainer.pos_label is not None):  # ClassifierExplainer
                metric = 'accuracy'
            elif hasattr(self.explainer, 'labels'):  # Another classifier check
                metric = 'accuracy'
            else:  # RegressionExplainer
                metric = 'mse'
        
        # Store parameters
        self.slice_features = slice_features
        self.slice_method = slice_method
        self.bins = bins
        self.metric = metric
        self.threshold = threshold
        self.min_samples = min_samples
        
        # Set subtitle based on explainer type
        if not hasattr(self, 'subtitle') or self.subtitle is None:
            self.subtitle = "Identify data slices where model performance is poor"
        
        # Set description if not provided
        if description is None:
            description = (
                "Weakspot analysis identifies regions in the feature space where "
                "your model performs significantly worse than its overall performance. "
                "Select features to analyze and adjust parameters to explore different "
                "slicing strategies."
            )
        self.description = description
        
        # Register dependencies
        self.register_dependencies("preds")
        if hasattr(self.explainer, 'pred_probas'):
            self.register_dependencies("pred_probas")

    def layout(self):
        """Generate the component layout."""
        # Get available metrics based on explainer type
        if (hasattr(self.explainer, 'pos_label') and 
            self.explainer.pos_label is not None) or hasattr(self.explainer, 'labels'):  # ClassifierExplainer
            metric_options = [
                {"label": "Accuracy", "value": "accuracy"},
                {"label": "Log Loss", "value": "log_loss"},
                {"label": "Brier Score", "value": "brier_score"},
            ]
        else:  # RegressionExplainer
            metric_options = [
                {"label": "Mean Squared Error", "value": "mse"},
                {"label": "Mean Absolute Error", "value": "mae"},
                {"label": "Mean Absolute Percentage Error", "value": "mape"},
            ]
        
        # Feature selector options (limit to 2 features max)
        feature_options = [
            {"label": col, "value": col} for col in self.explainer.columns
        ]
        
        return dbc.Card(
            [
                make_hideable(
                    dbc.CardHeader(
                        [
                            html.Div(
                                [
                                    html.H3(
                                        self.title,
                                        className="card-title",
                                        id="weakspot-title-" + self.name,
                                    ),
                                    make_hideable(
                                        html.H6(
                                            self.subtitle, 
                                            className="card-subtitle"
                                        ),
                                        hide=self.hide_subtitle,
                                    ),
                                    dbc.Tooltip(
                                        self.description,
                                        target="weakspot-title-" + self.name,
                                    ),
                                ]
                            ),
                        ]
                    ),
                    hide=self.hide_title,
                ),
                dbc.CardBody(
                    [
                        # Parameter controls row
                        dbc.Row(
                            [
                                make_hideable(
                                    dbc.Col(
                                        [
                                            dbc.Label("Features to analyze:"),
                                            dcc.Dropdown(
                                                id="weakspot-feature-selector-" + self.name,
                                                options=feature_options,
                                                value=self.slice_features,
                                                multi=True,
                                                placeholder="Select 1-2 features...",
                                            ),
                                            dbc.FormText(
                                                "Select 1 or 2 features for analysis",
                                                color="muted",
                                            ),
                                        ],
                                        md=3,
                                    ),
                                    hide=self.hide_feature_selector,
                                ),
                                make_hideable(
                                    dbc.Col(
                                        [
                                            dbc.Label("Slicing method:"),
                                            dbc.Select(
                                                id="weakspot-method-selector-" + self.name,
                                                options=[
                                                    {"label": "Histogram", "value": "histogram"},
                                                    {"label": "Decision Tree", "value": "tree"},
                                                ],
                                                value=self.slice_method,
                                                size="sm",
                                            ),
                                            dbc.FormText(
                                                "How to slice the feature space",
                                                color="muted",
                                            ),
                                        ],
                                        md=2,
                                    ),
                                    hide=self.hide_method_selector,
                                ),
                                make_hideable(
                                    dbc.Col(
                                        [
                                            dbc.Label("Performance metric:"),
                                            dbc.Select(
                                                id="weakspot-metric-selector-" + self.name,
                                                options=metric_options,
                                                value=self.metric,
                                                size="sm",
                                            ),
                                            dbc.FormText(
                                                "Metric to evaluate performance",
                                                color="muted",
                                            ),
                                        ],
                                        md=2,
                                    ),
                                    hide=self.hide_metric_selector,
                                ),
                                make_hideable(
                                    dbc.Col(
                                        [
                                            dbc.Label("Weakness threshold:"),
                                            dcc.Slider(
                                                id="weakspot-threshold-slider-" + self.name,
                                                min=1.0,
                                                max=2.0,
                                                step=0.1,
                                                value=self.threshold,
                                                marks={
                                                    1.0: "1.0x",
                                                    1.5: "1.5x",
                                                    2.0: "2.0x",
                                                },
                                                tooltip={"placement": "bottom", "always_visible": True},
                                            ),
                                            dbc.FormText(
                                                "Multiplier for weakness detection",
                                                color="muted",
                                            ),
                                        ],
                                        md=2,
                                    ),
                                    hide=self.hide_threshold_slider,
                                ),
                                make_hideable(
                                    dbc.Col(
                                        [
                                            dbc.Label("Bins:"),
                                            dbc.Input(
                                                id="weakspot-bins-input-" + self.name,
                                                type="number",
                                                min=3,
                                                max=50,
                                                value=self.bins,
                                                size="sm",
                                            ),
                                            dbc.FormText(
                                                "Number of bins (histogram only)",
                                                color="muted",
                                            ),
                                        ],
                                        md=1,
                                    ),
                                    hide=self.hide_bins_input,
                                ),
                            ],
                            className="mb-3",
                        ),
                        # Advanced parameters row
                        dbc.Row(
                            [
                                make_hideable(
                                    dbc.Col(
                                        [
                                            dbc.Label("Min samples per slice:"),
                                            dbc.Input(
                                                id="weakspot-min-samples-input-" + self.name,
                                                type="number",
                                                min=5,
                                                max=1000,
                                                value=self.min_samples,
                                                size="sm",
                                            ),
                                            dbc.FormText(
                                                "Minimum samples required per slice",
                                                color="muted",
                                            ),
                                        ],
                                        md=2,
                                    ),
                                    hide=self.hide_min_samples_input,
                                ),
                            ],
                            className="mb-3",
                        ),
                        # Results area
                        dbc.Row(
                            [
                                dbc.Col(
                                    [
                                        dcc.Loading(
                                            id="weakspot-loading-" + self.name,
                                            children=[
                                                dcc.Graph(
                                                    id="weakspot-graph-" + self.name,
                                                    config=dict(
                                                        modeBarButtonsToRemove=[
                                                            "lasso2d",
                                                            "select2d",
                                                            "autoScale2d",
                                                        ],
                                                        displaylogo=False,
                                                        toImageButtonOptions={
                                                            'format': 'png',
                                                            'filename': 'weakspot_analysis',
                                                            'height': 600,
                                                            'width': 800,
                                                            'scale': 1
                                                        }
                                                    ),
                                                ),
                                            ],
                                            type="dot",
                                            color="#1f77b4",
                                            style={"minHeight": "400px"},
                                        ),
                                    ],
                                    width=12,
                                ),
                            ]
                        ),
                        # Summary area
                        dbc.Row(
                            [
                                dbc.Col(
                                    [
                                        dcc.Loading(
                                            id="weakspot-summary-loading-" + self.name,
                                            children=[
                                                html.Div(
                                                    id="weakspot-summary-" + self.name,
                                                    className="mt-3",
                                                ),
                                            ],
                                            type="default",
                                            color="#1f77b4",
                                        ),
                                    ],
                                    width=12,
                                ),
                            ]
                        ),
                    ]
                ),
            ]
        )

    def component_callbacks(self, app):
        """Register component callbacks."""
        
        # Callback for dynamic metric selector based on explainer type
        @app.callback(
            Output("weakspot-metric-selector-" + self.name, "options"),
            [Input("weakspot-metric-selector-" + self.name, "id")],
            prevent_initial_call=False,
        )
        def update_metric_options(_):
            """Update metric options based on explainer type."""
            # Check if it's a classifier by looking for classification-specific attributes
            if (hasattr(self.explainer, 'pos_label') and 
                self.explainer.pos_label is not None) or hasattr(self.explainer, 'labels'):
                return [
                    {"label": "Accuracy", "value": "accuracy"},
                    {"label": "Log Loss", "value": "log_loss"},
                    {"label": "Brier Score", "value": "brier_score"},
                ]
            else:  # RegressionExplainer
                return [
                    {"label": "Mean Squared Error", "value": "mse"},
                    {"label": "Mean Absolute Error", "value": "mae"},
                    {"label": "Mean Absolute Percentage Error", "value": "mape"},
                ]
        
        # Main callback for weakspot analysis updates
        @app.callback(
            [
                Output("weakspot-graph-" + self.name, "figure"),
                Output("weakspot-summary-" + self.name, "children"),
            ],
            [
                Input("weakspot-feature-selector-" + self.name, "value"),
                Input("weakspot-method-selector-" + self.name, "value"),
                Input("weakspot-metric-selector-" + self.name, "value"),
                Input("weakspot-threshold-slider-" + self.name, "value"),
                Input("weakspot-bins-input-" + self.name, "value"),
                Input("weakspot-min-samples-input-" + self.name, "value"),
            ],
            prevent_initial_call=False,
        )
        def update_weakspot_analysis(
            slice_features, slice_method, metric, threshold, bins, min_samples
        ):
            """Update weakspot analysis based on parameter changes."""
            
            # Input validation
            validation_error = self._validate_inputs(
                slice_features, slice_method, metric, threshold, bins, min_samples
            )
            if validation_error:
                return validation_error
            
            try:
                # Sanitize inputs
                bins = max(3, min(50, bins or 10))
                min_samples = max(5, min(1000, min_samples or 20))
                threshold = max(1.0, min(5.0, threshold or 1.1))
                
                # Validate features exist in explainer
                invalid_features = [f for f in slice_features if f not in self.explainer.columns]
                if invalid_features:
                    return self._create_error_response(
                        f"Invalid features: {', '.join(invalid_features)}",
                        "Please select valid features from the dropdown."
                    )
                
                # Perform weakspot analysis
                result = self.explainer.calculate_weakspot_analysis(
                    slice_features=slice_features,
                    slice_method=slice_method,
                    bins=bins,
                    metric=metric,
                    threshold=threshold,
                    min_samples=min_samples,
                )
                
                # Get formatted data for plotting
                weakspot_data = self.explainer.get_weakspot_data(
                    slice_features=slice_features,
                    slice_method=slice_method,
                    bins=bins,
                    metric=metric,
                    threshold=threshold,
                    min_samples=min_samples,
                )
                
                # Generate plot based on number of features
                if len(slice_features) == 1:
                    fig = self._create_1d_plot(
                        weakspot_data, slice_features[0], metric, result
                    )
                else:  # 2 features
                    fig = self._create_2d_plot(
                        weakspot_data, slice_features, metric
                    )
                
                # # Generate summary
                # summary_component = self._create_summary_component(
                #     result, slice_features, metric
                # )
                
                return fig, None
                
            except ValueError as e:
                # Handle validation errors
                fig_error,_ =  self._create_error_response(
                    "Validation Error",
                    str(e)
                )
                return fig_error,None

            except Exception as e:
                # Handle unexpected errors
                fig_error,_ =  self._create_error_response(
                    "Analysis Error",
                    f"An unexpected error occurred: {str(e)}"
                )
                return fig_error,None
    
    def _validate_inputs(
        self, 
        slice_features: Optional[List[str]], 
        slice_method: Optional[str], 
        metric: Optional[str], 
        threshold: Optional[float], 
        bins: Optional[int], 
        min_samples: Optional[int]
    ) -> Optional[Tuple[Dict, Any]]:
        """Validate callback inputs and return error response if invalid.
        
        Args:
            slice_features: List of selected features
            slice_method: Selected slicing method
            metric: Selected performance metric
            threshold: Weakness threshold value
            bins: Number of bins for histogram method
            min_samples: Minimum samples per slice
            
        Returns:
            Tuple of (figure_dict, summary_component) if validation fails, None if valid
        """
        
        # Check if features are selected
        if not slice_features:
            return self._create_empty_state_response(
                "Please select at least one feature to analyze",
                "Select features from the dropdown above"
            )
        
        # Check maximum features
        if len(slice_features) > 2:
            return self._create_error_response(
                "Too Many Features",
                "Please select at most 2 features for analysis"
            )
        
        # Check if metric is provided
        if not metric:
            return self._create_error_response(
                "No Metric Selected",
                "Please select a performance metric"
            )
        
        # Check parameter ranges
        if bins is not None and (bins < 3 or bins > 50):
            return self._create_error_response(
                "Invalid Bins Parameter",
                "Number of bins must be between 3 and 50"
            )
        
        if min_samples is not None and (min_samples < 5 or min_samples > 1000):
            return self._create_error_response(
                "Invalid Min Samples Parameter",
                "Minimum samples must be between 5 and 1000"
            )
        
        if threshold is not None and (threshold < 1.0 or threshold > 5.0):
            return self._create_error_response(
                "Invalid Threshold Parameter",
                "Threshold must be between 1.0 and 5.0"
            )
        
        return None  # No validation errors
    
    def to_html(self, state_dict=None, add_header=True):
        """Generate static HTML representation of the component."""
        args = self.get_state_args(state_dict)
        
        try:
            # Perform weakspot analysis with current state
            result = self.explainer.calculate_weakspot_analysis(
                slice_features=args.get('slice_features', self.slice_features),
                slice_method=args.get('slice_method', self.slice_method),
                bins=args.get('bins', self.bins),
                metric=args.get('metric', self.metric),
                threshold=args.get('threshold', self.threshold),
                min_samples=args.get('min_samples', self.min_samples),
            )
            
            # Get formatted data for plotting
            weakspot_data = self.explainer.get_weakspot_data(
                slice_features=args.get('slice_features', self.slice_features),
                slice_method=args.get('slice_method', self.slice_method),
                bins=args.get('bins', self.bins),
                metric=args.get('metric', self.metric),
                threshold=args.get('threshold', self.threshold),
                min_samples=args.get('min_samples', self.min_samples),
            )
            
            # Generate plot
            slice_features = args.get('slice_features', self.slice_features)
            if len(slice_features) == 1:
                fig = self._create_1d_plot(
                    weakspot_data, slice_features[0], args.get('metric', self.metric), result
                )
            else:  # 2 features
                fig = self._create_2d_plot(
                    weakspot_data, slice_features, args.get('metric', self.metric)
                )
            
            # # Generate summary
            # summary = self.explainer.weakspot_summary(
            #     slice_features=args.get('slice_features', self.slice_features),
            #     slice_method=args.get('slice_method', self.slice_method),
            #     bins=args.get('bins', self.bins),
            #     metric=args.get('metric', self.metric),
            #     threshold=args.get('threshold', self.threshold),
            #     min_samples=args.get('min_samples', self.min_samples),
            # )
            
            # Create HTML content
            html_content = f"""
            <div class="card">
                <div class="card-header">
                    <h3>{self.title}</h3>
                    <h6 class="card-subtitle text-muted">{self.subtitle}</h6>
                </div>
                <div class="card-body">
                    <div class="row mb-3">
                        <div class="col-12">
                            <p><strong>Features:</strong> {', '.join(slice_features)}</p>
                            <p><strong>Method:</strong> {args.get('slice_method', self.slice_method)}</p>
                            <p><strong>Metric:</strong> {args.get('metric', self.metric)}</p>
                            <p><strong>Threshold:</strong> {args.get('threshold', self.threshold):.1f}x</p>
                        </div>
                    </div>
                    <div class="row">
                        <div class="col-12">
                            {to_html.plotly_html(fig)}
                        </div>
                    </div>
                    <div class="row mt-3">
                        <div class="col-12">
                        </div>
                    </div>
                </div>
            </div>
            """
            
        except Exception as e:
            # Fallback HTML for errors
            html_content = f"""
            <div class="card">
                <div class="card-header">
                    <h3>{self.title}</h3>
                </div>
                <div class="card-body">
                    <div class="alert alert-warning">
                        <h5>Analysis Error</h5>
                        <p>Unable to generate weakspot analysis: {str(e)}</p>
                        <p>Please check your data and parameters.</p>
                    </div>
                </div>
            </div>
            """
        
        if add_header:
            return to_html.add_header(html_content)
        return html_content
    
    def _create_1d_plot(self, weakspot_data: pd.DataFrame, slice_feature: str, metric: str, result: Dict) -> go.Figure:
        """Create 1D weakspot analysis plot.
        
        Args:
            weakspot_data: DataFrame with weakspot analysis results
            slice_feature: Name of the feature being analyzed
            metric: Performance metric name
            result: Dictionary containing analysis results
            
        Returns:
            Plotly figure for 1D weakspot visualization
        """
        from ..explainer_plots import plotly_weakspot_analysis
        
        return plotly_weakspot_analysis(
            weakspot_data=weakspot_data,
            slice_feature=slice_feature,
            metric=metric,
            threshold_value=result.get('threshold_value', 1.1),
            weak_regions=result.get('weak_regions', []),
            title=f"Weakspot Analysis: {slice_feature}",
        )
    
    def _create_2d_plot(self, weakspot_data: pd.DataFrame, slice_features: List[str], metric: str) -> go.Figure:
        """Create 2D weakspot analysis plot.
        
        Args:
            weakspot_data: DataFrame with weakspot analysis results
            slice_features: List of two feature names being analyzed
            metric: Performance metric name
            
        Returns:
            Plotly figure for 2D weakspot heatmap visualization
        """
        from ..explainer_plots import plotly_weakspot_heatmap
        
        return plotly_weakspot_heatmap(
            weakspot_data=weakspot_data,
            slice_features=slice_features,
            metric=metric,
            title=f"Weakspot Analysis: {slice_features[0]} vs {slice_features[1]}",
        )
    
    # def _create_summary_component(self, result: Dict, slice_features: List[str], metric: str) -> dbc.Alert:
    #     """Create summary component from analysis results.
        
    #     Args:
    #         result: Dictionary containing weakspot analysis results
    #         slice_features: List of feature names analyzed
    #         metric: Performance metric name
            
    #     Returns:
    #         Dash Bootstrap Alert component with analysis summary
    #     """
    #     return None
        # try:
        #     # Generate summary text
        #     summary = self.explainer.weakspot_summary(
        #         slice_features=slice_features,
        #         slice_method=result.get('slice_method', 'histogram'),
        #         bins=result.get('bins', 10),
        #         metric=metric,
        #         threshold=result.get('threshold_value', 1.1),
        #         min_samples=result.get('min_samples', 20),
        #     )
            
        #     # Determine alert color based on findings
        #     weak_regions = result.get('weak_regions', [])
        #     alert_color = "warning" if weak_regions else "success"
            
        #     return dbc.Alert(
        #         [
        #             html.H5("Analysis Summary", className="alert-heading"),
        #             html.P(summary),
        #             html.Hr() if weak_regions else None,
        #             html.P(
        #                 f"Found {len(weak_regions)} weak region(s)" if weak_regions 
        #                 else "No weak regions detected",
        #                 className="mb-0"
        #             ) if weak_regions or not weak_regions else None,
        #         ],
        #         color=alert_color,
        #         className="mt-3",
        #     )
        # except Exception as e:
        #     # Fallback summary if summary generation fails
        #     return dbc.Alert(
        #         [
        #             html.H5("Analysis Complete", className="alert-heading"),
        #             html.P("Analysis completed successfully, but summary generation failed."),
        #             html.P(f"Error: {str(e)}", className="text-muted small"),
        #         ],
        #         color="info",
        #         className="mt-3",
        #     )
    
    def _create_error_response(self, error_title: str, error_message: str) -> Tuple[Dict, dbc.Alert]:
        """Create error response with figure and summary.
        
        Args:
            error_title: Title for the error message
            error_message: Detailed error message
            
        Returns:
            Tuple of (error_figure_dict, error_alert_component)
        """
        error_fig = {
            "data": [],
            "layout": {
                "title": error_title,
                "xaxis": {"visible": False},
                "yaxis": {"visible": False},
                "annotations": [
                    {
                        "text": error_message,
                        "xref": "paper",
                        "yref": "paper",
                        "x": 0.5,
                        "y": 0.5,
                        "xanchor": "center",
                        "yanchor": "middle",
                        "showarrow": False,
                        "font": {"size": 14, "color": "red"},
                    }
                ],
                "plot_bgcolor": "white",
            },
        }
        
        error_summary = dbc.Alert(
            [
                html.H5(error_title, className="alert-heading"),
                html.P(error_message),
            ],
            color="danger",
            className="mt-3",
        )
        
        return error_fig, error_summary
    
    def _create_empty_state_response(self, title: str, message: str) -> Tuple[Dict, dbc.Alert]:
        """Create empty state response when no features are selected.
        
        Args:
            title: Title for the empty state message
            message: Detailed message for the empty state
            
        Returns:
            Tuple of (empty_figure_dict, info_alert_component)
        """
        empty_fig = {
            "data": [],
            "layout": {
                "title": title,
                "xaxis": {"visible": False},
                "yaxis": {"visible": False},
                "annotations": [
                    {
                        "text": message,
                        "xref": "paper",
                        "yref": "paper",
                        "x": 0.5,
                        "y": 0.5,
                        "xanchor": "center",
                        "yanchor": "middle",
                        "showarrow": False,
                        "font": {"size": 16, "color": "gray"},
                    }
                ],
                "plot_bgcolor": "white",
            },
        }
        
        return empty_fig, ""