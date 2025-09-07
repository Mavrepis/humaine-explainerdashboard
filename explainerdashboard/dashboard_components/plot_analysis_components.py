__all__ = [
    "PlotAnalysisComponent",
]

import base64
import re
from dash import html, dcc, Input, Output, State, callback_context
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc

from ..dashboard_methods import *
from .. import to_html


class PlotAnalysisComponent(ExplainerComponent):
    """AI-powered plot analysis component for explainerdashboard.
    
    Enables built-in plot capture from explainer and external image upload,
    providing intelligent insights about model behavior through LLM-based 
    visual analysis.
    """
    _state_props = dict(
        plot_type=("plot-analysis-type-", "value"),
        selected_plots=("plot-analysis-selected-", "value"),
        uploaded_files=("plot-analysis-upload-", "contents"),
        analysis_result=("plot-analysis-result-", "children"),
        api_key=("plot-analysis-apikey-", "value"),
        model_selection=("plot-analysis-model-", "value"),
        pos_label=("pos-label-", "value"),
    )

    def __init__(
        self,
        explainer,
        title="Plot Analysis",
        name=None,
        subtitle="AI-powered visualization insights",
        hide_title=False,
        hide_subtitle=False,
        hide_upload=False,
        hide_capture=False,
        hide_api_config=False,
        hide_model_selector=False,
        hide_selector=False,
        pos_label=None,
        max_file_size=10,  # MB
        supported_formats=None,
        default_model="gpt-4o",
        max_tokens=512,
        description=None,
        **kwargs,
    ):
        """Initialize PlotAnalysisComponent.

        Args:
            explainer (Explainer): explainer object (ClassifierExplainer/RegressionExplainer)
            title (str, optional): Component title. Defaults to "Plot Analysis".
            name (str, optional): unique component identifier. Defaults to None.
            subtitle (str, optional): Component subtitle. Defaults to "AI-powered visualization insights".
            hide_title (bool, optional): Hide title. Defaults to False.
            hide_subtitle (bool, optional): Hide subtitle. Defaults to False.
            hide_upload (bool, optional): Hide external upload functionality. Defaults to False.
            hide_capture (bool, optional): Hide built-in plot capture. Defaults to False.
            hide_api_config (bool, optional): Hide API configuration section. Defaults to False.
            hide_model_selector (bool, optional): Hide LLM model selection. Defaults to False.
            hide_selector (bool, optional): hide pos label selector. Defaults to False.
            pos_label ({int, str}, optional): initial pos label. Defaults to explainer.pos_label
            max_file_size (int, optional): Maximum upload file size (MB). Defaults to 10.
            supported_formats (list, optional): Supported image formats. Defaults to ["png", "jpg", "jpeg"].
            default_model (str, optional): Default LLM model. Defaults to "gpt-4o".
            max_tokens (int, optional): Maximum response tokens. Defaults to 512.
            description (str, optional): Tooltip description. Defaults to None.
        """
        super().__init__(explainer, title, name)
        
        # Store all the configuration attributes
        self.hide_upload = hide_upload
        self.hide_capture = hide_capture
        self.hide_api_config = hide_api_config
        self.hide_model_selector = hide_model_selector
        self.hide_selector = hide_selector
        self.max_file_size = max_file_size
        self.default_model = default_model
        self.max_tokens = max_tokens
        
        if supported_formats is None:
            self.supported_formats = ["png", "jpg", "jpeg"]
        else:
            self.supported_formats = supported_formats
            
        if description is None:
            self.description = """
            Upload images or capture built-in plots to get AI-powered insights about 
            model behavior, feature relationships, and potential biases through advanced 
            visual analysis using Large Language Models.
            """
        else:
            self.description = description
        
        # Initialize selector if classifier
        self.selector = PosLabelSelector(explainer, name=self.name, pos_label=pos_label)
        
        # Get available plots based on explainer type
        self.available_plots = self._get_available_plots()

    def _get_available_plots(self):
        """Get list of available plots based on explainer type."""
        plots = {}
        
        # Common plots for both classifiers and regressors
        plots["Feature Importance"] = [
            "plot_importances (permutation)",
            "plot_importances (shap)",
        ]
        
        plots["SHAP"] = [
            "plot_shap_summary (aggregate)",
            "plot_shap_summary (detailed)", 
            "plot_shap_dependence",
            "plot_shap_interaction_summary",
        ]
        
        if hasattr(self.explainer, 'decision_trees') and self.explainer.decision_trees:
            plots["Decision Trees"] = [
                "plot_decision_trees",
                "plot_decision_path",
            ]
        
        if self.explainer.is_classifier:
            plots["Classification"] = [
                "plot_confusion_matrix",
                "plot_roc_auc",
                "plot_pr_auc", 
                "plot_precision",
                "plot_cumulative_precision",
                "plot_classification",
                "plot_lift_curve",
            ]
        else:
            plots["Regression"] = [
                "plot_predicted_vs_actual",
                "plot_residuals",
                "plot_residuals_vs_feature",
            ]
            
        return plots
    
    def _figure_to_bytes(self, fig):
        """Convert a plotly figure to base64 encoded bytes.
        
        Args:
            fig: Plotly figure object
            
        Returns:
            str: Base64 encoded image data
        """
        try:
            # Try different image export methods for better compatibility
            try:
                # First try kaleido (preferred)
                img_bytes = fig.to_image(format="png", engine="kaleido")
            except Exception as kaleido_error:
                try:
                    # Fallback to orca if available
                    img_bytes = fig.to_image(format="png", engine="orca")
                except Exception as orca_error:
                    # Last resort: show helpful error message
                    raise Exception(
                        f"Image export failed. Please install compatible versions:\n"
                        f"Option 1: pip install -U plotly>=6.1.1 kaleido\n"
                        f"Option 2: pip install plotly==5.24.1 kaleido==0.2.1\n"
                        f"Kaleido error: {str(kaleido_error)}\n"
                        f"Orca error: {str(orca_error)}"
                    )
            
            return base64.b64encode(img_bytes).decode('utf-8')
            
        except Exception as e:
            print(f"Error converting figure to bytes: {str(e)}")
            return None

    def _encode_image(self, image_data):
        """Encode image data to base64."""
        if isinstance(image_data, str) and image_data.startswith('data:image'):
            # Already base64 encoded from upload
            return image_data.split(',')[1]
        elif isinstance(image_data, bytes):
            return base64.b64encode(image_data).decode('utf-8')
        else:
            return str(image_data)
    
    def _format_image_src(self, image_data):
        """Format image data for HTML img src attribute."""
        if isinstance(image_data, str):
            if image_data.startswith('data:image'):
                # Already properly formatted data URL
                return image_data
            else:
                # Assume it's base64 encoded, add data URL prefix
                return f"data:image/png;base64,{image_data}"
        elif isinstance(image_data, bytes):
            # Convert bytes to base64 data URL
            b64_data = base64.b64encode(image_data).decode('utf-8')
            return f"data:image/png;base64,{b64_data}"
        else:
            # Fallback
            return f"data:image/png;base64,{str(image_data)}"
    
    def _create_preview_display(self, images, names, title):
        """Create a preview display for images."""
        if not images:
            return None
            
        return dbc.Card([
            dbc.CardHeader([
                html.H6([
                    html.I(className="fas fa-eye me-2"),
                    title
                ], className="mb-0")
            ]),
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([
                        html.Div([
                            html.Img(
                                src=self._format_image_src(img_data),
                                style={
                                    "width": "100%",
                                    "maxHeight": "300px",
                                    "objectFit": "contain",
                                    "border": "1px solid #dee2e6",
                                    "borderRadius": "8px",
                                    "backgroundColor": "#f8f9fa",
                                    "cursor": "pointer"
                                },
                                className="mb-2",
                                title="Click to view full size"
                            ),
                            html.P(
                                names[i] if i < len(names) else f"Image {i+1}",
                                className="text-center text-muted small mb-0",
                                style={"fontSize": "0.85rem"}
                            )
                        ])
                    ], md=6 if len(images) == 2 else 4 if len(images) <= 3 else 3, className="mb-3")
                    for i, img_data in enumerate(images[:8])  # Limit to 8 images in preview
                ]),
                
                # Show message if more than 8 images
                html.P(
                    f"... and {len(images) - 8} more images",
                    className="text-muted small text-center mb-0"
                ) if len(images) > 8 else None,
                
                html.Hr(className="my-2"),
                html.P([
                    html.I(className="fas fa-info-circle me-1"),
                    f"Ready to analyze {len(images)} image(s). Click 'Analyze Plots' below to get AI insights."
                ], className="text-info small mb-0")
            ])
        ], color="light", outline=True, className="mb-3")
    
    def _markdown_to_html(self, markdown_text):
        """Convert markdown text to HTML components."""
        if not markdown_text:
            return html.P("No analysis available.")
        
        # Split by lines for processing
        lines = markdown_text.strip().split('\n')
        elements = []
        current_list_items = []
        
        for line in lines:
            line = line.strip()
            
            # Skip empty lines unless we're processing a list
            if not line:
                if current_list_items:
                    elements.append(html.Ul(current_list_items, className="mb-3"))
                    current_list_items = []
                continue
            
            # Headers
            if line.startswith('###'):
                if current_list_items:
                    elements.append(html.Ul(current_list_items, className="mb-3"))
                    current_list_items = []
                text = line[3:].strip()
                elements.append(html.H6(text, className="mt-3 mb-2 text-primary"))
            elif line.startswith('##'):
                if current_list_items:
                    elements.append(html.Ul(current_list_items, className="mb-3"))
                    current_list_items = []
                text = line[2:].strip()
                elements.append(html.H5(text, className="mt-3 mb-2 text-primary"))
            elif line.startswith('#'):
                if current_list_items:
                    elements.append(html.Ul(current_list_items, className="mb-3"))
                    current_list_items = []
                text = line[1:].strip()
                elements.append(html.H4(text, className="mt-3 mb-2 text-primary"))
            
            # List items
            elif line.startswith('- ') or line.startswith('* '):
                text = line[2:].strip()
                # Process text formatting
                formatted_text = self._process_text_formatting(text)
                if isinstance(formatted_text, list):
                    current_list_items.append(html.Li(formatted_text))
                else:
                    current_list_items.append(html.Li(formatted_text))
            
            # Numbered lists
            elif re.match(r'^\d+\. ', line):
                if current_list_items:
                    elements.append(html.Ul(current_list_items, className="mb-3"))
                    current_list_items = []
                text = re.sub(r'^\d+\. ', '', line)
                formatted_text = self._process_text_formatting(text)
                if isinstance(formatted_text, list):
                    elements.append(html.P(formatted_text, className="mb-2"))
                else:
                    elements.append(html.P(formatted_text, className="mb-2"))
            
            # Regular paragraphs
            else:
                if current_list_items:
                    elements.append(html.Ul(current_list_items, className="mb-3"))
                    current_list_items = []
                formatted_text = self._process_text_formatting(line)
                if isinstance(formatted_text, list):
                    elements.append(html.P(formatted_text, className="mb-2"))
                else:
                    elements.append(html.P(formatted_text, className="mb-2"))
        
        # Add any remaining list items
        if current_list_items:
            elements.append(html.Ul(current_list_items, className="mb-3"))
        
        return html.Div(elements) if elements else html.P("No analysis available.")
    
    def _process_text_formatting(self, text):
        """Process bold, italic, and other text formatting into proper Dash components."""
        if not text:
            return text
        
        # Split text by formatting patterns and create proper HTML components
        parts = []
        current_text = text
        
        # Process **bold** text
        import re
        bold_pattern = r'\*\*(.*?)\*\*'
        bold_matches = list(re.finditer(bold_pattern, current_text))
        
        if bold_matches:
            last_end = 0
            for match in bold_matches:
                # Add text before bold
                if match.start() > last_end:
                    parts.append(current_text[last_end:match.start()])
                # Add bold text
                parts.append(html.Strong(match.group(1)))
                last_end = match.end()
            # Add remaining text
            if last_end < len(current_text):
                parts.append(current_text[last_end:])
            return parts
        
        # Process *italic* text (if no bold found)
        italic_pattern = r'(?<!\*)\*([^*]+?)\*(?!\*)'
        italic_matches = list(re.finditer(italic_pattern, current_text))
        
        if italic_matches:
            last_end = 0
            for match in italic_matches:
                # Add text before italic
                if match.start() > last_end:
                    parts.append(current_text[last_end:match.start()])
                # Add italic text
                parts.append(html.Em(match.group(1)))
                last_end = match.end()
            # Add remaining text
            if last_end < len(current_text):
                parts.append(current_text[last_end:])
            return parts
        
        # Process `code` text (if no bold or italic found)
        code_pattern = r'`(.*?)`'
        code_matches = list(re.finditer(code_pattern, current_text))
        
        if code_matches:
            last_end = 0
            for match in code_matches:
                # Add text before code
                if match.start() > last_end:
                    parts.append(current_text[last_end:match.start()])
                # Add code text
                parts.append(html.Code(match.group(1), style={"backgroundColor": "#f8f9fa", "padding": "2px 4px", "borderRadius": "3px"}))
                last_end = match.end()
            # Add remaining text
            if last_end < len(current_text):
                parts.append(current_text[last_end:])
            return parts
        
        # Return plain text if no formatting found
        return text

    def _analyze_plots_with_llm(self, api_key, model="gpt-4o", max_tokens=512, images=None, user_prompt=None, history=None):
        """
        Analyze plots using OpenAI API, with support for conversational follow-ups.

        Args:
            api_key (str): Your OpenAI API key.
            model (str): The model to use (e.g., "gpt-4o").
            max_tokens (int): The maximum number of tokens to generate.
            images (list, optional): A list of image data for the initial analysis.
            user_prompt (str, optional): The user's text prompt. If None, a default prompt is used.
            history (list, optional): The existing conversation history. If None, a new conversation is started.

        Returns:
            tuple: A tuple containing the assistant's response (str) and the updated conversation history (list).
        """
        try:
            from openai import OpenAI
            client = OpenAI(api_key=api_key)

            # Initialize history with system prompt if starting a new conversation
            if history is None:
                system_instructions = """
You are a **laconic** (less than 512 tokens) explainable AI (XAI) expert.

# ANALYSIS SCOPE
- Model behavior patterns and feature relationships
- Statistical insights from visualizations
- Potential biases and anomalies
- Performance characteristics

# RULES
1) Provide concise observations on the model and feature relationships
2) DO NOT comment on inputs individually unless critically relevant
3) DO NOT repeat numbers or details already visible in the plots
4) Focus on non-trivial insights about model behavior
5) Highlight potential concerns or notable patterns

# OUTPUT FORMAT
- Start with overall model assessment
- Identify key feature relationships
- Note any concerning patterns or biases
- Suggest areas for further investigation

Focus on actionable insights for model improvement and validation.
"""
                history = [{"role": "system", "content": system_instructions}]

            # Use a default prompt for the initial image analysis if no specific user prompt is given
            current_prompt = user_prompt if user_prompt is not None else \
                "Analyze the provided machine learning visualization(s) to provide insights about the model's behavior, feature relationships, and any notable patterns or potential concerns."

            # Prepare the content for the user's message
            content = [{"type": "text", "text": current_prompt}]
            if images:
                for img in images:
                    b64_img = self._encode_image(img)
                    content.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64_img}"},
                    })
            
            # Create the message list to send to the API
            messages_to_send = history + [{"role": "user", "content": content}]

            response = client.chat.completions.create(
                model=model,
                messages=messages_to_send,
                max_tokens=max_tokens,
            )

            assistant_response = response.choices[0].message.content
            
            # The full history now includes the user's prompt and the assistant's reply
            updated_history = messages_to_send + [{"role": "assistant", "content": assistant_response}]

            return assistant_response, updated_history

        except Exception as e:
            error_message = f"Error analyzing plots: {str(e)}"
            # Return the error and the history before the failed call
            return error_message, history if history is not None else []

    def _capture_built_in_plot(self, plot_type, plot_name, **kwargs):
        """Capture a built-in plot from the explainer."""
        try:
            # Map plot names to explainer methods
            plot_method_map = {
                "plot_importances (permutation)": lambda: self.explainer.plot_importances(kind="permutation", **kwargs),
                "plot_importances (shap)": lambda: self.explainer.plot_importances(kind="shap", **kwargs),
                "plot_shap_summary (aggregate)": lambda: self.explainer.plot_importances(kind="shap", **kwargs),
                "plot_shap_summary (detailed)": lambda: self.explainer.plot_importances_detailed(**kwargs),
                "plot_shap_dependence": lambda: self.explainer.plot_shap_dependence(col=kwargs.get('col', 0), **kwargs),
                "plot_shap_interaction_summary": lambda: self.explainer.plot_shap_interaction_summary(**kwargs),
                "plot_confusion_matrix": lambda: self.explainer.plot_confusion_matrix(**kwargs),
                "plot_roc_auc": lambda: self.explainer.plot_roc_auc(**kwargs),
                "plot_pr_auc": lambda: self.explainer.plot_pr_auc(**kwargs),
                "plot_precision": lambda: self.explainer.plot_precision(**kwargs),
                "plot_cumulative_precision": lambda: self.explainer.plot_cumulative_precision(**kwargs),
                "plot_classification": lambda: self.explainer.plot_classification(**kwargs),
                "plot_lift_curve": lambda: self.explainer.plot_lift_curve(**kwargs),
                "plot_predicted_vs_actual": lambda: self.explainer.plot_predicted_vs_actual(**kwargs),
                "plot_residuals": lambda: self.explainer.plot_residuals(**kwargs),
                "plot_residuals_vs_feature": lambda: self.explainer.plot_residuals_vs_feature(col=kwargs.get('col', 0), **kwargs),
                "plot_decision_trees": lambda: self.explainer.plot_trees(**kwargs),
                "plot_decision_path": lambda: self.explainer.plot_tree_path(index=kwargs.get('index'), **kwargs),
            }
            
            if plot_name in plot_method_map:
                fig = plot_method_map[plot_name]()
                if fig:
                    # DELEGATE to the helper method
                    return self._figure_to_bytes(fig) 
                else:
                    return None
            else:
                return None
                
        except Exception as e:
            print(f"Error capturing plot {plot_name}: {str(e)}")
            return None

    def layout(self):
        """Return the component layout."""
        return dbc.Card(
            [
                make_hideable(
                    dbc.CardHeader(
                        [
                            html.Div(
                                [
                                    html.H3(
                                        self.title, id="plot-analysis-title-" + self.name
                                    ),
                                    make_hideable(
                                        html.H6(
                                            self.subtitle, className="card-subtitle"
                                        ),
                                        hide=self.hide_subtitle,
                                    ),
                                    dbc.Tooltip(
                                        self.description,
                                        target="plot-analysis-title-" + self.name,
                                    ),
                                ]
                            ),
                        ]
                    ),
                    hide=self.hide_title,
                ),
                dbc.CardBody(
                    [
                        # Configuration Row
                        dbc.Row(
                            [
                                make_hideable(
                                    dbc.Col(
                                        [
                                            dbc.Label("OpenAI API Key:"),
                                            dbc.Input(
                                                id="plot-analysis-apikey-" + self.name,
                                                type="password",
                                                placeholder="Enter your OpenAI API key...",
                                                size="sm",
                                            ),
                                            html.Small(
                                                "API key is stored locally and not sent to any server except OpenAI.",
                                                className="text-muted",
                                            ),
                                        ],
                                        md=4,
                                    ),
                                    hide=self.hide_api_config,
                                ),
                                make_hideable(
                                    dbc.Col(
                                        [
                                            dbc.Label("Model:"),
                                            dbc.Select(
                                                id="plot-analysis-model-" + self.name,
                                                options=[
                                                    {"label": "GPT-4o", "value": "gpt-4o"},
                                                    {"label": "GPT-4o mini", "value": "gpt-4o-mini"},
                                                    {"label": "GPT-4 Turbo", "value": "gpt-4-turbo"},
                                                ],
                                                value=self.default_model,
                                                size="sm",
                                            ),
                                        ],
                                        md=2,
                                    ),
                                    hide=self.hide_model_selector,
                                ),
                                make_hideable(
                                    dbc.Col([self.selector.layout()], md=2),
                                    hide=self.hide_selector,
                                ),
                            ],
                            className="mb-3",
                        ),
                        # Input Methods Tabs
                        dcc.Tabs(
                            id="plot-analysis-tabs-" + self.name,
                            value="capture",
                            children=[
                                make_hideable(
                                    dcc.Tab(
                                        label="Capture Built-in Plots",
                                        value="capture",
                                        children=[
                                            html.Div(
                                                [
                                                    dbc.Row(
                                                        [
                                                            dbc.Col(
                                                                [
                                                                    dbc.Label("Plot Category:"),
                                                                    dbc.Select(
                                                                        id="plot-analysis-type-" + self.name,
                                                                        options=[
                                                                            {"label": k, "value": k}
                                                                            for k in self.available_plots.keys()
                                                                        ],
                                                                        value=list(self.available_plots.keys())[0] if self.available_plots else None,
                                                                        size="sm",
                                                                    ),
                                                                ],
                                                                md=4,
                                                            ),
                                                        ],
                                                        className="mb-3 mt-3",
                                                    ),
                                                    dbc.Row(
                                                        [
                                                            dbc.Col(
                                                                [
                                                                    dbc.Label("Select Plots:"),
                                                                    dbc.Checklist(
                                                                        id="plot-analysis-selected-" + self.name,
                                                                        options=[],
                                                                        value=[],
                                                                        inline=False,
                                                                    ),
                                                                ],
                                                                md=12,
                                                            ),
                                                        ],
                                                        className="mb-3",
                                                    ),
                                                ]
                                            )
                                        ],
                                    ),
                                    hide=self.hide_capture,
                                ),
                                make_hideable(
                                    dcc.Tab(
                                        label="Upload External Images",
                                        value="upload",
                                        children=[
                                            html.Div(
                                                [
                                                    dcc.Upload(
                                                        id="plot-analysis-upload-" + self.name,
                                                        children=html.Div(
                                                            [
                                                                html.I(className="fas fa-cloud-upload-alt fa-3x mb-3"),
                                                                html.H5("Drag and Drop or Click to Upload"),
                                                                html.P(
                                                                    f"Supported formats: {', '.join(self.supported_formats).upper()}",
                                                                    className="text-muted",
                                                                ),
                                                                html.P(
                                                                    f"Max file size: {self.max_file_size}MB",
                                                                    className="text-muted",
                                                                ),
                                                            ],
                                                            style={
                                                                "textAlign": "center",
                                                                "padding": "50px",
                                                                "border": "2px dashed #ccc",
                                                                "borderRadius": "10px",
                                                                "margin": "20px 0",
                                                            },
                                                        ),
                                                        style={"width": "100%", "height": "200px"},
                                                        multiple=True,
                                                    ),
                                                    html.Div(id="plot-analysis-upload-status-" + self.name),
                                                ]
                                            )
                                        ],
                                    ),
                                    hide=self.hide_upload,
                                ),
                            ]
                        ),
                        
                        # Preview section
                        html.Div(id="plot-analysis-preview-" + self.name),
                        
                        # Analysis controls and results
                        dbc.Row([
                            dbc.Col([
                                dbc.Button(
                                    "Analyze Plots",
                                    id="plot-analysis-analyze-btn-" + self.name,
                                    color="primary",
                                    size="lg",
                                    className="me-2"
                                ),
                                dbc.Button(
                                    "Clear Results",
                                    id="plot-analysis-clear-btn-" + self.name,
                                    color="secondary",
                                    outline=True,
                                    size="lg"
                                )
                            ], className="text-center my-3")
                        ]),
                        
                        # Results section
                        html.Div(id="plot-analysis-results-" + self.name),
                        
                        # Follow-up section (hidden by default)
                        html.Div([
                            dbc.Card([
                                dbc.CardHeader([
                                    html.H6([
                                        html.I(className="fas fa-comments me-2"),
                                        "Ask Follow-up Questions"
                                    ])
                                ]),
                                dbc.CardBody([
                                    dbc.InputGroup([
                                        dbc.Input(
                                            id="plot-analysis-followup-input-" + self.name,
                                            placeholder="Ask a follow-up question about the analysis...",
                                            type="text"
                                        ),
                                        dbc.Button(
                                            "Send",
                                            id="plot-analysis-followup-btn-" + self.name,
                                            color="primary",
                                            outline=True
                                        )
                                    ])
                                ])
                            ], color="info", outline=True)
                        ], id="plot-analysis-followup-section-" + self.name, style={"display": "none"}, className="mt-3"),
                        
                        # Hidden store for conversation history
                        dcc.Store(id="plot-analysis-history-" + self.name, data=None)
                    ]
                )
            ]
        )
    
    def component_callbacks(self, app):
        """Register component callbacks."""
        
        # Update available plots based on selected category
        @app.callback(
            Output("plot-analysis-selected-" + self.name, "options"),
            [Input("plot-analysis-type-" + self.name, "value")],
        )
        def update_plot_options(plot_type):
            if plot_type and plot_type in self.available_plots:
                return [
                    {"label": plot, "value": plot}
                    for plot in self.available_plots[plot_type]
                ]
            return []

        # Handle file upload status
        @app.callback(
            Output("plot-analysis-upload-status-" + self.name, "children"),
            [Input("plot-analysis-upload-" + self.name, "contents")],
            [State("plot-analysis-upload-" + self.name, "filename")],
        )
        def update_upload_status(contents, filenames):
            if contents is not None:
                if isinstance(contents, list):
                    file_count = len(contents)
                    status_color = "success"
                    status_text = f"✓ {file_count} file(s) uploaded successfully"
                else:
                    status_color = "success"
                    status_text = "✓ 1 file uploaded successfully"
                    
                return dbc.Alert(
                    status_text,
                    color=status_color,
                    className="mt-2",
                )
            return None

        # Plot Preview callback - shows selected plots immediately
        @app.callback(
            Output("plot-analysis-preview-" + self.name, "children"),
            [
                Input("plot-analysis-selected-" + self.name, "value"),
                Input("plot-analysis-upload-" + self.name, "contents"),
                Input("plot-analysis-tabs-" + self.name, "value"),
            ],
            [
                State("plot-analysis-upload-" + self.name, "filename"),
                State("pos-label-" + self.name, "value"),
            ],
        )
        def update_plot_preview(selected_plots, uploaded_contents, active_tab, uploaded_filenames, pos_label):
            if active_tab == "capture" and selected_plots:
                # Show loading indicator while generating plots
                loading_content = dcc.Loading(
                    id="plot-preview-loading-" + self.name,
                    type="default",
                    children=[
                        dbc.Card([
                            dbc.CardBody([
                                html.Div([
                                    html.H5("Generating plot previews...", className="text-center mb-3"),
                                    html.P("Please wait while we create your selected plots.", className="text-center text-muted")
                                ], className="text-center py-4")
                            ])
                        ], color="light", className="mb-3")
                    ]
                )
                
                # Generate preview for built-in plots
                preview_images = []
                plot_names = []
                
                for plot_name in selected_plots:
                    try:
                        plot_kwargs = {"pos_label": pos_label} if pos_label is not None else {}
                        img_data = self._capture_built_in_plot(
                            None, plot_name, **plot_kwargs
                        )
                        if img_data:
                            preview_images.append(img_data)
                            plot_names.append(plot_name)
                    except Exception as e:
                        print(f"Error generating preview for {plot_name}: {e}")
                        continue
                
                if preview_images:
                    return self._create_preview_display(preview_images, plot_names, "Selected Plots Preview")
                else:
                    return dbc.Alert(
                        "⚠️ Unable to generate plot previews. Please check your selections and try again.",
                        color="warning",
                        className="mb-3"
                    )
                    
            elif active_tab == "upload" and uploaded_contents:
                # Generate preview for uploaded images
                if isinstance(uploaded_contents, list):
                    image_list = uploaded_contents
                    names = uploaded_filenames if uploaded_filenames else [f"Image {i+1}" for i in range(len(uploaded_contents))]
                else:
                    image_list = [uploaded_contents]
                    names = [uploaded_filenames] if uploaded_filenames else ["Uploaded Image"]
                
                return self._create_preview_display(image_list, names, "Uploaded Images Preview")
            
            return None

        # Main analysis callback
        @app.callback(
            [
                Output("plot-analysis-results-" + self.name, "children"),
                Output("plot-analysis-history-" + self.name, "data"),
                Output("plot-analysis-followup-section-" + self.name, "style"),
                Output("plot-analysis-analyze-btn-" + self.name, "children"),
                Output("plot-analysis-analyze-btn-" + self.name, "disabled"),
            ],
            [
                Input("plot-analysis-analyze-btn-" + self.name, "n_clicks"),
                Input("plot-analysis-clear-btn-" + self.name, "n_clicks"),
            ],
            [
                State("plot-analysis-tabs-" + self.name, "value"),
                State("plot-analysis-selected-" + self.name, "value"),
                State("plot-analysis-upload-" + self.name, "contents"),
                State("plot-analysis-upload-" + self.name, "filename"),
                State("plot-analysis-apikey-" + self.name, "value"),
                State("plot-analysis-model-" + self.name, "value"),
                State("pos-label-" + self.name, "value"),
                State("plot-analysis-history-" + self.name, "data"),
            ],
        )
        def analyze_or_clear_plots(
            analyze_clicks, clear_clicks, active_tab, selected_plots,
            uploaded_contents, uploaded_filenames, api_key, model, pos_label, history
        ):
            ctx = callback_context
            if not ctx.triggered:
                raise PreventUpdate
                
            button_id = ctx.triggered[0]["prop_id"].split(".")[0]
            
            # Handle clear button
            if button_id == "plot-analysis-clear-btn-" + self.name:
                return None, None, {"display": "none"}, "Analyze Plots", False
                
            # Handle analyze button
            if button_id == "plot-analysis-analyze-btn-" + self.name and analyze_clicks:
                
                # Show loading state immediately
                loading_content = dcc.Loading(
                    id="plot-analysis-loading-" + self.name,
                    type="default",
                    children=[
                        dbc.Card([
                            dbc.CardHeader([
                                html.H5("Analyzing Plots...")
                            ]),
                            dbc.CardBody([
                                html.Div([
                                    html.H6("AI Analysis in Progress", className="text-center mb-3"),
                                    html.P("Please wait while we analyze your plots using AI. This may take a few moments.", 
                                           className="text-center text-muted")
                                ], className="text-center py-4")
                            ])
                        ], color="info", outline=True, className="mt-3")
                    ]
                )
                
                # Validate API key
                if not api_key or not api_key.strip():
                    return dbc.Alert(
                        "⚠️ Please enter your OpenAI API key to proceed with analysis.",
                        color="warning",
                    ), history, {"display": "none"}, "Analyze Plots", False
                
                images_to_analyze = []
                
                try:
                    # Handle built-in plot capture
                    if active_tab == "capture" and selected_plots:
                        for plot_name in selected_plots:
                            plot_kwargs = {"pos_label": pos_label} if pos_label is not None else {}
                            img_data = self._capture_built_in_plot(
                                None, plot_name, **plot_kwargs
                            )
                            if img_data:
                                images_to_analyze.append(img_data)
                    
                    # Handle uploaded images
                    elif active_tab == "upload" and uploaded_contents:
                        if isinstance(uploaded_contents, list):
                            for content in uploaded_contents:
                                images_to_analyze.append(content)
                        else:
                            images_to_analyze.append(uploaded_contents)
                    
                    # Validate we have images to analyze
                    if not images_to_analyze:
                        return dbc.Alert(
                            "⚠️ Please select plots to capture or upload images before analyzing.",
                            color="warning",
                        ), history, {"display": "none"}, "Analyze Plots", False
                    
                    # Perform LLM analysis with conversation support
                    analysis_result, updated_history = self._analyze_plots_with_llm(
                        api_key=api_key,
                        model=(model or self.default_model),
                        max_tokens=self.max_tokens,
                        images=images_to_analyze,
                        history=history  # Pass existing history
                    )
                    
                    # Format and return results - simpler now since images are in preview
                    result_children = [
                        dbc.CardHeader([
                            html.H5([
                                html.I(className="fas fa-robot me-2"),
                                "AI Analysis Results"
                            ])
                        ]),
                        dbc.CardBody([
                            # AI Analysis section - main focus now
                            html.Div([
                                self._markdown_to_html(analysis_result)
                            ], className="border-start border-primary border-3 ps-3"),
                            
                            html.Hr(),
                            
                            html.Small([
                                html.I(className="fas fa-info-circle me-1"),
                                f"Analysis performed using {model or self.default_model} on {len(images_to_analyze)} image(s). ",
                                "Images are displayed in the preview area above."
                            ], className="text-muted"),
                        ])
                    ]
                    
                    result_card = dbc.Card(result_children, color="success", outline=True, className="mt-3")
                    
                    # Show follow-up section after successful analysis
                    followup_style = {"display": "block"}
                    
                    return result_card, updated_history, followup_style, "Analyze Plots", False
                    
                except Exception as e:
                    return dbc.Alert(
                        f"❌ Error during analysis: {str(e)}",
                        color="danger",
                    ), history, {"display": "none"}, "Analyze Plots", False
            
            raise PreventUpdate

        # Follow-up question callback
        @app.callback(
            [
                Output("plot-analysis-results-" + self.name, "children", allow_duplicate=True),
                Output("plot-analysis-history-" + self.name, "data", allow_duplicate=True),
                Output("plot-analysis-followup-input-" + self.name, "value"),
                Output("plot-analysis-followup-btn-" + self.name, "children"),
                Output("plot-analysis-followup-btn-" + self.name, "disabled"),
            ],
            [
                Input("plot-analysis-followup-btn-" + self.name, "n_clicks"),
                Input("plot-analysis-followup-input-" + self.name, "n_submit"),  # Enter key support
            ],
            [
                State("plot-analysis-followup-input-" + self.name, "value"),
                State("plot-analysis-history-" + self.name, "data"),
                State("plot-analysis-apikey-" + self.name, "value"),
                State("plot-analysis-model-" + self.name, "value"),
                State("plot-analysis-results-" + self.name, "children"),
            ],
            prevent_initial_call=True,
        )
        def handle_followup_question(
            send_clicks, input_submit, user_question, history, api_key, model, current_results
        ):
            if not user_question or not user_question.strip():
                raise PreventUpdate
                
            # Show loading state for the button
            loading_button = "Processing..."
                
            if not api_key or not api_key.strip():
                # Add error message to current results
                error_alert = dbc.Alert(
                    "⚠️ API key required for follow-up questions.",
                    color="warning",
                    className="mt-2"
                )
                if current_results:
                    if isinstance(current_results, dict) and "props" in current_results:
                        # Add error to existing results
                        current_results["props"]["children"][1]["props"]["children"].append(error_alert)
                    return current_results, history, "", "Send", False
                return error_alert, history, "", "Send", False
                
            if not history:
                # No conversation history, can't do follow-up
                error_alert = dbc.Alert(
                    "⚠️ Please run an initial analysis before asking follow-up questions.",
                    color="warning",
                    className="mt-2"
                )
                return error_alert, history, "", "Send", False
            
            try:
                # Show loading state in results while processing
                loading_card = dcc.Loading(
                    id="followup-loading-" + self.name,
                    type="default",
                    children=[
                        dbc.Card([
                            dbc.CardBody([
                                html.Div([
                                    html.H6("Processing your question...", className="text-center mb-3"),
                                    html.P(f'Analyzing: "{user_question[:50]}..."' if len(user_question) > 50 else f'Analyzing: "{user_question}"', 
                                           className="text-center text-muted small")
                                ], className="text-center py-3")
                            ])
                        ], color="light", outline=True, className="mt-3")
                    ]
                )
                
                # Perform follow-up analysis
                followup_result, updated_history = self._analyze_plots_with_llm(
                    api_key=api_key,
                    model=(model or self.default_model),
                    max_tokens=self.max_tokens,
                    images=None,  # No new images for follow-up
                    user_prompt=user_question,
                    history=history
                )
                
                # Create the new Q&A card
                new_qa_card = dbc.Card([
                    dbc.CardBody([
                        html.H6([
                            html.I(className="fas fa-user me-2"),
                            "Your Question:"
                        ], className="text-primary mb-2"),
                        html.P(user_question, className="mb-3 font-style-italic"),
                        
                        html.H6([
                            html.I(className="fas fa-robot me-2"),
                            "AI Response:"
                        ], className="text-success mb-2"),
                        html.Div([
                            self._markdown_to_html(followup_result)
                        ], className="border-start border-success border-3 ps-3"),
                    ])
                ], color="light", outline=True, className="mt-3")
                
                # Start with the new card as a list
                updated_children = [new_qa_card]

                # Check if current_results exists and has children to prepend
                if (current_results and 
                    isinstance(current_results, dict) and 
                    "props" in current_results and 
                    "children" in current_results["props"]):
                    
                    existing_children = current_results["props"]["children"]
                    if isinstance(existing_children, list):
                        # Prepend the existing conversation to the new card
                        updated_children = existing_children + updated_children
                    else:
                        # If it's not a list, just prepend the single component
                        updated_children = [existing_children] + updated_children

                return html.Div(updated_children), updated_history, "", "Send", False  # Clear input, reset button

            except Exception as e:
                # Error handling
                error_alert = dbc.Alert(f"❌ Error in follow-up: {str(e)}", color="danger", className="mt-2")
                
                # A slightly cleaner way to append the error
                if current_results:
                    # Check if children is a list, otherwise create one
                    if isinstance(current_results['props']['children'], list):
                        current_results['props']['children'].append(error_alert)
                    else:
                        current_results['props']['children'] = [current_results['props']['children'], error_alert]
                    return current_results, history, "", "Send", False
                else:
                    return error_alert, history, "", "Send", False

    def to_html(self, state_dict=None, add_header=True):
        """Return static HTML representation."""
        html_content = to_html.card(
            "<p>Plot Analysis Component - Interactive features not available in static export.</p>",
            title=self.title,
            subtitle=self.subtitle,
        )
        if add_header:
            return to_html.add_header(html_content)
        return html_content
