import dash
from dash import dcc, html, Input, Output, State, callback_context
import dash_bootstrap_components as dbc
from flask import Flask
import os

# Import the visualization modules
from stats_plotter import create_plotter_layout, register_plotter_callbacks
from cmd_latency_visualizer import DashboardApp

# Initialize Flask server with session support
server = Flask(__name__)
#server.secret_key = 'your-secret-key-change-in-production'
server.secret_key = os.environ.get('SECRET_KEY', os.urandom(24))

# Initialize Dash app
app = dash.Dash(
    __name__,
    server=server,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    suppress_callback_exceptions=True
)

# Main app layout
app.layout = dbc.Container([
    html.H1("Data Visualization Dashboard", className="mt-3 mb-4"),
    
    # Visualization type selector
    dbc.Row([
        dbc.Col([
            html.Label("Select Visualization Type:"),
            dcc.Dropdown(
                id='visualization-type',
                options=[
                    {'label': 'Stats Plotter', 'value': 'plotter'},
                    {'label': 'Command Latency Visualizer', 'value': 'latency'}
                ],
                value=None,
                clearable=False,
                className="mb-4"
            )
        ], width=6)
    ]),
    
    # Container for dynamic content
    html.Div(id='visualization-container')
    
], fluid=True)

# Callback to update the visualization container based on the selected type
@app.callback(
    Output('visualization-container', 'children'),
    Input('visualization-type', 'value')
)
def update_visualization_content(selected_type):
    if selected_type is None:
        return html.Div("Please select a visualization type to begin.", className="mt-4")
    
    if selected_type == 'plotter':
        return create_plotter_layout()
    elif selected_type == 'latency':
        # Return the latency visualizer layout
        latency_app = DashboardApp()
        return latency_app.get_upload_layout()
    
    return html.Div("Invalid visualization type selected.")

# Register the callbacks from the individual modules
if __name__ == '__main__':
    # Register plotter callbacks
    register_plotter_callbacks(app)
    
    # Register latency visualizer callbacks
    latency_app = DashboardApp()
    latency_app.register_callbacks(app)
    
    # Run the app
    app.run(debug=True, host="0.0.0.0", port=8050)
