import dash
from dash import dcc, html, Input, Output, State, callback_context
import dash_bootstrap_components as dbc
from flask import Flask
import os
from datetime import datetime

# Import your visualization modules
from stats_plotter import create_plotter_layout, register_plotter_callbacks
from cmd_latency_visualizer import DashboardApp
from perf_log_visualizer import PerfLogVisualizer

# Initialize Flask server
server = Flask(__name__)
server.secret_key = os.environ.get('SECRET_KEY', os.urandom(24))

# Initialize Dash app
app = dash.Dash(
    __name__,
    server=server,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    suppress_callback_exceptions=True
)

# Create instances of visualizers
plotter_layout = create_plotter_layout()
latency_app = DashboardApp()
perf_log_visualizer = PerfLogVisualizer()

# Main app layout
app.layout = dbc.Container([
    dcc.Store(id='visualization-selection-store', storage_type='session', data={'type': 'none'}),
    dcc.Store(id='session-timestamp', storage_type='session', data=str(datetime.now())),

    html.H1("Data Visualization Dashboard", className="mt-3 mb-4"),

    dbc.Row([
        dbc.Col([
            html.Label("Select Visualization Type:"),
            dcc.Dropdown(
                id='visualization-type',
                options=[
                    {'label': 'Select...', 'value': 'none'},
                    {'label': 'Stats Plotter', 'value': 'plotter'},
                    {'label': 'Command Latency Visualizer', 'value': 'latency'},
                    {'label': 'Performance Log Visualizer', 'value': 'perfLog'}
                ],
                value='none',
                clearable=False,
                className="mb-4"
            )
        ], width=6)
    ]),

    html.Div(id='visualization-container')
], fluid=True)

# Update store based on dropdown
@app.callback(
    Output('visualization-selection-store', 'data'),
    Input('visualization-type', 'value'),
    State('visualization-selection-store', 'data')
)
def update_selection_store(selected_type, current_data):
    if not selected_type or selected_type == 'none':
        return {'type': 'none'}
    current_data = current_data or {}
    current_data['type'] = selected_type
    return current_data

# Update visualizer based on store
@app.callback(
    Output('visualization-container', 'children'),
    Input('visualization-selection-store', 'data'),
    Input('session-timestamp', 'data'),
    prevent_initial_call=False
)
def update_visualization_content(store_data, timestamp):
    ctx = callback_context
    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else None

    if triggered_id == 'session-timestamp' and ctx.inputs_list[0] is not None:
        raise dash.exceptions.PreventUpdate

    if not store_data or store_data.get('type') == 'none':
        return html.Div("Please select a visualization type to begin.", className="mt-4")

    selected_type = store_data['type']
    print(f"Loading visualization: {selected_type}")

    if selected_type == 'plotter':
        return plotter_layout
    elif selected_type == 'latency':
        return latency_app.get_upload_layout()
    elif selected_type == 'perfLog':
        return perf_log_visualizer.get_upload_layout()

    return html.Div("Invalid visualization type selected.")

# Register all visualizer callbacks
if __name__ == '__main__':
    register_plotter_callbacks(app)
    latency_app.register_callbacks(app)
    perf_log_visualizer.register_callbacks(app)

    app.run(debug=True, host="0.0.0.0", port=8052)
