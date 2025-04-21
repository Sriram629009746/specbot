import dash
from dash import dcc, html, Input, Output, State, callback
import pandas as pd
import json
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import base64
import io
import dash_bootstrap_components as dbc


class Plotter:
    def __init__(self, df, variables_to_plot):
        self.df = df
        self.variables_to_plot = variables_to_plot

    def plot(self):
        # Create a subplot figure with as many rows as variables
        num_vars = len(self.variables_to_plot)
        fig = make_subplots(rows=num_vars, cols=1,
                            subplot_titles=self.variables_to_plot,
                            vertical_spacing=0.1)

        # Add a trace for each variable
        for i, var in enumerate(self.variables_to_plot, 1):
            fig.add_trace(
                go.Scatter(
                    x=self.df['index'],
                    y=self.df[var],
                    mode='lines+markers',
                    name=var
                ),
                row=i, col=1
            )

            # Update y-axis title for each subplot
            fig.update_yaxes(title_text=var, row=i, col=1)

        # Update layout - remove fixed width to allow stretching
        fig.update_layout(
            height=300 * num_vars,  # Scale height based on number of variables
            title_text="Metrics vs Index",
            showlegend=False,
            hovermode="x unified",
            margin=dict(l=50, r=50, t=100, b=50),  # Add some margin
            autosize=True  # Allow autosize
        )

        # Update x-axis title only for the bottom plot
        fig.update_xaxes(title_text="Index", row=num_vars, col=1)

        return fig


# Function to parse the uploaded csv file
def parse_contents(contents):
    if contents is None:
        return None
    
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    
    try:
        df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
        return df
    except Exception as e:
        return None


# Define the layout function
def create_plotter_layout():
    # Hardcoded configuration for products
    config = {
        "Product_A": ["read_count", "write_count", "temperature"],
        "Product_B": ["read_count", "write_count", "erase_count", "errors"],
        "Product_C": ["read_count", "write_count", "erase_count", "temperature", "errors"]
    }
    
    options = [{'label': product, 'value': product} for product in config.keys()]
    
    layout = html.Div([
        html.H3("Stats Plotter", className="mt-3 mb-3"),
        
        # File upload component
        dbc.Row([
            dbc.Col([
                dcc.Upload(
                    id='plotter-upload-data',
                    children=html.Div([
                        'Drag and Drop or ',
                        html.A('Select a CSV File')
                    ]),
                    style={
                        'width': '100%',
                        'height': '60px',
                        'lineHeight': '60px',
                        'borderWidth': '1px',
                        'borderStyle': 'dashed',
                        'borderRadius': '5px',
                        'textAlign': 'center',
                        'margin': '10px'
                    },
                    multiple=False,
                    className="upload-area"
                ),
                # Add status message area
                html.Div(id='plotter-upload-status', className="mt-2")
            ], width=12)
        ]),
        
        # Product dropdown
        dbc.Row([
            dbc.Col([
                html.Label("Select Product:"),
                dcc.Dropdown(
                    id='plotter-product-dropdown',
                    options=options,
                    disabled=True,
                    placeholder="Select a product",
                    className="mb-3"
                )
            ], width=12)
        ]),
        
        # Submit button
        dbc.Row([
            dbc.Col([
                dbc.Button(
                    "Generate Plot",
                    id="plotter-submit-button",
                    color="primary",
                    className="mb-3",
                    disabled=True
                )
            ], width={"size": 3, "offset": 0})
        ]),
        
        # Store for the uploaded data
        dcc.Store(id='plotter-stored-data'),
        
        # Store for the product config
        dcc.Store(id='plotter-stored-config', data=config),
        
        # Output graph
        html.Div(id='plotter-output-graph-container', className="mt-4")
    ])
    
    return layout


def register_plotter_callbacks(app):
    # Updated callback to store the uploaded data, enable product dropdown, and show status
    @app.callback(
        [Output('plotter-stored-data', 'data'),
         Output('plotter-product-dropdown', 'disabled'),
         Output('plotter-upload-status', 'children')],  # Add this output
        [Input('plotter-upload-data', 'contents'),
         Input('plotter-upload-data', 'filename')]  # Add filename input
    )
    def store_data(contents, filename):
        if contents is None:
            return None, True, ""

        df = parse_contents(contents)
        if isinstance(df, pd.DataFrame):
            # Success message with bootstrap styling
            success_message = dbc.Alert(
                f"File '{filename}' successfully uploaded! {len(df)} rows loaded.",
                color="success",
                dismissable=True
            )
            return df.to_json(date_format='iso', orient='split'), False, success_message

        # Error message if file parsing failed
        error_message = dbc.Alert(
            f"Failed to process file '{filename}'. Please ensure it's a valid CSV.",
            color="danger",
            dismissable=True
        )
        return None, True, error_message

    # Callback to enable submit button when product is selected
    @app.callback(
        Output('plotter-submit-button', 'disabled'),
        [Input('plotter-product-dropdown', 'value'),
         Input('plotter-stored-data', 'data')]
    )
    def enable_submit_button(selected_product, json_data):
        return not (selected_product and json_data)
    
    # Callback for generating the plot
    @app.callback(
        Output('plotter-output-graph-container', 'children'),
        [Input('plotter-submit-button', 'n_clicks')],
        [State('plotter-product-dropdown', 'value'),
         State('plotter-stored-data', 'data'),
         State('plotter-stored-config', 'data')],
        prevent_initial_call=True
    )
    def update_graph(n_clicks, selected_product, json_data, config):
        if not n_clicks or not selected_product or not json_data or not config:
            return html.Div("Please upload a CSV file, select a product, and click Generate Plot.")

        # Convert the stored JSON data back to a DataFrame
        df = pd.read_json(json_data, orient='split')

        # Get the variables to plot for the selected product
        variables_to_plot = config[selected_product]

        # Create a plotter instance and generate the plot
        plotter = Plotter(df, variables_to_plot)
        fig = plotter.plot()

        # Return graph with style to ensure it takes full width
        return dcc.Graph(
            figure=fig,
            style={'width': '100%', 'height': '100%'},
            config={'responsive': True}
        )


# This allows the module to be run standalone for testing
if __name__ == '__main__':
    app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
    app.layout = create_plotter_layout()
    register_plotter_callbacks(app)
    app.run_server(debug=True)
