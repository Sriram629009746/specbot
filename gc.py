import dash
from dash import dcc, html, Input, Output, State, ctx, MATCH, ALL
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import copy
import uuid
import dash_uploader as du
import os



class PlotterApp:
    def __init__(self):
        self.app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
        UPLOAD_FOLDER_ROOT = os.path.join(os.getcwd(), "uploads")
        du.configure_upload(self.app, UPLOAD_FOLDER_ROOT)
        self.df, self.config = self.generate_simulation_data()
        self.app.layout = self.build_layout()
        self.register_callbacks()

    def generate_simulation_data(self):
        np.random.seed(0)
        age = [f"A{i}" for i in range(1000)]
        df = pd.DataFrame({'age': age})
        for var in ['X', 'Y', 'Z', 'M', 'N', 'O']:
            df[var] = np.random.randn(1000).cumsum()
        for var in ['C1', 'C2']:
            df[var] = np.random.choice(['low', 'medium', 'high'], size=1000)

        config = {
            "Product_A": [
                {
                    "group_type": "numerical",
                    "plot_title": "Numerical Group 1",
                    "var_group": [
                        {"varname": "X", "default_enable": True},
                        {"varname": "Y", "default_enable": True},
                        {"varname": "Z", "default_enable": True}
                    ]
                },
                {
                    "group_type": "numerical",
                    "plot_title": "Numerical Group 2",
                    "var_group": [
                        {"varname": "M", "default_enable": True},
                        {"varname": "N", "default_enable": False},
                        {"varname": "O", "default_enable": True}
                    ]
                },
                {
                    "group_type": "categorical",
                    "plot_title": "Categorical Group",
                    "var_group": [
                        {"varname": "C1", "default_enable": True}
                    ]
                }
            ]
        }
        return df, config
    '''
    def build_layout(self):
        sidebar_store = dcc.Store(id='sidebar-toggle', data=True)
        return dbc.Container([
            sidebar_store,
            html.H2("Smart Plotter with Config and Sync", className="text-center mt-3 mb-3"),
            dbc.Row([
                dbc.Col([
                    du.Upload(
                        id='upload-data',
                        text='Drag and Drop or Select CSV File',
                        max_files=1,
                        filetypes=['csv'],
                        upload_id='data-upload',
                        disabled=False
                    ),
                    html.Div(id='upload-status', className='mb-3')
                ], width=12)
            ]),
            dbc.Row([
                dbc.Col([
                    html.Label("Select Product"),
                    dcc.Dropdown(
                        id='product-dropdown',
                        options=[{"label": k, "value": k} for k in self.config.keys()],
                        placeholder="Select a product",
                        disabled=True
                    ),
                    dbc.Button("Submit", id="submit-btn", className="mt-2", color="primary", disabled=True),
                    html.Div([
                        dbc.Button("☰", id="toggle-sidebar-btn", color="secondary", className="my-3"),
                        html.Div([
                            html.Div(id='variable-panel', className='mt-3'),
                            dbc.Button("Generate Plot", id="plot-btn", color="success", disabled=True, className="mt-3")
                        ], id='variable-panel-content')
                    ], id='variable-panel-wrapper')
                ], id='sidebar-col', width=3, style={"border-right": "1px solid #ccc"}),

                dbc.Col([
                    dcc.Loading(
                        id='plot-loading',
                        #type='default',
                        children=html.Div(id='plot-container', className='mt-3')
                    )
                ], id='plot-col', width=9)
            ]),
            dcc.Store(id='shared-xaxis-store')
        ], fluid=True)
    '''

    '''
    def build_layout(self):
        sidebar_store = dcc.Store(id='sidebar-toggle', data=True)
        return dbc.Container([
            sidebar_store,

            html.H2("Smart Plotter with Config and Sync", className="text-center mt-3 mb-3"),

            # Top row: Upload and Product Controls
            dbc.Row([
                dbc.Col([
                    du.Upload(
                        id='upload-data',
                        text='Drag and Drop or Select CSV File',
                        max_files=1,
                        filetypes=['csv'],
                        upload_id='data-upload',
                        disabled=False
                    ),
                    html.Div(id='upload-status', className='mb-3'),

                    html.Label("Select Product"),
                    dcc.Dropdown(
                        id='product-dropdown',
                        options=[{"label": k, "value": k} for k in self.config.keys()],
                        placeholder="Select a product",
                        disabled=True
                    ),
                    dbc.Button("Submit", id="submit-btn", className="mt-2", color="primary", disabled=True)
                ], width=12)
            ]),

            # Middle row: Toggle button + Sidebar + Plot area
            dbc.Row([
                # ☰ toggle button in a slim column (always visible)
                dbc.Col([
                    dbc.Button("☰", id="toggle-sidebar-btn", color="secondary", className="mt-4 mb-3")
                ], width="auto"),

                # Collapsible sidebar for checklist and plot button
                dbc.Col([
                    html.Div([
                        html.Div(id='variable-panel', className='mt-3'),
                        dbc.Button("Generate Plot", id="plot-btn", color="success", disabled=True, className="mt-3")
                    ], id='variable-panel-content')
                ], id='sidebar-col', width=3, style={"border-right": "1px solid #ccc"}),

                # Plot output area
                dbc.Col([
                    dcc.Loading(
                        id='plot-loading',
                        children=html.Div(id='plot-container', className='mt-3')
                    )
                ], id='plot-col', width=9)
            ]),

            # Hidden shared state
            dcc.Store(id='shared-xaxis-store')
        ], fluid=True)
    '''

    def build_layout(self):
        sidebar_store = dcc.Store(id='sidebar-toggle', data=True)
        return dbc.Container([
            sidebar_store,

            html.H2("Smart Plotter with Config and Sync", className="text-center mt-3 mb-3"),

            # Upload and Product Selection Row
            dbc.Row([
                dbc.Col([
                    du.Upload(
                        id='upload-data',
                        text='Drag and Drop or Select CSV File',
                        max_files=1,
                        filetypes=['csv'],
                        upload_id='data-upload',
                        disabled=False
                    ),
                    html.Div(id='upload-status', className='mb-3'),

                    html.Label("Select Product"),
                    dcc.Dropdown(
                        id='product-dropdown',
                        options=[{"label": k, "value": k} for k in self.config.keys()],
                        placeholder="Select a product",
                        disabled=True
                    ),
                    dbc.Button("Submit", id="submit-btn", className="mt-2", color="primary", disabled=True)
                ], width=12)
            ]),

            # Middle row: Toggle + Sidebar + Plot
            dbc.Row([
                # Toggle Button (always visible after submit)
                dbc.Col([
                    dbc.Button("☰ Toggle Checklist", id="toggle-sidebar-btn", color="secondary", className="mt-4")
                ], id="toggle-button-col", width=12, style={"display": "none"}),
            ]),

            dbc.Row([
                # Sidebar: checklist + generate button stacked vertically
                dbc.Col([
                    html.Div([
                        html.Div(id='variable-panel', className='mb-3'),
                        dbc.Button("Generate Plot", id="plot-btn", color="success", disabled=True)
                    ], id="variable-panel-content")
                ], id='sidebar-col', width=12, style={"border-right": "0px solid #ccc"}),

            ]),

            dbc.Row([

                # Plot area
                dbc.Col([
                    dcc.Loading(
                        id='plot-loading',
                        children=html.Div(id='plot-container', className='mt-3')
                    )
                ], id='plot-col', width=12)

            ]),




            dcc.Store(id='shared-xaxis-store')
        ], fluid=True)

    def build_variable_checklist(self, product):
        groups = self.config[product]
        checklist_blocks = []
        for i, group in enumerate(groups):
            group_id = f"group-{i}"
            checklist_blocks.append(html.Details([
                html.Summary(group['plot_title']),
                dcc.Checklist(
                    id={'type': 'var-checklist', 'index': i},
                    options=[{'label': v['varname'], 'value': v['varname']} for v in group['var_group']],
                    value=[v['varname'] for v in group['var_group'] if v['default_enable']]
                )
            ], open=True))
        return checklist_blocks


    def plot_numerical_group(self, df, variables, title, x_range=None):
        fig = go.Figure()
        for var in variables:
            fig.add_trace(go.Scatter(x=df['age'], y=df[var], mode='lines+markers', name=var))
        fig.update_layout(title=title, xaxis_title='Age', hovermode='x unified', autosize=True)
        #x_range = [100,200]
        if x_range:
            fig.update_layout(xaxis={'range': x_range})

        print("Num traces in plot", len(fig['data']))

        #return import copy

        return copy.deepcopy(fig)

    def plot_categorical_group(self, df, variables, title, x_range=None):
        #fig = make_subplots(rows=len(variables), cols=1, shared_xaxes=True, vertical_spacing=0.02)
        fig = go.Figure()
        #print(variables)
        if len(variables)>1:
            return {}
        for i, var in enumerate(variables, start=1):
            unique_vals = list(df[var].unique())
            mapping = {val: idx + 1 for idx, val in enumerate(unique_vals)}
            y_vals = df[var].map(mapping)
            fig.add_trace(go.Scatter(
                x=df['age'], y=y_vals,
                mode='lines+markers', name=var,
                text=df[var], hoverinfo='text+x+y'))
            fig.update_yaxes(title_text=var, tickvals=list(mapping.values()), ticktext=list(mapping.keys()))
        fig.update_layout(title=title, hovermode='x unified', height=300 * len(variables), autosize=True)
        #x_range = [100, 200]
        if x_range:
            fig.update_layout(xaxis={'range': x_range})

        #print("Num traces in plot", len(fig['data']))

        #return fig
        return copy.deepcopy(fig)

    def register_callbacks(self):

        @self.app.callback(
            [
                Output('upload-status', 'children'),
                Output('plot-container', 'children', allow_duplicate=True),
                Output('variable-panel', 'children',allow_duplicate=True),
                Output('product-dropdown', 'value'),
                Output('product-dropdown', 'disabled'),
                Output('submit-btn', 'disabled'),
                Output('plot-btn', 'disabled', allow_duplicate=True),
                Output('shared-xaxis-store', 'data', allow_duplicate=True),
            ],
            Input('upload-data', 'isCompleted'),
            State('upload-data', 'fileNames'),
            prevent_initial_call=True
        )
        def handle_upload(is_completed, filenames):
            if not is_completed or not filenames:
                raise dash.exceptions.PreventUpdate

            filename = filenames[0]
            full_path = os.path.join("uploads", "data-upload", filename)

            try:
                # ✅ Create dummy DataFrame from uploaded CSV
                dummy_df = pd.read_csv(full_path)
                print(f"CSV loaded successfully: {dummy_df.shape}")

            except Exception as e:
                print(f"Failed to load uploaded CSV: {e}")
                dummy_df = pd.DataFrame()

            # ✅ Still use simulation data for plotting
            self.df, self.config = self.generate_simulation_data()

            return (
                f"Uploaded file: {filename}",
                [],  # Clear plot-container
                [],  # Clear variable-panel
                None,  # Reset product-dropdown
                False,  # ✅ Enable product-dropdown
                False,  # ✅ Enable submit button
                True,  # Disable plot button until checklist is selected
                None  # Clear shared x-range
            )

        '''
        @self.app.callback(
            [Output('variable-panel-content', 'style'), Output('sidebar-toggle', 'data')],
            Input('toggle-sidebar-btn', 'n_clicks'),
            State('sidebar-toggle', 'data'),
            prevent_initial_call=True
        )
        def toggle_sidebar(n_clicks, is_open):
            if not n_clicks:
                raise dash.exceptions.PreventUpdate
            new_state = not is_open
            style = {"display": "block"} if new_state else {"display": "none"}
            return style, new_state
        '''

        @self.app.callback(
            [
                Output('sidebar-col', 'width'),
                Output('sidebar-col', 'style'),
                Output('plot-col', 'width'),
                Output('sidebar-toggle', 'data'),
            ],
            Input('toggle-sidebar-btn', 'n_clicks'),
            State('sidebar-toggle', 'data'),
            prevent_initial_call=True
        )
        def toggle_sidebar(n_clicks, is_open):
            new_state = not is_open
            if new_state:
                return 3, {"display": "block", "border-right": "0px solid #ccc"}, 9, new_state
            else:
                return 0, {"display": "none"}, 12, new_state

        '''
        @self.app.callback(
            Output("sidebar-controls-wrapper", "style"),
            Input("submit-btn", "n_clicks"),
            State("product-dropdown", "value"),
            prevent_initial_call=True
        )
        def show_sidebar_controls(n_clicks, product):
            if not n_clicks or not product:
                return {"display": "none"}
            return {"display": "block"}
        '''

        @self.app.callback(
            Output("toggle-button-col", "style"),
            #Output("toggle-button-col", "style"),
            Input("submit-btn", "n_clicks"),
            State("product-dropdown", "value"),
            prevent_initial_call=True
        )
        def show_toggle_button(n_clicks, product):
            if not n_clicks or not product:
                return {"display": "none"}
            return {"display": "block"}

        from copy import deepcopy

        from copy import deepcopy

        @self.app.callback(
            Output('shared-xaxis-store', 'data'),
            Input({'type': 'sync-graph', 'index': ALL, 'dummy': ALL}, 'relayoutData'),
            prevent_initial_call=True
        )
        def update_shared_xrange(relayout_datas):
            print(f"in update xrange")
            for relayout in relayout_datas:
                if relayout:
                    if 'xaxis.range[0]' in relayout and 'xaxis.range[1]' in relayout:
                        return [relayout['xaxis.range[0]'], relayout['xaxis.range[1]']]
                    elif 'xaxis.autorange' in relayout:
                        return None
            return dash.no_update

        @self.app.callback(
            Output('variable-panel', 'children'),
            Input('submit-btn', 'n_clicks'),
            State('product-dropdown', 'value')
        )
        def display_checklists(n, product):
            if not product:
                return dash.no_update
            return self.build_variable_checklist(product)

        @self.app.callback(
            Output('plot-btn', 'disabled'),
            Input({'type': 'var-checklist', 'index': ALL}, 'value')
        )
        def toggle_plot_button(all_selected):
            return not any(all_selected)

        '''
        @self.app.callback(
            Output('plot-container', 'children'),
            [Input('plot-btn', 'n_clicks'), Input('shared-xaxis-store', 'data')],
            [State('product-dropdown', 'value'), State({'type': 'var-checklist', 'index': ALL}, 'value')],
            prevent_initial_call=True
        )
        def generate_or_update_plots(n_clicks, shared_range, product, selections):

            print(f"in gen plot: {ctx.triggered_id}")
            trigger = ctx.triggered_id
            if not n_clicks and not shared_range:
                return dash.no_update
            if not product or not selections:
                return dash.no_update
            figs = []

            print(shared_range)

            if shared_range is not None:
                shared_range = None
            for i in range(len(self.config[product])):
                selected = selections[i] if i < len(selections) else []
                if not selected:
                    continue
                group = self.config[product][i]

                if group['group_type'] == 'numerical':
                    fig = self.plot_numerical_group(self.df, selected, group['plot_title'], None)
                else:
                    fig = self.plot_categorical_group(self.df, selected, group['plot_title'], None)
                figs.append(dcc.Graph(
                    id={'type': 'sync-graph', 'index': i},
                    figure=fig,
                    config={"responsive": True}
                ))
                print(len(figs))
            return figs
        '''

        @self.app.callback(
            Output('plot-container', 'children'),
            [Input('plot-btn', 'n_clicks'), Input('shared-xaxis-store', 'data')],
            [State('product-dropdown', 'value'), State({'type': 'var-checklist', 'index': ALL}, 'value')],
            prevent_initial_call=True
        )
        def generate_or_update_plots(n_clicks, shared_range, product, selections):

            trigger = ctx.triggered_id
            print(f"in gen: {trigger}")

            if not n_clicks and not shared_range:
                return dash.no_update
            if not product or not selections:
                return dash.no_update

            if trigger == 'plot-btn':
                shared_range = None

            figs = []
            for i, group in enumerate(self.config[product]):
                selected = selections[i] if i < len(selections) else []
                if not selected:
                    continue
                #print(shared_range)
                if group['group_type'] == 'numerical':
                    #print(selected, group['plot_title'], shared_range)
                    fig = self.plot_numerical_group(self.df, selected, group['plot_title'], shared_range)
                else:
                    #print(selected, group['plot_title'], shared_range)
                    fig = self.plot_categorical_group(self.df, selected, group['plot_title'], shared_range)

                #figs.append(dcc.Graph(
                #    id={'type': 'sync-graph', 'index': i},
                #    figure=fig,
                #    config={"responsive": True}
                #))
                # Add this line to preserve zoom state across updates
                #fig.update_layout(uirevision=str(uuid.uuid4()))  # force redraw
                #fig = copy.deepcopy(fig)

                figs.append(dcc.Graph(
                    id={'type': 'sync-graph', 'index': i, 'dummy': str(uuid.uuid4())},#'uid': str(uuid.uuid4())},
                    figure=copy.deepcopy(fig),
                    config={"responsive": True},
                    #key=str(uuid.uuid4())  # 👈 force DOM replacement
                ))

            #print("Returned Graph IDs:", [graph.id for graph in figs])


            return figs

    def run(self):
        self.app.run(debug=True)


if __name__ == '__main__':
    PlotterApp().run()
