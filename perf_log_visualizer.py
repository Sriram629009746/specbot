import pandas as pd
import numpy as np
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, State, ctx, no_update, callback_context
import dash_bootstrap_components as dbc
from plotly.subplots import make_subplots
import base64
import io
import uuid
import os
import glob
from flask import Flask

# Import your existing log parser here
# from log_parser import parse_log_file

# Placeholder for the parser function - replace with actual implementation
def parse(file_path):
    """
    Parse a log file and return a dataframe with the required columns.
    """
    # In a real implementation, this would read and parse the log file
    # For demonstration, we'll generate sample data

    # Create a safe seed from the filename
    try:
        if isinstance(file_path, str):
            base_name = os.path.basename(file_path).split('.')[0]
            # Try to convert to int if possible
            seed = int(base_name) if base_name.isdigit() else hash(base_name) % 10000
        else:
            seed = 42
    except:
        seed = 42

    np.random.seed(seed)

    # Generate sample data
    commands = ["read", "write", "trim", "verify", "erase", "format"]
    chunk_sizes = [0x4, 0x8, 0x10, 0x20, 0x40, 0x80]

    num_entries = 500

    data = {
        "index": np.arange(num_entries),
        "command": np.random.choice(commands, num_entries),
        "lba": np.random.randint(0, 1000, num_entries),
        "length": np.random.choice(chunk_sizes, num_entries)
    }

    df = pd.DataFrame(data)

    # Return the dataframe
    return df

class DataProcessor:
    def __init__(self, df):
        self.df = df
        self.cmd_colors = {
            "read": "blue",
            "write": "green",
            "trim": "orange",
            "verify": "purple",
            "erase": "red",
            "format": "brown"
        }
    
    def filter_by_index_range(self, idx_range):
        if idx_range is None:
            return self.df
        idx0, idx1 = idx_range
        return self.df[(self.df["index"] >= idx0) & (self.df["index"] <= idx1)]
    
    def get_chunk_size_distribution(self, command_type, idx_range=None):
        """Get chunk size distribution for a specific command type"""
        df = self.filter_by_index_range(idx_range) if idx_range else self.df
        cmd_df = df[df["command"] == command_type]
        
        if cmd_df.empty:
            return {"No Data": 1}  # Return dummy data if empty
        
        # Count occurrences of each chunk size
        chunk_counts = cmd_df["length"].value_counts().to_dict()
        
        # Format chunk sizes as hex strings
        formatted_counts = {f"0x{size:X}": count for size, count in chunk_counts.items()}
        
        return formatted_counts
    
    def calculate_cmd_stats(self, idx_range=None):
        """Calculate command-wise statistics for the given index range"""
        df = self.filter_by_index_range(idx_range) if idx_range else self.df
        
        stats = {}
        for cmd in df["command"].unique():
            cmd_df = df[df["command"] == cmd]
            stats[cmd] = {
                "count": len(cmd_df),
                "avg_length": cmd_df["length"].mean(),
                "total_length": cmd_df["length"].sum()
            }
        
        # Add overall stats
        stats["overall"] = {
            "count": len(df),
            "avg_length": df["length"].mean(),
            "total_length": df["length"].sum()
        }
        
        return stats


class VisualizationBuilder:
    def __init__(self, data_processor):
        self.data_processor = data_processor
    
    def build_cmd_index_plot(self, df, idx_range=None):
        """Build a plot showing commands vs index"""
        fig = go.Figure()
        
        # Create traces for each command type
        for cmd in sorted(df["command"].unique()):
            cmd_df = df[df["command"] == cmd]
            fig.add_trace(go.Scatter(
                x=cmd_df["index"],
                y=[cmd for _ in range(len(cmd_df))],  # Constant y-value for each command type
                mode="markers",
                marker=dict(color=self.data_processor.cmd_colors[cmd], size=8),
                name=cmd.capitalize(),
                hoverinfo="text",
                text=cmd_df.apply(
                    lambda row: f"Command: {row['command']}<br>Index: {row['index']}<br>LBA: {row['lba']}<br>Length: 0x{row['length']:X}",
                    axis=1)
            ))
        
        fig.update_layout(
            height=300,
            margin=dict(t=30, b=30, l=50, r=40),
            yaxis=dict(
                title="Command Type",
                categoryorder='array',
                categoryarray=sorted(df["command"].unique()),
                fixedrange=True
            ),
            xaxis=dict(title="Index"),
            title="Commands by Index"
        )
        
        if idx_range is not None:
            fig.update_xaxes(range=idx_range)
        
        return fig
    
    def build_lba_index_plot(self, df, idx_range=None):
        """Build a plot showing LBA vs index (address map)"""
        fig = go.Figure()
        
        # Create traces for each command type
        for cmd in sorted(df["command"].unique()):
            cmd_df = df[df["command"] == cmd]
            fig.add_trace(go.Scatter(
                x=cmd_df["index"],
                y=cmd_df["lba"],
                mode="markers",
                marker=dict(color=self.data_processor.cmd_colors[cmd], size=6),
                name=cmd.capitalize(),
                hoverinfo="text",
                text=cmd_df.apply(
                    lambda row: f"Command: {row['command']}<br>Index: {row['index']}<br>LBA: {row['lba']}<br>Length: 0x{row['length']:X}",
                    axis=1)
            ))
        
        fig.update_layout(
            height=300,
            margin=dict(t=30, b=30, l=50, r=40),
            yaxis=dict(title="LBA", fixedrange=True),
            xaxis=dict(title="Index"),
            title="Address Map (LBA vs Index)",
            legend=dict(orientation="h", yanchor="top", y=-0.15, xanchor="center", x=0.5)
        )
        
        if idx_range is not None:
            fig.update_xaxes(range=idx_range)
        
        return fig
    
    def build_cmd_count_bar_chart(self, df):
        """Build a bar chart showing the count of each command type"""
        cmd_counts = df["command"].value_counts().sort_index()
        
        fig = go.Figure(go.Bar(
            x=list(cmd_counts.index),
            y=list(cmd_counts.values),
            marker_color=[self.data_processor.cmd_colors[cmd] for cmd in cmd_counts.index],
            text=cmd_counts.values,
            textposition='auto',
        ))
        
        fig.update_layout(
            height=300,
            margin=dict(t=50, b=30),
            title="Command Count Distribution",
            xaxis=dict(title="Command Type", categoryorder='category ascending'),
            yaxis=dict(title="Count"),
        )
        
        return fig
    
    def build_chunk_size_charts(self, df, idx_range=None):
        """Build 1x2 subplot showing chunk size distribution for read and write operations"""
        # Create subplots
        fig = make_subplots(rows=1, cols=2,
                          subplot_titles=("Read Chunk Size Distribution", "Write Chunk Size Distribution"),
                          specs=[[{"type": "pie"}, {"type": "pie"}]])
        
        # Get chunk size distributions
        read_chunks = self.data_processor.get_chunk_size_distribution("read", idx_range)
        write_chunks = self.data_processor.get_chunk_size_distribution("write", idx_range)
        
        # Add read chunk size pie chart
        fig.add_trace(
            go.Pie(
                labels=list(read_chunks.keys()),
                values=list(read_chunks.values()),
                textinfo='percent+label',
                hole=0.4,
                marker=dict(colors=['lightblue', 'royalblue', 'darkblue', 'navy']),
                domain=dict(column=0),
                title=dict(text="Read")
            ),
            row=1, col=1
        )
        
        # Add write chunk size pie chart
        fig.add_trace(
            go.Pie(
                labels=list(write_chunks.keys()),
                values=list(write_chunks.values()),
                textinfo='percent+label',
                hole=0.4,
                marker=dict(colors=['lightgreen', 'forestgreen', 'green', 'darkgreen']),
                domain=dict(column=1),
                title=dict(text="Write")
            ),
            row=1, col=2
        )
        
        fig.update_layout(
            height=350,
            margin=dict(t=60, b=30, l=30, r=30),
            title="Chunk Size Distribution by Command Type"
        )
        
        return fig


import tempfile

class PerfLogVisualizer:
    def __init__(self):
        self.user_data = {}

        # Save uploaded files OUTSIDE project dir
        self.upload_folder = os.path.join(tempfile.gettempdir(), "uploaded_files")

        if not os.path.exists(self.upload_folder):
            os.makedirs(self.upload_folder)

    def get_upload_layout(self):
        return html.Div([
            html.H3("Performance Log Visualizer", className="mt-3 mb-4"),

            # File upload section
            dbc.Row([
                dbc.Col([
                    dcc.Upload(
                        id="perflog-upload-data",
                        children=html.Div([
                            "Drag and Drop or ",
                            html.A("Select Files")
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
                        multiple=True
                    ),
                    html.Div(
                        id="perflog-currently-displayed-log",
                        className="mt-3",
                        children=dbc.Alert("No file selected yet.", color="secondary")
                    ),
                    html.Div(id="perflog-upload-status", className="mt-2")
                ], width=12)
            ]),

            # File selection and processing section
            dbc.Row([
                dbc.Col([
                    html.Label("Select Log File:"),
                    dcc.Dropdown(
                        id="perflog-file-selector",
                        options=[],
                        value=None,
                        clearable=False,
                        className="mb-3"
                    ),
                ], width=6),
                dbc.Col([
                    dbc.Button(
                        "Submit",
                        id="perflog-submit-button",
                        color="primary",
                        className="mt-4 me-2"
                    ),
                    dbc.Button(
                        "Next Log",
                        id="perflog-next-button",
                        color="secondary",
                        className="mt-4 me-2",
                        disabled=True
                    ),
                    dbc.Button(
                        "Reset View",
                        id="perflog-reset-button",
                        color="info",
                        className="mt-4"
                    )
                ], width=6, className="d-flex justify-content-end")
            ]),

            # Processing status
            dbc.Row([
                dbc.Col([
                    html.Div(id="perflog-processing-status", className="mt-2")
                ], width=12)
            ]),

            # Loading indicator
            dcc.Loading(
                id="perflog-loading",
                type="circle",
                children=[
                    # Container for plots with empty graphs initially
                    html.Div(id="perflog-plots-container", className="mt-4", children=[
                        dbc.Row([
                            dbc.Col([
                                dcc.Graph(id="perflog-cmd-index-plot", figure={})
                            ], width=12)
                        ]),
                        dbc.Row([
                            dbc.Col([
                                dcc.Graph(id="perflog-lba-index-plot", figure={})
                            ], width=12)
                        ]),
                        dbc.Row([
                            dbc.Col([
                                dcc.Graph(id="perflog-cmd-count-chart", figure={})
                            ], width=12)
                        ]),
                        dbc.Row([
                            dbc.Col([
                                dcc.Graph(id="perflog-chunk-size-charts", figure={})
                            ], width=12)
                        ])
                    ])
                ]
            ),

            # Stats panel
            dbc.Row([
                dbc.Col([
                    html.Div(id="perflog-stats-panel", className="mt-4")
                ], width=12)
            ]),

            # Store components for maintaining state
            dcc.Store(id="perflog-uploaded-files", data=[]),
            dcc.Store(id="perflog-current-file-index", data=0),
            dcc.Store(id="perflog-index-range", data=None),
            dcc.Store(id="perflog-last-trigger", data=None),
            dcc.Store(id="perflog-user-id", data=None),
            dcc.Store(id="perflog-parser-data", data=None),
            dcc.Interval(id="perflog-init-interval", interval=1000, n_intervals=0, max_intervals=1)
        ])

    def get_plots_layout(self):
        return html.Div([
            dbc.Row([
                dbc.Col([
                    dcc.Graph(id="perflog-cmd-index-plot")
                ], width=12)
            ]),
            dbc.Row([
                dbc.Col([
                    dcc.Graph(id="perflog-lba-index-plot")
                ], width=12)
            ]),
            dbc.Row([
                dbc.Col([
                    dcc.Graph(id="perflog-cmd-count-chart")
                ], width=12)
            ]),
            dbc.Row([
                dbc.Col([
                    dcc.Graph(id="perflog-chunk-size-charts")
                ], width=12)
            ])
        ])
    
    def register_callbacks(self, app):

        @app.callback(
            Output('perflog-user-id', 'data'),
            Input('perflog-init-interval', 'n_intervals'),
            State('perflog-user-id', 'data')
        )
        def initialize_user_session(n_intervals, current_id):
            if current_id is None:
                new_id = str(uuid.uuid4())
                print(f"Created new user ID: {new_id}")
                return new_id
            print(f"Using existing user ID: {current_id}")
            return current_id

        @app.callback(
            [Output('perflog-upload-status', 'children'),
             Output('perflog-uploaded-files', 'data'),
             Output('perflog-file-selector', 'options', allow_duplicate=True),
             Output('perflog-file-selector', 'value'),
             Output('perflog-submit-button', 'disabled'),
             Output('perflog-cmd-index-plot', 'figure'),
             Output('perflog-lba-index-plot', 'figure'),
             Output('perflog-cmd-count-chart', 'figure'),
             Output('perflog-chunk-size-charts', 'figure'),
             Output('perflog-stats-panel', 'children'),
             Output('perflog-next-button', 'disabled')],
            Input('perflog-upload-data', 'contents'),
            [State('perflog-upload-data', 'filename'),
             State('perflog-uploaded-files', 'data'),
             State('perflog-user-id', 'data'),
             State('perflog-file-selector', 'value')],
            prevent_initial_call=True
        )
        def handle_upload(contents, filenames, existing_files, user_id, current_selection):
            if not contents or not filenames:
                return no_update, no_update, no_update, no_update, True, {}, {}, {}, {}, None, True

            user_id = user_id or str(uuid.uuid4())

            user_dir = os.path.join(self.upload_folder, user_id)
            if not os.path.exists(user_dir):
                os.makedirs(user_dir)

            uploaded_files = []
            for content, filename in zip(contents, filenames):
                content_type, content_string = content.split(',')
                decoded = base64.b64decode(content_string)
                file_path = os.path.join(user_dir, filename)

                with open(file_path, 'wb') as f:
                    f.write(decoded)

                uploaded_files.append({'name': filename, 'path': file_path})

            # REPLACE existing files
            existing_files = uploaded_files

            options = [{'label': file['name'], 'value': file['path']} for file in existing_files]
            selected_value = uploaded_files[0]['path'] if uploaded_files else None

            upload_message = dbc.Alert(
                f"Successfully uploaded {len(uploaded_files)} file(s)",
                color="success",
                dismissable=True,
                duration=4000
            )

            next_disabled = len(existing_files) <= 1

            return (upload_message, existing_files, options, selected_value,
                    False, {}, {}, {}, {}, None, next_disabled)

        @app.callback(
            [Output('perflog-processing-status', 'children'),
             Output('perflog-parser-data', 'data'),
             Output('perflog-next-button', 'disabled', allow_duplicate=True),
             Output('perflog-current-file-index', 'data', allow_duplicate=True),
             Output('perflog-currently-displayed-log', 'children')],
            [Input('perflog-submit-button', 'n_clicks'),
             Input('perflog-next-button', 'n_clicks')],
            [State('perflog-file-selector', 'value'),
             State('perflog-uploaded-files', 'data'),
             State('perflog-current-file-index', 'data'),
             State('perflog-user-id', 'data')],
            prevent_initial_call=True
        )
        def process_log_file(submit_clicks, next_clicks, selected_file, uploaded_files, current_index, user_id):
            ctx_msg = callback_context.triggered
            if not ctx_msg or not any([submit_clicks, next_clicks]):
                return no_update, no_update, no_update, no_update, no_update

            trigger = ctx_msg[0]['prop_id'].split('.')[0]

            if not uploaded_files:
                return no_update, no_update, True, no_update, no_update  # No files uploaded

            if trigger == 'perflog-submit-button':
                # User submits selected file
                file_path = selected_file
                file_index = next((i for i, f in enumerate(uploaded_files) if f['path'] == selected_file), 0)
            elif trigger == 'perflog-next-button':
                # Move to next file
                file_index = (current_index + 1) % len(uploaded_files)
                file_path = uploaded_files[file_index]['path']
            else:
                return no_update, no_update, no_update, no_update, no_update

            try:
                parsed_df = parse(file_path)

                data_processor = DataProcessor(parsed_df)
                viz_builder = VisualizationBuilder(data_processor)

                if user_id not in self.user_data:
                    self.user_data[user_id] = {}

                self.user_data[user_id]['data_processor'] = data_processor
                self.user_data[user_id]['viz_builder'] = viz_builder

                parser_data = {
                    'file_index': file_index,
                    'num_entries': len(parsed_df),
                    'file_path': file_path
                }

                processing_status = dbc.Alert(
                    f"Successfully processed: {os.path.basename(file_path)}",
                    color="success",
                    dismissable=True,
                    duration=4000
                )

                current_log_display = dbc.Alert(
                    f"Currently displaying: {os.path.basename(file_path)}",
                    color="info",
                    className="mt-2",
                    duration=0
                )

                next_disabled = len(uploaded_files) <= 1

                return processing_status, parser_data, next_disabled, file_index, current_log_display

            except Exception as e:
                error_status = dbc.Alert(
                    f"Error processing file: {str(e)}",
                    color="danger",
                    dismissable=True,
                )
                return error_status, None, True, no_update, None

        @app.callback(
            Output('perflog-current-file-index', 'data'),
            [Input('perflog-parser-data', 'data')],
            prevent_initial_call=True
        )
        def update_current_file_index(parser_data):
            if parser_data and 'file_index' in parser_data:
                return parser_data['file_index']
            return no_update

        @app.callback(
            Output('perflog-file-selector', 'options'),
            Input('perflog-uploaded-files', 'data')
        )
        def update_file_selector(uploaded_files):
            if not uploaded_files:
                return []

            options = [{'label': file['name'], 'value': file['path']} for file in uploaded_files]
            print(f"Updating file selector with {len(options)} options")
            return options

        @app.callback(
            [Output('perflog-cmd-index-plot', 'figure', allow_duplicate=True),
             Output('perflog-lba-index-plot', 'figure', allow_duplicate=True),
             Output('perflog-cmd-count-chart', 'figure', allow_duplicate=True),
             Output('perflog-chunk-size-charts', 'figure', allow_duplicate=True)],
            Input('perflog-parser-data', 'data'),
            State('perflog-user-id', 'data'),
            prevent_initial_call=True
        )
        def update_initial_graphs(parser_data, user_id):
            if not parser_data or user_id not in self.user_data:
                return no_update, no_update, no_update, no_update

            data_processor = self.user_data[user_id]['data_processor']
            viz_builder = self.user_data[user_id]['viz_builder']

            # Get full dataframe
            df = data_processor.df

            # Generate initial figures
            cmd_index_fig = viz_builder.build_cmd_index_plot(df)
            lba_index_fig = viz_builder.build_lba_index_plot(df)
            cmd_count_fig = viz_builder.build_cmd_count_bar_chart(df)
            chunk_size_fig = viz_builder.build_chunk_size_charts(df)

            return cmd_index_fig, lba_index_fig, cmd_count_fig, chunk_size_fig

        @app.callback(
            [Output('perflog-index-range', 'data'),
             Output('perflog-last-trigger', 'data')],
            [Input('perflog-cmd-index-plot', 'relayoutData'),
             Input('perflog-lba-index-plot', 'relayoutData'),
             Input('perflog-reset-button', 'n_clicks'),
             Input('perflog-parser-data', 'data')],
            [State('perflog-index-range', 'data'),
             State('perflog-last-trigger', 'data'),
             State('perflog-user-id', 'data')],
            prevent_initial_call=True
        )
        def update_index_range(relayout_cmd, relayout_lba, reset_clicks, parser_data, stored_idx_range, last_trigger,
                               user_id):
            """Update the index range based on user interactions or reset"""
            trigger_id = callback_context.triggered[0]['prop_id'].split('.')[0]

            # If we have a new file loaded or reset, reset the index range
            if trigger_id == 'perflog-reset-button' or trigger_id == 'perflog-parser-data':
                if parser_data and 'num_entries' in parser_data:
                    return [0, parser_data['num_entries'] - 1], 'reset'
                return no_update, no_update

            # Handle zooming in cmd-index plot
            if trigger_id == 'perflog-cmd-index-plot' and relayout_cmd:
                if "xaxis.range[0]" in relayout_cmd and "xaxis.range[1]" in relayout_cmd:
                    x0 = max(0, float(relayout_cmd["xaxis.range[0]"]))
                    x1 = float(relayout_cmd["xaxis.range[1]"])
                    return [x0, x1], 'perflog-cmd-index-plot'
                elif "xaxis.autorange" in relayout_cmd and relayout_cmd["xaxis.autorange"]:
                    if parser_data and 'num_entries' in parser_data:
                        return [0, parser_data['num_entries'] - 1], 'reset'

            # Handle zooming in lba-index plot
            if trigger_id == 'perflog-lba-index-plot' and relayout_lba:
                if "xaxis.range[0]" in relayout_lba and "xaxis.range[1]" in relayout_lba:
                    x0 = max(0, float(relayout_lba["xaxis.range[0]"]))
                    x1 = float(relayout_lba["xaxis.range[1]"])
                    return [x0, x1], 'perflog-lba-index-plot'
                elif "xaxis.autorange" in relayout_lba and relayout_lba["xaxis.autorange"]:
                    if parser_data and 'num_entries' in parser_data:
                        return [0, parser_data['num_entries'] - 1], 'reset'

            return no_update, no_update

        @app.callback(
            Output('perflog-cmd-index-plot', 'figure', allow_duplicate=True),
            [Input('perflog-index-range', 'data')],
            [State('perflog-user-id', 'data'),
             State('perflog-last-trigger', 'data')],
            prevent_initial_call=True
        )
        def update_cmd_index_plot(idx_range, user_id, last_trigger):
            if user_id not in self.user_data:
                return no_update

            # Don't update if this plot triggered the range change
            if last_trigger == 'perflog-cmd-index-plot':
                return no_update

            data_processor = self.user_data[user_id]['data_processor']
            viz_builder = self.user_data[user_id]['viz_builder']

            # Filter data based on index range
            filtered_df = data_processor.filter_by_index_range(idx_range)

            # Build and return the plot
            return viz_builder.build_cmd_index_plot(filtered_df, idx_range=idx_range)
        
        @app.callback(
            Output('perflog-lba-index-plot', 'figure', allow_duplicate=True),
            [Input('perflog-index-range', 'data')],
            [State('perflog-user-id', 'data'),
             State('perflog-last-trigger', 'data')],
            prevent_initial_call=True
        )
        def update_lba_index_plot(idx_range, user_id, last_trigger):
            if user_id not in self.user_data:
                return no_update
            
            # Don't update if this plot triggered the range change
            if last_trigger == 'perflog-lba-index-plot':
                return no_update
            
            data_processor = self.user_data[user_id]['data_processor']
            viz_builder = self.user_data[user_id]['viz_builder']
            
            # Filter data based on index range
            filtered_df = data_processor.filter_by_index_range(idx_range)
            
            # Build and return the plot
            return viz_builder.build_lba_index_plot(filtered_df, idx_range=idx_range)
        
        @app.callback(
            [Output('perflog-cmd-count-chart', 'figure', allow_duplicate=True),
             Output('perflog-chunk-size-charts', 'figure', allow_duplicate=True)],
            [Input('perflog-index-range', 'data')],
            [State('perflog-user-id', 'data')],
            prevent_initial_call=True
        )
        def update_additional_charts(idx_range, user_id):
            if user_id not in self.user_data:
                return no_update, no_update
            
            data_processor = self.user_data[user_id]['data_processor']
            viz_builder = self.user_data[user_id]['viz_builder']
            
            # Filter data based on index range
            filtered_df = data_processor.filter_by_index_range(idx_range)
            
            # Build and return the charts
            cmd_count_chart = viz_builder.build_cmd_count_bar_chart(filtered_df)
            chunk_size_charts = viz_builder.build_chunk_size_charts(filtered_df, idx_range)
            
            return cmd_count_chart, chunk_size_charts
        
        @app.callback(
            Output('perflog-stats-panel', 'children', allow_duplicate=True),
            [Input('perflog-index-range', 'data')],
            [State('perflog-user-id', 'data')],
            prevent_initial_call=True
        )
        def update_stats_panel(idx_range, user_id):
            if user_id not in self.user_data:
                return no_update
            
            data_processor = self.user_data[user_id]['data_processor']
            
            # Calculate statistics
            cmd_stats = data_processor.calculate_cmd_stats(idx_range)
            
            # Build the stats display
            stats_content = []
            
            # Add overall section
            stats_content.append(html.H4("Command Statistics"))
            
            # Command-specific stats
            for cmd in sorted(cmd_stats.keys()):
                if cmd == "overall":
                    continue
                    
                data = cmd_stats[cmd]
                color = data_processor.cmd_colors[cmd]
                
                cmd_card = dbc.Card(
                    dbc.CardBody([
                        html.H5(cmd.capitalize(), className="card-title", style={"color": color}),
                        html.P(f"Count: {data['count']}", className="card-text"),
                        html.P(f"Average Length: 0x{data['avg_length']:.2f}", className="card-text"),
                        html.P(f"Total Length: 0x{data['total_length']:.0f}", className="card-text"),
                    ]),
                    className="mb-3"
                )
                
                stats_content.append(cmd_card)
            
            # Add overall statistics
            overall = cmd_stats["overall"]
            overall_card = dbc.Card(
                dbc.CardBody([
                    html.H5("Overall Stats", className="card-title"),
                    html.P(f"Total Commands: {overall['count']}", className="card-text"),
                    html.P(f"Average Length: 0x{overall['avg_length']:.2f}", className="card-text"),
                    html.P(f"Total Length: 0x{overall['total_length']:.0f}", className="card-text"),
                ]),
                className="mb-3"
            )
            
            stats_content.append(overall_card)
            
            # Format the stats panel
            stats_panel = dbc.Row([
                dbc.Col(stats_content, width=12)
            ])
            
            return stats_panel


if __name__ == "__main__":
    # For standalone testing
    app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
    visualizer = PerfLogVisualizer()
    app.layout = visualizer.get_upload_layout()
    visualizer.register_callbacks(app)
    app.run(debug=True, host="0.0.0.0", port=8050)