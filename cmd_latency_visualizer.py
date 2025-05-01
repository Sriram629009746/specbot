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
from flask import Flask


# Import your existing log parser here
# from log_parser import parse_log_file

# Configuration
#MAX_CMDS_PER_PAGE = 100000  # Maximum commands to render per page
MAX_CMDS_PER_PAGE = 100  # Maximum commands to render per page
RANGE_INCREMENT = MAX_CMDS_PER_PAGE  # How many commands to increment for dropdown options

# Placeholder for your parser function - replace this with your actual parser
def parse_log_file(content):
    """
    Mock parser function - replace with your actual implementation
    Returns a dataframe with the expected columns
    """
    # In reality, this would process your log file and return a dataframe
    # For demonstration, we'll generate sample data
    np.random.seed(42)
    commands = ["read", "write", "trim", "verify", "erase", "format"]

    # Generate chunk sizes for read and write operations
    chunk_sizes = [0x4, 0x8, 0x10, 0x20, 0x40, 0x80]

    data = {
        "command": np.random.choice(commands, 500),
        "start time": sorted(np.random.randint(0, 50000, 500)),
        "duration": np.random.randint(10, 1000, 500),
        "position": np.random.randint(1, 33, 500),
        "status": np.random.choice(["success", "fail"], 500),
        "other details": [f"meta_{i}" for i in range(500)],
        "chunk_size": np.random.choice(chunk_sizes, 500)  # Added chunk size
    }
    df = pd.DataFrame(data)
    df["end time"] = df["start time"] + df["duration"]

    # In your real implementation, you would return:
    # csv_path, df = actual_parser_function(content)
    return "mock_path.csv", df


class DataProcessor:
    def __init__(self, df):
        self.df = df
        # Add index column to track command order
        #self.df = self.df.reset_index().rename(columns={"index": "cmd_index"})
        #self.df = self.df.reset_index().rename(columns={"index": "cmd_index"})

        print("erear")

        print(self.df.columns)
        self.cmd_colors = {
            "read": "blue",
            "write": "green",
            "trim": "orange",
            "verify": "purple",
            "erase": "red",
            "format": "brown"
        }

    def filter_by_time_range(self, x_range):
        if x_range is None:
            return self.df
        x0, x1 = x_range
        return self.df[(self.df["start time"] <= x1) & (self.df["end time"] >= x0)]

    def filter_by_index_range(self, idx_range):
        if idx_range is None:
            return self.df
        idx0, idx1 = idx_range
        return self.df[(self.df["cmd_index"] >= idx0) & (self.df["cmd_index"] <= idx1)]

    def compute_queue_depth(self, df, max_time):
        """Compute queue depth for the provided dataframe"""
        timeline = np.zeros(max_time + 1, dtype=int)
        for _, row in df.iterrows():
            timeline[row["start time"]:row["end time"] + 1] += 1
        return pd.Series(timeline)

    def get_max_time(self, df=None):
        """Get maximum time value from dataframe, optionally pass filtered df"""
        if df is None:
            df = self.df
        return int(df["end time"].max())

    def get_cmd_count(self):
        """Get total command count"""
        return len(self.df)

    def index_to_time_range(self, idx_range):
        if idx_range is None:
            return None

        idx0, idx1 = idx_range
        filtered = self.df[(self.df["cmd_index"] >= idx0) & (self.df["cmd_index"] <= idx1)]
        if filtered.empty:
            return None

        return [filtered["start time"].min(), filtered["end time"].max()]

    def time_to_index_range(self, time_range):
        if time_range is None:
            return None

        t0, t1 = time_range
        filtered = self.df[(self.df["start time"] <= t1) & (self.df["end time"] >= t0)]
        if filtered.empty:
            return None

        return [filtered["cmd_index"].min(), filtered["cmd_index"].max()]

    def calculate_cmd_stats(self, df=None, time_range=None):
        """Calculate command-wise statistics for the given time range or dataframe"""
        return None
        if df is None:
            df = self.filter_by_time_range(time_range) if time_range else self.df

        stats = {}
        for cmd in df["command"].unique():
            cmd_df = df[df["command"] == cmd]
            stats[cmd] = {
                "count": len(cmd_df),
                "mean_latency": cmd_df["duration"].mean(),
                "max_latency": cmd_df["duration"].max(),
                "min_latency": cmd_df["duration"].min(),
                "total_duration": cmd_df["duration"].sum()
            }

        # Add overall stats
        stats["overall"] = {
            "count": len(df),
            "mean_latency": df["duration"].mean(),
            "max_latency": df["duration"].max(),
            "min_latency": df["duration"].min(),
            "total_duration": df["duration"].sum()
        }

        return stats

    def get_chunk_size_distribution(self, command_type, df=None, time_range=None):
        """Get chunk size distribution for a specific command type"""
        if df is None:
            df = self.filter_by_time_range(time_range) if time_range else self.df
        cmd_df = df[df["command"] == command_type]

        if cmd_df.empty:
            return {"No Data": 1}  # Return dummy data if empty

        # Count occurrences of each chunk size
        chunk_counts = cmd_df["chunk_size"].value_counts().to_dict()

        # Format chunk sizes as hex strings
        formatted_counts = {f"0x{size:X}": count for size, count in chunk_counts.items()}

        return formatted_counts

class VisualizationBuilder:
    def __init__(self, data_processor):
        self.data_processor = data_processor

    def build_main_figure(self, df, x_range=None):
        max_time = int(df["end time"].max())
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                            row_heights=[0.6, 0.4], vertical_spacing=0.05,
                            subplot_titles=("Command Timeline", "Queue Depth"))

        for cmd in df["command"].unique():
            subset = df[df["command"] == cmd]
            x_vals = []
            y_vals = []
            texts = []

            for _, row in subset.iterrows():
                x_vals.extend([row["start time"], row["end time"], None])
                y_vals.extend([row["position"], row["position"], None])
                hover_text = (
                    f"Command: {row['command']}<br>"
                    f"Start: {row['start time']} µs<br>"
                    f"End: {row['end time']} µs<br>"
                    f"Latency: {row['end time'] - row['start time']} µs<br>"
                    f"Status: {row['status']}<br>"
                    f"Chunk Size: 0x{row['chunk_size']:X}<br>"
                    f"Details: {row['other details']}"
                )
                texts.extend([hover_text, hover_text, None])

            fig.add_trace(go.Scatter(
                x=x_vals,
                y=y_vals,
                mode="lines",
                line=dict(color=self.data_processor.cmd_colors[cmd], width=2),
                name=cmd,
                legendgroup=cmd,
                hoverinfo="text",
                text=texts
            ), row=1, col=1)

        depth = self.data_processor.compute_queue_depth(df, max_time)
        step_x = []
        step_y = []
        prev_qd = depth.iloc[0]
        step_x.append(0)
        step_y.append(prev_qd)
        for i in range(1, len(depth)):
            if depth.iloc[i] != prev_qd:
                step_x.extend([i, i])
                step_y.extend([prev_qd, depth.iloc[i]])
                prev_qd = depth.iloc[i]
        step_x.append(len(depth) - 1)
        step_y.append(prev_qd)

        fig.add_trace(go.Scatter(
            x=step_x,
            y=step_y,
            mode="lines",
            name="Queue Depth",
            showlegend=False,
            line=dict(color="black")
        ), row=2, col=1)

        tick_interval_us = 5000
        tick_vals = np.arange(0, max_time, tick_interval_us)
        tick_text = (tick_vals / 1000).astype(int).astype(str)

        fig.update_layout(
            height=700,
            margin=dict(t=50, b=40),
            yaxis=dict(title="Queue Position", fixedrange=True),
            yaxis2=dict(title="Depth", fixedrange=True)
        )

        fig.update_xaxes(
            tickvals=tick_vals,
            ticktext=tick_text,
            title="Time (ms)"
        )

        if x_range is not None:
            fig.update_xaxes(range=x_range)

        return fig

    def build_address_map(self, df, idx_range=None):
        fig = go.Figure()

        # Sort by cmd_index to ensure proper order
        df_sorted = df.sort_values('cmd_index')

        # Create traces for each command type
        for cmd in sorted(df_sorted["command"].unique()):
            cmd_df = df_sorted[df_sorted["command"] == cmd]
            fig.add_trace(go.Scatter(
                x=cmd_df["cmd_index"],
                y=cmd_df["position"],
                mode="markers",
                marker=dict(color=self.data_processor.cmd_colors[cmd], size=6),
                name=cmd.capitalize(),
                hoverinfo="text",
                text=cmd_df.apply(
                    lambda
                        row: f"Command: {row['command']}<br>Index: {row['cmd_index']}<br>Start time: {row['start time']}<br>Position: {row['position']}<br>Chunk Size: 0x{row['chunk_size']:X}",
                    axis=1)
            ))

        fig.update_layout(
            height=300,
            margin=dict(t=30, b=30, l=40, r=40),
            yaxis=dict(title="Position", fixedrange=True),
            xaxis=dict(title="Command Index"),
            title="Address Map (by Command Order)",
            # Move legend below the chart
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

    def build_chunk_size_charts(self, df, time_range=None):
        """Build 1x2 subplot showing chunk size distribution for read and write operations"""
        # Create subplots
        fig = make_subplots(rows=1, cols=2,
                            subplot_titles=("Read Chunk Size Distribution", "Write Chunk Size Distribution"),
                            specs=[[{"type": "pie"}, {"type": "pie"}]])

        # Get chunk size distributions
        read_chunks = self.data_processor.get_chunk_size_distribution("read", df, time_range)
        write_chunks = self.data_processor.get_chunk_size_distribution("write", df, time_range)

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

class DashboardApp:
    def __init__(self):
        # Initialize user session storage
        self.user_data = {}
        # Default cmd range size for rendering
        self.default_range_size = MAX_CMDS_PER_PAGE  # Can be modified as needed

    def get_upload_layout(self):
        """Return the upload layout for the latency visualizer component"""
        return html.Div([
            html.H3("Command Latency Visualizer", className="mt-3 mb-4"),

            # Upload component
            dbc.Row([
                dbc.Col([
                    dcc.Upload(
                        id='latency-upload-data',
                        children=html.Div([
                            'Drag and Drop or ',
                            html.A('Select a Log File')
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
                    html.Div(id='latency-upload-status', className="mt-2")
                ], width=12)
            ]),

            # Submit button (initially disabled)
            dbc.Row([
                dbc.Col([
                    dbc.Button(
                        "Process Log File",
                        id="latency-process-button",
                        color="primary",
                        className="mb-3",
                        disabled=True
                    )
                ], width={"size": 3, "offset": 0})
            ]),

            # Store for uploaded data
            dcc.Store(id='latency-uploaded-content'),
            dcc.Store(id='latency-user-id'),
            dcc.Store(id='latency-full-data-processed', data=False),
            dcc.Store(id='latency-total-cmd-count', data=0),
            dcc.Store(id='latency-current-range', data=[0, 0]),

            dcc.Interval(id="latency-init-trigger", interval=100, n_intervals=0, max_intervals=1),
            dcc.Store(id='latency-dashboard-initialized', data=False),

            # Dashboard content - hidden until a file is processed
            html.Div(id='latency-dashboard-content', style={'display': 'none'}, children=self.get_dashboard_content())
        ])

    def get_range_options(self, total_cmds, range_size=None):
        """Generate range options for dropdown based on total command count"""
        if range_size is None:
            range_size = self.default_range_size

        options = []
        for i in range(0, total_cmds, range_size):
            end = min(i + range_size, total_cmds)
            options.append({
                'label': f"{i:,} - {end:,}",
                'value': f"{i},{end}"
            })

        return options

    def get_dashboard_content(self):
        """Return the dashboard content component that appears after processing"""
        return html.Div([
            # Range selection controls
            dbc.Row([
                dbc.Col([
                    html.H5("Command Range Selection", className="mb-2")
                ], width=12)
            ]),

            dbc.Row([
                # Dropdown for predefined ranges
                dbc.Col([
                    html.Label("Select Command Range:"),
                    dcc.Dropdown(
                        id="latency-range-dropdown",
                        options=[],  # Will be populated dynamically
                        placeholder="Select a range...",
                        className="mb-2"
                    )
                ], width=4),

                # Custom range inputs
                dbc.Col(html.Div([
                    dbc.Checkbox(
                        id="latency-use-custom-range",
                        label="Use Custom Range",
                        value=False,
                        className="mb-2"
                    ),
                    html.Label("Custom Range:"),
                    dbc.Row([
                        dbc.Col(
                            dbc.Input(
                                id="latency-custom-start",
                                type="number",
                                placeholder="Start",
                                min=0,
                                step=1,
                                disabled=True
                            ),
                            width=5
                        ),
                        dbc.Col(
                            html.Span("to", className="pt-2"),
                            width=2,
                            className="text-center"
                        ),
                        dbc.Col(
                            dbc.Input(
                                id="latency-custom-end",
                                type="number",
                                placeholder="End",
                                min=1,
                                step=1,
                                disabled=True
                            ),
                            width=5
                        )
                    ])
                ])
                ),

                # Range control buttons
                dbc.Col([
                    dbc.Row([
                        dbc.Col([
                            dbc.Button(
                                "Load Selected Range",
                                id="latency-load-range-button",
                                color="primary",
                                className="w-100"
                            )
                        ], width=6),
                        dbc.Col([
                            dbc.Button(
                                "Next Range →",
                                id="latency-next-range-button",
                                color="secondary",
                                className="w-100"
                            )
                        ], width=6)
                    ])
                ], width=4)
            ]),

            # Status display and range info
            dbc.Row([
                dbc.Col([
                    # Processing status
                    html.Div(id="latency-processing-status", className="mt-2"),
                    # Current range display
                    html.Div(id="latency-current-range-display", className="mt-2 mb-3")
                ], width=12)
            ]),

            # Loading indicator
            dbc.Row([
                dbc.Col([
                    dcc.Loading(
                        id="latency-loading",
                        type="circle",
                        children=html.Div(id="latency-loading-output")
                    )
                ], width=12, className="text-center mb-3")
            ]),

            # Reset Button
            dbc.Row([
                dbc.Col([
                    dbc.Button("Reset View", id="latency-reset-button", color="secondary", className="mb-3")
                ], width={"size": 2, "offset": 10}, className="text-right")
            ]),

            # 1. Address Map and Stats Display (side by side)
            dbc.Row([
                dbc.Col([
                    dcc.Graph(id="latency-address-map")
                ], width=10),
                dbc.Col([
                    html.Div([
                        html.H4("Command Statistics", className="mb-3"),
                        html.Div(id="latency-stats-display", className="stats-panel")
                    ], className="h-100 d-flex flex-column justify-content-center")
                ], width=2)
            ]),

            # 2. Main figure (2x1 subplot)
            dbc.Row([
                dbc.Col([
                    dcc.Graph(id="latency-main-plot")
                ], width=12)
            ]),

            # 3. Command Count Bar Chart
            dbc.Row([
                dbc.Col([
                    dcc.Graph(id="latency-cmd-count-chart")
                ], width=12)
            ]),

            # 4. Chunk Size Distribution Charts (1x2 subplot for read/write)
            dbc.Row([
                dbc.Col([
                    dcc.Graph(id="latency-chunk-size-charts")
                ], width=12)
            ]),

            dcc.Store(id="latency-time-range"),
            dcc.Store(id="latency-index-range"),
            dcc.Store(id="latency-last-trigger"),
        ])

    def register_callbacks(self, app):
        """Register all callbacks for the latency visualizer component"""

        # Initialize user ID
        @app.callback(
            Output('latency-user-id', 'data'),
            Input('latency-user-id', 'modified_timestamp'),
            State('latency-user-id', 'data')
        )
        def init_user_session(ts, current_id):
            if current_id is None:
                # Generate a new user ID
                new_id = str(uuid.uuid4())
                return new_id
            return current_id

        # Enable process button when file is uploaded
        @app.callback(
            [Output('latency-process-button', 'disabled'),
             Output('latency-uploaded-content', 'data'),
             Output('latency-upload-status', 'children', allow_duplicate=True)],
            Input('latency-upload-data', 'contents'),
            Input('latency-upload-data', 'filename'),
            prevent_initial_call=True
        )
        def update_upload_status(contents, filename):
            if contents is None:
                return True, None, ""

            # Create success message with bootstrap styling
            upload_message = dbc.Alert(
                f"File '{filename}' selected and ready for processing.",
                color="info",
                dismissable=True
            )

            # Store the file contents for processing
            return False, {'contents': contents, 'filename': filename}, upload_message

        # Process file and show dashboard when process button is clicked
        @app.callback(
            [Output('latency-dashboard-content', 'style'),
             Output('latency-upload-status', 'children', allow_duplicate=True),
             Output('latency-dashboard-initialized', 'data'),
             Output('latency-full-data-processed', 'data'),
             Output('latency-total-cmd-count', 'data')],
            Input('latency-process-button', 'n_clicks'),
            State('latency-uploaded-content', 'data'),
            State('latency-user-id', 'data'),
            prevent_initial_call=True
        )
        def process_file(n_clicks, uploaded_data, user_id):
            if not n_clicks or not uploaded_data:
                return {'display': 'none'}, '', False, False, 0

            try:
                # Extract data from the uploaded content
                contents = uploaded_data['contents']
                filename = uploaded_data['filename']

                # Decode the file contents
                content_type, content_string = contents.split(',')
                decoded = base64.b64decode(content_string)

                # Parse the log file
                csv_path, df = parse_log_file(io.StringIO(decoded.decode('utf-8')))
                #print(df.head())

                total_cmd_count = len(df)

                # Store the data for this user
                self.user_data[user_id] = {
                    'full_df': df,  # Store the full dataframe
                    'data_processor': None,  # Will be initialized with range data
                    'viz_builder': None,  # Will be initialized later
                    'current_range': [0, min(self.default_range_size, total_cmd_count)]
                }

                return {'display': 'block'}, html.Div([
                    html.H5(f'Successfully parsed: {filename}', style={'color': 'green'}),
                    html.P(f"Total commands: {total_cmd_count:,}"),
                    html.Hr()
                ]), True, True, total_cmd_count

            except Exception as e:
                return {'display': 'none'}, html.Div([
                    html.H5('Error processing the file', style={'color': 'red'}),
                    html.P(str(e))
                ]), False, False, 0


        # Update dropdown options based on total command count
        @app.callback(
            Output('latency-range-dropdown', 'options'),
            Input('latency-total-cmd-count', 'data'),
            prevent_initial_call=True
        )
        def update_dropdown_options(total_cmd_count):
            if total_cmd_count <= 0:
                return []
            return self.get_range_options(total_cmd_count)

        # Update custom range end input max value
        @app.callback(
            [Output('latency-custom-start', 'max'),
             Output('latency-custom-end', 'max')],
            Input('latency-total-cmd-count', 'data'),
            prevent_initial_call=True
        )
        def update_custom_range_limits(total_cmd_count):
            if total_cmd_count <= 0:
                return 0, 0
            return total_cmd_count - 1, total_cmd_count

        @app.callback(
            [Output('latency-current-range', 'data'),
             Output('latency-processing-status', 'children'),
             Output('latency-loading-output', 'children')],
            [Input('latency-load-range-button', 'n_clicks'),
             Input('latency-next-range-button', 'n_clicks'),
             Input('latency-dashboard-initialized', 'data')],
            [State('latency-range-dropdown', 'value'),
             State('latency-custom-start', 'value'),
             State('latency-custom-end', 'value'),
             State('latency-current-range', 'data'),
             State('latency-total-cmd-count', 'data'),
             State('latency-user-id', 'data'),
             State('latency-use-custom-range', 'value')],
            prevent_initial_call=True
        )
        def load_selected_range(load_clicks, next_clicks, initialized,
                                dropdown_value, custom_start, custom_end,
                                current_range, total_cmd_count, user_id, use_custom_range):

            if not initialized or user_id not in self.user_data:
                return no_update, no_update, no_update

            trigger = ctx.triggered_id

            if trigger == 'latency-dashboard-initialized':
                start = 0
                end = min(self.default_range_size, total_cmd_count)

            elif trigger == 'latency-next-range-button':
                start = current_range[1]
                end = min(start + self.default_range_size, total_cmd_count)

                if start >= total_cmd_count:
                    return no_update, dbc.Alert("Already at the end of the command log.", color="warning"), no_update

            elif trigger == 'latency-load-range-button':
                if use_custom_range:
                    if custom_start is None or custom_end is None:
                        return no_update, dbc.Alert("Custom start/end must be provided.", color="danger"), no_update
                    if custom_start < 0 or custom_end <= custom_start or custom_end > total_cmd_count:
                        return no_update, dbc.Alert("Invalid custom range values.", color="danger"), no_update
                    if custom_end - custom_start > self.default_range_size:
                        return no_update, dbc.Alert(f"Custom range too large. Max {self.default_range_size} commands.",
                                                    color="danger"), no_update
                    start, end = custom_start, custom_end
                elif dropdown_value:
                    start, end = map(int, dropdown_value.split(','))
                else:
                    return no_update, dbc.Alert("Please select a range or enable custom range input.",
                                                color="warning"), no_update

            else:
                return no_update, no_update, no_update

            if start < 0 or end > total_cmd_count or start >= end:
                return no_update, dbc.Alert("Invalid range selected.", color="danger"), no_update

            try:
                full_df = self.user_data[user_id]['full_df']
                full_df = full_df.reset_index().rename(columns={"index": "cmd_index"})
                range_df = full_df[(full_df['cmd_index'] >= start) & (full_df['cmd_index'] < end)].copy()

                self.user_data[user_id]['data_processor'] = DataProcessor(range_df)
                self.user_data[user_id]['viz_builder'] = VisualizationBuilder(self.user_data[user_id]['data_processor'])
                self.user_data[user_id]['current_range'] = [start, end]

                return [start, end], dbc.Alert(f"Successfully loaded commands {start:,} to {end:,}",
                                               color="success"), None

            except Exception as e:
                return no_update, dbc.Alert(f"Error processing range: {str(e)}", color="danger"), no_update

        # Update current range display
        @app.callback(
            Output('latency-current-range-display', 'children'),
            Input('latency-current-range', 'data'),
            Input('latency-total-cmd-count', 'data'),
            prevent_initial_call=True
        )
        def update_range_display(current_range, total_cmd_count):
            if not current_range or total_cmd_count == 0:
                return ""

            start, end = current_range
            return html.Div([
                html.Strong("Currently Displaying: "),
                f"Commands {start:,} to {end:,} ",
                html.Span(f"({end - start:,} commands out of {total_cmd_count:,} total)")
            ], className="p-2 bg-light border rounded")

        # Enable/disable next range button
        @app.callback(
            Output('latency-next-range-button', 'disabled'),
            Input('latency-current-range', 'data'),
            Input('latency-total-cmd-count', 'data'),
            prevent_initial_call=True
        )
        def update_next_button_state(current_range, total_cmd_count):
            if not current_range or total_cmd_count == 0:
                return True

            start, end = current_range
            return end >= total_cmd_count

        # Update range stores
        @app.callback(
            [Output("latency-time-range", "data"),
             Output("latency-index-range", "data"),
             Output("latency-last-trigger", "data")],
            [Input("latency-current-range", "data"),
             Input("latency-main-plot", "relayoutData"),
             Input("latency-address-map", "relayoutData"),
             Input("latency-reset-button", "n_clicks")],
            [State("latency-time-range", "data"),
             State("latency-index-range", "data"),
             State("latency-last-trigger", "data"),
             State('latency-user-id', 'data')],
            prevent_initial_call=True
        )
        def update_range_stores(current_range, relayout_main, relayout_addr, reset_clicks,
                                stored_time_range, stored_idx_range, last_trigger, user_id):
            """This callback updates both time and index range stores based on user interactions"""
            if not current_range or user_id not in self.user_data:
                return no_update, no_update, no_update

            data_processor = self.user_data[user_id]['data_processor']
            trigger = ctx.triggered_id

            # Range changed or reset button
            if trigger == "latency-current-range" or trigger == "latency-reset-button":
                max_end = data_processor.get_max_time()
                time_range = [0, max_end]
                idx_range = [current_range[0], current_range[0] + len(data_processor.df) - 1]
                return time_range, idx_range, "reset"

            # Handle main plot time-based updates
            if trigger == "latency-main-plot":
                if not relayout_main:
                    return no_update, no_update, no_update

                time_range = None
                if "xaxis.range[0]" in relayout_main and "xaxis.range[1]" in relayout_main:
                    x0 = float(relayout_main["xaxis.range[0]"])
                    x1 = float(relayout_main["xaxis.range[1]"])
                    time_range = [x0, x1]
                elif "xaxis.autorange" in relayout_main and relayout_main["xaxis.autorange"]:
                    # Reset to full range
                    max_end = data_processor.get_max_time()
                    time_range = [0, max_end]

                if time_range:
                    # Convert time range to index range
                    idx_range = data_processor.time_to_index_range(time_range)
                    # Adjust indices to be relative to the full dataset
                    if idx_range:
                        idx_range = [idx + current_range[0] for idx in idx_range]
                    return time_range, idx_range, "latency-main-plot"

            # Handle address map index-based updates
            if trigger == "latency-address-map":
                if not relayout_addr:
                    return no_update, no_update, no_update

                idx_range = None
                if "xaxis.range[0]" in relayout_addr and "xaxis.range[1]" in relayout_addr:
                    x0 = float(relayout_addr["xaxis.range[0]"])
                    x1 = float(relayout_addr["xaxis.range[1]"])
                    # Adjust for current range offset
                    idx_range = [x0 + current_range[0], x1 + current_range[0]]
                elif "xaxis.autorange" in relayout_addr and relayout_addr["xaxis.autorange"]:
                    # Reset to full range
                    idx_range = [current_range[0], current_range[0] + len(data_processor.df) - 1]

                if idx_range:
                    # Convert index range to time range
                    time_range = data_processor.index_to_time_range(
                        [idx - current_range[0] for idx in idx_range])
                    return time_range, idx_range, "latency-address-map"

            return no_update, no_update, no_update

        @app.callback(
            Output("latency-main-plot", "figure"),
            [Input("latency-time-range", "data"),
             Input("latency-last-trigger", "data"),
             Input("latency-current-range", "data")],
            State('latency-user-id', 'data'),
            prevent_initial_call=True
        )
        def update_main_plot(time_range, last_trigger, current_range, user_id):
            if time_range is None or current_range is None or user_id not in self.user_data:
                return no_update

            # Don't update if this plot triggered the change (let plotly handle zoom/pan)
            if last_trigger == "latency-main-plot":
                return no_update

            data_processor = self.user_data[user_id]['data_processor']
            viz_builder = self.user_data[user_id]['viz_builder']
            filtered_df = data_processor.filter_by_time_range(time_range)
            fig = viz_builder.build_main_figure(filtered_df, x_range=time_range)

            # Update title to include range information
            start, end = current_range
            fig.update_layout(
                title_text=f"Command Timeline (Showing commands {start:,} to {end:,})"
            )

            return fig

        @app.callback(
            [Output('latency-custom-start', 'disabled'),
             Output('latency-custom-end', 'disabled')],
            Input('latency-use-custom-range', 'value'),
            prevent_initial_call=True
        )
        def toggle_custom_range_inputs(use_custom):
            return not use_custom, not use_custom

        @app.callback(
            Output("latency-address-map", "figure"),
            [Input("latency-index-range", "data"),
             Input("latency-time-range", "data"),
             Input("latency-last-trigger", "data"),
             Input("latency-current-range", "data")],
            State('latency-user-id', 'data'),
            prevent_initial_call=True
        )
        def update_address_map(idx_range, time_range, last_trigger, current_range, user_id):
            if time_range is None or current_range is None or user_id not in self.user_data:
                return no_update

            # Don't update if this plot triggered the change (let plotly handle zoom/pan)
            if last_trigger == "latency-address-map":
                return no_update

            data_processor = self.user_data[user_id]['data_processor']
            viz_builder = self.user_data[user_id]['viz_builder']
            filtered_df = data_processor.filter_by_time_range(time_range)

            # Adjust index range to be relative to the current range
            if idx_range:
                rel_idx_range = [idx - current_range[0] for idx in idx_range]
            else:
                rel_idx_range = None

            fig = viz_builder.build_address_map(filtered_df, idx_range=rel_idx_range)

            # Update title to include range information
            start, end = current_range
            fig.update_layout(
                title_text=f"Address Map (Showing commands {start:,} to {end:,})"
            )

            return fig

        @app.callback(
            [Output("latency-cmd-count-chart", "figure"),
             Output("latency-chunk-size-charts", "figure")],
            [Input("latency-time-range", "data"),
             Input("latency-current-range", "data")],
            State('latency-user-id', 'data'),
            prevent_initial_call=True
        )
        def update_command_charts(time_range, current_range, user_id):
            if time_range is None or current_range is None or user_id not in self.user_data:
                return no_update, no_update

            data_processor = self.user_data[user_id]['data_processor']
            viz_builder = self.user_data[user_id]['viz_builder']
            filtered_df = data_processor.filter_by_time_range(time_range)

            count_chart = viz_builder.build_cmd_count_bar_chart(filtered_df)
            chunk_charts = viz_builder.build_chunk_size_charts(filtered_df, time_range)

            # Update titles to include range information
            start, end = current_range
            count_chart.update_layout(
                title_text=f"Command Count Distribution (Commands {start:,} to {end:,})"
            )
            chunk_charts.update_layout(
                title_text=f"Chunk Size Distribution (Commands {start:,} to {end:,})"
            )

            return count_chart, chunk_charts

        @app.callback(
            Output("latency-stats-display", "children"),
            [Input("latency-time-range", "data"),
             Input("latency-current-range", "data")],
            State('latency-user-id', 'data'),
            prevent_initial_call=True
        )
        def update_stats_display(time_range, current_range, user_id):
            """Display command-wise statistics based on the selected time range"""
            if time_range is None or current_range is None or user_id not in self.user_data:
                return no_update

            data_processor = self.user_data[user_id]['data_processor']
            # Get statistics using the new method
            cmd_stats = data_processor.calculate_cmd_stats(time_range)

            # Build the stats display
            stats = []
            for cmd in sorted(cmd_stats.keys()):
                if cmd == "overall":
                    continue

                data = cmd_stats[cmd]
                color = data_processor.cmd_colors[cmd]

                stats.append(
                    html.Div([
                        html.Div([
                            html.Span(cmd.capitalize(), style={"color": color, "fontWeight": "bold"}),
                            html.Span(f" ({data['count']} commands)")
                        ]),
                        html.Div([
                            html.Span("Mean Latency: "),
                            html.Span(f"{data['mean_latency']:.2f} µs", style={"fontWeight": "bold"})
                        ])
                    ], className="stat-item mb-3")
                )

            # Add overall statistics
            overall = cmd_stats["overall"]
            stats.append(html.Hr())
            stats.append(
                html.Div([
                    html.Div([
                        html.Span("Total Commands: "),
                        html.Span(f"{overall['count']}", style={"fontWeight": "bold"})
                    ]),
                    html.Div([
                        html.Span("Overall Mean Latency: "),
                        html.Span(f"{overall['mean_latency']:.2f} µs", style={"fontWeight": "bold"})
                    ]),
                    html.Div([
                        html.Span("Max Latency: "),
                        html.Span(f"{overall['max_latency']:.2f} µs", style={"fontWeight": "bold"})
                    ])
                ], className="stat-item mt-2")
            )

            # Add time range info
            stats.append(html.Hr())
            stats.append(
                html.Div([
                    html.Div("Selected Time Range:"),
                    html.Div([
                        f"{time_range[0]:.0f} - {time_range[1]:.0f} µs ",
                        html.Span(f"({(time_range[1] - time_range[0]) / 1000:.2f} ms)")
                    ], style={"fontSize": "0.9em", "opacity": "0.8"})
                ], className="stat-item mt-2")
            )

            # Add command range info
            stats.append(html.Hr())
            stats.append(
                html.Div([
                    html.Div("Command Range:"),
                    html.Div([
                        f"{current_range[0]:,} - {current_range[1]:,} ",
                        html.Span(f"({current_range[1] - current_range[0]:,} commands)")
                    ], style={"fontSize": "0.9em", "opacity": "0.8"})
                ], className="stat-item mt-2")
            )

            return stats
        '''
        # Add dashboard content when process button is clicked
        @app.callback(
            Output('latency-dashboard-content', 'children'),
            Input('latency-process-button', 'n_clicks'),
            prevent_initial_call=True
        )
        def load_dashboard_content(n_clicks):
            if not n_clicks:
                return html.Div()
            return self.get_dashboard_content()
        '''

if __name__ == "__main__":
    # For standalone testing
    app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
    dashboard = DashboardApp()
    app.layout = dashboard.get_upload_layout()
    dashboard.register_callbacks(app)
    app.run(debug=True, host="0.0.0.0", port=8050)
