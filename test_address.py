from dash import Dash, dcc, html, Input, Output, State
import plotly.graph_objects as go
import pandas as pd
import numpy as np

# Device configuration
device_size_bytes = 1 * 1024 ** 4  # 1 TiB
lba_size_bytes = 4096  # 4 KB
total_lbas = device_size_bytes // lba_size_bytes

# Simulated command data
np.random.seed(42)
num_cmds = 40000
df = pd.DataFrame({
    'cmd index': np.arange(num_cmds),
    'cmd name': np.random.choice(['read', 'write', 'erase'], size=num_cmds, p=[0.4, 0.5, 0.1]),
    'LBA start': np.random.randint(0, total_lbas, size=num_cmds),
    'size': np.random.randint(1, 512, size=num_cmds),
})

# Define tick steps for dynamic range adjustments
tick_steps = [
    100 * 1024**3,  # 100 GiB
    10 * 1024**3,   # 10 GiB
    1 * 1024**3,    # 1 GiB
    100 * 1024**2,  # 100 MiB
    10 * 1024**2,   # 10 MiB
    1 * 1024**2,    # 1 MiB
    100 * 1024,     # 100 KiB
    10 * 1024,      # 10 KiB
    4096            # 4 KiB
]

# This function is used to format byte sizes into human-readable strings
def format_bytes(size):
    power = 1024
    n = 0
    labels = ['B', 'KB', 'MB', 'GB', 'TB']
    while size >= power and n < len(labels) - 1:
        size /= power
        n += 1
    return f"{size:.0f} {labels[n]}"


def format_bytes_new(num_bytes, idx):
    units = ['GB', 'GB', 'GB', 'MB', 'MB', 'MB', 'KB', 'KB', 'KB']
    multipliers = [100, 10, 1, 100, 10, 1, 100, 10, 4]

    return f"{num_bytes*multipliers[idx]/tick_steps[idx]} {units[idx]}"


def get_tick_step(byte_range):
    print(f"New {byte_range}\n")
    units = ['GB', 'GB', 'GB', 'MB', 'MB', 'MB', 'KB', 'KB', 'KB']
    for i, step in enumerate(tick_steps):
        #print(i, step)
        print("")
        print(byte_range/step)
        if (byte_range/step) < 11 and (byte_range/step) >=1:
            print(f"found step: {i},{step}, {byte_range/step}, {units[i]}")
            return i, step
    return len(tick_steps)-1, len(tick_steps)[-1]

def generate_ticks(byte_start, byte_end):
    #print(byte_start, byte_end)
    byte_range = byte_end - byte_start
    #print(byte_range)
    idx, tick_step = get_tick_step(byte_range)
    print(tick_step)
    first_tick = ((byte_start + tick_step - 1) // tick_step) * tick_step
    print(first_tick)
    byte_ticks = np.arange(first_tick, byte_end + 1, tick_step)
    print(byte_ticks)
    lba_ticks = byte_ticks / lba_size_bytes
    print(lba_ticks)
    tick_labels = [format_bytes(bt) for bt in byte_ticks]

    tick_labels = [format_bytes_new(bt, idx) for bt in byte_ticks]
    #print("here")
    print(tick_labels)
    return lba_ticks, tick_labels

def create_figure(y_range=None):
    colors = {'read': 'blue', 'write': 'red', 'erase': 'green'}

    data = [
        go.Scattergl(
            x=group['cmd index'],
            y=group['LBA start'],
            mode='markers',
            marker=dict(size=6, color=colors[cmd]),
            name=cmd
        )
        for cmd, group in df.groupby('cmd name')
    ]

    print("data done")

    if y_range is None:
        byte_start = 0
        byte_end = device_size_bytes
    else:
        byte_start, byte_end = [y * lba_size_bytes for y in y_range]

    print("if else done")

    ticks, labels = generate_ticks(byte_start, byte_end)

    print("tick done")
    fig = go.Figure(data)
    fig.update_layout(
        title='LBA Access Pattern',
        xaxis_title='Command Index',
        yaxis_title='Address (Bytes)',
        height=600
    )
    fig.update_yaxes(
        tickvals=ticks,
        ticktext=labels,
        autorange=not y_range,
        range=y_range if y_range else None
    )
    return fig

# Dash app setup
app = Dash(__name__)
initial_fig = create_figure()
app.layout = html.Div([
    html.Div([
        html.Label('Start Address:'),
        dcc.Input(id='start-addr', type='number', value=0),

        html.Label('End Address:'),
        dcc.Input(id='end-addr', type='number', value=1024),

        html.Label('Unit:'),
        dcc.Dropdown(
            id='addr-unit',
            options=[
                {'label': 'kB', 'value': 'KB'},
                {'label': 'MB', 'value': 'MB'},
                {'label': 'GB', 'value': 'GB'}
            ],
            value='MB',
            clearable=False
        ),
        html.Button('Apply Range', id='apply-range', n_clicks=0)
    ], style={'display': 'flex', 'gap': '10px', 'margin': '10px'}),



    dcc.Graph(id='address-plot', figure=initial_fig),
    dcc.Store(id='figure-store', data=initial_fig.to_dict())
])


@app.callback(
    Output('address-plot', 'figure'),
    Input('address-plot', 'relayoutData'),
    State('figure-store', 'data')
)
def update_ticks(relayout_data, fig_data):
    fig = go.Figure(fig_data)

    if not relayout_data or 'yaxis.range[0]' not in relayout_data:
        return fig

    y0 = relayout_data['yaxis.range[0]']
    y1 = relayout_data['yaxis.range[1]']
    print("new range:")
    print(y0, y1)
    new_range = [min(y0, y1), max(y0, y1)]
    return create_figure(y_range=new_range)

@app.callback(
    Output('address-plot', 'figure', allow_duplicate=True),
    Input('apply-range', 'n_clicks'),
    State('start-addr', 'value'),
    State('end-addr', 'value'),
    State('addr-unit', 'value'),
    prevent_initial_call = True
)
def apply_address_range(n_clicks, start_val, end_val, unit):
    print("in apply_address_range")
    if n_clicks == 0 or start_val is None or end_val is None:
        return create_figure()

    # Unit multipliers in bytes
    unit_multipliers = {
        'KB': 1024,
        'MB': 1024**2,
        'GB': 1024**3
    }

    start_bytes = start_val * unit_multipliers[unit]
    end_bytes = end_val * unit_multipliers[unit]

    # Convert bytes to LBA (4KB units)
    start_lba = start_bytes // lba_size_bytes
    end_lba = end_bytes // lba_size_bytes

    print(start_lba, end_lba)

    return create_figure(y_range=[start_lba, end_lba])


if __name__ == '__main__':
    app.run(debug=True, port=8052)
