# import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from io import StringIO


# Function to convert hex string to integer
def hex_to_int(hex_str):
    if isinstance(hex_str, str) and hex_str.startswith('0x'):
        return int(hex_str, 16)
    elif isinstance(hex_str, str) and hex_str.isdigit():
        return int(hex_str)
    else:
        try:
            return int(hex_str)
        except:
            return 0


# Function to process categorical variables for visualization
def process_categorical_data(df, cat_var_name, y_offset):
    # Get unique values in the categorical variable
    unique_vals = df[cat_var_name].unique()

    # Create a color mapping for the categorical values
    color_map = {}
    colors = ['red', 'blue', 'gray']

    for i, val in enumerate(unique_vals):
        color_map[val] = colors[i % len(colors)]

    # Create segments for visualization
    segments = []
    annotations = []
    for i in range(len(df) - 1):

        color = color_map[df.iloc[i][cat_var_name]]
        segment = {
            'x0': i,
            'x1': i+1,
            'y0': y_offset,  # Offset to create parallel lines for different categorical variables
            'y1': y_offset,
            'color': color,
            'name': f"{cat_var_name}: {df.iloc[i][cat_var_name]}"
        }

        print(df.iloc[i]['Age_int'])
        segments.append(segment)

        # Add annotation for the segment
        mid_x = (df.iloc[i]['Age_int'] + df.iloc[i + 1]['Age_int']) / 2
        annotations.append({
            'x': mid_x,
            'y': y_offset * 1.1,  # Slightly above the line
            #'text': str(df.iloc[i][cat_var_name]),
            'showarrow': False,
            'font': {'size': 10, 'color': 'black'}
        })

    annotations.append({
        'x': df['Age_int'].min() - 2,  # Position to the left of the plot
        'y': y_offset,
        'text': cat_var_name,
        'showarrow': False,
        'font': {'size': 12, 'color': 'black'},#, 'bold': True},
        #'xanchor': 'right'
    })

    return segments, color_map, annotations


# Sidebar for file upload and configurati
# Sample data for demonstration if no file is uploaded

# uploaded_file = pd.read_csv()
if True:
    sample_data = pd.DataFrame({
        'Age': ['0xF', '0xD', '0xB', '0x9', '0x7', '0x6', '0x5', '0x3', '0x2', '0x1'],
        'PowerOffCount': [75, 82, 90, 95, 100, 75, 82, 90, 95, 100],
        'Temperature': [35, 38, 42, 45, 50, 35, 38, 42, 45, 50],
        'ErrorRate': [0.5, 1.2, 2.1, 3.0, 3.8, 0.5, 1.2, 2.1, 3.0, 3.8],
        'PerformanceScore': [850, 800, 750, 700, 650, 850, 800, 750, 700, 650],
        'WriteType': ['Sequential', 'Random', 'None', 'Sequential', 'Random', 'Sequential', 'Random', 'None',
                      'Sequential', 'Random'],
        'ReadType': ['Seq', 'Ran', 'Seq', 'NA', 'Seq', 'Seq', 'Ran', 'Seq',
                     'NA', 'Seq']
    })

    # st.dataframe(sample_data)

    # Convert Age to integer for sorting
    sample_data['Age_int'] = sample_data['Age'].apply(hex_to_int)

    # Sort data by Age (descending)
    sample_data = sample_data.sort_values('Age_int', ascending=False).reset_index(drop=True)

    y_min = 0
    y_max = (sample_data[['PowerOffCount', 'Temperature', 'ErrorRate', 'PerformanceScore']].max().max()) * 1.1

    # Calculate y-positions for categorical variables (at the top of the plot)
    write_y_position = y_max * 1.05  # 5% above the top of the plot
    read_y_position = y_max * 1.12  # 12% above the top of the plot

    # Process categorical variables
    write_segments, write_color_map, write_annotations = process_categorical_data(sample_data, 'WriteType', write_y_position)
    #read_segments, read_color_map, read_annotations = process_categorical_data(sample_data, 'ReadType',read_y_position)

    # print(write_segments)

    # Create figure with multiple y-axes
    numerical_vars = ['PowerOffCount', 'Temperature', 'ErrorRate', 'PerformanceScore']

    # Create a subplot for each numerical variable
    fig = make_subplots(rows=1, cols=1, shared_xaxes=True,specs=[[{'secondary_y': True}]])

    # Generate colors for numerical variables
    num_colors = ['rgba(31, 119, 180, 1)', 'rgba(255, 127, 14, 1)',
                  'rgba(44, 160, 44, 1)', 'rgba(214, 39, 40, 1)']

    # Create axis configuration
    y_axis_config = {}

    print()

    # Add numerical variables with different y-axes
    for i, var in enumerate(numerical_vars):
        print(var)
        yaxis_name = f"yaxis{i + 1 if i > 0 else ''}"
        print(yaxis_name)

        # Add trace
        fig.add_trace(
            go.Scatter(
                x=sample_data['Age'],  # Use original hex values
                y=sample_data[var],
                mode='lines+markers',
                name=var,
                line=dict(color=num_colors[i], width=2),
                marker=dict(size=5, color=num_colors[i], line=dict(width=1, color='black')),
                hovertemplate=f"{var}: %{{y}}<extra></extra>",
                yaxis=f"y{i + 1}" if i > 0 else "y"
            ),

            #secondary_y=True if i > 0 else False
            #secondary_y=True
        )

        # Configure the y-axis with updated property names
        # position = 0 if i == 0 else 1 - (i * 0.08)

        # position = 0 if i == 0 else (i * 0.04)
        position = (i * 0.06)
        y_axis_config[yaxis_name] = dict(
            title=dict(
                text=var,
                font=dict(color=num_colors[i])
            ),
            tickfont=dict(color=num_colors[i]),
            anchor="free" if i > 0 else "x",
            overlaying="y" if i > 0 else None,
            side="left",
            position=position #if i > 0 else 0
        )


    # check
    # Add categorical variables as shapes
    '''
    for segment in write_segments:# + read_segments:
        fig.add_shape(
            type="line",
            x0=segment['x0'],
            y0=segment['y0'] * max(sample_data[numerical_vars].max()) / 3,
            x1=segment['x1'],
            y1=segment['y1'] * max(sample_data[numerical_vars].max()) / 3,
            line=dict(color=segment['color'], width=3),
            name=segment['name']
        )
    '''
    # Collect all annotations
    #all_annotations = write_annotations# + read_annotations


    # Update layout with configured axes
    layout_update = {
        "title_text": "Disk Drive Data Visualization (Sample)",
        "template": "plotly_white",
        "hovermode": "x unified",
        "height": 700,
        "legend_title_text": 'Variables',
        "xaxis": {
            "title": "Age (Hex Values)",
            "showgrid": True,
            "zeroline": False,
            "type": 'category',  # Use category type to preserve hex values
            "domain": [0.3, 1],
            #"type":'linear'
        },
        #"annotations": all_annotations
    }

    # Add the y-axis configs to the layout update
    layout_update.update(y_axis_config)

    # Update layout
    fig.update_layout(**layout_update)


    # Add categorical variable legends manually
    for write_type, color in write_color_map.items():
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode='lines',
                name=f'Write: {write_type}',
                line=dict(color=color, width=3)
            )
        )

    """
    for read_type, color in read_color_map.items():
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode='lines',
                name=f'Read: {read_type}',
                line=dict(color=color, width=3)
            )
        )
    """
    # Show plot
    # st.plotly_chart(fig, use_container_width=True)
    fig.update_yaxes(visible=False)
    fig.show()

    # Show instructions
