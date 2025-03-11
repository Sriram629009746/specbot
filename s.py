import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.colors as pc
import pandas as pd
import streamlit as st

class DataPlotter:
    def __init__(self, df, age_column, variable_groups, group_types):
        self.df = df
        self.preprocess_data()
        self.age_column = age_column
        self.variable_groups = variable_groups
        self.group_types = group_types
        self.category_positions = {}
        self.high_contrast_colors = pc.qualitative.Set1
        #self.fig = make_subplots(rows=len(variable_groups), cols=1, row_heights=[1 / len(variable_groups)] * len(variable_groups), shared_xaxes=True, vertical_spacing=0.05)
        num_numerical = sum(1 for t in group_types if t == 'numerical')
        num_categorical = len(group_types) - num_numerical
        total = num_numerical * 0.7 + num_categorical * 0.3

        row_heights = [(0.7 / total) if t == 'numerical' else (0.3 / total) for t in group_types]

        self.fig = make_subplots(rows=len(variable_groups), cols=1, row_heights=row_heights, shared_xaxes=True,
                                 vertical_spacing=0.1)

    def preprocess_data(self):
        self.df.columns = self.df.columns.str.strip()

        #return df

    def plot_numerical(self, row, columns, legend_group):
        for col in columns:
            self.fig.add_trace(go.Scatter(
                x=self.df[self.age_column],
                y=self.df[col],
                mode='lines+markers',
                name=col,
                legendgroup=legend_group,
                showlegend=True
            ), row=row, col=1)
        self.fig.update_yaxes(fixedrange=True, row=row, col=1)

    def plot_categorical(self, row, columns):
        for i, var in enumerate(columns):
            if var not in self.category_positions:
                self.category_positions[var] = 0.2 + 0.2 * i
            unique_values = self.df[var].unique()
            color_map = {value: self.high_contrast_colors[i % len(self.high_contrast_colors)] for i, value in enumerate(unique_values)}

            for value in unique_values:
                self.fig.add_trace(go.Scatter(
                    x=[None],
                    y=[None],
                    mode='lines',
                    line=dict(color=color_map[value], width=3),
                    name=f'{var}: {value}',
                    legendgroup=f'categorical_{var}',
                    showlegend=True
                ))

            for i in range(1, len(self.df)):
                self.fig.add_trace(go.Scatter(
                    x=[self.df[self.age_column].iloc[i - 1], self.df[self.age_column].iloc[i]],
                    y=[self.category_positions[var]] * 2,
                    mode='lines',
                    line=dict(color=color_map[self.df[var].iloc[i]], width=3),
                    name=f'{var}: {self.df[var].iloc[i]}',
                    legendgroup=f'categorical_{var}',
                    showlegend=False,
                    text=f'{var}:{self.df[var].iloc[i]}',
                    hovertemplate='(%{x}, %{text})<extra></extra>'
                ), row=row, col=1)

        self.fig.update_yaxes(
            tickvals=list(self.category_positions.values()),
            ticktext=list(self.category_positions.keys()),
            row=row, col=1,
            fixedrange=True
        )

    def configure_xaxis(self):
        for row in range(1, len(self.variable_groups) + 1):
            self.fig.update_xaxes(
                tickmode='array',
                tickvals=self.df[self.age_column],
                ticktext=[str(a) for a in self.df[self.age_column]],
                row=row, col=1,
                showticklabels=True,
                range=[0, 20],
                title_text=self.age_column
            )

    def configure_layout(self):
        self.fig.update_layout(
            height=900,
            title_text='Numerical and Categorical Variables Subplot',
            legend_tracegroupgap=50,
            dragmode='pan'
        )

    def plot(self):
        for row, (columns, var_type) in enumerate(zip(self.variable_groups, self.group_types), start=1):
            if var_type == 'numerical':
                self.plot_numerical(row=row, columns=columns, legend_group=f'numerical_{row}')
            elif var_type == 'categorical':
                self.plot_categorical(row=row, columns=columns)

        self.configure_xaxis()
        self.configure_layout()
        st.plotly_chart(self.fig,use_container_width=True)

        self.fig.show()

# Streamlit interface
st.title('Data Visualization App')

uploaded_file = st.file_uploader('Upload your CSV file', type='csv')

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    age_column = 'Age'

    numerical_cols = ['Temp', 'Poweroff Count']
    categorical_cols = ['Day of Week', 'Month', 'Month1']
    stats_cols = ['Error Rate', 'Failure Rate']

    variable_groups = [numerical_cols, categorical_cols, stats_cols]
    group_types = ['numerical', 'categorical', 'numerical']

    plotter = DataPlotter(df, age_column, variable_groups, group_types)
    plotter.plot()
