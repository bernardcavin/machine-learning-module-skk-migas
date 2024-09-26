from dash_pydantic_form import ModelForm, fields
from pydantic import BaseModel, Field, ValidationError
from typing import List, Optional, Union, Literal
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import plotly.express as px
import plotly.graph_objects as go
from typing import List, Dict
from dash import Input, Output, callback, html, State, dcc, MATCH, ALL
from dash.exceptions import PreventUpdate
import dash_mantine_components as dmc
from pages.sections.regression.utils import create_table_description, session_get_file_path, session_df_to_file

class DataPreprocessingSchema(BaseModel):
    missing_value_handling: Literal["remove", "fill_mean", "fill_median", "fill_mode", "custom"] = Field(
        "remove", 
        description="Method for handling missing values"
    )
    custom_value: Optional[float] = Field(
        0, 
        description="Custom value for filling missing data, used if missing_value_handling is 'custom'"
    )
    scaling_method: Optional[Literal["standard", "minmax", "robust"]] = Field(
        "standard", 
        description="Scaling method to be applied to numerical features"
    )
    outlier_handling: Optional[Literal["remove", "cap", "none"]] = Field(
        "remove", 
        description="Method for handling outliers"
    )
    cap_value: Optional[float] = Field(
        0, 
        description="Value to cap outliers, used if outlier_handling is 'cap'"
    )

def preprocess_and_visualize_series(series: pd.Series, schema: DataPreprocessingSchema):
    visualizations = {
        'before':{},
        'after':{}
    }

    # Initial Data Distribution
    fig_hist = px.histogram(series, x=series.name, title=f"Distribution of {series.name} Before Preprocessing", template='simple_white')
    fig_hist.update_layout(margin=dict(l=20, r=20, t=50, b=20))
    fig_box = px.box(series, y=series.name, title=f"Box Plot of {series.name} Before Preprocessing", template='simple_white')
    fig_box.update_layout(margin=dict(l=20, r=20, t=50, b=20))

    visualizations['before'] = {'histogram': fig_hist, 'box': fig_box}

    # Handling missing values
    if schema.missing_value_handling == "remove":
        series = series.dropna()
    elif schema.missing_value_handling == "fill_mean":
        series = series.fillna(series.mean())
    elif schema.missing_value_handling == "fill_median":
        series = series.fillna(series.median())
    elif schema.missing_value_handling == "fill_mode":
        series = series.fillna(series.mode()[0])
    elif schema.missing_value_handling == "custom" and schema.custom_value is not None:
        series = series.fillna(schema.custom_value)

    # Handling outliers
    if schema.outlier_handling == "remove":
        mean = series.mean()
        std = series.std()
        series = series[(np.abs(series - mean) <= (3 * std))]
    elif schema.outlier_handling == "cap" and schema.cap_value is not None:
        series = pd.Series(np.where(series > schema.cap_value, schema.cap_value, series.values), name=series.name)

    # Scaling numerical feature
    if schema.scaling_method:
        scaler = None
        if schema.scaling_method == "standard":
            scaler = StandardScaler()
        elif schema.scaling_method == "minmax":
            scaler = MinMaxScaler()
        elif schema.scaling_method == "robust":
            scaler = RobustScaler()
        
        series = pd.Series(scaler.fit_transform(series.values.reshape(-1, 1)).flatten(), name=series.name)

        # Visualize scaled data
        fig_hist_scaled = px.histogram(series, x=series.name, title=f"Distribution of {series.name} After Scaling ({schema.scaling_method})", template='simple_white')
        fig_hist_scaled.update_layout(margin=dict(l=20, r=20, t=50, b=20))
        fig_box_scaled = px.box(series, y=series.name, title=f"Box Plot of {series.name} After Scaling ({schema.scaling_method})", template='simple_white')
        fig_box_scaled.update_layout(margin=dict(l=20, r=20, t=50, b=20))

        visualizations['after'] = {'histogram': fig_hist_scaled, 'box': fig_box_scaled}

    return visualizations

def preprocess_all_and_visualize(df: pd.DataFrame, column_schemas):

    # Apply preprocessing to each column according to the schema provided
    for column, schema in column_schemas.items():
        # Handle missing values
        if schema['missing_value_handling'] == "remove":
            df = df.dropna(subset=[column])
        elif schema['missing_value_handling'] == "fill_mean":
            df[column] = df[column].fillna(df[column].mean())
        elif schema['missing_value_handling'] == "fill_median":
            df[column] = df[column].fillna(df[column].median())
        elif schema['missing_value_handling'] == "fill_mode":
            df[column] = df[column].fillna(df[column].mode()[0])
        elif schema['missing_value_handling'] == "custom" and schema.get('custom_value') is not None:
            df[column] = df[column].fillna(schema['custom_value'])

        # Handle outliers
        if schema['outlier_handling'] == "remove":
            mean = df[column].mean()
            std = df[column].std()
            df = df[(np.abs(df[column] - mean) <= (3 * std))]
        elif schema['outlier_handling'] == "cap" and schema.get('cap_value') is not None:
            df[column] = np.where(df[column] > schema['cap_value'], schema['cap_value'], df[column])

        # Scale numerical features
        if schema.get('scaling_method'):
            scaler = None
            if schema['scaling_method'] == "standard":
                scaler = StandardScaler()
            elif schema['scaling_method'] == "minmax":
                scaler = MinMaxScaler()
            elif schema['scaling_method'] == "robust":
                scaler = RobustScaler()

            df[column] = scaler.fit_transform(df[[column]])

    # Generate scatter plot matrix for numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    scatter_fig = None
    if len(numeric_cols) > 1:
        scatter_fig = px.scatter_matrix(df, dimensions=numeric_cols, title="Scatter Matrix After Preprocessing", template='simple_white')
        scatter_fig.update_layout(margin=dict(l=20, r=20, t=50, b=20))
        
    session_df_to_file(df, 'preprocessed')

    # Return the processed table description and scatter plot
    return create_table_description(df), scatter_fig

def create_form(column):

    form = ModelForm(
        DataPreprocessingSchema,
        "data_preprocessing",
        column,
        fields_repr={
            "missing_value_handling": fields.Select(
                options_labels={"remove": "Remove", "fill_mean": "Fill mean", "fill_median": "Fill median", "fill_mode": "Fill mode", "custom": "Custom"},
                description="Method for handling missing values",
                default="fill_mean",
                required=True,
            ),
            "custom_value": {
                "visible": ("missing_value_handling", "==", "custom"),
            },
            "scaling_method": fields.RadioItems(
                options_labels={"standard": "Standard", "minmax": "MinMax", "robust": "Robust"},
                description="Scaling method to be applied to numerical features",
                default="standard",
                required=True,
            ),
            "outlier_handling": fields.RadioItems(
                options_labels={"remove": "Remove", "cap": "Cap", "none": "None"},
                description="Method for handling outliers",
                default="none",
                required=True,
            ),
            "cap_value": {
                "visible": ("outlier_handling", "==", "cap"),
            },
        },
    )
    
    return dmc.Stack(
        [
            form,
            dmc.Flex(
                dmc.Button("Apply", color="blue", id={'type': 'apply_preprocessing', 'form_id': column}, n_clicks=0, w=200),
                justify="end",
            ),
            html.Div(id={"type": "output", 'form_id': column}),
        ]
    )

tab_style = {
  'border': f'2px solid {dmc.theme.DEFAULT_THEME["colors"]["blue"][6]}',
  'border-radius': '5px',
  'bg-color': dmc.theme.DEFAULT_THEME["colors"]["red"][6]
}

def layout():
    
    df = pd.read_csv(session_get_file_path('rawdata',extension='csv'))

    columns = df.select_dtypes(include=[np.number]).columns

    tab_list = dmc.Paper(
        dmc.Stack(
            [
                dmc.Title("Columns", size="md"),
                dmc.TabsList(
                    dmc.SimpleGrid(
                        [
                            dmc.TabsTab(column, value=column, style=tab_style) for column in columns
                        ],
                        cols=5,
                        w='100%'
                    ),
                ),
            ],
            m=20,
            gap=20
        ),
        withBorder=True,
        shadow=0
    )

    tabs = dmc.Tabs(
        dmc.Stack(
            [tab_list] + [
                dmc.Paper(
                    [
                        dmc.TabsPanel(
                            create_form(column), 
                            value=column, 
                            m=20
                        ) for column in columns
                    ], 
                    withBorder=True, 
                    shadow=0)]
            ),
        variant="pills",
        value=columns[0],
    )

    layout = dmc.Stack(
        [
            create_table_description(df),
            tabs,
            dmc.Button("Apply All Preprocessing", color="green", id='apply_preprocessing', n_clicks=0),
            html.Div(id="output"),
        ]
    )
    
    return layout

@callback(
    Output({"type": "output", 'form_id': MATCH},'children'),
    Input({"type": "apply_preprocessing", 'form_id': MATCH}, 'n_clicks'),
    State(ModelForm.ids.main("data_preprocessing", MATCH), "data"),
    State({"type": "apply_preprocessing", 'form_id': MATCH}, 'id'),
)
def preprcess_each_column(n_clicks, form_data, button_id):
    if n_clicks > 0:
        
        column = button_id['form_id']
        
        try:
            df = pd.read_csv(session_get_file_path('rawdata', extension='csv'))
            
            visualizations = preprocess_and_visualize_series(df[column], DataPreprocessingSchema(**form_data))
            
            graphs = [
                dcc.Graph(figure=visualizations['before']['histogram']),
                dcc.Graph(figure=visualizations['after']['histogram']),
                dcc.Graph(figure=visualizations['before']['box']),
                dcc.Graph(figure=visualizations['after']['box'])
            ]
            
            output = dmc.Stack(
                [
                    dmc.SimpleGrid(
                        graphs,
                        cols=4,
                    ),
                ]
            )
            
        except ValidationError as exc:
            return html.Div(
                [
                    dmc.Alert(
                        "There was an error preprocessing your data.",
                        color="red",
                        variant="filled",
                    )
                ]
            )
        return output
    else:
        raise PreventUpdate
    
@callback(
    Output("output", "children"),
    Input('apply_preprocessing', 'n_clicks'),
    State(ModelForm.ids.main("data_preprocessing", ALL), "data"),
    State({"type": "apply_preprocessing", 'form_id': ALL}, 'id'),
)
def preprocess_all(n_clicks, form_datas, button_ids):
    if n_clicks > 0:
        
        columns = [button_id['form_id'] for button_id in button_ids]
        
        try:
            
            columns_schemas = {columns[i]: form_datas[i] for i in range(len(columns))}
            
            df = pd.read_csv(session_get_file_path('rawdata', extension='csv'))
            
            result = preprocess_all_and_visualize(df, columns_schemas)

            output = dmc.Stack(
                [
                    result[0],
                    dcc.Graph(figure=result[1], style={'height': '1000px'})
                ]
            )
            
            return output
            
        except Exception as exc:
            return html.Div(
                [
                    dmc.Alert(
                        "There was an error preprocessing your data.",
                        color="red",
                        variant="filled",
                    )
                ]
            )
        return output
    else:
        raise PreventUpdate
            