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
<<<<<<< HEAD
from pages.sections.regression.utils import create_table_description, session_get_file_path, session_df_to_file
=======
from pages.sections.regression.utils import create_table_description, session_get_file_path, session_df_to_file, continue_button, reset_button, session_dict_to_json
>>>>>>> f23d340 (regression done)

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

def preprocess_all_and_visualize(df: pd.DataFrame, column_schemas, preprocessed_file_name='preprocessed'):

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
        
    session_df_to_file(df, preprocessed_file_name)

    # Return the processed table description and scatter plot
    return create_table_description(df), scatter_fig

def create_form(column):

<<<<<<< HEAD
    form = ModelForm(
        DataPreprocessingSchema,
        "data_preprocessing",
        column,
        fields_repr={
            "missing_value_handling": fields.Select(
                options_labels={"remove": "Remove", "fill_mean": "Fill mean", "fill_median": "Fill median", "fill_mode": "Fill mode", "custom": "Custom"},
=======
    form = dmc.Stack(
        [
            dmc.Select(
                id={'class':'regression', 'type':"missing_value_handling", 'column': column},
                label="Missing Value Handling",
>>>>>>> f23d340 (regression done)
                description="Method for handling missing values",
                default="fill_mean",
                required=True,
            ),
<<<<<<< HEAD
            "custom_value": {
                "visible": ("missing_value_handling", "==", "custom"),
            },
            "scaling_method": fields.RadioItems(
                options_labels={"standard": "Standard", "minmax": "MinMax", "robust": "Robust"},
=======
            dmc.NumberInput(
                id={'class':'regression', 'type':"custom_value", 'column': column},
                label="Custom Value",
                description="Custom value for filling missing data, used if missing_value_handling is 'custom'",
                value=0,
                required=False,
                display="none",
            ),
            dmc.Select(
                id={'class':'regression', 'type':"scaling_method", 'column': column},
                label="Scaling Method",
>>>>>>> f23d340 (regression done)
                description="Scaling method to be applied to numerical features",
                default="standard",
                required=True,
            ),
<<<<<<< HEAD
            "outlier_handling": fields.RadioItems(
                options_labels={"remove": "Remove", "cap": "Cap", "none": "None"},
=======
            dmc.Select(
                id={'class':'regression', 'type':"outlier_handling", 'column': column},
                label="Outlier Handling",
>>>>>>> f23d340 (regression done)
                description="Method for handling outliers",
                default="none",
                required=True,
            ),
<<<<<<< HEAD
            "cap_value": {
                "visible": ("outlier_handling", "==", "cap"),
            },
        },
=======
            dmc.NumberInput(
                id={'class':'regression', 'type':"cap_value", 'column': column},
                label="Cap Value",
                description="Value to cap outliers, used if outlier_handling is 'cap'",
                value=0,
                required=False,
                display="none",
            ),
        ]
>>>>>>> f23d340 (regression done)
    )
    
    return dmc.Stack(
        [
            form,
            dmc.Flex(
<<<<<<< HEAD
                dmc.Button("Apply", color="blue", id={'type': 'apply_preprocessing', 'form_id': column}, n_clicks=0, w=200),
                justify="end",
            ),
            html.Div(id={"type": "output", 'form_id': column}),
=======
                dmc.Button("Apply", color="blue", id={'class':'regression', 'type': 'apply_preprocessing', 'column': column}, n_clicks=0, w=200),
                justify="end",
            ),
            html.Div(id={'class':'regression', 'type': "output", 'column': column}),
>>>>>>> f23d340 (regression done)
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
<<<<<<< HEAD
            dmc.Button("Apply All Preprocessing", color="green", id='apply_preprocessing', n_clicks=0),
            html.Div(id="output"),
=======
            dmc.Button("Apply All Preprocessing", color="green", id='regression-apply_preprocessing', n_clicks=0),
            html.Div(id="regression-output"),
            dmc.Group(
                [
                    reset_button,
                    html.Div(
                        id='regression-proceed-output',
                    )
                ],
                justify="space-between",
            )
>>>>>>> f23d340 (regression done)
        ]
    )
    
    return layout

@callback(
<<<<<<< HEAD
    Output({"type": "output", 'form_id': MATCH},'children'),
    Input({"type": "apply_preprocessing", 'form_id': MATCH}, 'n_clicks'),
    State(ModelForm.ids.main("data_preprocessing", MATCH), "data"),
    State({"type": "apply_preprocessing", 'form_id': MATCH}, 'id'),
)
def preprcess_each_column(n_clicks, form_data, button_id):
=======
    Output({'class':'regression', 'type': "custom_value", 'column': MATCH}, 'display'),
    Input({'class':'regression', 'type': "missing_value_handling", 'column': MATCH}, 'value'),
    prevent_initial_call=True
)
def toggle_custom_value(value):
    if value == "custom":
        return "block"
    else:
        return "none"

@callback(
    Output({'class':'regression', 'type': "cap_value", 'column': MATCH}, 'display'),
    Input({'class':'regression', 'type': "outlier_handling", 'column': MATCH}, 'value'),
    prevent_initial_call=True
)
def toggle_cap_value(value):
    if value == "cap":
        return "block"
    else:
        return "none"

@callback(
    Output({'class':'regression', 'type': "output", 'column': MATCH},'children'),
    Input({'class':'regression', 'type': "apply_preprocessing", 'column': MATCH}, 'n_clicks'),
    State({'class':'regression', 'type': 'missing_value_handling', 'column': MATCH}, 'value'),
    State({'class':'regression', 'type': 'custom_value', 'column': MATCH}, 'value'),
    State({'class':'regression', 'type': 'scaling_method', 'column': MATCH}, 'value'),
    State({'class':'regression', 'type': 'outlier_handling', 'column': MATCH}, 'value'),
    State({'class':'regression', 'type': 'cap_value', 'column': MATCH}, 'value'),
    State({'class':'regression', 'type': "apply_preprocessing", 'column': MATCH}, 'id'),
)
def preprcess_each_column(n_clicks, missing_value_handling, custom_value, scaling_method, outlier_handling, cap_value, button_id):
>>>>>>> f23d340 (regression done)
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
<<<<<<< HEAD
    Output("output", "children"),
    Input('apply_preprocessing', 'n_clicks'),
    State(ModelForm.ids.main("data_preprocessing", ALL), "data"),
    State({"type": "apply_preprocessing", 'form_id': ALL}, 'id'),
=======
    Output("regression-output", "children"),
    Output("regression-proceed-output", "children", allow_duplicate=True),
    Input('regression-apply_preprocessing', 'n_clicks'),
    State({'class':'regression', 'type': 'missing_value_handling', 'column': ALL}, 'value'),
    State({'class':'regression', 'type': 'custom_value', 'column': ALL}, 'value'),
    State({'class':'regression', 'type': 'scaling_method', 'column': ALL}, 'value'),
    State({'class':'regression', 'type': 'outlier_handling', 'column': ALL}, 'value'),
    State({'class':'regression', 'type': 'cap_value', 'column': ALL}, 'value'),
    State({'class':'regression', 'type': "apply_preprocessing", 'column': ALL}, 'id'),
    prevent_initial_call = True
>>>>>>> f23d340 (regression done)
)
def preprocess_all(n_clicks, form_datas, button_ids):
    if n_clicks > 0:
        
        columns = [button_id['form_id'] for button_id in button_ids]
        
        try:
            
<<<<<<< HEAD
            columns_schemas = {columns[i]: form_datas[i] for i in range(len(columns))}
            
            df = pd.read_csv(session_get_file_path('rawdata', extension='csv'))
=======
            schemas = [{
                    'missing_value_handling': missing_value_handling,
                    'custom_value': custom_value,
                    'scaling_method': scaling_method,
                    'outlier_handling': outlier_handling,
                    'cap_value': cap_value
                } for missing_value_handling, custom_value, scaling_method, outlier_handling, cap_value in zip(missing_value_handlings, custom_values, scaling_methods, outlier_handlings, cap_values)
            ]
            
            columns_schemas = {columns[i]: schemas[i] for i in range(len(columns))}
            session_dict_to_json(columns_schemas, 'preprocessing')
            
            df = pd.read_csv(session_get_file_path('rawdata', extension='csv'), index_col=0)
>>>>>>> f23d340 (regression done)
            
            result = preprocess_all_and_visualize(df, columns_schemas)

            output = dmc.Stack(
                [
                    result[0],
                    dcc.Graph(figure=result[1], style={'height': '1000px'})
                ]
            )
            
            return output
            
        except Exception as exc:
<<<<<<< HEAD
=======
            
            print(exc)
            
>>>>>>> f23d340 (regression done)
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
            