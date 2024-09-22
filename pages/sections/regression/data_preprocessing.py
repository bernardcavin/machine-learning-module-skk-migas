from dash_pydantic_form import ModelForm, fields
from pydantic import BaseModel, Field, ValidationError
from typing import List, Optional, Union, Literal
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import plotly.express as px
import plotly.graph_objects as go
from typing import List, Dict
from dash import Input, Output, callback, html, State, dcc
from dash.exceptions import PreventUpdate
import dash_mantine_components as dmc
from pages.sections.regression.utils import create_table_description, session_get_file_path

class DataPreprocessingSchema(BaseModel):
    missing_value_handling: Literal["remove", "fill_mean", "fill_median", "fill_mode", "custom"] = Field(
        ..., 
        description="Method for handling missing values"
    )
    custom_value: Optional[float] = Field(
        None, 
        description="Custom value for filling missing data, used if missing_value_handling is 'custom'"
    )
    scaling_method: Optional[Literal["standard", "minmax", "robust"]] = Field(
        None, 
        description="Scaling method to be applied to numerical features"
    )
    encoding_method: Optional[Literal["onehot", "label"]] = Field(
        None, 
        description="Encoding method for categorical features"
    )
    outlier_handling: Optional[Literal["remove", "cap", "none"]] = Field(
        None, 
        description="Method for handling outliers"
    )
    cap_value: Optional[float] = Field(
        None, 
        description="Value to cap outliers, used if outlier_handling is 'cap'"
    )
    
def preprocess_and_visualize(df: pd.DataFrame, schema: DataPreprocessingSchema):
    visualizations = {
        'before':{},
        'after':{}
    }

    # Initial Data Distribution
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        figures = {}
        fig = px.histogram(df, x=col, title=f"Distribution of {col} Before Preprocessing")
        fig.update_layout(
            margin=dict(l=20, r=20, t=20, b=20),
        )
        
        figures['histogram'] = fig

        fig_box = px.box(df, y=col, title=f"Box Plot of {col} Before Preprocessing")
        fig_box.update_layout(
            margin=dict(l=20, r=20, t=20, b=20),
        )
        figures['box'] = fig_box

        visualizations['before'][col] = figures

    # Handling missing values
    if schema.missing_value_handling == "remove":
        df = df.dropna()
    elif schema.missing_value_handling == "fill_mean":
        df = df.fillna(df.mean(numeric_only=True))
    elif schema.missing_value_handling == "fill_median":
        df = df.fillna(df.median(numeric_only=True))
    elif schema.missing_value_handling == "fill_mode":
        df = df.fillna(df.mode(numeric_only=True).iloc[0])
    elif schema.missing_value_handling == "custom" and schema.custom_value is not None:
        df = df.fillna(schema.custom_value)

    # Handling outliers
    if schema.outlier_handling == "remove":
        df = df[(np.abs(df[numeric_cols] - df[numeric_cols].mean()) <= (3 * df[numeric_cols].std())).all(axis=1)]
    elif schema.outlier_handling == "cap" and schema.cap_value is not None:
        for col in numeric_cols:
            df[col] = np.where(df[col] > schema.cap_value, schema.cap_value, df[col])

    # Scaling numerical features
    if schema.scaling_method:
        scaler = None
        if schema.scaling_method == "standard":
            scaler = StandardScaler()
        elif schema.scaling_method == "minmax":
            scaler = MinMaxScaler()
        elif schema.scaling_method == "robust":
            scaler = RobustScaler()
        
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

        # Visualize scaled data
        for col in numeric_cols:
            figures = {}
            fig_scaled = px.histogram(df, x=col, title=f"Distribution of {col} After Scaling ({schema.scaling_method})")
            figures['histogram'] = fig_scaled
            fig_scaled.update_layout(
                margin=dict(l=20, r=20, t=20, b=20),
            )
            
            fig_box_scaled = px.box(df, y=col, title=f"Box Plot of {col} After Scaling ({schema.scaling_method})")
            figures['box'] = fig_box_scaled
            fig_box_scaled.update_layout(
                margin=dict(l=20, r=20, t=20, b=20),
            )
            
            visualizations['after'][col] = figures

    # Encoding categorical features
    if schema.encoding_method == "onehot":
        df = pd.get_dummies(df)
    elif schema.encoding_method == "label":
        for col in df.select_dtypes(include=['object']).columns:
            df[col] = df[col].astype('category').cat.codes

    # Scatter plot matrix to show relationships between features after processing
    if len(numeric_cols) > 1:
        scatter_fig = px.scatter_matrix(df, dimensions=numeric_cols, title="Scatter Matrix After Preprocessing")
        scatter_fig.update_layout(
            margin=dict(l=20, r=20, t=20, b=20),
        )

    return create_table_description(df), visualizations, scatter_fig

form = ModelForm(
    DataPreprocessingSchema,
    "regression",
    "data_processing",
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
        "encoding_method": fields.RadioItems(
            options_labels={"onehot": "One-hot", "label": "Label"},
            description="Encoding method for categorical features",
            default="onehot",
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

layout = dmc.Stack(
    [
        form,
        dmc.Button("Apply", variant="outline", color="blue", id='apply_preprocessing', n_clicks=0),
        html.Div(id="output"),
    ]
)

@callback(
    Output("output", "children"),
    Input('apply_preprocessing', 'n_clicks'),
    State(ModelForm.ids.main("regression", "data_processing"), "data"),
)
def use_form_data(n_clicks, form_data: dict):
    if n_clicks > 0:
        try:
            df = pd.read_csv(session_get_file_path('rawdata'))
            
            result = preprocess_and_visualize(df,DataPreprocessingSchema(**form_data))
            
            graphs=[]
            visualizations = result[1]
            scatter_matrix = result[2]
            
            for col in visualizations['before'].keys():
                graphs.append(dcc.Graph(figure=visualizations['before'][col]['histogram']))
                graphs.append(dcc.Graph(figure=visualizations['after'][col]['histogram']))
                graphs.append(dcc.Graph(figure=visualizations['before'][col]['box']))
                graphs.append(dcc.Graph(figure=visualizations['after'][col]['box']))
            
            output = dmc.Stack(
                [
                    result[0],
                    dmc.SimpleGrid(
                        graphs,
                        cols=4,
                    ),
                    dcc.Graph(figure=scatter_matrix)
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