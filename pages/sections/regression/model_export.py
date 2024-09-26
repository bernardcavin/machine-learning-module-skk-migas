from dash_pydantic_form import ModelForm, fields
from pydantic import BaseModel, Field, ValidationError
from typing import List, Optional, Union, Literal
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import plotly.express as px
import plotly.graph_objects as go
from typing import List, Dict
from dash import Input, Output, callback, html, State, dcc, MATCH, ALL, no_update
from dash.exceptions import PreventUpdate
import dash_mantine_components as dmc
from pages.sections.regression.utils import create_table_description, session_get_file_path, session_df_to_file, session_dict_to_json
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
import pickle, joblib, io

class ModelExportSchema(BaseModel):
    export_format: Literal["pickle", "joblib"] = Field(
        ..., description="The format to export the trained model."
    )
    filename: str = Field(
        ..., description="The name of the file (without extension) to save the model as."
    )
    
def export_model(model, schema: ModelExportSchema) -> str:

    if schema.export_format == "pickle":
            # Export the model as Pickle
            filename = f"{schema.filename}.pkl"
            output = io.BytesIO()
            pickle.dump(model, output)
            output.seek(0)
            return dcc.send_bytes(output.read(), filename=filename)

    elif schema.export_format == "joblib":
        # Export the model as Joblib
        filename = f"{schema.filename}.joblib"
        output = io.BytesIO()
        joblib.dump(model, output)
        output.seek(0)
        return dcc.send_bytes(output.read(), filename=filename)

def layout():

    form = ModelForm(
        ModelExportSchema,
        "model_export",
        "main",
        fields_repr={
            "export_format": fields.Select(
                options_labels={"pickle": "Pickle", "joblib": "Joblib"},
                description="The format to export the trained model.",
            )
        },
    )
    
    layout = dmc.Stack(
        [
            dmc.Paper(
                html.Div(
                    form,
                    style={'margin':20}
                ),
                withBorder=True,
                shadow=0,
            ),
            dmc.Button("Download", color="green", id='apply_model_export', n_clicks=0),
            html.Div(id="model-export-output"),
            dcc.Download(id="download-model"),
        ]
    )
    
    return layout

@callback(
    Output("model-export-output", "children"),
    Input('apply_model_export', 'n_clicks'),
    State(ModelForm.ids.main("model_export", 'main'), "data"),
)
def apply_model_export(n_clicks, form_data):
    if n_clicks > 0:
        
        try:
            
            with open(session_get_file_path('model', extension='joblib'), 'rb') as file:
                model = joblib.load(file)
            
            export_model(model, ModelExportSchema(**form_data))
            
            return export_model(model, ModelExportSchema(**form_data))
            
        except ValidationError as exc:
            return html.Div(
                [
                    dmc.Alert(
                        "There was an error applying feature selection.",
                        color="red",
                        variant="filled",
                    )
                ]
            )

    else:
        raise PreventUpdate