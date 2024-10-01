from dash_pydantic_form import ModelForm, fields
from pydantic import BaseModel, Field, ValidationError
from typing import Literal
from dash import Input, Output, callback, html, State, dcc, no_update
from dash.exceptions import PreventUpdate
import dash_mantine_components as dmc
from pages.sections.clustering.utils import parse_validation_errors, session_get_file_path, continue_button, reset_button
import pickle, joblib, io, dash

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
            dmc.Button("Download", color="green", id='clustering-apply_model_export', n_clicks=0),
            html.Div(id="clustering-model-export-output"),
            dcc.Download(id="clustering-download-model"),
            dmc.Group(
                [
                    reset_button,
                    html.Div(
                        id='clustering-proceed-output',
                    )
                ],
                justify="space-between",
            )
        ]
    )
    
    return layout

@callback(
    Output("clustering-model-export-output", "children"),
    Output("clustering-download-model", "data"),
    Input('clustering-apply_model_export', 'n_clicks'),
    State(ModelForm.ids.main("model_export", 'main'), "data"),
    prevent_initial_call=True
)
def apply_model_export(n_clicks, form_data):
    if n_clicks > 0:
        
        try:
            
            with open(session_get_file_path('model', extension='joblib'), 'rb') as file:
                model = joblib.load(file)
            
            export_model(model, ModelExportSchema(**form_data))
            
            return no_update, export_model(model, ModelExportSchema(**form_data))
            
            
        except ValidationError as exc:
            return html.Div(
                [
                    dmc.Alert(
                        parse_validation_errors(exc),
                        color="red",
                        variant="filled",
                        withCloseButton=True
                    )
                ]
            ), no_update
        
        except Exception as exc:
            return html.Div(
                [
                    dmc.Alert(
                        "There was an error downloading the model.",
                        color="red",
                        variant="filled",
                        withCloseButton=True
                    )
                ]
            ), no_update

    else:
        raise PreventUpdate