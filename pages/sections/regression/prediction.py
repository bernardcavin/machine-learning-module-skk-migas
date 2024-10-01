import dash_mantine_components as dmc
from dash import dcc, html, Input, Output, State, callback, no_update
from dash.exceptions import PreventUpdate
from dash_iconify import DashIconify
import flask
import uuid
from pydantic import BaseModel, Field
from pages.sections.regression.utils import session_df_to_file, session_get_file_path, session_delete_file, parse_contents, render_upload_header, reset_button, continue_button, create_table_description, session_json_to_dict
from typing import Literal, Optional
from dash_pydantic_form import ModelForm, fields
import pandas as pd
import plotly.graph_objects as go
from plotly.graph_objs import Figure
from pages.sections.regression.data_preprocessing import preprocess_all_and_visualize
import joblib

session = {}

upload_button = dcc.Upload(
    id='regression-prediction-upload-data',
    children=dmc.Stack([
        dmc.Flex(DashIconify(icon="ic:outline-cloud-upload", height=30),justify='center'),
        dmc.Text('Drag and Drop or Select Files'),
        dmc.Text("Accepted file types: .txt, .csv, .tsv .xslx .las .", c='#1890ff', ta='center', style={"fontSize": 12},),
    ], gap=0, m='10px', justify='center'),
    style={
        'width': '100%',
        'height': '100%',
        'lineHeight': '60px',
        'borderWidth': '1px',
        'borderStyle': 'dashed',
        'borderRadius': '5px',
        'textAlign': 'center',
        'borderColor': '#1890ff',
        'color': '#1890ff',
        'fontSize': '16px'
    },
)

class PredictionFeatureSelectionSchema(BaseModel):
    selected_features: list[str] = Field( # type: ignore
        default_factory=list
    )
    prediction_column_name: str = Field(
        description="Prediction column name",
        default="prediction"
    )
    plot_y_column: str = Field(
        description="Column to plot on the y-axis",
        default="prediction"
    )

def create_prediction_plot(features_df: pd.DataFrame, predictions: list, y_axis: str) -> Figure:
    # Creating a scatter plot comparing input features with the predictions
    fig = go.Figure()
    
    fig.add_trace(
        go.Scatter(
            x=predictions,
            y=features_df[y_axis],
            mode='markers',
        )
    )

    # Add layout information
    fig.update_layout(
        title=f"Predicted Values vs {y_axis}",
        xaxis_title="Predicted Values",
        yaxis_title=y_axis,
        template="plotly_white"
    )
    
    return fig

def predict_and_visualize_regression_model(schema: PredictionFeatureSelectionSchema) -> dict:
    
    preprocessing_dict = session_json_to_dict('preprocessing')

    feature_df = pd.read_csv(session_get_file_path('predictionData', extension='csv'), index_col=0)
    feature_df = feature_df[schema.selected_features]
    processed_preprocessed_dict = {
        column: preprocessing_dict.get(
            column,
            preprocessing_dict.get(list(preprocessing_dict.keys())[0])
            ) for column in feature_df.columns
    }
    
    result = preprocess_all_and_visualize(feature_df, processed_preprocessed_dict, preprocessed_file_name='preprocessedPredictionData')

    with open(session_get_file_path('model', extension='joblib'), 'rb') as file:
        model = joblib.load(file)
    
    predictions = model.predict(feature_df)
    feature_df[feature_df.index.name if feature_df.index.name is not None else 'index'] = feature_df.index
    visualization = create_prediction_plot(feature_df, predictions, schema.plot_y_column)
    
    feature_df[schema.prediction_column_name] = predictions
    
    session_df_to_file(feature_df, 'predictedData')
    
    return {
        'result_table':result[0], 
        "visualization": [
            result[1],
            visualization
        ]
    }


def layout():
    
    layout = dmc.Stack(
        [
            html.Div(
                upload_button,
                id='regression-prediction-upload-header'
            ),
            html.Div(
                id='regression-prediction-upload-output-data',
            ),
            dmc.Group(
                [
                    reset_button,
                    html.Div(
                        id='regression-proceed-output',
                    )
                ],
                justify="space-between",
            )
        ],
        gap='md'
    )
    return layout

@callback(
    Output('regression-prediction-upload-output-data', 'children', allow_duplicate=True),
    Output('regression-prediction-upload-header', 'children', allow_duplicate=True),
    Input('regression-prediction-upload-data', 'contents'),
    State('regression-prediction-upload-data', 'filename'),
    prevent_initial_call=True
)
def upload_data_prediction(contents, filename):
    if contents is not None:
        
        try:
                
            if flask.session.get('session_id', None) is None:
                flask.session['session_id'] = str(uuid.uuid4())

            df = parse_contents(contents, filename, 'predictionData')
            upload_header = render_upload_header(filename, 'predictionData')
            
            feature_selection_dict = session_json_to_dict('feature_selection')
            
            df[df.index.name if df.index.name is not None else 'index'] = df.index
    
            form = ModelForm(
                PredictionFeatureSelectionSchema,
                "prediction_feature_selection",
                "main",
                fields_repr={
                    "selected_features": fields.MultiSelect(
                        data_getter=lambda: df.columns.to_list(),
                        description=f"Select features from the uploaded data. Must be a subset of the original features: {', '.join(feature_selection_dict.get('selected_features'))}",
                        required=True,
                    ),
                    "prediction_column_name": fields.Text(
                        description="Prediction column name",
                        default=f"predicted_{feature_selection_dict.get('target')}",
                        required=True,
                    ),
                    "plot_y_column": fields.Select(
                        data_getter=lambda: df.columns.to_list(),
                        description="Column to plot on the y-axis",
                        required=True,
                    ),
                },
            )
            
            output = dmc.Stack(
                [
                    create_table_description(df),
                    dmc.Paper(
                        html.Div(
                            form,
                            style={'margin':20}
                        ),
                        withBorder=True,
                        shadow=0,
                    ),
                    dmc.Button("Predict", color="blue", id='regression-apply_prediction', n_clicks=0),
                    html.Div(id="regression-prediction-output"),
                ]
            )
            
            return output, upload_header
            
        except Exception as e:
            
            print(e)

            upload_output = dmc.Alert(
                'There was an error processing this file.',
                color="red",
                variant="filled",
            )

            return upload_output, no_update
    
    else:
        
        raise PreventUpdate

@callback(
    Output('regression-prediction-upload-output-data', 'children'),
    Output('regression-prediction-upload-header', 'children'),
    Input('regression-predictionData-remove-data', 'n_clicks'),
)
def remove_data(n_clicks):

    if n_clicks > 0:
        session_delete_file('predictionData')
        # reset_session()
        return upload_button, None
    else:
        raise PreventUpdate

@callback(
    Output("regression-prediction-output", "children"),
    Output("regression-proceed-output", "children", allow_duplicate=True),
    Input('regression-apply_prediction', 'n_clicks'),
    State(ModelForm.ids.main("prediction_feature_selection", 'main'), "data"),
    prevent_initial_call = True
)
def apply_prediction(n_clicks, form_data):
    if n_clicks > 0:
        
        # try:
            
        schema = PredictionFeatureSelectionSchema(**form_data)
        
        result = predict_and_visualize_regression_model(schema)
        
        output = dmc.Stack(
            [
                result['result_table'],
                dcc.Graph(figure=result['visualization'][0], style={'height': '1000px'}),
                dcc.Graph(figure=result['visualization'][1], style={'height': '1000px'}),
                dmc.Button("Download Predicted Data", color="green", id='regression-download_predicted_data_button', n_clicks=0),
                dcc.Download(id="regression-download-predicted-data"),
            ]
        )
        
        return output, continue_button
            
        # except Exception as exc:
        #     print(exc)
        #     return html.Div(
        #         [
        #             dmc.Alert(
        #                 "There was an error applying prediction.",
        #                 color="red",
        #                 variant="filled",
        #             )
        #         ]
        #     ), no_update

    else:
        raise PreventUpdate

@callback(
    Output("regression-download-predicted-data", "data"),
    Input('regression-download_predicted_data_button', 'n_clicks'),
    prevent_initial_call=True
)
def download_predicted_data(n_clicks):
    if n_clicks > 0:
        return dcc.send_file(session_get_file_path('predictedData', extension='csv'), filename='predictedData.csv')
    else:
        raise PreventUpdate