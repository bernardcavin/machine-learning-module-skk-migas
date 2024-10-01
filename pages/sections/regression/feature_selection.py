from dash_pydantic_form import ModelForm, fields
from pydantic import BaseModel, Field, ValidationError
from typing import Optional, Literal
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Input, Output, callback, html, State, dcc, no_update
from dash.exceptions import PreventUpdate
import dash_mantine_components as dmc
from pages.sections.regression.utils import parse_validation_errors, create_table_description, session_get_file_path, session_df_to_file, session_dict_to_json, continue_button, reset_button
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression

class FeatureSelectionSchema(BaseModel):
    method: Literal["manual", "RFE", "auto"] = Field(
        'manual', 
        description="Feature selection method",
    )
    selected_features: Optional[list[str]] = Field( # type: ignore
        description="List of selected features if manual selection is used",
        default_factory=list
    )
    target: str = Field(
        description="Target column",
    )

def feature_selection(df: pd.DataFrame, target: str, schema) -> pd.DataFrame:
    if schema.method == "manual" and schema.selected_features:
        features = schema.selected_features
        return df[features + [target]]
    elif schema.method == "RFE":
        X = df.drop(columns=[target])
        y = df[target]
        model = LinearRegression()
        rfe = RFE(model, n_features_to_select=5)
        rfe.fit(X, y)
        selected_features = X.columns[rfe.support_]
        return df[selected_features.to_list() + [target]]
    else:
        return df

def create_scatter_matrix_and_heatmap(df: pd.DataFrame):
    # Scatter matrix
    scatter_matrix_fig = px.scatter_matrix(df, template='simple_white')
    
    # Correlation heatmap
    corr_matrix = df.corr()
    heatmap_fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale='Viridis',
    ))
    
    heatmap_fig.update_layout(
        template='simple_white',
    )
    
    return scatter_matrix_fig, heatmap_fig

def layout():
        
    df = pd.read_csv(session_get_file_path('preprocessed' ,extension='csv'), index_col=0)
    
    columns = df.columns.to_list()
        
    form = ModelForm(
        FeatureSelectionSchema,
        "feature_selection",
        "main",
        fields_repr={
            "method": fields.Select(
                options_labels={"manual": "Manual", "RFE": "RFE", "auto": "Auto"},
                description="Feature selection method",
                required=True,
            ),
            "selected_features": {
                "visible": ("method", "==", "manual"),
            },
            "selected_features": fields.MultiSelect(
                data_getter=lambda: columns,
                description="List of selected features if manual selection is used",
            ),
            'target': fields.Select(
                data_getter=lambda: columns,
                description="Target column",
                required=True,
            ),
        },
    )
    
    scatter_matrix_fig, heatmap_fig = create_scatter_matrix_and_heatmap(df)

    layout = dmc.Stack(
        [
            create_table_description(df),
            dmc.SimpleGrid(
                [
                    dcc.Graph(figure=scatter_matrix_fig),
                    dcc.Graph(figure=heatmap_fig),
                ],
                cols=2
            ),
            dmc.Paper(
                html.Div(
                    form,
                    style={'margin':20}
                ),
                withBorder=True,
                shadow=0,
            ),
            dmc.Button("Apply", color="blue", id='regression-apply_feature_selection', n_clicks=0),
            html.Div(id="regression-feature-selection-output"),
            dmc.Group(
                [
                    reset_button,
                    html.Div(
                        id='regression-proceed-output',
                    )
                ],
                justify="space-between",
            )
        ]
    )
    
    return layout

@callback(
    Output("regression-feature-selection-output", "children"),
    Output("regression-proceed-output", "children", allow_duplicate=True),
    Input('regression-apply_feature_selection', 'n_clicks'),
    State(ModelForm.ids.main("feature_selection", 'main'), "data"),
    prevent_initial_call = True
)
def apply_feature_selection(n_clicks, form_data):
    if n_clicks > 0:
        
        try:
            df = pd.read_csv(session_get_file_path('preprocessed', extension='csv'), index_col=0)
            
            session_dict_to_json(form_data, 'feature_selection')
            
            schema = FeatureSelectionSchema(**form_data)

            df = feature_selection(df, schema.target, schema)
            
            scatter_matrix_fig, heatmap_fig = create_scatter_matrix_and_heatmap(df)
            
            output = dmc.Stack(
                [
                    create_table_description(df),
                    dmc.SimpleGrid(
                        [
                            dcc.Graph(figure=scatter_matrix_fig),
                            dcc.Graph(figure=heatmap_fig),
                        ],
                        cols=2
                    ),
                ]
            )
            
            session_df_to_file(df, 'feature_selected')
            
            return output, continue_button
            
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
                        "There was an error applying feature selection.",
                        color="red",
                        variant="filled",
                        withCloseButton=True
                    )
                ]
            ), no_update

    else:
        raise PreventUpdate