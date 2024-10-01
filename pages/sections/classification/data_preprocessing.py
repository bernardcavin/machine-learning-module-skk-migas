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
from pages.sections.classification.utils import parse_validation_errors, create_table_description, session_get_file_path, session_df_to_file, continue_button, reset_button, session_dict_to_json
import pandas.api.types as ptypes
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
import category_encoders as ce

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

class LabeledDataProcessing(BaseModel):
    missing_value_handling: Literal["remove", "custom"] = Field(
        "remove", 
        description="Method for handling missing values"
    )
    custom_value: Optional[float] = Field(
        0, 
        description="Custom value for filling missing data, used if missing_value_handling is 'custom'"
    )
    encoding_method: Literal["label", "onehot", "frequency", "ordinal", "binary"] = Field("label", description="Encoding method for categorical features.")

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
    
def preprocess_label_and_visualize_series(series: pd.Series, schema: LabeledDataProcessing):
    
    if schema.missing_value_handling == "remove":
        series = series.dropna()
    elif schema.missing_value_handling == "custom" and schema.custom_value is not None:
        series = series.fillna(schema.custom_value)
    
    if schema.encoding_method == "label":
        le = LabelEncoder()
        encoded_series = pd.Series(le.fit_transform(series), index=series.index, name=series.name)
        
    elif schema.encoding_method == "onehot":
        # Create one-hot encoded DataFrame and merge it back as a Series
        one_hot_encoded = pd.get_dummies(series, prefix='series')
        encoded_series = one_hot_encoded.idxmax(axis=1)  # Return the column names for each value (optional)
        
    elif schema.encoding_method == "frequency":
        frequency = series.value_counts()
        encoded_series = series.map(frequency)
        
    elif schema.encoding_method == "ordinal":
        oe = OrdinalEncoder()
        reshaped_series = series.values.reshape(-1, 1)
        encoded_values = oe.fit_transform(reshaped_series)
        encoded_series = pd.Series(encoded_values.flatten(), index=series.index, name=series.name)
        
    elif schema.encoding_method == "binary":
        binary_encoder = ce.BinaryEncoder(cols=[series.name])
        temp_df = pd.DataFrame({series.name: series})
        binary_encoded = binary_encoder.fit_transform(temp_df)
        encoded_series = binary_encoded.iloc[:, :binary_encoded.shape[1]].apply(lambda x: ''.join(x.astype(str)), axis=1)

    # elif schema.encoding_method == "hash":
    #     hash_encoder = ce.HashingEncoder(cols=[series.name], n_components=10)
    #     temp_df = pd.DataFrame({series.name: series})
    #     hash_encoded = hash_encoder.fit_transform(temp_df)
    #     encoded_series = hash_encoded.sum(axis=1)  # Optional: Summarize encoded values as a single series (depends on use-case)
    
    series = encoded_series
    
    fig_bar = px.bar(series.value_counts().reset_index(), 
            x=series.name, y='count', 
            labels={ 'count': 'Count'}, 
            title='Frequency of Categories')
    fig_bar.update_layout(template='plotly_white')
    fig_pie = px.pie(series.value_counts().reset_index(), 
                names=series.name, values='count', 
                labels={ 'count': 'Count'}, 
                title='Proportion of Categories')
    
    return [fig_bar, fig_pie]
    
def preprocess_all_and_visualize(df: pd.DataFrame, column_schemas, preprocessed_file_name='preprocessed'):

    # Apply preprocessing to each column according to the schema provided
    for column, schema in column_schemas.items():
        
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

        if ptypes.is_numeric_dtype(df[column]):
            if schema.get('scaling_method', None):
                scaler = None
                if schema['scaling_method'] == "standard":
                    scaler = StandardScaler()
                elif schema['scaling_method'] == "minmax":
                    scaler = MinMaxScaler()
                elif schema['scaling_method'] == "robust":
                    scaler = RobustScaler()

                df[column] = scaler.fit_transform(df[[column]])
                
        elif ptypes.is_string_dtype(df[column]):
        
            if schema.encoding_method == "label":
                le = LabelEncoder()
                df[column] = pd.df[column](le.fit_transform(df[column]), index=df[column].index, name=df[column].name)
        
            elif schema.encoding_method == "onehot":
                # Create one-hot encoded DataFrame and merge it back as a df[column]
                one_hot_encoded = pd.get_dummies(df[column], prefix='df[column]')
                df[column] = one_hot_encoded.idxmax(axis=1)  # Return the column names for each value (optional)
                
            elif schema.encoding_method == "frequency":
                frequency = df[column].value_counts()
                df[column] = df[column].map(frequency)
                
            elif schema.encoding_method == "ordinal":
                oe = OrdinalEncoder()
                reshaped_series = df[column].values.reshape(-1, 1)
                encoded_values = oe.fit_transform(reshaped_series)
                df[column] = pd.df[column](encoded_values.flatten(), index=df[column].index, name=df[column].name)
                
            elif schema.encoding_method == "binary":
                binary_encoder = ce.BinaryEncoder(cols=[df[column].name])
                temp_df = pd.DataFrame({df[column].name: df[column]})
                binary_encoded = binary_encoder.fit_transform(temp_df)
                df[column] = binary_encoded.iloc[:, :binary_encoded.shape[1]].apply(lambda x: ''.join(x.astype(str)), axis=1)

            # elif schema.encoding_method == "hash":
            #     hash_encoder = ce.HashingEncoder(cols=[df[column].name], n_components=10)
            #     temp_df = pd.DataFrame({df[column].name: df[column]})
            #     hash_encoded = hash_encoder.fit_transform(temp_df)
            #     df[column] = hash_encoded.sum(axis=1) 
            

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

    form = dmc.Stack(
        [
            dmc.Select(
                id={'class':'classification', 'type':"missing_value_handling", 'column': column},
                label="Missing Value Handling",
                description="Method for handling missing values",
                data=[
                    {"label": "Remove", "value": "remove"},
                    {"label": "Fill mean", "value": "fill_mean"},
                    {"label": "Fill median", "value": "fill_median"},
                    {"label": "Fill mode", "value": "fill_mode"},
                    {"label": "Custom", "value": "custom"},
                ],
                value="fill_mean",
                required=True,
            ),
            dmc.NumberInput(
                id={'class':'classification', 'type':"custom_value", 'column': column},
                label="Custom Value",
                description="Custom value for filling missing data, used if missing_value_handling is 'custom'",
                value=0,
                required=False,
                display="none",
            ),
            dmc.Select(
                id={'class':'classification', 'type':"scaling_method", 'column': column},
                label="Scaling Method",
                description="Scaling method to be applied to numerical features",
                data=[
                    {"label": "Standard", "value": "standard"},
                    {"label": "MinMax", "value": "minmax"},
                    {"label": "Robust", "value": "robust"},
                ],
                value="standard",
                required=True,
            ),
            dmc.Select(
                id={'class':'classification', 'type':"outlier_handling", 'column': column},
                label="Outlier Handling",
                description="Method for handling outliers",
                data=[
                    {"label": "Remove", "value": "remove"},
                    {"label": "Cap", "value": "cap"},
                    {"label": "None", "value": "none"},
                ],
                value="none",
                required=True,
            ),
            dmc.NumberInput(
                id={'class':'classification', 'type':"cap_value", 'column': column},
                label="Cap Value",
                description="Value to cap outliers, used if outlier_handling is 'cap'",
                value=0,
                required=False,
                display="none",
            ),
        ]
    )
    
    
    
    # ModelForm(
    #     DataPreprocessingSchema,
    #     "data_preprocessing",
    #     column,
    #     fields_repr={
    #         "missing_value_handling": fields.Select(
    #             options_labels={"remove": "Remove", "fill_mean": "Fill mean", "fill_median": "Fill median", "fill_mode": "Fill mode", "custom": "Custom"},
    #             description="Method for handling missing values",
    #             default="fill_mean",
    #             required=True,
    #         ),
    #         "custom_value": {
    #             "visible": ("missing_value_handling", "==", "custom"),
    #         },
    #         "scaling_method": fields.RadioItems(
    #             options_labels={"standard": "Standard", "minmax": "MinMax", "robust": "Robust"},
    #             description="Scaling method to be applied to numerical features",
    #             default="standard",
    #             required=True,
    #         ),
    #         "outlier_handling": fields.RadioItems(
    #             options_labels={"remove": "Remove", "cap": "Cap", "none": "None"},
    #             description="Method for handling outliers",
    #             default="none",
    #             required=True,
    #         ),
    #         "cap_value": {
    #             "visible": ("outlier_handling", "==", "cap"),
    #         },
    #     },
    # )
    
    return dmc.Stack(
        [
            form,
            dmc.Flex(
                dmc.Button("Apply", color="blue", id={'class':'classification', 'type': 'numeric', 'column': column}, n_clicks=0, w=200),
                justify="end",
            ),
            html.Div(id={'class':'classification', 'type': "output", 'column': column}),
        ]
    )

def create_form_labeled(column):

    form = dmc.Stack(
        [
            dmc.Select(
                id={'class':'classification', 'type':"label_missing_value_handling", 'column': column},
                label="Missing Value Handling",
                description="Method for handling missing values",
                data=[
                    {"label": "Remove", "value": "remove"},
                    {"label": "Custom", "value": "custom"},
                ],
                value="remove",
                required=True,
            ),
            dmc.NumberInput(
                id={'class':'classification', 'type':"label_custom_value", 'column': column},
                label="Custom Value",
                description="Custom value for filling missing data, used if missing_value_handling is 'custom'",
                value=0,
                required=False,
                display="none",
            ),
            dmc.Select(
                id={'class':'classification', 'type':"label_encoding_method", 'column': column},
                label="Encoding Method",
                description="Encoding method to be applied to categorical features",
                data=[
                    {"label":"Label", "value": "label"},
                    {"label":"One-Hot", "value": "onehot"},
                    {"label":"Frequency", "value": "frequency"},
                    {"label":"Ordinal", "value": "ordinal"},
                    {"label":"Binary", "value": "binary"},
                    # {"label":"Hash", "value": "hash"},
                ],
                value="label",
                required=True,
            ),
        ]
    )
    
    
    
    # ModelForm(
    #     DataPreprocessingSchema,
    #     "data_preprocessing",
    #     column,
    #     fields_repr={
    #         "missing_value_handling": fields.Select(
    #             options_labels={"remove": "Remove", "fill_mean": "Fill mean", "fill_median": "Fill median", "fill_mode": "Fill mode", "custom": "Custom"},
    #             description="Method for handling missing values",
    #             default="fill_mean",
    #             required=True,
    #         ),
    #         "custom_value": {
    #             "visible": ("missing_value_handling", "==", "custom"),
    #         },
    #         "scaling_method": fields.RadioItems(
    #             options_labels={"standard": "Standard", "minmax": "MinMax", "robust": "Robust"},
    #             description="Scaling method to be applied to numerical features",
    #             default="standard",
    #             required=True,
    #         ),
    #         "outlier_handling": fields.RadioItems(
    #             options_labels={"remove": "Remove", "cap": "Cap", "none": "None"},
    #             description="Method for handling outliers",
    #             default="none",
    #             required=True,
    #         ),
    #         "cap_value": {
    #             "visible": ("outlier_handling", "==", "cap"),
    #         },
    #     },
    # )
    
    return dmc.Stack(
        [
            form,
            dmc.Flex(
                dmc.Button("Apply", color="blue", id={'class':'classification', 'type': 'label', 'column': column}, n_clicks=0, w=200),
                justify="end",
            ),
            html.Div(id={'class':'classification', 'type': "label-output", 'column': column}),
        ]
    )

tab_style = {
  'border': f'2px solid {dmc.theme.DEFAULT_THEME["colors"]["blue"][6]}',
  'border-radius': '5px',
  'bg-color': dmc.theme.DEFAULT_THEME["colors"]["red"][6]
}

def layout():
    
    df = pd.read_csv(session_get_file_path('rawData',extension='csv'), index_col=0)

    columns = df.columns

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
                            create_form(column) if ptypes.is_numeric_dtype(df[column]) else create_form_labeled(column),
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
            dmc.Button("Apply All Preprocessing", color="green", id='classification-apply_preprocessing', n_clicks=0),
            html.Div(id="classification-output"),
            dmc.Group(
                [
                    reset_button,
                    html.Div(
                        id='classification-proceed-output',
                    )
                ],
                justify="space-between",
            )
        ]
    )
    
    return layout

@callback(
    Output({'class':'classification', 'type': "custom_value", 'column': MATCH}, 'display'),
    Input({'class':'classification', 'type': "missing_value_handling", 'column': MATCH}, 'value'),
    prevent_initial_call=True
)
def toggle_custom_value(value):
    if value == "custom":
        return "block"
    else:
        return "none"

@callback(
    Output({'class':'classification', 'type': "label_custom_value", 'column': MATCH}, 'display'),
    Input({'class':'classification', 'type': "label_missing_value_handling", 'column': MATCH}, 'value'),
    prevent_initial_call=True
)
def toggle_custom_value_label(value):
    if value == "custom":
        return "block"
    else:
        return "none"

@callback(
    Output({'class':'classification', 'type': "cap_value", 'column': MATCH}, 'display'),
    Input({'class':'classification', 'type': "outlier_handling", 'column': MATCH}, 'value'),
    prevent_initial_call=True
)
def toggle_cap_value(value):
    if value == "cap":
        return "block"
    else:
        return "none"


@callback(
    Output({'class':'classification', 'type': "label-output", 'column': MATCH},'children'),
    Input({'class':'classification', 'type': "label", 'column': MATCH}, 'n_clicks'),
    State({'class':'classification', 'type': 'label_missing_value_handling', 'column': MATCH}, 'value'),
    State({'class':'classification', 'type': 'label_custom_value', 'column': MATCH}, 'value'),
    State({'class':'classification', 'type': 'label_encoding_method', 'column': MATCH}, 'value'),
    State({'class':'classification', 'type': "label", 'column': MATCH}, 'id'),
    prevent_initial_call=True
)
def preprcess_labeled_column(n_clicks, missing_value_handling, custom_value, encoding_method, button_id):
    if n_clicks > 0:
        
        column = button_id['column']
        
        try:
            df = pd.read_csv(session_get_file_path('rawData', extension='csv'), index_col=0)
            
            schema = LabeledDataProcessing(
                missing_value_handling=missing_value_handling,
                custom_value=custom_value,
                encoding_method=encoding_method,
            )
            
            visualizations = preprocess_label_and_visualize_series(df[column], schema)

            graphs = [
                dcc.Graph(figure=visualization) for visualization in visualizations
            ]
            
            output = dmc.Stack(
                [
                    dmc.SimpleGrid(
                        graphs,
                        cols=len(graphs),
                    ),
                ]
            )
            
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
            )
            
        except Exception as exc:
            
            print(exc)
            
            return html.Div(
                [
                    dmc.Alert(
                        "There was an error preprocessing your data.",
                        color="red",
                        variant="filled",
                        withCloseButton=True
                    )
                ]
            )
            
            
        return output
    else:
        raise PreventUpdate

@callback(
    Output({'class':'classification', 'type': "output", 'column': MATCH},'children'),
    Input({'class':'classification', 'type': "numeric", 'column': MATCH}, 'n_clicks'),
    State({'class':'classification', 'type': 'missing_value_handling', 'column': MATCH}, 'value'),
    State({'class':'classification', 'type': 'custom_value', 'column': MATCH}, 'value'),
    State({'class':'classification', 'type': 'scaling_method', 'column': MATCH}, 'value'),
    State({'class':'classification', 'type': 'outlier_handling', 'column': MATCH}, 'value'),
    State({'class':'classification', 'type': 'cap_value', 'column': MATCH}, 'value'),
    State({'class':'classification', 'type': "numeric", 'column': MATCH}, 'id'),
)
def preprcess_each_column(n_clicks, missing_value_handling, custom_value, scaling_method, outlier_handling, cap_value, button_id):
    if n_clicks > 0:
        
        column = button_id['column']
        
        try:
            df = pd.read_csv(session_get_file_path('rawData', extension='csv'), index_col=0)
            
            schema = DataPreprocessingSchema(
                missing_value_handling=missing_value_handling,
                custom_value=custom_value,
                scaling_method=scaling_method,
                outlier_handling=outlier_handling,
                cap_value=cap_value
            )
            
            visualizations = preprocess_and_visualize_series(df[column], schema)

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
                        parse_validation_errors(exc),
                        color="red",
                        variant="filled",
                        withCloseButton=True
                    )
                ]
            )
            
        except Exception as exc:
            
            return html.Div(
                [
                    dmc.Alert(
                        "There was an error preprocessing your data.",
                        color="red",
                        variant="filled",
                        withCloseButton=True
                    )
                ]
            )
            
            
        return output
    else:
        raise PreventUpdate
    
@callback(
    Output("classification-output", "children"),
    Output("classification-proceed-output", "children", allow_duplicate=True),
    Input('classification-apply_preprocessing', 'n_clicks'),
    State({'class':'classification', 'type': 'missing_value_handling', 'column': ALL}, 'value'),
    State({'class':'classification', 'type': 'custom_value', 'column': ALL}, 'value'),
    State({'class':'classification', 'type': 'scaling_method', 'column': ALL}, 'value'),
    State({'class':'classification', 'type': 'outlier_handling', 'column': ALL}, 'value'),
    State({'class':'classification', 'type': 'cap_value', 'column': ALL}, 'value'),
    State({'class':'classification', 'type': 'label_missing_value_handling', 'column': ALL}, 'value'),
    State({'class':'classification', 'type': 'label_custom_value', 'column': ALL}, 'value'),
    State({'class':'classification', 'type': 'label_encoding_method', 'column': ALL}, 'value'),
    State({'class':'classification', 'type': "numeric", 'column': ALL}, 'id'),
    State({'class':'classification', 'type': "label", 'column': ALL}, 'id'),
    prevent_initial_call = True
)
def preprocess_all(n_clicks, missing_value_handlings, custom_values, scaling_methods, outlier_handlings, cap_values, label_missing_value_handlings, label_custom_values, label_encoding_methods, numeric_button_ids, label_button_ids):
    if n_clicks > 0:
        
        numeric_columns = [button_id['column'] for button_id in numeric_button_ids]
        
        label_columns = [button_id['column'] for button_id in label_button_ids]
        
        try:
            
            numeric_schemas = [{
                    'missing_value_handling': missing_value_handling,
                    'custom_value': custom_value,
                    'scaling_method': scaling_method,
                    'outlier_handling': outlier_handling,
                    'cap_value': cap_value
                } for missing_value_handling, custom_value, scaling_method, outlier_handling, cap_value in zip(missing_value_handlings, custom_values, scaling_methods, outlier_handlings, cap_values)
            ]
            
            label_schemas = [{
                    'missing_value_handling': label_missing_value_handling,
                    'custom_value': label_custom_value,
                    'encoding_method': label_encoding_method
                } for label_missing_value_handling, label_custom_value, label_encoding_method in zip(label_missing_value_handlings, label_custom_values, label_encoding_methods)
            ]
            
            schemas = numeric_schemas + label_schemas
            columns = numeric_columns + label_columns
            
            columns_schemas = {columns[i]: schemas[i] for i in range(len(numeric_columns))}
            session_dict_to_json(columns_schemas, 'preprocessing')
            
            df = pd.read_csv(session_get_file_path('rawData', extension='csv'), index_col=0)
            
            result = preprocess_all_and_visualize(df, columns_schemas)

            output = dmc.Stack(
                [
                    result[0],
                    dcc.Graph(figure=result[1], style={'height': '1000px'})
                ]
            )
            
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
                        "There was an error preprocessing your data.",
                        color="red",
                        variant="filled",
                        withCloseButton=True
                    )
                ]
            ), no_update

    else:
        raise PreventUpdate
            