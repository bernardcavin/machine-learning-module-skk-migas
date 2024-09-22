import dash_mantine_components as dmc
from dash import dcc, html, Input, Output, State, callback, dash_table, no_update
from dash.exceptions import PreventUpdate
from dash_iconify import DashIconify
import base64
import io
import pandas as pd
from pages.sections.regression.utils import session_df_to_file, session_delete_file

session = {}

def render_upload_header(filename):
    return dmc.Group(
        [
            dmc.ActionIcon(
                id="remove-data",
                color="red",
                size="md",
                radius="sm",
                children=DashIconify(icon="ic:outline-delete", height=20),
                n_clicks=0
            ),
            dmc.Text(
                filename,
                size="md",
            ),
        ],
        gap=5,
        mb="md",
    )

def parse_contents(contents, filename):
    content_type, content_string = contents.split(',')

    decoded = base64.b64decode(content_string)
    
    if 'csv' in filename:

        df = pd.read_csv(
            
            io.StringIO(decoded.decode('utf-8')))
        
    elif 'xls' in filename:

        df = pd.read_excel(io.BytesIO(decoded))
    
    session_df_to_file(df, 'rawdata')
    
    # df = df.describe()
    df = df.head(5)
    df = df.round(2)

    df.reset_index(inplace=True)
    
    return html.Div(
        dmc.Table(
            striped=True,
            highlightOnHover=True,
            withColumnBorders=True,
            withTableBorder=True,
            withRowBorders=True,
            data={
                "head": df.columns.to_list(),
                "body": df.values.tolist(),
            }
        ),
        style={
            "width": "550px",
            "overflow-x": "scroll",
            }
    )

upload_button = dcc.Upload(
    id='upload-data',
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

layout = dmc.Container(
    [
        html.Div(
            upload_button,
            id='upload-header'
        ),
        html.Div(
            id='output-data',
        ),
    ],
)

@callback(
    Output('output-data', 'children', allow_duplicate=True),
    Output('upload-header', 'children', allow_duplicate=True),
    Input('upload-data', 'contents'),
    State('upload-data', 'filename'),
    prevent_initial_call=True
)
def upload_data_processing(contents, filename):
    if contents is not None:
        
        upload_header = no_update
        
        try:
            
            upload_output = parse_contents(contents, filename)
            upload_header = render_upload_header(filename)
            
        except Exception as e:
            
            print(e)
            
            upload_output = dmc.Alert(
                'There was an error processing this file.',
                color="red",
                variant="filled",
            )

        return upload_output, upload_header
    
    else:
        
        raise PreventUpdate

@callback(
    Output('output-data', 'children'),
    Output('upload-header', 'children'),
    Input('remove-data', 'n_clicks'),
)
def remove_data(n_clicks):

    if n_clicks > 0:
        session_delete_file('rawdata')
        return upload_button, None
    else:
        raise PreventUpdate