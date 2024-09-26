import dash_mantine_components as dmc
from dash import dcc, html, Input, Output, State, callback, no_update
from dash.exceptions import PreventUpdate
from dash_iconify import DashIconify
import flask
import uuid
from pages.sections.regression.utils import session_delete_file, parse_contents, render_upload_header

session = {}

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
    fluid=True
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
            
            flask.session['session_id'] = str(uuid.uuid4())
            
            upload_output = parse_contents(contents, filename)
            upload_header = render_upload_header(filename)
            
        except Exception as e:

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
        # reset_session()
        return upload_button, None
    else:
        raise PreventUpdate