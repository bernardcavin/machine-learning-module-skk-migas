import dash_mantine_components as dmc
from dash import dcc, html, Input, Output, State, callback, no_update
from dash.exceptions import PreventUpdate
from dash_iconify import DashIconify
import flask
import uuid
from pages.sections.regression.utils import session_delete_file, parse_contents, render_upload_header

session = {}

upload_button = dcc.Upload(
    id='regression-upload-data',
    children=dmc.Stack([
        dmc.Flex(DashIconify(icon="ic:outline-cloud-upload", height=30),justify='center'),
        dmc.Text('Drag and Drop or Select Files'),
        dmc.Text("Accepted file types: .csv, .xslx, .las .", c='#1890ff', ta='center', style={"fontSize": 12},),
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

<<<<<<< HEAD
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
=======
def layout():
    
    layout = dmc.Stack(
        [
            html.Div(
                upload_button,
                id='regression-upload-header'
            ),
            html.Div(
                id='regression-output-data',
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
    Output('regression-output-data', 'children', allow_duplicate=True),
    Output('regression-upload-header', 'children', allow_duplicate=True),
    Output('regression-proceed-output', 'children', allow_duplicate=True),
    Input('regression-upload-data', 'contents'),
    State('regression-upload-data', 'filename'),
>>>>>>> f23d340 (regression done)
    prevent_initial_call=True
)
def upload_data_processing(contents, filename):
    if contents is not None:
        
        upload_header = no_update
        
        try:
<<<<<<< HEAD
            
            flask.session['session_id'] = str(uuid.uuid4())
            
            upload_output = parse_contents(contents, filename)
            upload_header = render_upload_header(filename)
=======
                
            if flask.session.get('session_id', None) is None:
                flask.session['session_id'] = str(uuid.uuid4())

            df = parse_contents(contents, filename, 'rawData')
            upload_header = render_upload_header(filename, 'regression-rawData')
            
            # df = df.describe()
            df = df.head(5)
            df = df.round(2)

            df.reset_index(inplace=True)
            
            upload_output = html.Div(
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
                )
            )
>>>>>>> f23d340 (regression done)
            
        except Exception as e:

            upload_output = dmc.Alert(
                'There was an error processing this file.',
                color="red",
                variant="filled",
                withCloseButton=True
            )

        return upload_output, upload_header
    
    else:
        
        raise PreventUpdate

@callback(
    Output('regression-output-data', 'children'),
    Output('regression-upload-header', 'children'),
    Input('regression-rawData-remove-data', 'n_clicks'),
)
def remove_data(n_clicks):

    if n_clicks > 0:
        session_delete_file('rawData')
        # reset_session()
        return upload_button, None
    else:
        raise PreventUpdate