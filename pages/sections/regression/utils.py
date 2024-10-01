import time
import os
import shutil
import pandas as pd
import dash_mantine_components as dmc
from dash import html
import threading
import flask
import json
from dash_iconify import DashIconify
import base64
import io
import numpy as np
import joblib


<<<<<<< HEAD
def render_upload_header(filename):
=======
loader = dmc.Flex(dmc.Loader(color="blue", size="xl"), justify="center")
continue_button  = dmc.Popover(
        [
            dmc.PopoverTarget(dmc.Button("Next")),
            dmc.PopoverDropdown(
                dmc.Stack(
                    [
                        dmc.Text("Are you sure you want to proceed?"),
                        dmc.Button("Yes", id="next-button", n_clicks=0, color="green", variant="filled"),
                    ],
                    gap=5
                )
            ),
        ],
        id='next-popover',
        width=200,
        position="bottom",
        withArrow=True,
        shadow="md",
        zIndex=2000,
    )

@callback(
    Output("next-popover", "opened"),
    Input("next-button", "n_clicks"),
    prevent_initial_call=True
)
def toggle_next_modal(n_clicks):
    if n_clicks > 0:
        return False

reset_button = dmc.Button(
    "Reset",
    id="toggle-reset-modal",
    n_clicks=0,
    color="red",
    variant="filled",
)

def render_upload_header(filename, prefix):
>>>>>>> f23d340 (regression done)
    return dmc.Group(
        [
            dmc.ActionIcon(
                id=f"{prefix}-remove-data",
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

<<<<<<< HEAD
def parse_contents(contents, filename):

    content_type, content_string = contents.split(',')

    decoded = base64.b64decode(content_string)
    
    if 'csv' in filename:

        df = pd.read_csv(
            
            io.StringIO(decoded.decode('utf-8')))
=======
def parse_contents(contents, filename, finalfilename):
>>>>>>> f23d340 (regression done)
        
    elif 'xls' in filename:

        df = pd.read_excel(io.BytesIO(decoded))
        
    df = df.select_dtypes(include=[np.number])
    
<<<<<<< HEAD
    create_session()
    
    session_df_to_file(df, 'rawdata')
=======
    session_df_to_file(df, finalfilename)
>>>>>>> f23d340 (regression done)
    
    return df

def create_timed_folder(folder_path):

    os.makedirs(folder_path, exist_ok=True)

    time.sleep(3600)

    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)

def create_session():
    path = 'sessions/' + flask.session['session_id']
    background_thread = threading.Thread(target=create_timed_folder, args=(path,))
    background_thread.daemon = True
    background_thread.start()

def session_df_to_file(df, filename):
    df.to_csv('sessions/' + flask.session['session_id'] + '/' + filename + '.csv', index=False)
    
def session_delete_file(filename):
    os.remove('sessions/' + flask.session['session_id'] + '/' + filename + '.csv')

def session_get_file_path(filename, extension=None):
    if extension:
        return 'sessions/' + flask.session['session_id'] + '/' + filename + '.' + extension
    else:
        return 'sessions/' + flask.session['session_id'] + '/' + filename


def create_table_description(df: pd.DataFrame):

    df = df.describe()
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
        # style={
        #     "width": "550px",
        #     "overflow-x": "scroll",
        #     }
    )

def session_dict_to_json(dict, filename):
    path = 'sessions/' + flask.session['session_id'] + '/' + filename + '.json'
    with open(path, 'w') as json_file:
        json.dump(dict, json_file)

def session_json_to_dict(filename):
    path = 'sessions/' + flask.session['session_id'] + '/' + filename + '.json'
    with open(path, 'r') as json_file:
        return json.load(json_file)

def session_save_model(model, filename):
    path = 'sessions/' + flask.session['session_id'] + '/' + filename + '.joblib'
    joblib.dump(model, path)

# from pages.regression import step_pages

# def reset_session():
#     for i, step_page in enumerate(step_pages):
#         flask.session[step_page.name] = False if i!=0 else True