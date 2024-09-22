import time
import os
import shutil
import pandas as pd
import dash_mantine_components as dmc
from dash import html
import threading
import flask

def create_timed_folder(folder_path):

    os.makedirs(folder_path, exist_ok=True)

    time.sleep(3600)

    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)


def session_df_to_file(df, filename):
    
    path = 'sessions/' + flask.session['session_id']
    
    background_thread = threading.Thread(target=create_timed_folder, args=(path,))
    background_thread.daemon = True
    background_thread.start()

    df.to_csv('sessions/' + flask.session['session_id'] + '/' + filename + '.csv', index=False)
    
def session_delete_file(filename):
    os.remove('sessions/' + flask.session['session_id'] + '/' + filename + '.csv')

def session_get_file_path(filename):
    return 'sessions/' + flask.session['session_id'] + '/' + filename + '.csv'


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
        style={
            "width": "550px",
            "overflow-x": "scroll",
            }
    )