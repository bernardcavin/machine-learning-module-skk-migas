
from flask import Flask
from dash import Dash, _dash_renderer
import dash_mantine_components as dmc
import dash
_dash_renderer._set_react_version("18.2.0")

server = Flask(__name__)

server.secret_key = 'secret!'

app = dash.Dash(
    __name__, 
    server=server,
    use_pages=True, 
    suppress_callback_exceptions=True, 
    external_stylesheets=dmc.styles.ALL
)

app.layout = dmc.MantineProvider(
    dash.page_container
)

if __name__ == "__main__":
    app.run_server(debug=True)


