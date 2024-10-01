import base64
import io
import dash
from dash import dcc, html, Input, Output, State
import plotly.graph_objects as go
import lasio

# Initialize the Dash app
app = dash.Dash(__name__)

app.layout = html.Div([
    dcc.Upload(
        id='upload-las',
        children=html.Button('Upload File'),
        multiple=False  # Allow multiple files to be uploaded
    ),
    dcc.Graph(id='las-graph')
])

@app.callback(
    Output('las-graph', 'figure'),
    Input('upload-las', 'contents')
)
def update_graph(contents):
    if contents is None:
        return go.Figure()

    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string.encode('utf-8'))  # Ensure string is converted to bytes
    print(io.BytesIO(decoded))
    las = lasio.read(io.BytesIO(decoded))

    # Assuming you want to plot the first curve in the LAS file
    if len(las.keys()) > 0:
        curve_name = las.keys()[0]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=las[curve_name], y=las.index, mode='lines', name=curve_name))
        fig.update_layout(title=curve_name, xaxis_title='Value', yaxis_title='Depth', yaxis_autorange='reversed')
        return fig
    return go.Figure()

if __name__ == '__main__':
    app.run_server(debug=True)