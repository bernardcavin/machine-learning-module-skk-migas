import dash
from dash import html, dcc, Output, Input, callback, MATCH, ALL, no_update
from dash.exceptions import PreventUpdate
import flask
import uuid
dash.register_page(__name__)

from pages.sections.regression import feature_selection, model_training, upload_data, data_preprocessing, model_export, prediction
import dash_mantine_components as dmc
from dash import callback, Output, Input, State, ctx
from dash_iconify import DashIconify
from pydantic import BaseModel
from pages.sections.utils import loader

def get_icon(icon):
    return DashIconify(icon=icon, height=20)

class StepPage(BaseModel):
    
    title: str
    name: str
    icon: str
    description: str
    layout: object

step_pages = [
    StepPage(
        title="Data Upload",
        name="data_upload",
        icon="ic:outline-cloud-upload",
        description="Upload your data",
        layout=upload_data.layout
    ),
    StepPage(
        title="Data Preprocessing",
        name="data_preprocessing",
        icon="mdi:gear",
        description="Preprocess your data",
        layout=data_preprocessing.layout
    ),
    StepPage(
        title="Feature Selection",
        name="feature_selection",
        icon="mdi:filter-variant",
        description="Select features",
        layout=feature_selection.layout
    ),
    StepPage(
        title="Model Training",
        name="model_training",
        icon="mdi:cog",
        description="Train regression models",
        layout=model_training.layout
    ),
    StepPage(
        title="Prediction",
        name="prediction",
        icon="mdi:cog",
        description="Predict",
        layout=prediction.layout
    ),
    StepPage(
        title="Model Export",
        name="model_export",
        icon="mdi:download",
        description="Export model",
        layout=model_export.layout
    ),
]

layout_dict = {step_page.name: step_page.layout for step_page in step_pages}

min_step = 0
max_step = len(step_pages)
active = 0

def render_step_page(step_page: StepPage):
    
    return dmc.Container(
        dmc.Stack(
            [
                dmc.Group(
                    [
                        DashIconify(icon=step_page.icon, height=30),
                        dmc.Title(step_page.title, size=30),
                    ],
                    gap=5
                ),
                html.Div(
                    step_page.layout if not callable(step_page.layout) else step_page.layout()
                )
            ]
        ),
        fluid=True,
    )

content = html.Div(
    render_step_page(step_pages[0]),
    id='regression-content',
)



def layout():
    
    states = []
    
    layout = dmc.Container(
        states + [
            dmc.Grid(
                [
                    dmc.GridCol(
                        html.Div(
                            dmc.Stack(
                                [
                                    dcc.Location(id='url'),
                                    dmc.Modal(
                                        title="Are you sure you want to reset?",
                                        id='regression-reset-modal',
                                        children=[
                                            dmc.Flex(
                                                dmc.Button("Yes, reset", color="red", variant="filled", id='regression-reset-button', n_clicks=0),
                                                justify="end",
                                            )
                                        ]
                                    ),
                                    dmc.Stack(
                                        [
                                            dmc.Group(
                                                [
                                                    dmc.ActionIcon(
                                                        id='regression-go-back-button',
                                                        color="blue",
                                                        size="md",
                                                        radius="sm",
                                                        children=DashIconify(icon="ic:baseline-arrow-back", height=20),
                                                        n_clicks=0,
                                                        variant='transparent',
                                                    ),
                                                    # DashIconify(icon="carbon:chart-logistic-regression", height=30),
                                                    dmc.Title("Regression", size=30),
                                                ],
                                                gap=5
                                            ),
                                            dmc.Text("Regression is a machine learning technique used to predict a continuous numerical value based on one or more input features.", c="gray", size="sm"),
                                        ],
                                        gap=0
                                    ),
                                    dmc.Divider(label='Workflow', labelPosition='center'),
                                    dmc.Stepper(
                                        id="regression-stepper",
                                        active=active,
                                        orientation='vertical',
                                        children=[
                                            dmc.StepperStep(
                                                label=step_page.title,
                                                description=step_page.title,
                                                icon=get_icon(icon=step_page.icon),
                                                progressIcon=get_icon(icon=step_page.icon),
                                                completedIcon=get_icon(icon="material-symbols:done"),
                                                children=None,
                                            )
                                            for step_page in step_pages
                                        ],
                                    ),
                                ],
                            ),
                            style={
                                'position': 'sticky',
                                'top': 20,
                                'z-index': 1000,
                            }
                        ),
                        span=2,
                    ),
                    dmc.GridCol(
                        dmc.Card(
                            content,
                            withBorder=True,
                            radius='md',
                            shadow='sm',
                        ),
                        span=10,
                    ),
                ]
                
            )
        ],
        fluid=True,
        m=20,
    )
    
    return layout

# @callback(
#     Output("next-modal", "opened"),
#     Input("toggle-next-modal", "n_clicks"),
#     State("next-modal", "opened"),
#     prevent_initial_call=True
# )
# def toggle_next_modal(n_clicks, opened):
#     if n_clicks > 0:
#         return not opened

@callback(
    Output("regression-reset-modal", "opened"),
    Input("regression-toggle-reset-modal", "n_clicks"),
    State("regression-reset-modal", "opened"),
    prevent_initial_call=True
)
def toggle_reset_modal(n_clicks, opened):
    if n_clicks > 0:
        return not opened

@callback(
    Output("url", "pathname", allow_duplicate=True),
    Input("regression-go-back-button", "n_clicks"),
    prevent_initial_call=True
)
def go_back(n_clicks):
    if n_clicks > 0:
        return '/'

@callback(
    Output("url", "pathname", allow_duplicate=True),
    Output("url",'refresh', allow_duplicate=True),
    Input("regression-reset-button", "n_clicks"),
    State("url", "pathname"),
    prevent_initial_call=True,
)
def reset_url(n_clicks, path):
    if n_clicks > 0:
        if path[-1] == '/':
            return path[:-1], True
        else:
            return path+'/', True
    else:
        raise PreventUpdate

@callback(
    Output("regression-stepper", "active", allow_duplicate=True),
    Output("regression-content", "children"),
    Input("regression-next-button", "n_clicks"),
    State("regression-stepper", "active"),
    prevent_initial_call=True,
)
def next_page(next_, current):
    if next_ > 0:
        step = current if current is not None else active
        step = step + 1 if step < max_step else step
        if step <= max_step:
            step_page = step_pages[step]
            # states = [no_update for step_page in step_pages]
            # states[step] = True
            
            layout = render_step_page(step_page)
            
        else:
            raise PreventUpdate
        
        return step, layout
    else:
        raise PreventUpdate

# @callback(
#     Output({'type':'step_page','index':MATCH}, "children"),
#     Input({'type':'state','index':MATCH}, "data"),
#     prevent_initial_call=True,
# )
# def update_tab(active):
    
#     page = ctx.triggered_id['index']
#     had_opened = flask.session[page]
    
#     if not had_opened:
        
#         layout = layout_dict[page]
        
#         content = layout() if callable(layout) else layout
        
#         flask.session[page] = True
        
#         return content
    
#     else:
        
#         raise dash.exceptions.PreventUpdate
