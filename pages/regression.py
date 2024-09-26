import dash
from dash import html, dcc, Output, Input, callback, MATCH, ALL, no_update
import flask
import uuid
dash.register_page(__name__)

from pages.sections.regression import feature_selection, model_training, upload_data, data_preprocessing, model_export
import dash_mantine_components as dmc
from dash import callback, Output, Input, State, ctx
from dash_iconify import DashIconify
from pydantic import BaseModel

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

loader = dmc.Flex(dmc.Loader(color="blue", size="xl"), justify="center")

content = dmc.Tabs(
    [
        dmc.TabsPanel(
            step_page.layout if not callable(step_page.layout) else step_page.layout() if i == 0 else loader,
            id={'type':'step_page','index':step_page.name},
            value=step_page.name,
        ) for i,step_page in enumerate(step_pages)
    ],
    id='regression-content',
    value=step_pages[0].name,
)


def layout():
    
    states = []
    
    for i, step_page in enumerate(step_pages):
        
        flask.session[step_page.name] = False if i!=0 else True
        states.append(dcc.Store(id={'type':'state','index':step_page.name}, data=False))
        
    layout = dmc.Container(
        states + [
            dmc.Grid(
                [
                    dmc.GridCol(
                        dmc.Stack(
                            [
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
                            mx="md"
                        ),
                        span=2,
                    ),
                    dmc.GridCol(
                        dmc.Card(
                            dmc.Stack(
                                [
                                    content,
                                    dmc.Group(
                                        justify="space-between",
                                        children=[
                                            dmc.Button("Back", id="regression-back-button", variant="default"),
                                            dmc.Button("Next step", id="regression-next-button"),
                                        ],
                                    ),
                                ]
                            ),
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

@callback(
    Output("regression-stepper", "active", allow_duplicate=True),
    Output("regression-content", "value"),
    Output({'type':'state','index':ALL}, "data"),
    Input("regression-back-button", "n_clicks"),
    Input("regression-next-button", "n_clicks"),
    State("regression-stepper", "active"),
    prevent_initial_call=True,
)
def update_with_icons(back, next_, current):
    button_id = ctx.triggered_id

    step = current if current is not None else active
    if button_id == "regression-back-button":
        step = step - 1 if step > min_step else step
    else:
        step = step + 1 if step < max_step else step
        
    name = step_pages[step].name
    
    states = [no_update for step_page in step_pages]
    states[step] = True
    
    return step, name, states

@callback(
    Output({'type':'step_page','index':MATCH}, "children"),
    Input({'type':'state','index':MATCH}, "data"),
    prevent_initial_call=True,
)
def update_tab(active):
    
    page = ctx.triggered_id['index']
    had_opened = flask.session[page]
    
    if not had_opened:
        
        layout = layout_dict[page]
        
        content = layout() if callable(layout) else layout
        
        flask.session[page] = True
        
        return content
    
    else:
        
        raise dash.exceptions.PreventUpdate

# dmc.StepperStep(
#     label="Data Upload",
#     description="Upload your data",
#     icon=get_icon(icon="ic:outline-cloud-upload"),
#     progressIcon=get_icon(icon="ic:outline-cloud-upload"),
#     completedIcon=get_icon(icon="material-symbols:done"),
# ),
# dmc.StepperStep(
#     label='Data Preprocessing',
#     description="Preprocess your data",
#     icon=get_icon(icon="ic:outline-cloud-download"),
#     progressIcon=get_icon(icon="ic:outline-cloud-download"),
#     completedIcon=get_icon(icon="material-symbols:done"),
# ),
# dmc.StepperStep(
#     label="Feature Selection",
#     description="Select features",
#     icon=get_icon(icon="ic:outline-filter-list"),
#     progressIcon=get_icon(icon="ic:outline-filter-list"),
#     completedIcon=get_icon(icon="material-symbols:done"),
# ),
# dmc.StepperStep(
#     label="Model Selection",
#     description="Select model",
#     icon=get_icon(icon="ic:outline-build"),
#     progressIcon=get_icon(icon="ic:outline-build"),
#     completedIcon=get_icon(icon="material-symbols:done"),
# ),
# dmc.StepperStep(
#     label='Paremeter Tuning',
#     description="Tune parameters",
#     icon=get_icon(icon="ic:outline-tune"),
#     progressIcon=get_icon(icon="ic:outline-tune"),
#     completedIcon=get_icon(icon="material-symbols:done"),
# ),
# dmc.StepperStep(
#     label="Model Training",
#     description="Train model",
#     icon=get_icon(icon="ic:outline-train"),
#     progressIcon=get_icon(icon="ic:outline-train"),
#     completedIcon=get_icon(icon="material-symbols:done"),
# ),
# dmc.StepperStep(
#     label="Model Evaluation",
#     description="Evaluate model",
#     icon=get_icon(icon="ic:outline-check"),
#     progressIcon=get_icon(icon="ic:outline-check"),
#     completedIcon=get_icon(icon="material-symbols:done"),
# ),
# dmc.StepperStep(
#     label="Model Export",
#     description="Export model",
#     icon=get_icon(icon="ic:outline-download"),
#     progressIcon=get_icon(icon="ic:outline-download"),
#     completedIcon=get_icon(icon="material-symbols:done"),
# ),
# dmc.StepperCompleted(
#     children=[
#         dmc.Text(
#             "Completed, click back button to get to previous step",
#             ta="center",
#         )
#     ]
# )