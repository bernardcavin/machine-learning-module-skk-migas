import dash
from dash import html, dcc, Output, Input, callback
import flask
import uuid
dash.register_page(__name__)

from pages.sections.regression import upload_data, data_preprocessing
import dash_mantine_components as dmc
from dash import callback, Output, Input, State, ctx
from dash_iconify import DashIconify
from pydantic import BaseModel

def get_icon(icon):
    return DashIconify(icon=icon, height=20)

class StepPage(BaseModel):
    
    title: str
    icon: str
    description: str
    layout: object

step_pages = [
    StepPage(
        title="Data Upload",
        icon="ic:outline-cloud-upload",
        description="Upload your data",
        layout=upload_data.layout
    ),
    StepPage(
        title="Data Preprocessing",
        icon="mdi:gear",
        description="Preprocess your data",
        layout=data_preprocessing.layout
    ),
    StepPage(
        title="Feature Selection",
        icon="mdi:filter-variant",
        description="Select features",
        layout=html.Div()
    ),
    StepPage(
        title="Model Selection",
        icon="mdi:cog",
        description="Select model",
        layout=html.Div()
    ),
    StepPage(
        title="Model Training",
        icon="mdi:train",
        description="Train model",
        layout=html.Div()
    ),
    StepPage(
        title="Model Evaluation",
        icon="mdi:chart-line-variant",
        description="Evaluate model",
        layout=html.Div()
    ),
    StepPage(
        title="Model Export",
        icon="mdi:download",
        description="Export model",
        layout=html.Div()
    ),
]

min_step = 0
max_step = len(step_pages)
active = 0

content = dmc.Tabs(
    [
        dmc.TabsPanel(step_pages[i].layout, value=step_pages[i].title) for i in range(len(step_pages))
    ],
    orientation="vertical",
    value=step_pages[0].title,
    id="regression-tab"
)

def layout():
    
    flask.session['session_id'] = str(uuid.uuid4())

    layout = dmc.Container(
        [
            dmc.Grid(
                [
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
                        span=8,
                    ),
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
                                        )
                                        for step_page in step_pages
                                    ],
                                ),
                            ],
                            mx="md"
                        ),
                        span=4,
                        
                    ),
                ]
                
            )
        ]
    )

    return layout

@callback(
    Output("regression-stepper", "active", allow_duplicate=True),
    Input("regression-back-button", "n_clicks"),
    Input("regression-next-button", "n_clicks"),
    State("regression-stepper", "value"),
    prevent_initial_call=True,
)
def update_with_icons(back, next_, current):
    button_id = ctx.triggered_id
    step = current if current is not None else active
    if button_id == "regression-back-button":
        step = step - 1 if step > min_step else step
    else:
        step = step + 1 if step < max_step else step
    return step

@callback(
    Output("regression-tab", "value"),
    Input("regression-stepper", "active"),
    prevent_initial_call=True
)
def update_tab(active):
    return step_pages[active].title

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