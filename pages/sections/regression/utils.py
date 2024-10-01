import dash_mantine_components as dmc
from dash import Output, Input, callback
from pages.sections.utils import *

loader = dmc.Flex(dmc.Loader(color="blue", size="xl"), justify="center")
continue_button  = dmc.Popover(
        [
            dmc.PopoverTarget(dmc.Button("Next")),
            dmc.PopoverDropdown(
                dmc.Stack(
                    [
                        dmc.Text("Are you sure you want to proceed?"),
                        dmc.Button("Yes", id="regression-next-button", n_clicks=0, color="green", variant="filled"),
                    ],
                    gap=5
                )
            ),
        ],
        id='regression-next-popover',
        width=200,
        position="bottom",
        withArrow=True,
        shadow="md",
        zIndex=2000,
    )

@callback(
    Output("regression-next-popover", "opened"),
    Input("regression-next-button", "n_clicks"),
    prevent_initial_call=True
)
def toggle_next_modal(n_clicks):
    if n_clicks > 0:
        return False

reset_button = dmc.Button(
    "Reset",
    id="regression-toggle-reset-modal",
    n_clicks=0,
    color="red",
    variant="filled",
)