import dash
import dash_mantine_components as dmc
from dash import dcc
from enum import Enum

dash.register_page(__name__, path="/")

class PageState(Enum):
    ONLINE = 'ONLINE'
    OFFLINE = 'OFFLINE'

def FeatureCard(title, image, description, link, page_state: PageState):
    
    if page_state == PageState.ONLINE:
        badge = dmc.Badge("Online", color="green")
    else:
        badge = dmc.Badge("Offline", color="red")
    
    return dmc.Card(
        children=[
            dmc.CardSection(
                dmc.Image(
                    src=image,
                    h=160,
                    alt="Norway",
                )
            ),
            dmc.Group(
                [
                    dmc.Text(title, fw=500),
                    badge,
                ],
                justify="space-between",
                mt="md",
                mb="xs",
            ),
            dmc.Text(
                description,
                size="sm",
                c="dimmed",
            ),
            dmc.Anchor(
                dmc.Button(
                    "Go",
                    color="blue",
                    fullWidth=True,
                    mt="md",
                    radius="md",
                    disabled=True if page_state == PageState.OFFLINE else False
                ),
                href=link,
            )
        ],
        withBorder=True,
        shadow="sm",
        radius="md",
        w=350,
    )

layout = dmc.SimpleGrid(
    [
        dmc.Center(
            [
                dmc.Stack(
                    [
                        dmc.SimpleGrid(
                            [
                                FeatureCard(
                                    title="Regression",
                                    image='assets/regression.png',
                                    description="Regression is a supervised learning task that involves predicting a continuous target variable based on one or more input features.",
                                    link="/regression",
                                    page_state=PageState.ONLINE
                                ),
                                FeatureCard(
                                    title="Classification",
                                    image="assets/classification.png",
                                    description="Classification is a supervised learning task that involves predicting a discrete target variable based on one or more input features.",
                                    link="/classification",
                                    page_state=PageState.ONLINE
                                ),
                                FeatureCard(
                                    title="Clustering",
                                    image="assets/clustering.png",
                                    description="Clustering is a supervised learning task that involves grouping data points based on their similarity",
                                    link="/clustering",
                                    page_state=PageState.ONLINE
                                ),
                            ],
                            cols=3,
                            w=1100
                        ),
                    ]
                )
            ],
            h=720
        )
    ]
)
