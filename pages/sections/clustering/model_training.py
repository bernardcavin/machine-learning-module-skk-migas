from dash_pydantic_form import ModelForm, fields
from pydantic import BaseModel, Field
from typing import Optional, Literal
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from dash import Input, Output, callback, html, State, dcc, no_update
import dash_mantine_components as dmc
from pages.sections.clustering.utils import parse_validation_errors, session_get_file_path, session_json_to_dict, session_save_model,reset_button, continue_button
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from pydantic import ValidationError
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.metrics import silhouette_samples, silhouette_score, davies_bouldin_score, calinski_harabasz_score

class ModelTrainingSchema(BaseModel):
    # Clustering model parameters
    # General clustering model parameters
    ml_model_type: Literal["kmeans", "dbscan", "agglomerative"] = Field(..., description="Type of clustering model to train.")
    
    # Parameters for KMeans
    n_clusters: Optional[int] = Field(3, description="Number of clusters for KMeans and Agglomerative Clustering.")
    init: Optional[Literal["k-means++", "random"]] = Field("k-means++", description="Method for initialization (for KMeans).")
    n_init: Optional[int] = Field(10, description="Number of time the k-means algorithm will be run with different centroid seeds (for KMeans).")
    max_iter: Optional[int] = Field(300, description="Maximum number of iterations for a single run (for KMeans).")

    # Parameters for DBSCAN
    eps: Optional[float] = Field(0.5, description="The maximum distance between two samples for them to be considered as in the same neighborhood (for DBSCAN).")
    min_samples: Optional[int] = Field(5, description="The number of samples (or total weight) in a neighborhood for a point to be considered a core point (for DBSCAN).")

    # Parameters for Agglomerative Clustering
    affinity: Optional[Literal["euclidean", "l1", "l2", "manhattan", "cosine"]] = Field("euclidean", description="Metric used to compute the linkage (for Agglomerative Clustering).")
    linkage: Optional[Literal["ward", "complete", "average", "single"]] = Field("ward", description="Linkage criterion to use (for Agglomerative Clustering).")

    # Visualization parameters
    x_axis: str = Field(..., description="The feature to use for the x-axis in the scatter plot.")
    y_axis: str = Field(..., description="The feature to use for the y-axis in the scatter plot.")
    show_centers: bool = Field(False, description="Whether to show the cluster centers in the plot (for KMeans).")

    # Evaluation parameters
    metrics: list[str] = Field(
        default_factory=list, description="List of clustering evaluation metrics to compute."
    )


def train_clustering_model(features_df: pd.DataFrame, schema: ModelTrainingSchema):
    
    if schema.ml_model_type == "kmeans":
        model = KMeans(
            n_clusters=schema.n_clusters,
            init=schema.init,
            n_init=schema.n_init,
            max_iter=schema.max_iter,
            random_state=42
        )
    elif schema.ml_model_type == "dbscan":
        model = DBSCAN(
            eps=schema.eps,
            min_samples=schema.min_samples
        )
    elif schema.ml_model_type == "agglomerative":
        model = AgglomerativeClustering(
            n_clusters=schema.n_clusters,
            affinity=schema.affinity,
            linkage=schema.linkage
        )
    else:
        raise ValueError(f"Unsupported clustering model type: {schema.model_type}")

    # Fit the model and get cluster labels
    cluster_labels = model.fit_predict(features_df)

    # Step 2: Evaluate the clustering results using specified metrics
    evaluation_results = {}
    if "silhouette" in schema.metrics:
        silhouette_avg = silhouette_score(features_df, cluster_labels)
        evaluation_results["silhouette_score"] = silhouette_avg
    if "davies_bouldin" in schema.metrics:
        davies_bouldin = davies_bouldin_score(features_df, cluster_labels)
        evaluation_results["davies_bouldin_score"] = davies_bouldin
    if "calinski_harabasz" in schema.metrics:
        calinski_harabasz = calinski_harabasz_score(features_df, cluster_labels)
        evaluation_results["calinski_harabasz_score"] = calinski_harabasz

    # Step 3: Create the visualizations
    visualizations = {}

    # Scatter plot showing clusters
    features_df['Cluster'] = cluster_labels
    plot_df = features_df.copy()
    plot_df['Labeled Cluster'] = [f"Cluster {clst}" for clst in cluster_labels]
    # fig = px.scatter(features_df, x=schema.x_axis, y=schema.y_axis, color='Cluster',
    #                  title='Cluster Scatter Plot', labels={'color': 'Cluster'}, template='plotly_white')
    fig = px.scatter(
        plot_df,
        x=schema.x_axis,
        y=schema.y_axis,
        color='Labeled Cluster',
        title=f'Scatter Plot of {schema.x_axis} vs {schema.y_axis} with Cluster',
        labels={'color': 'Cluster'},
    )

    # Update layout for better visualization
    fig.update_layout(
        xaxis_title=schema.x_axis,
        yaxis_title=schema.y_axis,
        legend_title='Cluster',
        template='plotly_white'
    )
    visualizations['cluster_scatter'] = fig

    # Silhouette plot if applicable
    if "silhouette" in schema.metrics:
        silhouette_vals = silhouette_samples(features_df.drop(columns=['Cluster']), cluster_labels)
        fig_silhouette = go.Figure()
        y_lower = 0

        for i in range(len(set(cluster_labels))):
            ith_cluster_silhouette_values = silhouette_vals[cluster_labels == i]
            ith_cluster_silhouette_values.sort()
            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i

            fig_silhouette.add_trace(
                go.Bar(
                    x=ith_cluster_silhouette_values,
                    y=np.arange(y_lower, y_upper),
                    orientation='h',
                    name=f'Cluster {i}'
                )
            )
            y_lower = y_upper

        fig_silhouette.add_trace(go.Scatter(x=[silhouette_avg, silhouette_avg], y=[0, len(silhouette_vals)],
                                            mode='lines', line=dict(color='red', dash='dash'), name='Average Silhouette'))
        fig_silhouette.update_layout(title='Silhouette Plot', xaxis_title='Silhouette Coefficient Values', yaxis_title='Samples', template='plotly_white')
        visualizations['silhouette_plot'] = fig_silhouette

    return model, evaluation_results, visualizations

def layout():
    
    df = pd.read_csv(session_get_file_path('preprocessed', extension='csv'), index_col=0)

    form = ModelForm(
        ModelTrainingSchema,
        "model_selection",
        "main",
        fields_repr={
            "ml_model_type": fields.Select(
                options_labels={"kmeans": "K-Means", "dbscan": "DBSCAN", "agglomerative": "Agglomerative"},
                description="Type of clustering model to train", required=True),
            "n_clusters":{
                "visible": ("ml_model_type", "==", "kmeans"),
                "visible": ("ml_model_type", "==", "agglomerative"),
            },
            "init":{
                "visible": ("ml_model_type", "==", "kmeans"),
            },
            "n_init":{
                "visible": ("ml_model_type", "==", "kmeans"),
            },
            "max_iter":{
                "visible": ("ml_model_type", "==", "kmeans"),
            },
            "eps":{
                "visible": ("ml_model_type", "==", "dbscan"),
            },
            "min_samples":{
                "visible": ("ml_model_type", "==", "dbscan"),
            },
            "affinity":{
                "visible": ("ml_model_type", "==", "agglomerative"),
            },
            "linkage":{
                "visible": ("ml_model_type", "==", "agglomerative"),
            },
            "x_axis": fields.Select(
                data_getter=lambda: df.columns.to_list(),
                description="The feature to use for the x-axis in the scatter plot.",
            ),
            "y_axis": fields.Select(
                data_getter=lambda: df.columns.to_list(),
                description="The feature to use for the y-axis in the scatter plot.",
            ),
            "show_centers":{
                "visible": ("ml_model_type", "==", "kmeanss"),
            },
            "metrics": fields.MultiSelect(data_getter=lambda: ["silhouette", "davies_bouldin", "calinski_harabasz"], description="Metrics to evaluate the model", required=True),
        },
    )

    layout = dmc.Stack(
        [
            form,
            dmc.Button("Apply", color="blue", id='clustering-apply_model_selection', n_clicks=0),
            html.Div(id="clustering-model-selection-output"),
            dmc.Group(
                [
                    reset_button,
                    html.Div(
                        id='clustering-proceed-output',
                    )
                ],
                justify="space-between",
            )
        ]
    )
    
    return layout

@callback(
    Output("clustering-model-selection-output", "children"),
    Output("clustering-proceed-output", "children", allow_duplicate=True),
    Input('clustering-apply_model_selection', 'n_clicks'),
    State(ModelForm.ids.main("model_selection", 'main'), "data"),
    prevent_initial_call = True
)
def apply_model_training(n_clicks, form_data):
    if n_clicks > 0:
        
        # try:
            
        # feature_selection_dict = session_json_to_dict('feature_selection')
        df = pd.read_csv(session_get_file_path('preprocessed', extension='csv'), index_col=0)
        
        model, scores, visualizations = train_clustering_model(df, ModelTrainingSchema(**form_data))
        
        visualizations = [dcc.Graph(figure=figure) for figure in visualizations.values()]
        
        session_save_model(model, 'model')
        
        return dmc.Stack(
            [
                dmc.Table(
                    data={
                        "head": ["Metric", "Score"],
                        "body": [
                            [metric, score] for metric, score in scores.items()
                        ],
                    }
                ),
                dmc.SimpleGrid(visualizations, cols=len(visualizations)),
            ]
        ), continue_button
            
            
        # except ValidationError as exc:
        #     return html.Div(
        #         [
        #             dmc.Alert(
        #                 parse_validation_errors(exc),
        #                 color="red",
        #                 variant="filled",
        #                 withCloseButton=True
        #             )
        #         ]
        #     ), no_update
        
        # except Exception as exc:
        #     return html.Div(
        #         [
        #             dmc.Alert(
        #                 "There was an error applying model training.",
        #                 color="red",
        #                 variant="filled",
        #                 withCloseButton=True
        #             )
        #         ]
        #     ), no_update
