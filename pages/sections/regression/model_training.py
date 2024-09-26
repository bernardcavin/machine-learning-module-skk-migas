from dash_pydantic_form import ModelForm, fields
from pydantic import BaseModel, Field, ValidationError
from typing import List, Optional, Union, Literal
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import plotly.express as px
import plotly.graph_objects as go
from typing import List, Dict
from dash import Input, Output, callback, html, State, dcc, MATCH, ALL
from dash.exceptions import PreventUpdate
import dash_mantine_components as dmc
from pages.sections.regression.utils import create_table_description, session_get_file_path, session_df_to_file, session_json_to_dict, session_save_model
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
import json
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

class ModelTrainingSchema(BaseModel):
    ml_model_type: Literal["linear", "ridge", "lasso", "elasticnet", "decision_tree"] = Field(
        ..., description="Type of regression model to train"
    )
    validation_method: Literal["train_test_split", "cross_validation"] = Field(
        ..., description="Validation method to be used"
    )
    test_size: Optional[float] = Field(
        0.2, description="Test size ratio for train-test split"
    )
    cross_validation_folds: Optional[int] = Field(
        5, description="Number of folds for cross-validation"
    )
    tuning_method: Literal["manual", "grid_search", "random_search"] = Field(
        ..., description="Hyperparameter tuning method"
    )
    # Model-specific parameters
    alpha: Optional[float] = Field(
        None, description="Regularization strength for Ridge, Lasso, ElasticNet"
    )
    max_depth: Optional[int] = Field(
        None, description="Maximum depth of the tree for DecisionTreeRegressor"
    )
    min_samples_split: Optional[int] = Field(
        None, description="Minimum samples required to split an internal node for DecisionTreeRegressor"
    )
    l1_ratio: Optional[float] = Field(
        None, description="The ElasticNet mixing parameter (only for ElasticNet)"
    )

    # Evaluation metrics
    metrics: list[str] = Field(
        description="List of metrics to evaluate the model",
        default_factory=list
    )

def select_train_evaluate(df: pd.DataFrame, target: str, schema: ModelTrainingSchema):
    # Prepare data for training
    X = df.drop(columns=[target])
    y = df[target]

    # Define the model
    if schema.ml_model_type == "linear":
        model = LinearRegression()
    elif schema.ml_model_type == "ridge":
        model = Ridge(alpha=schema.alpha or 1.0)
    elif schema.ml_model_type == "lasso":
        model = Lasso(alpha=schema.alpha or 1.0)
    elif schema.ml_model_type == "elasticnet":
        model = ElasticNet(alpha=schema.alpha or 1.0, l1_ratio=schema.l1_ratio or 0.5)
    elif schema.ml_model_type == "decision_tree":
        model = DecisionTreeRegressor(max_depth=schema.max_depth, min_samples_split=schema.min_samples_split or 2)

    # Handle validation method
    if schema.validation_method == "train_test_split":
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=schema.test_size)
    else:
        X_train, X_test, y_train, y_test = X, None, y, None  # For cross-validation, we'll use the entire dataset

    # Hyperparameter tuning
    if schema.tuning_method == "manual":
        # Parameters are already set during model initialization
        pass
    elif schema.tuning_method in ["grid_search", "random_search"]:
        param_grid = {}
        if schema.ml_model_type in ["ridge", "lasso", "elasticnet"]:
            param_grid['alpha'] = [0.1, 1.0, 10.0] if schema.alpha is None else [schema.alpha]
        if schema.ml_model_type == "elasticnet":
            param_grid['l1_ratio'] = [0.1, 0.5, 0.9] if schema.l1_ratio is None else [schema.l1_ratio]
        if schema.ml_model_type == "decision_tree":
            param_grid['max_depth'] = [None, 5, 10, 15] if schema.max_depth is None else [schema.max_depth]
            param_grid['min_samples_split'] = [2, 5, 10] if schema.min_samples_split is None else [schema.min_samples_split]

        if schema.tuning_method == "grid_search":
            search = GridSearchCV(model, param_grid=param_grid, cv=schema.cross_validation_folds)
        else:
            search = RandomizedSearchCV(model, param_distributions=param_grid, cv=schema.cross_validation_folds)

        search.fit(X_train, y_train)
        model = search.best_estimator_

    # Training the model
    if schema.validation_method == "train_test_split":
        model.fit(X_train, y_train)

    # Evaluate the model
    evaluation_results = {}
    visualizations = {}

    if schema.validation_method == "train_test_split":
        y_pred = model.predict(X_test)

        # Calculate evaluation metrics
        if "R2" in schema.metrics:
            evaluation_results["R2"] = r2_score(y_test, y_pred)
        if "RMSE" in schema.metrics:
            evaluation_results["RMSE"] = np.sqrt(mean_squared_error(y_test, y_pred))
        if "MAE" in schema.metrics:
            evaluation_results["MAE"] = mean_absolute_error(y_test, y_pred)

        # Create visualizations
        # Predicted vs Actual Plot
        fig_pred_actual = px.scatter(
            x=y_test,
            y=y_pred,
            labels={'x': 'Actual Values', 'y': 'Predicted Values'},
            title='Predicted vs Actual Values'
        )
        fig_pred_actual.add_trace(
            go.Scatter(x=y_test, y=y_test, mode='lines', name='Ideal Fit')
        )
        fig_pred_actual.update_layout(template='simple_white')
        visualizations['predicted_vs_actual'] = fig_pred_actual

        # Residual Plot
        residuals = y_test - y_pred
        fig_residuals = px.scatter(
            x=y_pred,
            y=residuals,
            labels={'x': 'Predicted Values', 'y': 'Residuals'},
            title='Residual Plot'
        )
        fig_residuals.add_hline(y=0, line_dash="dash", line_color="red")
        fig_residuals.update_layout(template='simple_white')
        visualizations['residuals'] = fig_residuals

    else:
        # For cross-validation, we will evaluate the metrics across the folds
        if "R2" in schema.metrics:
            r2_scores = cross_val_score(model, X_train, y_train, cv=schema.cross_validation_folds, scoring='r2')
            evaluation_results["R2"] = np.mean(r2_scores)
        if "RMSE" in schema.metrics:
            neg_mse_scores = cross_val_score(model, X_train, y_train, cv=schema.cross_validation_folds, scoring='neg_mean_squared_error')
            evaluation_results["RMSE"] = np.sqrt(-np.mean(neg_mse_scores))
        if "MAE" in schema.metrics:
            neg_mae_scores = cross_val_score(model, X_train, y_train, cv=schema.cross_validation_folds, scoring='neg_mean_absolute_error')
            evaluation_results["MAE"] = -np.mean(neg_mae_scores)
        # Note: Visualizations are not generated for cross-validation in this example

    # Feature Importance (if applicable)
    if schema.ml_model_type == "decision_tree":
        importance = model.feature_importances_
        feature_names = X.columns
        fig_feature_importance = px.bar(
            x=feature_names,
            y=importance,
            labels={'x': 'Features', 'y': 'Importance'},
            title='Feature Importance'
        )
        fig_feature_importance.update_layout(template='simple_white')
        visualizations['feature_importance'] = fig_feature_importance

    return model, evaluation_results, visualizations

def layout():

    form = ModelForm(
        ModelTrainingSchema,
        "model_selection",
        "main",
        fields_repr={
            "ml_model_type": fields.Select(
                options_labels={"linear": "Linear", "ridge": "Ridge", "lasso": "Lasso", "elasticnet": "Elastic Net", "decision_tree": "Decision Tree"},
                description="Type of regression model to train",
                required=True,
            ),
            "validation_method": fields.Select(
                options_labels={"train_test_split": "Train-Test Split", "cross_validation": "Cross-Validation"},
                description="Validation method",
                required=True,
            ),
            "test_size": {
                "visible": ("validation_method", "==", "train_test_split"),
            },
            "cross_validation_folds": {
                "visible": ("validation_method", "==", "cross_validation"),
            },
            "alpha": {
                "visible": ("ml_model_type", "==", "ridge"),
                "visible": ("tuning_method", "==", "manual"),
            },
            "l1_ratio": {
                "visible": ("ml_model_type", "==", "elasticnet"),
                "visible": ("tuning_method", "==", "manual"),
            },
            "max_depth": {
                "visible": ("ml_model_type", "==", "decision_tree"),
                "visible": ("tuning_method", "==", "manual"),
            },
            "min_samples_split": {
                "visible": ("ml_model_type", "==", "decision_tree"),
                "visible": ("tuning_method", "==", "manual"),
            },
            "tuning_method": fields.Select(
                options_labels={"manual": "Manual", "grid_search": "Grid Search", "random_search": "Random Search"},
                description="Hyperparameter tuning method",
                required=True,
            ),
            "metrics": fields.MultiSelect(
                data_getter=lambda: ["R2", "RMSE", "MAE"],
                description="Metrics to evaluate the model",
                required=True,
            )
        },
    )

    layout = dmc.Stack(
        [
            form,
            dmc.Button("Apply", color="blue", id='apply_model_selection', n_clicks=0),
            html.Div(id="model-selection-output"),
        ]
    )
    
    return layout

@callback(
    Output("model-selection-output", "children"),
    Input('apply_model_selection', 'n_clicks'),
    State(ModelForm.ids.main("model_selection", 'main'), "data"),
)
def apply_model_training(n_clicks, form_data):
    if n_clicks > 0:
        
        try:
            
            feature_selection_dict = session_json_to_dict('feature_selection')
            target = feature_selection_dict.get('target')
            df = pd.read_csv(session_get_file_path('preprocessed', extension='csv'))
            
            model, scores, visualizations = select_train_evaluate(df, target, ModelTrainingSchema(**form_data))
            
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
            )
        except Exception as exc:
            return html.Div(
                [
                    dmc.Alert(
                        "There was an error applying model selection.",
                        color="red",
                        variant="filled",
                    )
                ]
            )
