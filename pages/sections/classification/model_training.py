from dash_pydantic_form import ModelForm, fields
from pydantic import BaseModel, Field
from typing import Optional, Literal
import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, callback, html, State, dcc, no_update
import dash_mantine_components as dmc
from pages.sections.classification.utils import parse_validation_errors, session_get_file_path, session_json_to_dict, session_save_model,reset_button, continue_button
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
import numpy as np
from pydantic import ValidationError
from typing import List
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    precision_recall_curve,
    confusion_matrix,
)

class ModelTrainingSchema(BaseModel):
    # Model Selection
    ml_model_type: Literal["logistic_regression", "decision_tree", "random_forest", "svm"] = Field(
        ..., description="Type of classification model to train."
    )
    
    # Parameter Tuning
    tuning_method: Literal["manual", "grid_search", "random_search"] = Field(
        ..., description="Hyperparameter tuning method to be used."
    )
    
    # Model-specific parameters
    C: Optional[float] = Field(None, description="Regularization strength for Logistic Regression and SVM.")
    max_depth: Optional[int] = Field(None, description="Maximum depth of the tree for Decision Tree and Random Forest.")
    n_estimators: Optional[int] = Field(None, description="Number of trees in the forest for Random Forest.")
    
    # Model Evaluation
    metrics: list[str] = Field(
        ..., description="List of evaluation metrics to compute.", default_factory=list
    )

def select_train_evaluate(df: pd.DataFrame, target: str, schema: ModelTrainingSchema):
    # Prepare data for training
    X = df.drop(columns=[target])
    y = df[target]

    # Model Selection
    if schema.ml_model_type == "logistic_regression":
        model = LogisticRegression()
    elif schema.ml_model_type == "decision_tree":
        model = DecisionTreeClassifier()
    elif schema.ml_model_type == "random_forest":
        model = RandomForestClassifier()
    elif schema.ml_model_type == "svm":
        model = SVC(probability=True)  # Enable probability estimates for ROC curve visualization
    else:
        raise ValueError(f"Unsupported model type: {schema.ml_model_type}")

    # Hyperparameter Tuning
    if schema.tuning_method == "manual":
        # For manual tuning, set the parameters directly
        if isinstance(model, (LogisticRegression, SVC)) and schema.C is not None:
            model.C = schema.C
        if isinstance(model, (DecisionTreeClassifier, RandomForestClassifier)) and schema.max_depth is not None:
            model.max_depth = schema.max_depth
        if isinstance(model, RandomForestClassifier) and schema.n_estimators is not None:
            model.n_estimators = schema.n_estimators

    elif schema.tuning_method in ["grid_search", "random_search"]:
        param_grid = {}
        if isinstance(model, (LogisticRegression, SVC)):
            param_grid["C"] = [0.01, 0.1, 1.0, 10.0] if schema.C is None else [schema.C]
        if isinstance(model, (DecisionTreeClassifier, RandomForestClassifier)):
            param_grid["max_depth"] = [3, 5, 10, 15] if schema.max_depth is None else [schema.max_depth]
        if isinstance(model, RandomForestClassifier):
            param_grid["n_estimators"] = [50, 100, 200] if schema.n_estimators is None else [schema.n_estimators]

        if schema.tuning_method == "grid_search":
            search = GridSearchCV(model, param_grid=param_grid, cv=5)
        else:
            search = RandomizedSearchCV(model, param_distributions=param_grid, cv=5)

        search.fit(X, y)
        model = search.best_estimator_

    # Train the model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)

    # Model Evaluation
    y_pred = model.predict(X_test)
    evaluation_results = evaluate_classification_model(y_test, y_pred, schema.metrics)

    # Create Visualizations
    visualizations = create_classification_visualizations(model, X_test, y_test, y_pred, schema.metrics)

    return model, evaluation_results, visualizations,  # Return the visualizations

# Helper function to evaluate the classification model
def evaluate_classification_model(y_test, y_pred, metrics: List[str]) -> dict:
    evaluation_results = {}
    if "accuracy" in metrics:
        evaluation_results["accuracy"] = accuracy_score(y_test, y_pred)
    if "precision" in metrics:
        evaluation_results["precision"] = precision_score(y_test, y_pred, average="weighted")
    if "recall" in metrics:
        evaluation_results["recall"] = recall_score(y_test, y_pred, average="weighted")
    if "f1" in metrics:
        evaluation_results["f1"] = f1_score(y_test, y_pred, average="weighted")
    return evaluation_results

# Helper function to create Plotly visualizations for classification models
def create_classification_visualizations(model, X_test, y_test, y_pred, metrics: List[str]) -> dict:
    visualizations = {}

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    fig_cm = go.Figure(data=go.Heatmap(z=cm, x=model.classes_, y=model.classes_, colorscale="Blues"))
    fig_cm.update_layout(title="Confusion Matrix", xaxis_title="Predicted", yaxis_title="Actual")
    visualizations["confusion_matrix"] = fig_cm

    # ROC Curve (if applicable)
    if len(model.classes_) == 2:  # ROC Curve is applicable only for binary classification
        y_prob = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name="ROC Curve"))
        fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash"), name="Random Classifier"))
        fig_roc.update_layout(title="ROC Curve", xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
        visualizations["roc_curve"] = fig_roc

    # Precision-Recall Curve (if applicable)
    if len(model.classes_) == 2:  # Precision-Recall Curve is applicable only for binary classification
        precision, recall, _ = precision_recall_curve(y_test, y_prob)
        fig_pr = go.Figure()
        fig_pr.add_trace(go.Scatter(x=recall, y=precision, mode="lines", name="Precision-Recall Curve"))
        fig_pr.update_layout(title="Precision-Recall Curve", xaxis_title="Recall", yaxis_title="Precision")
        visualizations["precision_recall_curve"] = fig_pr

    # Feature Importance (for models that support it)
    if hasattr(model, "feature_importances_"):
        feature_importance = model.feature_importances_
        fig_fi = go.Figure(data=go.Bar(x=X_test.columns, y=feature_importance))
        fig_fi.update_layout(title="Feature Importance", xaxis_title="Features", yaxis_title="Importance")
        visualizations["feature_importance"] = fig_fi

    return visualizations

def layout():

    form = ModelForm(
        ModelTrainingSchema,
        "model_selection",
        "main",
        fields_repr={
            "ml_model_type": fields.Select(
                options_labels={"logistic_regression": "Logistic Regression", "decision_tree": "Decision Tree", "random_forest": "Random Forest", "svm": "SVM"},
                description="Type of classification model to train",
                required=True,
            ),
            "tuning_method": fields.Select(
                options_labels={"manual": "Manual", "grid_search": "Grid Search", "random_search": "Random Search"},
                description="Hyperparameter tuning method",
                required=True,
            ),
            "C":{
                "visible": ("ml_model_type", "==", "logistic_regression"),
                "visible": ("ml_model_type", "==", "svm"),
                "visible": ("tuning_method", "==", "manual"),
            },
            "max_depth":{
                "visible": ("ml_model_type", "==", "decision_tree"),
                "visible": ("ml_model_type", "==", "random_forest"),
                "visible": ("tuning_method", "==", "manual"),
            },
            "n_estimators":{
                "visible": ("ml_model_type", "==", "random_forest"),
                "visible": ("tuning_method", "==", "manual"),
            },
            "metrics": fields.MultiSelect(
                data_getter=lambda: ["accuracy", "precision", "recall", "f1"],
                description="Metrics to evaluate the model",
                required=True,
            )
        },
    )

    layout = dmc.Stack(
        [
            form,
            dmc.Button("Apply", color="blue", id='classification-apply_model_selection', n_clicks=0),
            html.Div(id="classification-model-selection-output"),
            dmc.Group(
                [
                    reset_button,
                    html.Div(
                        id='classification-proceed-output',
                    )
                ],
                justify="space-between",
            )
        ]
    )
    
    return layout

@callback(
    Output("classification-model-selection-output", "children"),
    Output("classification-proceed-output", "children", allow_duplicate=True),
    Input('classification-apply_model_selection', 'n_clicks'),
    State(ModelForm.ids.main("model_selection", 'main'), "data"),
    prevent_initial_call = True
)
def apply_model_training(n_clicks, form_data):
    if n_clicks > 0:
        
        try:
            
            feature_selection_dict = session_json_to_dict('feature_selection')
            target = feature_selection_dict.get('target')
            df = pd.read_csv(session_get_file_path('preprocessed', extension='csv'), index_col=0)
            df = df[feature_selection_dict.get('selected_features')+[target]]
            
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
            ), continue_button
            
            
        except ValidationError as exc:
            return html.Div(
                [
                    dmc.Alert(
                        parse_validation_errors(exc),
                        color="red",
                        variant="filled",
                        withCloseButton=True
                    )
                ]
            ), no_update
        
        except Exception as exc:
            print(exc)
            
            return html.Div(
                [
                    dmc.Alert(
                        "There was an error applying model training.",
                        color="red",
                        variant="filled",
                        withCloseButton=True
                    )
                ]
            ), no_update

