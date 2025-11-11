# pipelines/pipeline_v2.py
from kfp import dsl
from kfp.dsl import InputPath, OutputPath

# ------------- Components -------------

@dsl.component(
    base_image="python:3.11-slim",
    packages_to_install=["pandas==2.2.2"],
)
def preprocess(raw_csv_uri: str, processed_csv: OutputPath(str)):
    """Load CSV (local or gs://), drop NAs, ensure 'target' exists, write cleaned CSV."""
    import pandas as pd

    df = pd.read_csv(raw_csv_uri)
    df = df.dropna()
    assert "target" in df.columns, "CSV must contain a 'target' column"
    df.to_csv(processed_csv, index=False)
    print(f"[preprocess] wrote -> {processed_csv}")


@dsl.component(
    base_image="python:3.11-slim",
    packages_to_install=[
        "pandas==2.2.2",
        "scikit-learn==1.5.2",
        "joblib==1.4.2",
        "mlflow==2.15.1",
    ],
)
def train(
    processed_csv: InputPath(str),
    model_dir: OutputPath(str),
    mlflow_tracking_uri: str = "",
    mlflow_experiment: str = "linreg",
) -> float:
    """
    Train a LinearRegression model and save it to model_dir.
    Returns: train MSE (float) as the component's primitive output.
    """
    import os, joblib, pandas as pd
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error

    # optional MLflow
    has_mlflow = False
    try:
        import mlflow
        import mlflow.sklearn
        has_mlflow = True
    except Exception:
        pass

    # load data
    df = pd.read_csv(processed_csv)
    y = df.pop("target").values
    X = df.values

    # fit & metric
    model = LinearRegression().fit(X, y)
    preds = model.predict(X)
    mse = float(mean_squared_error(y, preds))

    # persist model
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(model, os.path.join(model_dir, "model.joblib"))
    print(f"[train] model saved -> {model_dir}, train_mse={mse:.6f}")

    # log to MLflow if configured
    if has_mlflow and mlflow_tracking_uri:
        mlflow.set_tracking_uri(mlflow_tracking_uri)
        mlflow.set_experiment(mlflow_experiment)
        with mlflow.start_run() as run:
            mlflow.log_metric("train_mse", mse)
            mlflow.sklearn.log_model(model, "model")
            print(f"[train] mlflow run_id={run.info.run_id}")

    return mse  # <-- primitive output


@dsl.component(
    base_image="python:3.11-slim",
    packages_to_install=[
        "pandas==2.2.2",
        "scikit-learn==1.5.2",
        "joblib==1.4.2",
        "mlflow==2.15.1",
    ],
)
def evaluate(
    processed_csv: InputPath(str),
    model_dir: InputPath(str),
    mlflow_tracking_uri: str = "",
) -> float:
    """
    Evaluate the saved model on the (demo) dataset.
    Returns: eval MSE (float) as the component's primitive output.
    """
    import os, joblib, pandas as pd
    from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

    # optional MLflow
    has_mlflow = False
    try:
        import mlflow
        has_mlflow = True
    except Exception:
        pass

    # load model & data
    model = joblib.load(os.path.join(model_dir, "model.joblib"))
    df = pd.read_csv(processed_csv)
    y = df.pop("target").values
    X = df.values

    # metrics
    preds = model.predict(X)
    mse = float(mean_squared_error(y, preds))
    r2 = float(r2_score(y, preds))
    mae = float(mean_absolute_error(y, preds))
    print(f"[evaluate] eval_mse={mse:.6f}, r2={r2:.6f}, mae={mae:.6f}")

    if has_mlflow and mlflow_tracking_uri:
        mlflow.set_tracking_uri(mlflow_tracking_uri)
        with mlflow.start_run():
            mlflow.log_metric("eval_mse", mse)
            mlflow.log_metric("eval_r2", r2)
            mlflow.log_metric("eval_mae", mae)

    return mse  # <-- primitive output


# ------------- Pipeline -------------

@dsl.pipeline(name="linreg-v2-pipeline")
def pipeline(
    raw_csv_uri: str,
    mse_threshold: float = 10.0,
    mlflow_tracking_uri: str = "",
    mlflow_experiment: str = "linreg",
):
    """
    Lightweight DSL v2 pipeline:
      preprocess -> train -> (if train_mse <= threshold) evaluate
    """
    cleaned = preprocess(raw_csv_uri=raw_csv_uri)

    train_task = train(
        processed_csv=cleaned.outputs["processed_csv"],
        mlflow_tracking_uri=mlflow_tracking_uri,
        mlflow_experiment=mlflow_experiment,
    )

    # Gate evaluation using the primitive float output from train()
    with dsl.If(train_task.output <= mse_threshold):
        evaluate(
            processed_csv=cleaned.outputs["processed_csv"],
            model_dir=train_task.outputs["model_dir"],
            mlflow_tracking_uri=mlflow_tracking_uri,
        )
