# kfp_run/run_pipeline.py
"""
Submit a Vertex AI Pipelines (KFP v2) run from a compiled JSON spec.

Usage (typical):
  python kfp_run/run_pipeline.py \
    --project YOUR_PROJECT \
    --region us-central1 \
    --template pipelines/linreg_v2_pipeline.json \
    --raw-csv-uri gs://YOUR_BUCKET/kfp-demo/train.csv \
    --mse-threshold 10.0

Optional:
  --mlflow-tracking-uri https://<your-mlflow-url>
  --service-account vertex-pipelines-sa@YOUR_PROJECT.iam.gserviceaccount.com
  --display-name linreg-v2-demo
  --no-cache
  --labels env=dev,app=linreg
"""

import argparse
import os
import sys
from typing import Dict

from google.cloud import aiplatform


def parse_kv_labels(s: str) -> Dict[str, str]:
    """
    Parse labels like: "env=dev,app=linreg,team=ml"
    """
    out = {}
    if not s:
        return out
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise argparse.ArgumentTypeError(f"Bad label '{part}'. Use key=value.")
        k, v = part.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def main():
    ap = argparse.ArgumentParser(description="Run a Vertex AI pipeline job (KFP v2).")
    ap.add_argument("--project", default=os.getenv("PROJECT_ID"), required=False, help="GCP project ID")
    ap.add_argument("--region", default=os.getenv("REGION", "us-central1"), help="Vertex AI region")
    ap.add_argument("--template", default=os.getenv("PIPELINE_SPEC", "pipelines/linreg_v2_pipeline.json"),
                    help="Path to compiled pipeline JSON")
    ap.add_argument("--display-name", default=os.getenv("DISPLAY_NAME", "linreg-v2-demo"),
                    help="Pipeline job display name")

    # Pipeline parameters (match your pipeline signature)
    ap.add_argument("--raw-csv-uri", required=True, help="gs:// path to training CSV (must include 'target' column)")
    ap.add_argument("--mse-threshold", type=float, default=float(os.getenv("MSE_THRESHOLD", 10.0)),
                    help="Gate for evaluation step")
    ap.add_argument("--mlflow-tracking-uri", default=os.getenv("MLFLOW_TRACKING_URI", ""),
                    help="Optional MLflow tracking URI")
    ap.add_argument("--mlflow-experiment", default=os.getenv("MLFLOW_EXPERIMENT", "linreg"),
                    help="Optional MLflow experiment name")

    # Execution options
    ap.add_argument("--service-account", default=os.getenv("VERTEX_PIPELINES_SA", ""),
                    help="Optional SA email to run the pipeline")
    ap.add_argument("--labels", type=parse_kv_labels, default=parse_kv_labels(os.getenv("LABELS", "")),
                    help="Comma-separated key=value labels, e.g., env=dev,app=linreg")
    ap.add_argument("--no-cache", action="store_true", help="Disable Vertex caching for this run")

    args = ap.parse_args()

    # Basic validation
    if not args.project:
        ap.error("--project (or env PROJECT_ID) is required.")
    if not args.region:
        ap.error("--region (or env REGION) is required.")
    if not os.path.exists(args.template):
        ap.error(f"Pipeline spec not found: {args.template}. Compile it first.")

    # Initialize Vertex
    aiplatform.init(project=args.project, location=args.region)

    # Build pipeline params (exact names must match your pipeline signature)
    params = {
        "raw_csv_uri": args.raw_csv_uri,
        "mse_threshold": args.mse_threshold,
        "mlflow_tracking_uri": args.mlflow_tracking_uri,
        "mlflow_experiment": args.mlflow_experiment,
    }

    # Create job
    job = aiplatform.PipelineJob(
        display_name=args.display_name,
        template_path=args.template,
        parameter_values=params,
        enable_caching=not args.no_cache,
        labels=args.labels or None,
    )

    # Run (optionally with a service account)
    if args.service_account:
        job.run(sync=False, service_account=args.service_account)
    else:
        job.run(sync=False)

    # Print identifiers + a Console URL
    resource_name = job.resource_name or ""  # e.g., projects/123/locations/us-central1/pipelineJobs/PIPELINE_JOB_ID
    print(f"\nSubmitted pipeline job:\n  resource: {resource_name}")

    # Try to extract the job ID from the resource name for a clickable URL
    try:
        job_id = resource_name.split("/")[-1]
        url = (
            f"https://console.cloud.google.com/vertex-ai/locations/{args.region}"
            f"/pipelines/runs/{job_id}?project={args.project}"
        )
        print(f"Open in Console:\n  {url}\n")
    except Exception:
        print("Open Vertex AI → Pipelines to view the run.\n")

    print("Tip:\n  If you hit GCS permission errors, grant your Vertex runner service account read/write "
          f"on gs://{args.raw_csv_uri.split('/')[2]} (your bucket).")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n[ERROR] {e}\n", file=sys.stderr)
        sys.exit(1)
