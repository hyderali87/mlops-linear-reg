# pipelines/pipeline_compile.py
from pathlib import Path
from kfp.compiler import Compiler
from pipelines.pipeline_v2 import pipeline  # import the pipeline() defined above

if __name__ == "__main__":
    out_path = Path(__file__).parent / "linreg_v2_pipeline.json"
    print(f"[compile] writing spec to: {out_path.resolve()}")
    Compiler().compile(pipeline_func=pipeline, package_path=str(out_path))
    print("[compile] done.")
