"""
MLflow UI and tracking utilities for cataract detection experiments.
"""

import os
import subprocess
import sys
import mlflow
import pandas as pd


def start_mlflow_ui(port=5000, backend_store_uri=None):
    """
    Start the MLflow UI server.
    
    Args:
        port: Port number for the UI (default 5000)
        backend_store_uri: Backend store URI (default ./mlruns)
    """
    backend_store_uri = backend_store_uri or "./mlruns"
    
    print(f"Starting MLflow UI on http://localhost:{port}")
    print(f"Backend store: {backend_store_uri}")
    print("\nPress Ctrl+C to stop the server.\n")
    
    # Use mlflow command line tool
    cmd = [
        sys.executable, "-m", "mlflow",
        "ui",
        "--backend-store-uri", backend_store_uri,
        "--port", str(port)
    ]
    
    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\nMLflow UI stopped.")


def list_experiments():
    """List all MLflow experiments."""
    experiments = mlflow.search_experiments()
    
    print("\n" + "="*80)
    print("MLflow Experiments")
    print("="*80)
    
    for exp in experiments:
        print(f"\nExperiment: {exp.name} (ID: {exp.experiment_id})")
        print(f"  Artifact location: {exp.artifact_location}")
        
        # List runs in this experiment
        runs = mlflow.search_runs(experiment_ids=[exp.experiment_id])
        print(f"  Runs: {len(runs)}")
        
        for idx, run in runs.head(10).iterrows():
            run_id = run['run_id']
            test_acc = run['metrics.test_acc'] if 'metrics.test_acc' in run.index else 'N/A'
            status = run['status']
            print(f"    - {run_id}: status={status}, test_acc={test_acc}")
        
        if len(runs) > 10:
            print(f"    ... and {len(runs) - 10} more")
    
    print("\n" + "="*80)


def get_best_run(experiment_name="cataract_detection", metric="test_acc"):
    """Get the best run for a given experiment and metric."""
    try:
        exp = mlflow.get_experiment_by_name(experiment_name)
        if exp is None:
            print(f"Experiment '{experiment_name}' not found.")
            return None
        
        runs = mlflow.search_runs(experiment_ids=[exp.experiment_id], order_by=[f"metrics.{metric} DESC"])
        
        if len(runs) == 0:
            print(f"No runs found in experiment '{experiment_name}'.")
            return None
        
        best_run = runs.iloc[0]
        print(f"\nBest run in '{experiment_name}':")
        print(f"  Run ID: {best_run['run_id']}")
        print(f"  {metric}: {best_run.get(f'metrics.{metric}', 'N/A')}")
        
        print(f"  Parameters:")
        for col in best_run.index:
            if col.startswith('params.'):
                param_name = col.replace('params.', '')
                if pd.notna(best_run[col]):
                    print(f"    {param_name}: {best_run[col]}")
        
        return best_run
        
    except Exception as e:
        print(f"Error: {e}")
        return None


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="MLflow utilities")
    subparsers = parser.add_subparsers(dest="command", help="Command")
    
    # UI command
    ui_parser = subparsers.add_parser("ui", help="Start MLflow UI")
    ui_parser.add_argument("--port", type=int, default=5000, help="Port number")
    ui_parser.add_argument("--backend-store", default="file:./mlruns", help="Backend store URI")
    
    # List command
    subparsers.add_parser("list", help="List all experiments and runs")
    
    # Best command
    best_parser = subparsers.add_parser("best", help="Get best run")
    best_parser.add_argument("--experiment", default="cataract_detection", help="Experiment name")
    best_parser.add_argument("--metric", default="test_acc", help="Metric to optimize")
    
    args = parser.parse_args()
    
    if args.command == "ui":
        start_mlflow_ui(port=args.port, backend_store_uri=args.backend_store)
    elif args.command == "list":
        list_experiments()
    elif args.command == "best":
        get_best_run(experiment_name=args.experiment, metric=args.metric)
    else:
        parser.print_help()
