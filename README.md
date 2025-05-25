# End-to-End MLOps Pipelines: Titanic Survival Prediction

This repository demonstrates scalable MLOps workflows for the Titanic dataset https://www.kaggle.com/competitions/titanic/data, incrementally integrating modern tools for data versioning, experiment tracking, model deployment, and CI/CD. Each project is a self-contained pipeline, from data ingestion to model evaluation, with progressive complexity.

# Projects Overview
1. Baseline Pipeline
   
     Techniques: Raw Python scripting, no configuration management.
     
     Steps:
     
     Data loading → EDA (matplotlib/seaborn) → Feature engineering  → Model training (Logistic Regression/Random Forest) → Evaluation (metrics).
     
     Output: Saved model (pickle), evaluation reports.

2. Structured Pipeline with Hydra
   
     Techniques: Configuration management using Hydra.
     
     Features:
     
     YAML configs for data paths, hyperparameters, and preprocessing steps.
     
     Dynamic experiment reproducibility via CLI overrides.

3. Data Versioning with DVC

     Tools: DVC + Git.
     
     Workflow:
     
     Track datasets, models, and metrics.
     
     Pipeline orchestration (dvc.yaml) for reproducible data processing and training.

4. Experiment Tracking with MLflow
   
     Integration: MLflow for logging metrics, artifacts, and hyperparameters.
     
     Features:
     
     Compare runs across models.
     
     Log feature importance plots and validation curves.

5. Deployment
   
     A. Real-time API (FastAPI + Docker)
     
      Serve models via REST API with FastAPI.
       
      Containerize with Docker and deploy locally.
     
     
     B. Batch Serving (Prefect + MotherDuck)
     
      Schedule batch predictions using Prefect.
       
      Store results in MotherDuck (DuckDB cloud).

  6. CI/CD Pipeline (Gitlab)
   
     Automated testing (pytest) on push/PR.
     
     Build Docker images and deploy on merge to main.



# Repository Structure

.  
├── 1_baseline/                  # Basic script-based pipeline (not configured)
├── 2_hydra_config/             # Hydra-managed configs  
├── 3_dvc_data_versioning/       # DVC pipelines  
├── 4_mlflow_tracking/           # Experiment logging  
├── 5_deployment/  
│   ├── fastapi_docker/          # Real-time API  
│   └── prefect_batch/           # Batch processing  
├── 6_ci_cd/                     # Gitlab workflows  
