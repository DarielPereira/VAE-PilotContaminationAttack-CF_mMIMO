# cVAE-PilotContaminationAttack-CF_mMIMO
Using conditional variational autoencoders (cVAEs) to detect and mitigate multi-target pilot contamination attacks in CF-mMIMO communication systems

## Description of the commits
Brief descriptions of the submission scripts that automate training, evaluation and attack simulations.

- `scripts/submit_train.sh`  
  Runs the training pipeline locally or on a cluster. Typically accepts a configuration file from `configs/` and resource options (GPU, epochs).  
  Example: `bash scripts/submit_train.sh configs/train.yaml`

- `scripts/submit_eval.sh`  
  Launches evaluations using pre-trained checkpoints, producing metrics and figures in `results/`.  
  Example: `bash scripts/submit_eval.sh models/checkpoint.pth`

- `scripts/submit_attack.sh`  
  Simulates multi-target pilot contamination attacks on predefined scenarios and saves results for analysis. Useful for robustness testing.  
  Example: `bash scripts/submit_attack.sh configs/attack_scenario.yaml`

- `scripts/submit_inference.sh`  
  Runs inference on new data using a trained model and exports predictions.  
  Example: `bash scripts/submit_inference.sh models/best_model.pth data/new_inputs/`

### To do
- Document input/output data formats in `data/` and provide download/preprocessing examples.
- Add example configuration files in `configs/` and templates for reproducible experiments.
- Provide instructions for running on Windows (PowerShell) and on clusters (SLURM / PBS) if applicable.
- Add basic unit tests for `src/` and configure CI.
- Containerize the environment (Docker/Podman) and publish an image for reproducibility.
- Improve logging and metrics (TensorBoard / MLflow) and include analysis notebooks.
- Define checkpoint naming policy and storage location under `models/`.

## File structure
- `README.md` - Central project document (this file).
- `requirements.txt` / `environment.yml` - Project dependencies.
- `src/` - Main source code (models, training, evaluation, utilities).
  - `src/models.py` - Implementations of cVAE and related models.
  - `src/train.py` - Training script.
  - `src/eval.py` - Evaluation and metrics script.
  - `src/data.py` - Data loaders and preprocessing.
- `notebooks/` - Notebooks for experimentation and visualization.
- `data/` - Raw and processed data (if applicable, add download instructions).
- `models/` - Saved weights and checkpoints.
- `results/` - Experiment outputs: logs, figures, metrics.
- `scripts/` - Utility scripts for execution and deployment.
  - `scripts/submit_train.sh`
  - `scripts/submit_eval.sh`
  - `scripts/submit_attack.sh`
  - `scripts/submit_inference.sh`
- `configs/` - Configuration files and parameters for experiments.
- `LICENSE` - Project license.

## Credits and contact
Add authors, bibliographic references and contact information for the project maintainer.
