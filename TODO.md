# TODO: Project Reorganization to Target Structure

## Steps (to be marked as completed progressively):

- [x] Step 1: Create all required directories (.github/workflows/, Membangun_Model/, MLproject/, monitoring & logging/, preprosecessing/workflow/)
- [x] Step 2: Handle .github/workflows/ci-mlflow-training.yml (copy/rename ci_workflow.txt)
- [x] Step 3: Copy files to Membangun_Model/ (modelling.py, modelling_tuning.py, preprocessed_corn_data.csv, requirements.txt)
- [ ] Step 4: Copy files to MLproject/ (MLproject, conda.yaml, modelling.py, modelling_tuning.py, preprocessed_corn_data.csv, X*.csv, y*.csv)
- [ ] Step 5: Copy/move files to monitoring & logging/ (inference.py, prometheus_exporter.py, prometheus-corn.yml)
- [ ] Step 6: Verify structure with tree / dir listing
- [ ] Step 7: Optional cleanup (delete originals if desired), git add/commit
