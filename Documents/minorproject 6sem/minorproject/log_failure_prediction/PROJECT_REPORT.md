# Self-Learning Log-Based Failure Prediction Using LSTM

## 1. Project Overview
This project predicts potential system failures by learning patterns from HDFS log event sequences using an LSTM neural network.

The model takes a short sequence of events (length 5) and predicts whether the pattern indicates:
- `0`: Normal/Success
- `1`: Failure/Anomaly

---

## 2. Dataset Used
Current dataset files are in `dataset/`.

Main file used for training:
- `dataset/Event_traces.csv`

Important columns:
- `Label`: `Success` or `Fail`
- `Features`: EventId sequence (example: `[E5,E22,E5,E5,E11,...]`)

Related files in dataset folder:
- `anomaly_label.csv`
- `Event_occurrence_matrix.csv`
- `HDFS.log_templates.csv`
- `HDFS.log`

---

## 3. Project Structure
```text
log_failure_prediction/
|
|- dataset/
|  |- Event_traces.csv
|  |- anomaly_label.csv
|  |- Event_occurrence_matrix.csv
|  |- HDFS.log_templates.csv
|  |- HDFS.log
|
|- preprocessing.py
|- sequence_generator.py
|- lstm_model.py
|- train_model.py
|- predict.py
|- predict_manual.py
|- main.py
|- requirements.txt
|- PROJECT_REPORT.md
|
|- artifacts/
   |- lstm_failure_predictor.keras
   |- label_encoder.joblib
   |- config.json
   |- plots/
      |- training_history.png
      |- confusion_matrix.png
      |- metrics_bar.png
```

---

## 4. Step-by-Step Pipeline

### Step 1: Data Loading
Implemented in `preprocessing.py`.
- Reads `Event_traces.csv`
- Validates required columns: `Label`, `Features`

### Step 2: Data Preprocessing
Implemented in `preprocessing.py`.
- Extracts EventIds from `Features` text using regex (`E1`, `E2`, ...)
- Maps labels:
  - `Success`/`Normal` -> `0`
  - `Fail`/`Anomaly` -> `1`
- Encodes EventIds to numeric values via `LabelEncoder`

### Step 3: Sequence Generation
Implemented in `sequence_generator.py`.
- Uses sliding windows of length 5 from each trace
- Each window inherits parent trace label
- Output:
  - `X`: numeric sequences for LSTM
  - `y`: binary labels

### Step 4: Subset Selection and Balancing
Implemented in `train_model.py`.
- Uses `max_records=10000`
- Uses `balanced_subset=True` (exact 5000 success + 5000 fail traces)
- Additional train balancing by oversampling minority windows

### Step 5: Train/Test/Validation Split
Implemented in `train_model.py`.
- Splits data into train/test
- Creates validation set from train
- Ensures test contains anomaly samples

### Step 6: LSTM Model
Implemented in `lstm_model.py`.
Architecture:
- `Embedding`
- `LSTM` (with dropout)
- `Dense` hidden layer
- `Dense(sigmoid)` output

Loss/Optimizer:
- Optimizer: `Adam`
- Loss: `binary_crossentropy`
- Metric: `accuracy`

### Step 7: Model Training
Implemented in `train_model.py`.
- Uses callbacks:
  - `EarlyStopping`
  - `ReduceLROnPlateau`
- Current main config (`main.py`):
  - `epochs=4`
  - `sequence_length=5`
  - `target_precision=0.75`

### Step 8: Threshold Tuning
Implemented in `train_model.py`.
- Uses validation probabilities to choose decision threshold
- If `target_precision` is set, threshold is selected to satisfy that precision target (if possible)

### Step 9: Evaluation
Calculated in `train_model.py`.
- `Accuracy`
- `Loss`
- `Precision`
- `Recall`
- `F1 Score`
- `Confusion Matrix`

### Step 10: Prediction
Implemented in `predict.py`.
- Input: one sequence of 5 events (e.g., `E5,E22,E5,E5,E11`)
- Output:
  - anomaly probability
  - decision threshold
  - final prediction (`True`/`False`)

Manual CLI prediction is in `predict_manual.py`.

---

## 5. How to Run

### 5.1 Install dependencies
```powershell
pip install -r requirements.txt
```

### 5.2 Train and evaluate
```powershell
python main.py
```

### 5.3 Manual prediction
```powershell
python predict_manual.py
```
Enter sequence like:
```text
E5,E22,E5,E5,E11
```

---

## 6. Understanding Metrics

- `Precision`: Out of predicted failures, how many are truly failures.
- `Recall`: Out of actual failures, how many the model catches.
- `F1`: Balance between precision and recall.
- `Confusion Matrix [[TN, FP], [FN, TP]]`:
  - `TN`: normal predicted normal
  - `FP`: normal predicted failure
  - `FN`: failure predicted normal
  - `TP`: failure predicted failure

If precision is high and recall is lower, model is conservative (fewer false alarms, more missed failures).
If recall is high and precision is lower, model is aggressive (catches more failures, more false alarms).

---

## 7. Artifacts and Visualizations
Generated after training:
- `artifacts/lstm_failure_predictor.keras`
- `artifacts/label_encoder.joblib`
- `artifacts/config.json`
- `artifacts/plots/training_history.png`
- `artifacts/plots/confusion_matrix.png`
- `artifacts/plots/metrics_bar.png`

---

## 8. Example Sequences for Manual Testing
Likely `True` (failure):
- `E5,E5,E22,E7,E11`
- `E22,E5,E5,E5,E25`
- `E22,E5,E5,E5,E18`

Likely `False` (normal):
- `E5,E22,E5,E5,E11`
- `E5,E5,E22,E5,E11`
- `E5,E5,E5,E22,E11`

---

## 9. Conclusion
This project implements an end-to-end LSTM-based failure prediction system for HDFS logs with:
- robust preprocessing
- balanced subset training
- threshold tuning for precision targets
- manual inference support
- saved plots and artifacts for analysis/reporting

The model is configurable to prioritize either precision or recall depending on reliability requirements.

---

## 10. Submission Polish Updates
To support minor-project submission and viva demo, a polished structure and documentation set has been added under `project/`:

- Detailed report-ready sections: `project/REPORT_SECTIONS.md`
- Architecture pipeline diagram: `project/architecture_diagram.md`
- Result interpretation notes: `project/evaluation/results_analysis.md`
- Streamlit demo UI: `project/prediction/demo_streamlit.py`

### Highlighted Contributions
- Threshold optimization instead of fixed 0.5 decision threshold
- Unknown event detection in inference
- Sequence-based log learning with LSTM
- Manual and Streamlit-based anomaly testing interface
- Full ML pipeline with saved model artifacts and evaluation plots
