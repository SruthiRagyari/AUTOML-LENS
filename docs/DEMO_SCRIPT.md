# AutoML-Lens: Live Demonstration Script (7–10 Minutes)

This runbook guides a seamless, professional live demonstration of **AutoML-Lens** for examiners, project evaluators, or conference attendees.

---

## Pre-Demo Checklist

1. **Verify Backend**:
   ```bash
   cd backend
   python -m uvicorn app.main:app --reload --port 8000
   ```
   Confirm [http://localhost:8000/api/health](http://localhost:8000/api/health) returns `status: "healthy"`.

2. **Verify Frontend**:
   ```bash
   cd frontend
   npm run dev
   ```
   Open [http://localhost:5173](http://localhost:5173) in your browser.

3. **Demo Datasets Available**:
   - `demo_data/classification.csv` (Customer Churn, 300 rows x 12 cols).
   - Alternatively: `benchmarks/data/wine+quality/winequality-red.csv` (1,599 rows x 12 cols).

---

## Step-by-Step Live Presentation Guide

### Step 1: Start & System Overview (0:00 – 0:45)
- **What to Click**: Open browser to `http://localhost:5173`.
- **What to Show**: Dashboard displaying existing experiments, metrics cards, and clean UI navigation.
- **What to Say**:
  > *"AutoML-Lens is a structured, LLM-guided AutoML system for tabular data. We're looking at the main dashboard showing experiment history. Notice that all metrics shown here represent real cross-validation scores and post-selection holdout evaluations."*
- **Expected Result**: Dashboard renders experiment table and action buttons cleanly.

---

### Step 2: Upload Real Dataset (0:45 – 1:30)
- **What to Click**: Click the **"+ New Experiment"** button in the header. Drag and drop `demo_data/classification.csv` into the dropzone.
- **What to Show**: The instant upload progress, row/column count preview, and automatic target column suggestion (`Churn`).
- **What to Say**:
  > *"When we drop a dataset, the backend immediately analyzes the file headers, infers data types, detects delimiters, and automatically identifies the most probable target column—in this case, 'Churn'. Notice the dataset readiness badge confirming it is suitable for automated modeling."*
- **Expected Result**: File card displays `300 rows | 12 columns | Target: Churn`.

---

### Step 3: Dataset Profiling (1:30 – 2:30)
- **What to Click**: Click **"Create Experiment"**, which opens the 7-step Experiment Workspace on Step 1: **Dataset**. Click **"Profile Dataset"**.
- **What to Show**: The interactive data profiling tables showing column data types, missing value percentages, unique value counts, and numeric distributions.
- **What to Say**:
  > *"The DatasetProfiler calculates exact summary statistics across all features. Crucially, the 20% holdout split has already been separated by our deterministic splitter. The statistics shown here and passed to downstream components come strictly from the training partition, eliminating data leakage."*
- **Expected Result**: Step 1 displays complete profiling metrics without errors.

---

### Step 4: LLM Pipeline Planning (2:30 – 3:30)
- **What to Click**: Click **"Next: AI Analysis"** (Step 2) -> Click **"Run AI Analysis"**.
- **What to Show**: The generated **AutoML Pipeline Plan** panel displaying the recommended problem type, primary metric (`f1_weighted`), candidate models, preprocessing strategy, and technical reasoning.
- **What to Say**:
  > *"Here is our core contribution: the LLM functions as a structured decision layer. It reads the dataset profile and outputs a declarative JSON specification—not raw Python code. Notice that it selected f1_weighted to guard against class imbalance and proposed four complementary algorithms."*
- **Expected Result**: Pipeline Plan card populates with validated plan parameters.

---

### Step 5: Pipeline Validation & Safety Constraints (3:30 – 4:15)
- **What to Click**: Scroll down to the **"Validation & Registry Diagnostics"** section in Step 2.
- **What to Show**: Validation badge showing `Status: Valid (Registry-Constrained)` and diagnostic chips showing approved models and feature operations.
- **What to Say**:
  > *"Before any code runs, the PipelinePlanValidator inspects the plan against our ModelRegistry and OperationRegistry. If the LLM proposes an unvetted library or attempts to transform the target column, the validator rejects the invalid operation and logs a warning. This completely eliminates arbitrary code execution risks."*
- **Expected Result**: Diagnostic card shows all proposed candidate models match system registries.

---

### Step 6: Model Training & Real-Time Progress (4:15 – 5:30)
- **What to Click**: Click **"Next: Models"** -> Click **"Fast Demo"** (trains bounded candidates for fast demonstration).
- **What to Show**: Live progress stepper streaming stages: *Initializing -> Preprocessing -> Training Candidates -> Optuna Tuning -> Model Fusion -> Evaluating Holdout*.
- **What to Say**:
  > *"We now trigger training. The backend uses K-fold cross-validation with fold-safe preprocessing. Notice the live progress updates: imputers and scalers are being fit inside each fold. The 20% holdout split remains completely locked in storage."*
- **Expected Result**: Progress bar completes and automatically transitions to Step 6: **Results**.

---

### Step 7: Model Comparison & Selection (5:30 – 6:30)
- **What to Click**: On Step 6: **Results**, view the **Leaderboard Table** and **Metric Comparison Chart**.
- **What to Show**: Candidate models ranked by cross-validation score, displaying CV Mean, CV Std, Holdout Score, Training Time, and Best Hyperparameters.
- **What to Say**:
  > *"Here is the empirical leaderboard. The algorithms are ranked strictly by cross-validation performance on the training split. Only after the ranking is frozen does the system evaluate the 20% holdout partition, giving us an honest, unbiased measure of generalization."*
- **Expected Result**: Leaderboard clearly shows the winning estimator with separate CV and holdout columns.

---

### Step 8: Model Fusion & Ensemble Optimization (6:30 – 7:15)
- **What to Click**: Scroll to the **"Model Fusion / Ensemble"** section on the Results page.
- **What to Show**: The ensemble candidates card displaying greedy weighted average and stacking weights.
- **What to Say**:
  > *"AutoML-Lens generates out-of-fold predictions and uses Nelder-Mead optimization to find optimal model weights. Crucially, the ensemble is selected as the overall winner only if its cross-validation score strictly beats the best individual model."*
- **Expected Result**: Ensemble weights table displays participating candidate models and their relative percentages.

---

### Step 9: Model Explainability with SHAP (7:15 – 8:00)
- **What to Click**: Click **"Next: Explainability"** (Step 7).
- **What to Show**: The SHAP feature importance plot and natural language explanation summary.
- **What to Say**:
  > *"Machine learning models shouldn't be black boxes. AutoML-Lens automatically runs TreeSHAP on the winning model to compute exact Shapley values. We can clearly see which features—such as contract type or total charges—contributed most heavily to churn risk."*
- **Expected Result**: Feature importance bar chart and summary paragraph render cleanly.

---

### Step 10: Real-Time & Batch Prediction (8:00 – 8:45)
- **What to Click**: Navigate to the **"Prediction"** tab in the main navigation. Select our active experiment from the dropdown.
- **What to Show**: The pre-filled feature form. Change a numeric value and click **"Generate Prediction"**.
- **What to Say**:
  > *"The winning model is immediately deployable. The UI automatically renders an input schema matching the dataset's features. When we submit, the backend applies the exact fitted transformation pipeline and outputs the predicted class along with confidence probabilities."*
- **Expected Result**: Prediction card displays output class (e.g., `Churn: No (78% confidence)`).

---

### Step 11: Scientific Provenance (8:45 – 9:15)
- **What to Click**: Return to the Results page and highlight the **"Scientific Reproducibility & Provenance"** card.
- **What to Show**: Master seed, SHA-256 dataset hash, and Git commit revision.
- **What to Say**:
  > *"For academic and audit compliance, every experiment captures its full cryptographic provenance: the raw dataset's SHA-256 hash, the exact Git commit SHA, the random seed, and the metric contract. Any researcher can reproduce these exact results."*
- **Expected Result**: Provenance panel shows verified hash and git revision.

---

### Step 12: Multi-Seed Research Benchmarks (9:15 – 10:00)
- **What to Click**: Scroll to the **"Multi-Seed Empirical Benchmark Evaluation"** section -> Click **"Open Research Report (HTML)"**.
- **What to Show**: The standalone interactive HTML research report opening in a new tab, displaying the 18 multi-seed runs across Adult Census and Wine Quality.
- **What to Say**:
  > *"Finally, we evaluated our system across 18 benchmark runs on UCI Adult Census and Wine Quality Red over three seeds. On Adult Census, LLM-guided planning improved holdout F1 by +0.0072. On Wine Quality, the deterministic baseline had a slight 0.0012 RMSE advantage. We report both positive and negative results honestly, providing a sound, reproducible contribution to AutoML research. Thank you!"*
- **Expected Result**: Standalone HTML report opens and displays complete empirical tables and charts.
