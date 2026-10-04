# AutoML-Lens: Final Project Presentation Outline

**Target Audience**: Academic Examiners, Viva Committee, Industry ML Practitioners.  
**Format**: 14 Slides (~12–15 Minutes).

---

### Slide 1: Title & Project Overview
- **Title**: AutoML-Lens: Structured LLM-Guided AutoML with Registry-Constrained Pipeline Planning
- **Subtitle**: Safe, Reproducible Automated Machine Learning for Tabular Data
- **Presenter**: Engineering Research Team
- **Key Visual**: High-level visual banner showing React UI, FastAPI logo, and pipeline flow.
- **Presenter Speech**:
  > "Good morning, respected examiners. Today we present AutoML-Lens, an automated machine learning framework that introduces structured Large Language Model planning into tabular data pipelines while enforcing strict safety air-gaps, fold-safe execution, and zero data leakage."

---

### Slide 2: The Problem: Limitations of Traditional AutoML
- **Brute-Force Inefficiency**: Exhaustive search over high-dimensional hyperparameter spaces requires prohibitive computational budgets.
- **Rigid Heuristics**: Fixed rule sets (e.g., if rows > 10,000, use HistGradientBoosting) fail to reason about feature semantics or domain context.
- **Trial & Error Bottlenecks**: Engineers spend hours configuring standard pipelines for repetitive tabular tasks.
- **Key Visual**: Diagram contrasting a massive, blind combinatorial grid search against a focused, semantically guided search.
- **Presenter Speech**:
  > "Traditional AutoML frameworks like Auto-sklearn and TPOT are mathematically principled but semantically blind. They treat column names as opaque indices and cannot infer that 'Experience' relates to 'Salary', leading to massive compute waste traversing implausible configurations."

---

### Slide 3: Motivation: Why LLMs, and Why NOT Code Generation?
- **Semantic Reasoning**: LLMs understand column semantics, target distributions, and standard ML practices.
- **The Code Generation Trap**: Prompting LLMs to generate Python scripts causes:
  - Arbitrary code execution security risks.
  - Critical data leakage (fitting scalers before CV splitting).
  - High failure rates due to syntax errors and hallucinated APIs.
- **The AutoML-Lens Solution**: Decouple **intelligent planning** from **deterministic execution**.
- **Key Visual**: Split graphic: Left = "Unsafe Code Generation (Hallucinations, Leakage, Vulnerabilities)", Right = "AutoML-Lens Declarative Plan (Safe, Schema-Enforced, Deterministic)".
- **Presenter Speech**:
  > "While LLMs possess rich ML knowledge, directly asking an LLM to write Python code is dangerous. Studies show up to 40% of generated data science code suffers from subtle leakage or syntax crashes. Our core innovation is using the LLM strictly as a declarative JSON planner."

---

### Slide 4: Proposed Solution: AutoML-Lens
- **Full-Stack Architecture**: React 18 frontend + FastAPI backend.
- **Structured Pipeline Plan**: Declarative JSON schema output by LLM (Gemini, OpenAI, or Fallback).
- **Registry-Constrained Safety**: Every model and transform must exist in vetted system registries.
- **Fold-Safe Execution**: Transformers fit strictly inside CV folds.
- **Post-Selection Holdout**: 20% holdout split is isolated until final evaluation.
- **Key Visual**: Hero screenshot of the AutoML-Lens web application showing the 7-step pipeline.
- **Presenter Speech**:
  > "AutoML-Lens solves this by treating the LLM as an intelligent advisor whose suggestions are validated against a strict system registry. It executes pipelines with rigorous cross-validation, Bayesian optimization, and model fusion ensembles."

---

### Slide 5: System Architecture & Workflow
- **Data Ingestion**: Multi-format ingestion (CSV/XLSX) with automatic delimiter detection.
- **Profiling**: In-depth statistics (types, cardinality, missingness) computed on training split only.
- **Planning & Validation**: LLM proposes plan -> `PipelinePlanValidator` enforces constraints.
- **Engine Execution**: K-Fold CV -> Preprocessing -> Optuna HPO -> OOF Stacking -> Selection.
- **Serving**: SHAP explanations, real-time inference, and self-contained research reports.
- **Key Visual**: System architecture diagram from `docs/ARCHITECTURE_DIAGRAM.md`.
- **Presenter Speech**:
  > "Here is our complete architecture. Notice the red box representing the holdout split: it is detached at the very beginning and remains completely invisible to the LLM, the preprocessor, and the cross-validation engine until the final pipeline is selected."

---

### Slide 6: Structured LLM Pipeline Planning
- **Context Injection**: Serialized training profile passed to LLM (zero holdout rows exposed).
- **AutoMLPipelinePlan Schema**:
  - Inferred task & primary optimization metric.
  - Preprocessing strategy (imputation, scaling, encoding).
  - Recommended candidate algorithm set.
  - Declarative feature operations.
  - Hyperparameter tuning focus and trial budgets.
- **Deterministic Fallback**: Instant local planning without external API keys or network latency.
- **Key Visual**: Code snippet showing JSON representation of an `AutoMLPipelinePlan`.
- **Presenter Speech**:
  > "The LLM receives statistical context—not raw rows—and returns a typed Pydantic schema. If an API key is missing or the provider fails, the system instantly engages a deterministic fallback planner without interrupting the user workflow."

---

### Slide 7: The Safety Layer: PipelinePlanValidator
- **Zero Arbitrary Code**: Schema validation only; no Python `exec()` or `eval()`.
- **Registry Confinement**: Models checked against `ModelRegistry` (17 vetted estimators).
- **Leakage Prevention**: Prohibits any feature transform targeting the dependent variable.
- **Task Verification**: Validates metric-task alignment (e.g., rejects RMSE on classification).
- **Autonomous Repair**: Silently corrects malformed parameters using deterministic defaults.
- **Key Visual**: Flowchart showing invalid LLM recommendations being filtered/repaired into an executable plan.
- **Presenter Speech**:
  > "The PipelinePlanValidator acts as a strict air-gap. If the LLM hallucinates an unsupported library like 'XGBoostUltra' or tries to transform the target column, the validator strips the invalid instruction and logs a warning."

---

### Slide 8: Fold-Safe Preprocessing & Data Isolation
- **The Data Snooping Threat**: Standard pipelines often fit scalers across the entire dataset.
- **Fold-Safe Execution**:
  - Transformers are instantiated independently for each cross-validation fold.
  - Means, medians, and encoders fit strictly on $K-1$ training folds.
  - Validation fold is transformed using strictly frozen parameters.
- **100% Holdout Isolation**: Evaluated strictly once after model selection is finalized.
- **Key Visual**: Diagram illustrating K-fold data partitioning with per-fold preprocessing pipelines.
- **Presenter Speech**:
  > "We enforce scientific rigor through fold-safe preprocessing. Preprocessing pipelines are instantiated inside each fold. Imputation statistics and scalers never see validation rows, eliminating subtle leakage."

---

### Slide 9: Hyperparameter Optimization & Model Fusion
- **Optuna Bayesian Tuning**: Tree-structured Parzen Estimators (TPE) search bounded hyperparameter spaces.
- **Median Trial Pruning**: Terminates unpromising parameter sets early to save compute.
- **Out-of-Fold (OOF) Ensembling**:
  - Collects unbiased cross-validation predictions.
  - Optimizes greedy model weights via Nelder-Mead coordinate search.
  - Trains regularized stacking meta-learners.
- **Honest Selection**: Ensemble is selected only if it strictly outperforms the best single model on CV.
- **Key Visual**: Optuna trial convergence plot alongside model fusion weight distribution chart.
- **Presenter Speech**:
  > "We combine candidate models using both Bayesian hyperparameter search via Optuna and out-of-fold model fusion. An ensemble is only promoted to winner if it genuinely beats the best individual model on cross-validation."

---

### Slide 10: Model Explainability & Inference
- **Integrated SHAP**:
  - TreeSHAP for tree ensembles (Random Forest, Gradient Boosting).
  - KernelSHAP for linear models and kernel estimators.
- **Interpretability Visuals**: Summary beeswarm plots, mean $|\text{SHAP}|$ feature importance rankings.
- **Production Inference**:
  - Interactive single-record prediction with probability gauge.
  - High-throughput batch CSV scoring.
- **Key Visual**: Screenshot of the Explainability and Prediction tabs in the application UI.
- **Presenter Speech**:
  > "Once trained, the winning model is immediately explainable via SHAP, showing which features drove predictions. Users can perform interactive what-if predictions directly in the browser or upload CSVs for batch inference."

---

### Slide 11: Experimental Benchmark Setup
- **Rigorous Multi-Seed Protocol**:
  - Seed list: `[42, 123, 456]` ($N=3$ seeds per condition, 18 total experimental runs).
  - Benchmarks: UCI Adult Census Income (Classification) & UCI Wine Quality Red (Regression).
  - Identical 3-fold CV and Optuna trial budgets.
- **Three Evaluated Conditions**:
  1. Condition A: Deterministic Baseline.
  2. Condition B: LLM Model-Only (Ablation).
  3. Condition C: Full LLM-Guided Pipeline Planning.
- **Key Visual**: Summary table outlining dataset characteristics and experimental parameters.
- **Presenter Speech**:
  > "To evaluate whether LLM planning adds measurable value, we ran an empirical multi-seed benchmark across 18 runs under identical random seeds, fold splits, and trial budgets."

---

### Slide 12: Empirical Results & Key Findings
- **Adult Census Classification (`f1_weighted` $\uparrow$)**:
  - Deterministic Baseline: $0.8148 \pm 0.0187$
  - Full LLM-Guided: $0.8220 \pm 0.0101$
  - **$\Delta = +0.0072$** (LLM improvement driven by `svm_clf` on seed 456).
- **Wine Quality Regression (`rmse` $\downarrow$)**:
  - Deterministic Baseline: $0.5544 \pm 0.0307$
  - Full LLM-Guided: $0.5556 \pm 0.0298$
  - **$\Delta = -0.0012$** (Deterministic baseline advantage driven by `knn_reg` in ensemble).
- **Key Visual**: Bar charts comparing Deterministic vs LLM holdout scores across both datasets.
- **Presenter Speech**:
  > "Here are our measured results: On Adult Census, LLM planning improved mean F1 by +0.0072 because it smartly included an SVM classifier. On Wine Quality, the deterministic baseline had a slight 0.0012 RMSE advantage because its default heuristic included KNN. This proves LLM guidance is an adaptive heuristic, not an infallible magic bullet."

---

### Slide 13: Ablation Analysis & Scientific Integrity
- **The Ablation Question**: Does gain come from model selection or feature operations?
- **Finding**: Conditions B (Model-Only) and C (Full LLM) achieved **identical holdout means** ($0.8220$ on Adult, $0.5556$ on Wine).
- **Conclusion**: Under realistic budgets, **candidate model space selection is the primary driver of performance**, while tree models absorb simple feature transforms.
- **Scientific Honesty**: We do NOT claim universal superiority or AGI; we provide empirical proof of safe, registry-constrained planning.
- **Key Visual**: Visual comparison showing identical performance bars between Model-Only and Full LLM conditions.
- **Presenter Speech**:
  > "Our ablation study revealed that model selection was the primary driver of performance variation. This demonstrates the importance of empirical honesty: we present measured reality rather than fabricated claims of state-of-the-art superiority."

---

### Slide 14: Conclusion & Future Work
- **Summary of Contributions**:
  1. Decoupled, safe LLM pipeline planning without code generation hazards.
  2. Registry-constrained validation air-gap.
  3. Strict fold-safe preprocessing and untouched holdout discipline.
  4. Full-stack open-source platform with 285 passing tests and production frontend.
- **Future Directions**: Multi-table relational AutoML, GPU-accelerated gradient boosting backends, multi-objective Pareto optimization.
- **Questions & Discussion**: Thank you! We welcome your questions.
- **Key Visual**: Concluding slide with GitHub repository link, test badge (285 passed), and open Q&A prompt.
- **Presenter Speech**:
  > "AutoML-Lens bridges the gap between generative AI and empirical machine learning engineering. Thank you for your time, and we look forward to your questions."
