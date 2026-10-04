"""Deterministic fallback LLM provider - works without any API key."""
from typing import Optional
from app.llm.base import LLMProvider, DatasetAnalysisResult


class FallbackLLMProvider(LLMProvider):
    """Provides deterministic recommendations based on dataset characteristics.
    Always available - requires no API key or internet connection."""

    def get_provider_name(self) -> str:
        return "Fallback (Deterministic)"

    def is_available(self) -> bool:
        return True

    async def plan_pipeline(self, dataset_context: dict):
        """Generate a deterministic rule-based AutoML pipeline plan."""
        from app.services.pipeline_planner import generate_deterministic_plan
        return generate_deterministic_plan(dataset_context)

    async def analyze_dataset(self, dataset_context: dict) -> DatasetAnalysisResult:
        """Analyze dataset using rule-based heuristics."""
        shape = dataset_context.get("shape", [0, 0])
        columns = dataset_context.get("columns", [])
        dtypes = dataset_context.get("dtypes", {})
        missing_pct = dataset_context.get("missing_percentages", {})
        unique_counts = dataset_context.get("unique_counts", {})
        target_col = dataset_context.get("target_column", "")
        target_stats = dataset_context.get("target_stats", {})

        # Determine problem type
        target_dtype = str(dtypes.get(target_col, "object"))
        target_unique = unique_counts.get(target_col, 999)

        if "float" in target_dtype and target_unique > 20:
            problem_type = "regression"
        elif "int" in target_dtype and target_unique > 20:
            problem_type = "regression"
        elif target_dtype == "object" or target_unique <= 20:
            problem_type = "classification"
        else:
            problem_type = "classification"

        # Build preprocessing recommendations
        preprocessing = []
        warnings = []
        n_rows = shape[0] if shape else 0

        for col, pct in missing_pct.items():
            if col == target_col:
                continue
            if pct > 30:
                preprocessing.append({
                    "column": col, "action": "consider_drop",
                    "reason": f"{pct:.1f}% missing values - may be unreliable"
                })
                warnings.append(f"Column '{col}' has {pct:.1f}% missing values.")
            elif pct > 0:
                dtype = str(dtypes.get(col, "object"))
                action = "median_imputation" if "float" in dtype or "int" in dtype else "mode_imputation"
                preprocessing.append({
                    "column": col, "action": action,
                    "reason": f"{pct:.1f}% missing values"
                })

        # Check for high cardinality
        for col in columns:
            if col == target_col:
                continue
            dtype = str(dtypes.get(col, ""))
            uc = unique_counts.get(col, 0)
            if dtype == "object" and uc > 50:
                preprocessing.append({
                    "column": col, "action": "frequency_encoding",
                    "reason": f"High cardinality ({uc} unique values)"
                })
                warnings.append(f"Column '{col}' has high cardinality ({uc} unique values).")
            elif dtype == "object" and n_rows > 0 and uc / max(n_rows, 1) > 0.9:
                warnings.append(f"Column '{col}' appears to be an identifier.")

        # Feature engineering recommendations
        feature_eng = []
        num_cols = [c for c in columns if c != target_col and ("float" in str(dtypes.get(c, "")) or "int" in str(dtypes.get(c, "")))]
        if len(num_cols) >= 2 and len(num_cols) <= 8:
            feature_eng.append({
                "name": "numerical_interactions",
                "description": "Create interaction features between correlated numerical columns",
                "type": "interaction"
            })

        # Select candidate models
        if problem_type == "classification":
            if n_rows < 1000:
                candidate_models = [
                    "logistic_regression", "random_forest_clf",
                    "gradient_boosting_clf", "knn_clf"
                ]
            else:
                candidate_models = [
                    "logistic_regression", "random_forest_clf",
                    "gradient_boosting_clf", "hist_gradient_boosting_clf"
                ]
            recommended_metric = "f1_weighted"
        else:
            if n_rows < 1000:
                candidate_models = [
                    "ridge", "random_forest_reg",
                    "gradient_boosting_reg", "knn_reg"
                ]
            else:
                candidate_models = [
                    "ridge", "random_forest_reg",
                    "gradient_boosting_reg", "hist_gradient_boosting_reg"
                ]
            recommended_metric = "neg_root_mean_squared_error"

        # Build reasoning
        reasoning_parts = [
            f"The dataset contains {shape[0]} rows and {shape[1]} columns.",
            f"The target column '{target_col}' has {target_unique} unique values "
            f"and dtype '{target_dtype}', indicating a {problem_type} task.",
        ]
        n_missing = sum(1 for v in missing_pct.values() if v > 0)
        if n_missing > 0:
            reasoning_parts.append(f"{n_missing} columns have missing values requiring imputation.")
        n_cat = sum(1 for c in columns if str(dtypes.get(c, "")) == "object" and c != target_col)
        if n_cat > 0:
            reasoning_parts.append(f"{n_cat} categorical columns will be encoded.")
        reasoning_parts.append(
            f"Recommended models are selected for {'small' if n_rows < 1000 else 'medium-to-large'} "
            f"dataset size with {problem_type} objective."
        )
        reasoning = " ".join(reasoning_parts)

        # Rule-based understanding for the AI panel; no model was consulted.
        problem_understanding = (
            f"Rule-based analysis: {shape[0]} rows x {shape[1]} columns; "
            f"'{target_col}' marks a {problem_type} task with {target_unique} "
            f"unique values. No LLM was consulted for this assessment."
        )
        # Feature-engineering stays deterministic: no operation is proposed
        # through the registry, the FeatureEngineer's own heuristics run instead.
        raw = {
            "problem_type": problem_type,
            "target_column": target_col,
            "reasoning": reasoning,
            "problem_understanding": problem_understanding,
            "preprocessing": preprocessing,
            "feature_engineering": feature_eng,
            "suggested_operations": [],
            "useful_feature_candidates": [c for c in num_cols if c != target_col][:8],
            "potentially_irrelevant_columns": [
                col for col, pct in missing_pct.items()
                if col != target_col and pct > 50
            ],
            "leakage_warnings": [],
            "modelling_considerations": [
                f"Dataset has {n_rows} rows "
                f"({'small' if n_rows < 1000 else 'moderate-to-large'}); "
                "favour regularised models if signal is weak."
            ],
            # Structured model recommendations produced by the same
            # deterministic heuristics - no LLM was consulted for these picks.
            "model_recommendations": [
                {
                    "model_id": m,
                    "reason": (
                        f"Rule-based pick for a {problem_type} task: "
                        f"{shape[0]} rows, {shape[1]} columns."
                    ),
                    "suitability": "deterministic heuristic, registry-validated",
                    "strengths": "safe fast default confirmed in the registry",
                    "limitations": "not tailored by a language model",
                }
                for m in candidate_models
            ],
            "model_selection_source": "deterministic_defaults",
            "candidate_models": candidate_models,
            "recommended_metric": recommended_metric,
            "optimization_strategy": "optuna",
            "warnings": warnings,
            # Rule-based heuristics: there is no model confidence to report.
            "confidence": None,
            "operation_source": "deterministic_defaults",
            "provider_used": self.get_provider_name(),
            "is_fallback": True,
        }
        return self.validate_analysis(raw, dataset_context)

    async def explain_results(self, results_context: dict) -> str:
        """Generate deterministic explanation of experiment results."""
        best_model = results_context.get("best_model", "Unknown")
        best_display = results_context.get("best_display_name", best_model)
        metric_name = results_context.get("metric_name", "score")
        best_score = results_context.get("best_score")
        problem_type = results_context.get("problem_type", "task")
        all_results = results_context.get("all_results", [])
        best_params = results_context.get("best_params", {})
        n_trials = results_context.get("n_trials", 0)

        parts = []
        parts.append(f"## Experiment Results Summary\n")
        parts.append(
            f"The best performing model for this {problem_type} task was "
            f"**{best_display}**, achieving a {metric_name} score of "
            f"**{best_score:.4f}**." if best_score is not None
            else f"The best performing model was **{best_display}**."
        )

        if len(all_results) > 1:
            sorted_results = sorted(
                [r for r in all_results if r.get("score") is not None],
                key=lambda x: x["score"], reverse=True
            )
            if len(sorted_results) >= 2:
                runner_up = sorted_results[1]
                parts.append(
                    f"\nThe runner-up was {runner_up.get('display_name', runner_up.get('model_name', 'N/A'))} "
                    f"with a score of {runner_up['score']:.4f}."
                )

        if best_params:
            params_str = ", ".join(f"{k}={v}" for k, v in best_params.items())
            parts.append(f"\nOptimal hyperparameters: {params_str}.")

        if n_trials > 0:
            parts.append(f"\nOptuna explored {n_trials} hyperparameter configurations to find the best setup.")

        parts.append(
            "\n\n*Note: This explanation was generated in Fallback Mode (no LLM API configured). "
            "For richer insights, configure a Gemini or OpenAI API key.*"
        )
        return "\n".join(parts)

    async def chat(self, message: str, experiment_context: Optional[dict] = None,
                   history: Optional[list] = None, context_mode: str = "general", **kwargs) -> str:
        """Answer questions using pattern matching and experiment data with honest fallback notices."""
        msg = message.lower().strip()
        ctx = experiment_context or {}

        # 1. Project / AutoML queries
        if context_mode == "project" or any(k in msg for k in [
            "target", "metric", "f1", "accuracy", "rmse", "r2", "r-squared",
            "model", "best model", "winner", "feature", "preprocessing",
            "confusion matrix", "hyperparameter", "optuna", "experiment", "automl"
        ]):
            if "target" in msg and ("what" in msg or "which" in msg or "name" in msg):
                target = ctx.get("target_column", "not set")
                ptype = ctx.get("problem_type", "not determined")
                return (
                    f"### Target Column & Problem Type\n\n"
                    f"- **Target Column**: `{target}`\n"
                    f"- **Problem Type**: `{ptype}`\n\n"
                    f"*(Answered via Deterministic Rule Engine. For general AI queries, configure a Gemini or OpenAI API key.)*"
                )

            if any(k in msg for k in ["best", "winner", "winning model"]) and ("model" in msg or "winner" in msg):
                best = ctx.get("best_model_name", "Pending training completion")
                score = ctx.get("best_score")
                metric = ctx.get("primary_metric", "primary metric")
                score_str = f"{score:.4f}" if isinstance(score, (int, float)) else "N/A"
                return (
                    f"### Winning Pipeline Overview\n\n"
                    f"- **Best Model**: `{best}`\n"
                    f"- **Validation Score ({metric})**: `{score_str}`\n\n"
                    f"*(Answered via Deterministic Rule Engine.)*"
                )

            if any(k in msg for k in ["what is f1", "explain f1", "f1 score"]):
                return (
                    "### F1-Score (Harmonic Mean of Precision & Recall)\n\n"
                    "The **F1-Score** balances precision and recall:\n\n"
                    "$$\\text{F1} = 2 \\times \\frac{\\text{Precision} \\times \\text{Recall}}{\\text{Precision} + \\text{Recall}}$$\n\n"
                    "- **Precision**: What proportion of positive identifications was actually correct?\n"
                    "- **Recall**: What proportion of actual positives was identified correctly?\n"
                    "- **F1-Weighted**: Averages individual class F1 scores weighted by the number of true instances in each class."
                )

            if any(k in msg for k in ["what is accuracy", "explain accuracy"]):
                return (
                    "### Classification Accuracy\n\n"
                    "**Accuracy** is the ratio of correct predictions to total predictions:\n\n"
                    "$$\\text{Accuracy} = \\frac{\\text{True Positives} + \\text{True Negatives}}{\\text{Total Samples}}$$\n\n"
                    "> **Caution**: In imbalanced datasets (e.g., 95% negative, 5% positive), a naive model predicting all negatives achieves 95% accuracy while remaining useless. For this reason, AutoML-Lens defaults to **F1-Weighted** or **Balanced Accuracy** for imbalanced classification."
                )

            if any(k in msg for k in ["what is rmse", "explain rmse"]):
                return (
                    "### Root Mean Squared Error (RMSE)\n\n"
                    "**RMSE** measures the standard deviation of prediction residuals in continuous regression:\n\n"
                    "$$\\text{RMSE} = \\sqrt{\\frac{1}{n} \\sum_{i=1}^n (y_i - \\hat{y}_i)^2}$$\n\n"
                    "- Because errors are squared before averaging, RMSE penalizes large outlier errors more heavily than MAE (Mean Absolute Error).\n"
                    "- Lower values indicate better fit, with 0 representing perfect prediction."
                )

            if any(k in msg for k in ["what is r2", "explain r2", "r-squared", "r squared"]):
                return (
                    "### Coefficient of Determination (R² Score)\n\n"
                    "**R²** measures the proportion of variance in the dependent variable explained by the model:\n\n"
                    "- **$R^2 = 1.0$**: Perfect prediction.\n"
                    "- **$R^2 = 0.0$**: Model performs identically to constantly predicting the empirical mean.\n"
                    "- **$R^2 < 0.0$**: Model performs worse than the mean baseline (severe overfitting or model mismatch)."
                )

            if "feature" in msg and ("important" in msg or "importance" in msg):
                top = ctx.get("top_features", [])
                if top:
                    feat_list = "\n".join(f"{i+1}. `{f}`" for i, f in enumerate(top[:10]))
                    return f"### Top Informative Features (SHAP Importance)\n\n{feat_list}\n\n*(Computed via TreeSHAP/KernelSHAP)*"
                return "Feature importance data is not yet computed. Complete training to view SHAP rankings."

            if "preprocessing" in msg or "preprocess" in msg:
                preproc = ctx.get("preprocessing_summary", "No preprocessing information recorded yet.")
                if isinstance(preproc, dict):
                    parts = [f"- **{k}**: `{v}`" for k, v in preproc.items()]
                    preproc = "\n".join(parts)
                return f"### Fold-Safe Preprocessing Directives\n\n{preproc}"

            if any(k in msg for k in ["what is automl", "explain automl"]):
                return (
                    "### Automated Machine Learning (AutoML)\n\n"
                    "**AutoML** automates the end-to-end lifecycle of machine learning:\n\n"
                    "1. **Data Profiling**: Inferred types, cardinality, missingness audit.\n"
                    "2. **Structured Planning**: Registry-constrained pipeline generation.\n"
                    "3. **Fold-Safe Preprocessing**: Imputers and scalers fit strictly on training folds.\n"
                    "4. **Bayesian HPO**: Optuna Tree-structured Parzen Estimator (TPE) parameter search.\n"
                    "5. **Model Fusion**: Out-of-fold greedy weighted ensemble and stacking.\n"
                    "6. **Explainability**: SHAP value feature importance analysis."
                )

        # 2. General Knowledge / Coding / Conceptual queries (honest fallback + answer if known)
        general_intents = {
            "quantum computing": (
                "### Quantum Computing Overview\n\n"
                "**Quantum computing** is a multidisciplinary field comprising aspects of computer science, physics, and mathematics that utilizes quantum mechanics to solve complex problems faster than classical computers.\n\n"
                "- **Qubits**: Unlike classical bits (0 or 1), quantum bits can exist in a **superposition** of states.\n"
                "- **Entanglement**: Qubits can be linked such that the state of one instantaneously affects another, regardless of distance.\n"
                "- **Applications**: Quantum chemistry, drug discovery, integer factorization (Shor's algorithm), and combinatorial optimization."
            ),
            "reverse a string": (
                "### Python: Reverse a String\n\n"
                "In Python, the most idiomatic and efficient approach is using **slice notation**:\n\n"
                "```python\ndef reverse_string(s: str) -> str:\n    return s[::-1]\n\n# Example:\ntext = 'AutoML-Lens'\nprint(reverse_string(text))  # sneL-LMotuA\n```\n\n"
                "Alternatively, using `reversed()` and `''.join()`:\n"
                "```python\ndef reverse_string_alt(s: str) -> str:\n    return ''.join(reversed(s))\n```"
            ),
            "capital of france": "The capital of France is **Paris**.",
            "difference between ai and ml": (
                "### Difference Between AI and ML\n\n"
                "- **Artificial Intelligence (AI)**: The broad discipline of building computational systems capable of performing tasks that typically require human cognition (reasoning, perception, language understanding, problem solving).\n"
                "- **Machine Learning (ML)**: A primary subset of AI focused on training statistical models on empirical data so they learn patterns and generalize to unseen instances without hardcoded rules."
            ),
            "pasta": (
                "### Classic Italian Pasta Aglio e Olio\n\n"
                "**Ingredients**:\n"
                "- 200g Spaghetti\n"
                "- 4 cloves Garlic (thinly sliced)\n"
                "- 4 tbsp Extra Virgin Olive Oil\n"
                "- 1/2 tsp Red pepper chili flakes\n"
                "- Fresh parsley (chopped) & Salt\n\n"
                "**Instructions**:\n"
                "1. Boil pasta in salted water until 1 minute before al dente.\n"
                "2. In a skillet, warm olive oil over medium-low heat; gently sauté sliced garlic and red pepper flakes until golden (do not burn).\n"
                "3. Transfer pasta directly into the skillet with 1/4 cup starchy pasta cooking water.\n"
                "4. Toss vigorously for 1 minute to emulsify the sauce into a glossy coating.\n"
                "5. Garnish with chopped fresh parsley and serve immediately."
            ),
            "python": (
                "### Python Programming Language\n\n"
                "**Python** is a high-level, interpreted, dynamically typed programming language known for its clean syntax, readability, and extensive scientific computing ecosystem (`numpy`, `pandas`, `scikit-learn`, `fastapi`)."
            ),
        }

        for key, ans in general_intents.items():
            if key in msg:
                return (
                    f"{ans}\n\n"
                    f"---\n"
                    f"*Notice: Provided via Local Fallback Engine. For open-ended general dialogue, configure `GEMINI_API_KEY` in `backend/.env`.*"
                )

        # Standard informative fallback message
        return (
            f"### AutoML-Lens AI Assistant (Fallback Mode)\n\n"
            f"I received your question: *\"{message}\"*\n\n"
            f"The backend is currently operating in **Deterministic Fallback Mode** (no external LLM API key configured). "
            f"In this mode, all AutoML pipeline planning, model training, cross-validation, and Optuna HPO execute locally with 100% functionality.\n\n"
            f"**To enable full general-purpose AI chat (like ChatGPT)**:\n"
            f"1. Open `backend/.env`\n"
            f"2. Set `LLM_PROVIDER=gemini` (or `openai`)\n"
            f"3. Add your `GEMINI_API_KEY` (from [Google AI Studio](https://aistudio.google.com/apikey))\n"
            f"4. Restart the backend server\n\n"
            f"*(For questions about this experiment's metrics, dataset, or models, switch to **Project Mode** above or ask about 'target', 'best model', or 'metrics'.)*"
        )
