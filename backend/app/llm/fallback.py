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

    async def chat(self, message: str, experiment_context: Optional[dict] = None) -> str:
        """Answer questions using pattern matching and experiment data."""
        msg = message.lower().strip()
        ctx = experiment_context or {}

        # Target question
        if "target" in msg and ("what" in msg or "which" in msg):
            target = ctx.get("target_column", "not set")
            ptype = ctx.get("problem_type", "not determined")
            return (
                f"Your target column is **{target}** and the detected problem type is **{ptype}**.\n\n"
                f"_Note: Running in Fallback Mode (no LLM API configured)._"
            )

        # Metric explanations
        if any(k in msg for k in ["what is f1", "explain f1", "f1 score"]):
            return (
                "**F1 Score** is the harmonic mean of precision and recall. It provides a balanced measure "
                "that accounts for both false positives and false negatives. F1-weighted averages F1 across "
                "all classes, weighted by their support (number of samples).\n\n"
                "_Note: Running in Fallback Mode._"
            )
        if any(k in msg for k in ["what is accuracy", "explain accuracy"]):
            return (
                "**Accuracy** is the ratio of correct predictions to total predictions. While intuitive, "
                "it can be misleading for imbalanced datasets where a model could achieve high accuracy "
                "by always predicting the majority class.\n\n_Note: Running in Fallback Mode._"
            )
        if any(k in msg for k in ["what is rmse", "explain rmse"]):
            return (
                "**RMSE (Root Mean Squared Error)** measures the average magnitude of prediction errors. "
                "It penalizes larger errors more heavily than MAE due to the squaring operation. "
                "Lower RMSE indicates better model performance.\n\n_Note: Running in Fallback Mode._"
            )
        if any(k in msg for k in ["what is r2", "explain r2", "r-squared", "r squared"]):
            return (
                "**R² (R-squared)** measures how well the model explains the variance in the target variable. "
                "A value of 1.0 means perfect prediction, 0.0 means the model is no better than predicting "
                "the mean, and negative values indicate worse-than-mean predictions.\n\n"
                "_Note: Running in Fallback Mode._"
            )

        # Preprocessing
        if "preprocessing" in msg or "preprocess" in msg:
            preproc = ctx.get("preprocessing_summary", "No preprocessing information available.")
            if isinstance(preproc, dict):
                parts = []
                for k, v in preproc.items():
                    parts.append(f"- **{k}**: {v}")
                preproc = "\n".join(parts)
            return f"**Preprocessing Applied:**\n{preproc}\n\n_Note: Running in Fallback Mode._"

        # Feature importance
        if "feature" in msg and ("important" in msg or "importance" in msg):
            top = ctx.get("top_features", [])
            if top:
                feat_list = "\n".join(f"{i+1}. {f}" for i, f in enumerate(top[:10]))
                return f"**Top Important Features:**\n{feat_list}\n\n_Note: Running in Fallback Mode._"
            return "Feature importance data is not yet available. Run training first.\n\n_Note: Running in Fallback Mode._"

        # Best model
        if "best" in msg and ("model" in msg or "winner" in msg):
            best = ctx.get("best_model_name", "not yet determined")
            score = ctx.get("best_score")
            metric = ctx.get("primary_metric", "")
            text = f"The best model is **{best}**"
            if score is not None:
                text += f" with a {metric} score of **{score:.4f}**"
            text += "."
            return f"{text}\n\n_Note: Running in Fallback Mode._"

        # Confusion matrix
        if "confusion" in msg and "matrix" in msg:
            return (
                "A **confusion matrix** shows how predictions compare to actual values for each class. "
                "Rows represent actual classes, columns represent predicted classes. The diagonal shows "
                "correct predictions, while off-diagonal elements show misclassifications. This helps "
                "identify which classes the model confuses most often.\n\n_Note: Running in Fallback Mode._"
            )

        # Hyperparameters
        if "hyperparameter" in msg or "hyperparam" in msg:
            return (
                "**Hyperparameter optimization** was performed using Optuna, which intelligently "
                "searches the parameter space to find the configuration that maximizes model performance. "
                "Parameters like learning rate, tree depth, and regularization strength were tuned "
                "using cross-validation.\n\n_Note: Running in Fallback Mode._"
            )

        # Why specific model
        if "why" in msg and any(m in msg for m in ["random forest", "logistic", "gradient", "ridge", "lasso", "svm", "knn"]):
            return (
                "Models are selected based on dataset characteristics including size, number of features, "
                "feature types, and problem complexity. Tree-based models (Random Forest, Gradient Boosting) "
                "often perform well on tabular data due to their ability to capture non-linear relationships "
                "and handle mixed feature types.\n\n_Note: Running in Fallback Mode._"
            )

        # Default response
        return (
            "I'm the AutoML-Lens assistant. I can help you understand your experiment results. "
            "Try asking about:\n"
            "- Your target column and problem type\n"
            "- What specific metrics mean (F1, accuracy, RMSE, R²)\n"
            "- What preprocessing was applied\n"
            "- Which features are most important\n"
            "- Which model performed best and why\n"
            "- What a confusion matrix shows\n"
            "- How hyperparameter optimization works\n\n"
            "_Note: Running in Fallback Mode (no LLM API configured). "
            "Configure a Gemini or OpenAI API key for richer responses._"
        )
