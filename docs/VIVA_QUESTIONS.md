# AutoML-Lens: Comprehensive Viva Examination Question Bank (50+ Questions & Answers)

This document provides technical, rigorous answers to questions anticipated during thesis defense, viva voce examinations, and technical presentations.

---

## 1. Basic Project Questions

### Q1. What is AutoML-Lens in simple terms?
**Answer**: AutoML-Lens is an end-to-end Automated Machine Learning framework for tabular data. It takes a raw CSV or Excel dataset, automatically profiles its properties, uses an LLM (or a local rule-based fallback) to design a structured pipeline plan, validates and executes that plan safely using cross-validation and Optuna hyperparameter optimization, constructs model fusion ensembles, and delivers explainable predictions with an interactive web UI.

### Q2. What was your personal contribution to this project?
**Answer**: Designed and implemented the complete software and ML architecture: the decoupled LLM pipeline planner, the declarative `PipelinePlanValidator`, fold-safe preprocessing routines, the multi-seed benchmarking framework, SHAP explainability integration, FastAPI REST endpoints, and the React 18 interactive frontend.

### Q3. What problem does this system solve that scikit-learn or standard libraries do not?
**Answer**: Standard scikit-learn provides manual building blocks. An engineer must manually inspect data, choose imputers, write encoding pipelines, select models, write search loops, and assemble stacking ensembles. AutoML-Lens automates this entire lifecycle with semantic intelligence and scientific safety guarantees.

### Q4. What technologies make up the tech stack?
**Answer**: 
- **Backend**: Python 3.11, FastAPI, Pydantic v2, SQLAlchemy, SQLite.
- **ML / AI**: scikit-learn 1.6.1, Optuna 4.2.0, SHAP, NumPy, Pandas.
- **LLM Integrations**: Google Gemini API, OpenAI API, plus a local deterministic rule-based engine.
- **Frontend**: React 18, Vite, Chart.js / canvas plots, custom CSS styling.

---

## 2. AI/ML Questions

### Q5. What supervised learning tasks does AutoML-Lens support?
**Answer**: Tabular binary classification, multiclass classification, and continuous regression.

### Q6. Which algorithms are supported in the Model Registry?
**Answer**: 17 algorithms:
- **Classification (8)**: Logistic Regression, Random Forest, Gradient Boosting, HistGradientBoosting, Support Vector Classifier (SVC), K-Nearest Neighbors, Decision Tree, Gaussian Naive Bayes.
- **Regression (9)**: Ridge, Lasso, ElasticNet, Random Forest, Gradient Boosting, HistGradientBoosting, Support Vector Regressor (SVR), K-Nearest Neighbors, Decision Tree Regressor.

### Q7. How does the system determine whether a task is classification or regression?
**Answer**: The `DatasetProfiler` inspects the target column: if the target is string/object or boolean, or an integer with low unique cardinality ($\le 20$ unique values), it infers classification. If the target is floating-point or integer with high unique cardinality ($>20$), it infers regression. The user can also explicitly override this detection.

### Q8. What is the difference between GradientBoostingClassifier and HistGradientBoostingClassifier?
**Answer**: Standard `GradientBoostingClassifier` evaluates exact split thresholds over continuous features, scaling $O(N \cdot D)$. `HistGradientBoostingClassifier` discretizes continuous variables into 256 integer bins (similar to LightGBM), dramatically reducing training time to $O(N)$ and supporting native missing value handling.

---

## 3. AutoML Questions

### Q9. Why is AutoML needed if data scientists already exist?
**Answer**: AutoML automates mechanical, repetitive tasks: baseline establishment, hyperparameter sweeping, and boilerplate preprocessing. This allows data scientists to focus on problem framing, domain feature engineering, and business impact.

### Q10. How does AutoML-Lens differ from Auto-sklearn or TPOT?
**Answer**: Auto-sklearn uses Bayesian optimization over a massive, fixed configuration space with meta-learning; TPOT uses genetic programming. Both are computationally heavy and treat features as anonymous numbers. AutoML-Lens incorporates LLMs as a semantic planning layer to prune candidate model spaces before execution while strictly guaranteeing safety via registry validation.

### Q11. Is this framework completely autonomous?
**Answer**: It is semi-autonomous. It can run 100% end-to-end automatically with default settings, but also allows domain experts to review, edit, or override the LLM-generated pipeline plan before training starts.

---

## 4. LLM Questions

### Q12. Why use an LLM in an AutoML pipeline?
**Answer**: LLMs possess rich semantic knowledge about relationships between feature names, data types, and typical ML modeling strategies. For example, an LLM recognizes that an income column in a census dataset may be skewed and suggests logarithmic scaling, or that text-like categorical descriptions benefit from frequency encoding.

### Q13. Why NOT simply ask ChatGPT to generate Python code?
**Answer**: Prompting LLMs to generate raw executable code introduces severe hazards:
1. Arbitrary code execution security risks.
2. Silent data leakage (e.g., fitting scalers on test splits).
3. Frequent execution crashes due to syntax errors, hallucinated package versions, or deprecated parameters.
AutoML-Lens restricts the LLM to outputting a declarative JSON specification that our deterministic engine validates and executes safely.

### Q14. What happens if the user has no Gemini or OpenAI API key, or if the LLM fails?
**Answer**: The system immediately engages a deterministic rule-based fallback planner. It generates a valid, reproducible pipeline plan locally without network calls, errors, or delays.

### Q15. Does the LLM see the actual raw data rows?
**Answer**: No. To preserve user privacy, minimize token usage, and prevent data snooping, the LLM is only provided with aggregated statistical profile summaries (column names, inferred dtypes, null percentages, and cardinality) computed strictly from the 80% training partition.

---

## 5. Feature Engineering Questions

### Q16. What feature engineering operations are supported?
**Answer**: Bounded mathematical transformations (`log`, `sqrt`, `square`, `abs`, `reciprocal`), scaling transforms (`zscore`, `minmax`, `robust_scale`), pairwise interactions (product, ratio, difference), and categorical frequency encoding.

### Q17. How do you prevent mathematical errors like division by zero or log of negative numbers?
**Answer**: The `FeatureEngineer` applies defensive bounds: `log` applies `np.log1p(np.maximum(x, 0))`; ratios add an epsilon ($\epsilon = 1e-6$) or clip denominators; invalid values (NaN, inf) are imputed with median statistics.

### Q18. How do you prevent target leakage during feature engineering?
**Answer**: The `PipelinePlanValidator` inspects all proposed feature operations and explicitly rejects any operation that specifies the target column as an input feature or output target.

---

## 6. Hyperparameter Optimization Questions

### Q19. Why use Optuna instead of Grid Search or Random Search?
**Answer**: Grid search scales exponentially with parameter count ($O(k^d)$), while random search samples blindly. Optuna uses Tree-structured Parzen Estimators (TPE), a Bayesian optimization method that constructs a probability model of the objective function to focus exploration on high-performing parameter regions.

### Q20. How does trial pruning work in your system?
**Answer**: Optuna's `MedianPruner` compares the intermediate validation score of an active trial against the median score of previous trials at the same step. If a trial underperforms the historical median, it is terminated early, conserving compute for promising configurations.

### Q21. How do you ensure Optuna is reproducible?
**Answer**: The Optuna study instantiates `TPESampler(seed=master_seed)`. The master experiment seed is propagated to the sampler, trial splits, and model constructors.

---

## 7. Ensemble Questions

### Q22. What ensemble techniques are implemented?
**Answer**: Out-of-fold (OOF) greedy weighted averaging, soft probability voting, uniform averaging, and ridge/logistic stacking.

### Q23. What are Out-of-Fold (OOF) predictions and why are they necessary?
**Answer**: OOF predictions are generated during cross-validation: for each fold, predictions are recorded on the validation fold that the model was not trained on. Combining these creates a full prediction vector over the training set that is unbiased by model overfitting, allowing meta-learners or weights to be tuned without data leakage.

### Q24. Can an ensemble perform worse than a single model? How do you prevent this?
**Answer**: Yes, poorly weighted ensembles or overfitted stacking meta-learners can degrade holdout performance. AutoML-Lens prevents this through strict selection logic: an ensemble is promoted to the overall winner **only if its CV score strictly exceeds the best individual model's CV score**. Otherwise, the best individual model is chosen.

---

## 8. Data Leakage Questions

### Q25. What is data leakage, and why is it dangerous in AutoML?
**Answer**: Data leakage occurs when information from outside the training dataset (such as validation or test splits) is inadvertently used to train a model or fit transformers. It produces falsely optimistic cross-validation scores that collapse in production.

### Q26. How does AutoML-Lens guarantee fold-safe preprocessing?
**Answer**: Rather than preprocessing the whole dataset beforehand, transformers (`StandardScaler`, `SimpleImputer`, `OneHotEncoder`) are fit inside each fold loop on the $K-1$ training folds only. The validation fold is transformed using the frozen parameters learned on the training slice.

### Q27. How is the holdout split protected from leakage?
**Answer**: The 20% holdout split is separated immediately after dataset ingestion using a deterministic random seed. It is completely isolated from the profiler context sent to the LLM, the cross-validation splits, Optuna trials, and ensemble weight optimization. It is evaluated **strictly once** after the final winning pipeline has been selected.

---

## 9. Evaluation Questions

### Q28. Why use Cross-Validation instead of a single train/test split for model selection?
**Answer**: Single validation splits have high variance and can lead to selecting a model that happened to perform well on that specific slice. K-fold cross-validation averages performance across $K$ distinct validation splits, providing an unbiased estimator of generalization error.

### Q29. Which evaluation metrics are supported?
**Answer**: 
- **Classification**: `accuracy`, `balanced_accuracy`, `f1_weighted`, `f1_macro`, `roc_auc`, `precision_weighted`, `recall_weighted`.
- **Regression**: `neg_root_mean_squared_error` (`rmse`), `neg_mean_squared_error` (`mse`), `neg_mean_absolute_error` (`mae`), `r2`.

### Q30. How does the system handle metric directionality?
**Answer**: All internal optimization and ranking logic queries `is_higher_better(metric)`. For metrics where lower is better (RMSE, MAE), scores are sign-adjusted or ranked in ascending order, ensuring that comparison deltas always indicate true improvement.

---

## 10. Research Methodology Questions

### Q31. What experimental conditions were evaluated in your benchmark?
**Answer**: 
1. **Condition A (`deterministic`)**: Standard heuristic AutoML baseline without LLM.
2. **Condition B (`llm_model_only` ablation)**: LLM candidate model selection only; default preprocessing and zero feature operations.
3. **Condition C (`llm_guided`)**: Full LLM pipeline planning (model selection, feature operations, HPO focus, ensemble configuration).

### Q32. What datasets and seeds were used in the benchmark?
**Answer**: Evaluated on UCI Adult Census Income (binary classification) and UCI Wine Quality Red (regression) across 3 reproducible seeds (`[42, 123, 456]`), yielding 18 total experimental benchmark runs.

### Q33. Did the LLM improve performance on Adult Census?
**Answer**: Yes. On Adult Census, LLM-guided planning achieved a mean holdout F1 of $0.8220 \pm 0.0101$ compared to $0.8148 \pm 0.0187$ for the deterministic baseline ($+0.0072$ improvement). On seed 456, the LLM included `svm_clf`, which scored $0.8176$ on holdout, outperforming tree models on that partition.

### Q34. What happened on Wine Quality Red? Why did the deterministic baseline win?
**Answer**: On Wine Quality Red, the deterministic baseline achieved lower holdout RMSE ($0.5544 \pm 0.0307$ vs. $0.5556 \pm 0.0298$). The deterministic rule set included `knn_reg`, which received $0.30$ weight in the winning ensemble on seed 456. The LLM did not suggest KNN. This demonstrates that LLMs are not universally superior and can omit effective classical heuristics.

### Q35. What was the outcome of your ablation study?
**Answer**: Conditions B (Model-Only) and C (Full LLM) yielded identical holdout means on both datasets ($0.8220$ on Adult, $0.5556$ on Wine). This demonstrates that under the evaluated experimental budget, performance gains were driven predominantly by **model candidate space selection** rather than automated feature engineering.

---

## 11. Architecture Questions

### Q36. Why choose FastAPI over Flask or Django?
**Answer**: FastAPI provides native asynchronous request processing (`async/await`), automatic OpenAPI/Swagger documentation, high execution speed (built on Starlette/Uvicorn), and strict request/response data validation via Pydantic v2.

### Q37. How does the frontend track training progress in real time?
**Answer**: The backend runs training asynchronously in worker threads while recording phase milestones and elapsed times in the database. The React frontend polls `GET /api/experiments/{id}` and streams progress updates through a dedicated UI stepper.

### Q38. How is experiment reproducibility ensured?
**Answer**: Every experiment stores a cryptographic SHA-256 hash of the dataset, the exact Git commit SHA, the master random seed, and serialized configuration dictionaries in the SQLite database.

---

## 12. Security & Safety Questions

### Q39. How is arbitrary code execution prevented when using LLMs?
**Answer**: The system never executes code generated by the LLM. There are no calls to `exec()`, `eval()`, or dynamic script runners. The LLM produces JSON data conforming to a strict schema, and our Python backend executes only pre-compiled functions from vetted internal libraries.

### Q40. How do you protect against prompt injection or malicious dataset uploads?
**Answer**: The LLM prompt only receives numerical summaries and sanitized column headers. Uploaded files are validated for MIME type, sanitized, and stored with generated UUID paths outside the web root.

### Q41. Are API keys exposed to the client browser?
**Answer**: No. All API keys (`GEMINI_API_KEY`, `OPENAI_API_KEY`) reside exclusively in server-side environment variables loaded in `backend/.env`. The frontend never accesses or receives these secrets.

---

## 13. Difficult Examiner Questions

### Q42. Is this reinforcement learning (RL)?
**Answer**: No. Reinforcement learning requires Markov decision processes, policy networks, and reward backpropagation. AutoML-Lens employs Bayesian optimization via Optuna and prompt-based LLM planning, not reinforcement learning.

### Q43. Is this Neural Architecture Search (NAS)?
**Answer**: No. NAS searches continuous or discrete layers, connections, and cell topologies of deep neural networks. AutoML-Lens optimizes classical tabular algorithms, hyperparameters, and ensemble combinations.

### Q44. Why did you only evaluate on 2 datasets and 3 seeds? Is that enough for a research claim?
**Answer**: The 2 datasets represent canonical, contrasting tabular challenges (high-dimensional mixed classification vs. pure continuous regression). The 3 seeds ($N=3$, 18 runs) provide initial variance bounds while keeping computational runtime bounded on local commodity hardware. We explicitly state in our documentation and paper that our findings represent **descriptive empirical evidence**, not an asymptotic proof of universal superiority.

### Q45. If the ablation showed feature engineering had zero impact on the mean holdout score, isn't that component useless?
**Answer**: No. The ablation demonstrates that for tree-based models and normalized estimators under an Optuna tuning budget, simple monotonic transforms do not provide additive margin on these specific datasets. However, the feature engineering engine is mathematically proven to be fold-safe, leak-free, and operational for datasets that require explicit non-linear expansions.

### Q46. Could an examiner claim you 'cherry-picked' results?
**Answer**: No, precisely because we openly report that the deterministic baseline beat the LLM on Wine Quality Red, and that feature operations produced identical holdout means in ablation. Cherry-picked work would conceal negative results; our work reports honest empirical measurements.

### Q47. How do you know SHAP explanations are mathematically sound?
**Answer**: We use the official `shap` library developed by Lundberg & Lee. For tree ensembles, TreeSHAP computes exact Shapley values in polynomial time based on cooperative game theory axioms (efficiency, symmetry, dummy, additivity).

### Q48. What happens if an uploaded dataset has severe class imbalance (e.g., 99% vs 1%)?
**Answer**: The system uses stratified splits for both the holdout split and CV folds to preserve class ratios. The profiler flags class imbalance, and the LLM/fallback planner assigns `f1_weighted` or `balanced_accuracy` rather than naive accuracy.

### Q49. Why should an enterprise adopt AutoML-Lens instead of commercial platforms like DataRobot or AWS SageMaker Canvas?
**Answer**: AutoML-Lens is fully open-source, runs entirely on-premise without cloud dependencies, requires zero subscription licensing, supports transparent local fallback without external API costs, and provides full code auditability.

### Q50. What is the single most important takeaway from your research?
**Answer**: That LLMs are valuable in AutoML as **structured, semantic planners rather than code generators**, and that combining declarative LLM planning with registry-constrained execution and fold-safe verification creates a safe, competitive, and scientifically sound AutoML system.
