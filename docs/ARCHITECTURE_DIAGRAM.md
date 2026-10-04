# AutoML-Lens: System Architecture & Workflow Diagram

This document illustrates the end-to-end technical architecture of **AutoML-Lens**, highlighting the strict separation between training/cross-validation and the untouched holdout test partition.

---

## 1. System Architecture Diagram (Mermaid)

```mermaid
flowchart TD
    classDef userLayer fill:#1e293b,stroke:#3b82f6,stroke-width:2px,color:#f8fafc;
    classDef ingestLayer fill:#0f172a,stroke:#64748b,stroke-width:1px,color:#e2e8f0;
    classDef llmLayer fill:#312e81,stroke:#6366f1,stroke-width:2px,color:#e0e7ff;
    classDef trainLayer fill:#14532d,stroke:#22c55e,stroke-width:2px,color:#f0fdf4;
    classDef holdoutLayer fill:#7f1d1d,stroke:#ef4444,stroke-width:2px,color:#fef2f2;
    classDef servingLayer fill:#3b0764,stroke:#a855f7,stroke-width:2px,color:#faf5ff;

    User(["👤 User / Client UI (React 18)"]):::userLayer
    API["⚡ FastAPI REST Gateway"]:::userLayer

    subgraph DataIngestion ["1. Data Ingestion & Isolation Protocol"]
        Upload["📁 Dataset Upload (.csv, .xlsx)"]:::ingestLayer
        Profile["📊 Dataset Profiler (dtypes, cardinality, missingness)"]:::ingestLayer
        Split{"🔀 Deterministic Split (Seed-Controlled)"}:::ingestLayer
        TrainSplit["🟢 Training Split (80%)"]:::trainLayer
        HoldoutSplit["🔴 100% Isolated Holdout (20%)"]:::holdoutLayer
    end

    subgraph LLMPlanning ["2. Structured LLM Pipeline Planning"]
        Context["📝 Training Profile Context (Zero Holdout Leakage)"]:::llmLayer
        LLMOrFallback{"🤖 Gemini / OpenAI / Deterministic Fallback"}:::llmLayer
        RawPlan["📋 Structured Pipeline Plan (JSON)"]:::llmLayer
        Validator{"🛡️ PipelinePlanValidator (Registry-Constrained)"}:::llmLayer
        SanitizedPlan["✅ Validated AutoML Pipeline Plan"]:::llmLayer
    end

    subgraph AutoMLCore ["3. Fold-Safe AutoML Training Engine"]
        Registries[("📚 Model & Operation Registries")]:::trainLayer
        FoldSafeCV["🔁 Stratified K-Fold Splitter (K=3 or 5)"]:::trainLayer
        Preprocessor["⚙️ PreprocessingEngine (Fit on Train Folds Only)"]:::trainLayer
        FeatureOps["🧪 FeatureEngineer (Controlled Operations)"]:::trainLayer
        OptunaHPO["🎯 Optuna Bayesian HPO (TPE Sampler)"]:::trainLayer
        Candidates["🏁 Candidate Estimator Training"]:::trainLayer
        OOFGen["📊 Out-of-Fold (OOF) Prediction Generator"]:::trainLayer
        EnsembleOpt["🤝 Model Fusion & Stacking Optimizer"]:::trainLayer
        CVSelection{"🏆 CV-Only Winner Selection"}:::trainLayer
    end

    subgraph HoldoutEval ["4. Scientific Holdout Evaluation (Post-Selection Only)"]
        WinningPipeline["✨ Selected Best Pipeline / Ensemble"]:::trainLayer
        FinalScore["🎯 Holdout Evaluator (Evaluated Strictly ONCE)"]:::holdoutLayer
    end

    subgraph Serving ["5. Serving, Explainability & Reporting"]
        SHAPExplainer["🔍 Explainer (TreeSHAP & KernelSHAP)"]:::servingLayer
        PredictorServ["🔮 Predictor & Batch Inference Engine"]:::servingLayer
        Reporter["📑 ReportGenerator (HTML / Markdown / Provenance)"]:::servingLayer
    end

    %% Workflow Connections
    User --> API
    API --> Upload
    Upload --> Profile
    Profile --> Split
    Split --> TrainSplit
    Split --> HoldoutSplit

    TrainSplit --> Context
    Context --> LLMOrFallback
    LLMOrFallback --> RawPlan
    RawPlan --> Validator
    Registries -.-> Validator
    Validator --> SanitizedPlan

    SanitizedPlan --> FoldSafeCV
    TrainSplit --> FoldSafeCV
    FoldSafeCV --> Preprocessor
    Preprocessor --> FeatureOps
    FeatureOps --> Candidates
    Candidates <--> OptunaHPO
    Candidates --> OOFGen
    OOFGen --> EnsembleOpt
    EnsembleOpt --> CVSelection
    Candidates --> CVSelection

    CVSelection --> WinningPipeline
    WinningPipeline --> FinalScore
    HoldoutSplit --> FinalScore

    WinningPipeline --> SHAPExplainer
    WinningPipeline --> PredictorServ
    FinalScore --> Reporter
    SHAPExplainer --> Reporter
    Reporter --> API
```

---

## 2. Text / ASCII Architectural Flow

```
                      +------------------------------------------+
                      |         USER / FRONTEND CLIENT           |
                      |          (React 18 + Vite SPA)           |
                      +------------------------------------------+
                                           |
                                           v
                      +------------------------------------------+
                      |           FASTAPI REST GATEWAY           |
                      +------------------------------------------+
                                           |
                                           v
                      +------------------------------------------+
                      |         DATASET INGESTION & READ         |
                      |   (CSV / Excel, Delimiter Detection)     |
                      +------------------------------------------+
                                           |
                                           v
                      +------------------------------------------+
                      |             DATASET PROFILER             |
                      | (Shape, Types, Missingness, Cardinality) |
                      +------------------------------------------+
                                           |
                       +-------------------+--------------------+
                       |                                        |
                       v                                        v
        +-------------------------------+       +-------------------------------+
        |        TRAIN SPLIT (80%)      |       |       HOLDOUT SPLIT (20%)     |
        |   Used for CV, HPO & Ensembles|       |     STRICTLY ISOLATED UNTIL   |
        +-------------------------------+       |         FINAL EVALUATION      |
                       |                        +-------------------------------+
                       v                                        :
        +-------------------------------+                       :
        |  LLM PIPELINE PLANNER ENGINE  |                       :
        | (Gemini / OpenAI / Fallback)  |                       :
        +-------------------------------+                       :
                       |                                        :
                       v                                        :
        +-------------------------------+                       :
        |     PIPELINE PLAN VALIDATOR   |                       :
        | (Registry Constraints, Safety)|                       :
        +-------------------------------+                       :
                       |                                        :
                       v                                        :
        +-------------------------------+                       :
        |      FOLD-SAFE PREPROCESSING  |                       :
        |  (Fit strictly on train folds)|                       :
        +-------------------------------+                       :
                       |                                        :
                       v                                        :
        +-------------------------------+                       :
        |   CONTROLLED FEATURE OPS      |                       :
        |  (Mathematical transformations)                       :
        +-------------------------------+                       :
                       |                                        :
                       v                                        :
        +-------------------------------+                       :
        |   MODEL TRAINING & OPTUNA HPO |                       :
        |   (Tree, Linear, Kernel Models)                       :
        +-------------------------------+                       :
                       |                                        :
                       v                                        :
        +-------------------------------+                       :
        | MODEL FUSION & ENSEMBLE TUNING|                       :
        |  (Leakage-Free OOF Stacking)  |                       :
        +-------------------------------+                       :
                       |                                        :
                       v                                        :
        +-------------------------------+                       :
        |       CV-BASED SELECTION      |                       :
        |  (Winner chosen on CV score)  |                       :
        +-------------------------------+                       :
                       |                                        :
                       v                                        v
        +---------------------------------------------------------------+
        |                  FINAL HOLDOUT EVALUATION                     |
        |     (Evaluated strictly once on the untouched 20% holdout)    |
        +---------------------------------------------------------------+
                       |
                       +-------------------+--------------------+
                       |                   |                    |
                       v                   v                    v
        +---------------------+ +--------------------+ +-----------------+
        | SHAP EXPLAINABILITY | | REAL-TIME PREDICT  | | RESEARCH REPORT |
        | (Values & Plots)    | | (Single & Batch)   | | (HTML & MD)     |
        +---------------------+ +--------------------+ +-----------------+
```

---

## 3. Strict Holdout Isolation Contract

A core scientific guarantee of AutoML-Lens is **zero data leakage**:

1. **Pre-Planning Partition**: The dataset is split into `train` (80%) and `holdout` (20%) using a deterministic random seed before profiling context is passed to the LLM.
2. **Context Restriction**: The LLM prompt receives statistical summaries (column names, types, missing rates, value ranges) derived **strictly from the 80% training partition**. The holdout partition is invisible to the planner.
3. **Fold-Safe Isolation**: Inside the cross-validation loop, all imputation statistics, standardizers, one-hot encoders, and custom feature transforms are fit strictly on each fold's training slice ($K-1$ folds) and applied to the validation fold ($1$ fold).
4. **Post-Selection Evaluation**: Model ranking and ensemble weighting are determined exclusively by out-of-fold cross-validation performance. The 20% holdout split is evaluated **exactly once** after the final pipeline has been fixed.
