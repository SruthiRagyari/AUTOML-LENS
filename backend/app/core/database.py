"""SQLite database setup using SQLAlchemy."""
import json
from datetime import datetime, timezone
from pathlib import Path
from sqlalchemy import (
    create_engine, Column, Integer, String, Float, Text,
    DateTime, Boolean, ForeignKey, Enum as SAEnum
)
from sqlalchemy.orm import sessionmaker, declarative_base, relationship
from sqlalchemy.pool import StaticPool

Base = declarative_base()


class Dataset(Base):
    __tablename__ = "datasets"

    id = Column(Integer, primary_key=True, autoincrement=True)
    filename = Column(String(255), nullable=False)
    original_filename = Column(String(255), nullable=False)
    file_path = Column(String(500), nullable=False)
    file_size = Column(Integer)
    rows = Column(Integer)
    columns = Column(Integer)
    profile_json = Column(Text)  # JSON string of profile result
    uploaded_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    experiments = relationship("Experiment", back_populates="dataset")


class Experiment(Base):
    __tablename__ = "experiments"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(255))
    dataset_id = Column(Integer, ForeignKey("datasets.id"), nullable=False)
    target_column = Column(String(255))
    problem_type = Column(String(50))  # classification / regression
    primary_metric = Column(String(50))
    mode = Column(String(50), default="llm_assisted")  # baseline / automl / llm_assisted
    n_folds = Column(Integer, default=5)
    n_trials = Column(Integer, default=20)
    status = Column(String(50), default="created")  # created/profiling/analyzing/preprocessing/training/completed/failed
    llm_provider = Column(String(100))
    llm_analysis_json = Column(Text)
    preprocessing_json = Column(Text)
    feature_engineering_json = Column(Text)
    results_json = Column(Text)
    best_model_name = Column(String(255))
    best_score = Column(Float)
    explainability_json = Column(Text)
    report_path = Column(String(500))
    error_message = Column(Text)
    task_description = Column(Text)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))

    dataset = relationship("Dataset", back_populates="experiments")
    training_runs = relationship("TrainingRun", back_populates="experiment")
    predictions = relationship("Prediction", back_populates="experiment")


class TrainingRun(Base):
    __tablename__ = "training_runs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    experiment_id = Column(Integer, ForeignKey("experiments.id"), nullable=False)
    model_name = Column(String(255), nullable=False)
    display_name = Column(String(255))
    status = Column(String(50), default="queued")  # queued/running/completed/failed
    baseline_metrics_json = Column(Text)
    optimized_metrics_json = Column(Text)
    best_params_json = Column(Text)
    cv_scores_json = Column(Text)
    optimization_history_json = Column(Text)
    training_time = Column(Float)
    prediction_time = Column(Float)
    error_message = Column(Text)
    model_path = Column(String(500))
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    experiment = relationship("Experiment", back_populates="training_runs")


class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    experiment_id = Column(Integer, ForeignKey("experiments.id"), nullable=False)
    prediction_type = Column(String(50))  # single / batch
    input_json = Column(Text)
    result_json = Column(Text)
    batch_file_path = Column(String(500))
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    experiment = relationship("Experiment", back_populates="predictions")


# Engine and session factory
_engine = None
_SessionLocal = None


def get_engine():
    global _engine
    if _engine is None:
        db_path = Path("automl_lens.db")
        _engine = create_engine(
            f"sqlite:///{db_path}",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
            echo=False,
        )
    return _engine


def get_session_factory():
    global _SessionLocal
    if _SessionLocal is None:
        _SessionLocal = sessionmaker(
            autocommit=False, autoflush=False, bind=get_engine()
        )
    return _SessionLocal


def get_db():
    """Dependency for FastAPI routes."""
    SessionLocal = get_session_factory()
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """Create all tables."""
    engine = get_engine()
    Base.metadata.create_all(bind=engine)
