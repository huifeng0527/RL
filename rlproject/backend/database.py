"""SQLAlchemy database models for rehabilitation evaluation system."""
from sqlalchemy import create_engine, Column, Integer, String, Float, Boolean, DateTime, ForeignKey, JSON, Text, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from datetime import datetime
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATABASE_PATH = os.path.join(BASE_DIR, "rehab_eval.db")

engine = create_engine(f"sqlite:///{DATABASE_PATH}", echo=False)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class Patient(Base):
    __tablename__ = "patients"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), nullable=False)
    gender = Column(String(10), nullable=True)  # 'M' or 'F'
    birth_date = Column(DateTime, nullable=True)
    diagnosis = Column(Text, nullable=True)
    notes = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    sessions = relationship("Session", back_populates="patient", cascade="all, delete-orphan")


class Session(Base):
    __tablename__ = "sessions"

    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(Integer, ForeignKey("patients.id"), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    total_score = Column(Float, nullable=True)

    # Legacy 4-task scores
    sprint_score = Column(Float, nullable=True)
    tracking_score = Column(Float, nullable=True)
    league_score = Column(Float, nullable=True)
    boundary_score = Column(Float, nullable=True)

    # Six-task scores
    rapid_reach_score = Column(Float, nullable=True)
    continuous_tracking_score = Column(Float, nullable=True)
    moving_target_interception_score = Column(Float, nullable=True)
    adaptive_boundary_challenge_score = Column(Float, nullable=True)
    rhythmic_switching_score = Column(Float, nullable=True)
    mirror_mapping_reach_score = Column(Float, nullable=True)

    notes = Column(Text, nullable=True)  # 新增：评估备注
    video_path = Column(String(500), nullable=True)  # 评估录像路径

    patient = relationship("Patient", back_populates="sessions")

    # Legacy 4-task relationships
    sprint = relationship("EvalSprint", back_populates="session", uselist=False, cascade="all, delete-orphan")
    tracking = relationship("EvalTracking", back_populates="session", uselist=False, cascade="all, delete-orphan")
    league = relationship("EvalLeague", back_populates="session", uselist=False, cascade="all, delete-orphan")
    boundary = relationship("EvalBoundary", back_populates="session", uselist=False, cascade="all, delete-orphan")

    # Six-task relationships
    rapid_reach = relationship("EvalRapidReach", back_populates="session", uselist=False, cascade="all, delete-orphan")
    continuous_tracking = relationship("EvalContinuousTracking", back_populates="session", uselist=False, cascade="all, delete-orphan")
    moving_target_interception = relationship("EvalMovingTargetInterception", back_populates="session", uselist=False, cascade="all, delete-orphan")
    adaptive_boundary_challenge = relationship("EvalAdaptiveBoundaryChallenge", back_populates="session", uselist=False, cascade="all, delete-orphan")
    rhythmic_switching = relationship("EvalRhythmicSwitching", back_populates="session", uselist=False, cascade="all, delete-orphan")
    mirror_mapping_reach = relationship("EvalMirrorMappingReach", back_populates="session", uselist=False, cascade="all, delete-orphan")


class EvalSprint(Base):
    __tablename__ = "eval_sprint"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(Integer, ForeignKey("sessions.id"), unique=True, nullable=False)

    catch_times = Column(JSON, nullable=False)  # List of 5 catch times
    peak_vels = Column(JSON, nullable=False)  # List of 5 peak velocities

    session = relationship("Session", back_populates="sprint")


class EvalTracking(Base):
    __tablename__ = "eval_tracking"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(Integer, ForeignKey("sessions.id"), unique=True, nullable=False)

    rmse_list = Column(JSON, nullable=False)  # Time series of cross-track errors
    jerk_list = Column(JSON, nullable=False)  # Time series of jerks

    session = relationship("Session", back_populates="tracking")


class EvalLeague(Base):
    __tablename__ = "eval_league"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(Integer, ForeignKey("sessions.id"), unique=True, nullable=False)

    is_caught = Column(Boolean, nullable=False)
    survival_time = Column(Float, nullable=False)  # seconds
    dist_list = Column(JSON, nullable=False)  # Time series of distances

    session = relationship("Session", back_populates="league")


class EvalBoundary(Base):
    __tablename__ = "eval_boundary"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(Integer, ForeignKey("sessions.id"), unique=True, nullable=False)

    min_x = Column(Float, nullable=False)
    max_x = Column(Float, nullable=False)
    min_y = Column(Float, nullable=False)
    max_y = Column(Float, nullable=False)
    vel_list = Column(JSON, nullable=False)  # Time series of velocities

    session = relationship("Session", back_populates="boundary")


class EvalRapidReach(Base):
    __tablename__ = "eval_rapid_reach"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(Integer, ForeignKey("sessions.id"), unique=True, nullable=False)

    catch_times = Column(JSON, nullable=False, default=list)
    peak_vels = Column(JSON, nullable=False, default=list)
    successes = Column(JSON, nullable=False, default=list)
    target_positions = Column(JSON, nullable=False, default=list)
    reaction_times = Column(JSON, nullable=False, default=list)
    movement_times = Column(JSON, nullable=False, default=list)
    endpoint_errors = Column(JSON, nullable=False, default=list)

    session = relationship("Session", back_populates="rapid_reach")


class EvalContinuousTracking(Base):
    __tablename__ = "eval_continuous_tracking"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(Integer, ForeignKey("sessions.id"), unique=True, nullable=False)

    rmse_list = Column(JSON, nullable=False, default=list)
    jerk_list = Column(JSON, nullable=False, default=list)
    mean_error = Column(Float, nullable=True)
    max_error = Column(Float, nullable=True)
    target_loss_rate = Column(Float, nullable=True)
    trajectory_names = Column(JSON, nullable=False, default=list)

    session = relationship("Session", back_populates="continuous_tracking")


class EvalMovingTargetInterception(Base):
    __tablename__ = "eval_moving_target_interception"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(Integer, ForeignKey("sessions.id"), unique=True, nullable=False)

    total_trials = Column(Integer, nullable=False, default=0)
    successes = Column(JSON, nullable=False, default=list)
    timing_errors = Column(JSON, nullable=False, default=list)
    spatial_errors = Column(JSON, nullable=False, default=list)
    early_count = Column(Integer, nullable=False, default=0)
    late_count = Column(Integer, nullable=False, default=0)
    reaction_times = Column(JSON, nullable=False, default=list)

    session = relationship("Session", back_populates="moving_target_interception")


class EvalAdaptiveBoundaryChallenge(Base):
    __tablename__ = "eval_adaptive_boundary_challenge"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(Integer, ForeignKey("sessions.id"), unique=True, nullable=False)

    reachable_radii = Column(JSON, nullable=False, default=list)
    reachable_area = Column(Float, nullable=True)
    directional_asymmetry = Column(Float, nullable=True)
    boundary_control_times = Column(JSON, nullable=False, default=list)
    boundary_violation_count = Column(Integer, nullable=False, default=0)
    recovery_times = Column(JSON, nullable=False, default=list)
    min_x = Column(Float, nullable=True)
    max_x = Column(Float, nullable=True)
    min_y = Column(Float, nullable=True)
    max_y = Column(Float, nullable=True)
    vel_list = Column(JSON, nullable=False, default=list)

    session = relationship("Session", back_populates="adaptive_boundary_challenge")


class EvalRhythmicSwitching(Base):
    __tablename__ = "eval_rhythmic_switching"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(Integer, ForeignKey("sessions.id"), unique=True, nullable=False)

    beat_times = Column(JSON, nullable=False, default=list)
    target_sequence = Column(JSON, nullable=False, default=list)
    response_times = Column(JSON, nullable=False, default=list)
    timing_errors = Column(JSON, nullable=False, default=list)
    correct_count = Column(Integer, nullable=False, default=0)
    early_count = Column(Integer, nullable=False, default=0)
    late_count = Column(Integer, nullable=False, default=0)
    miss_count = Column(Integer, nullable=False, default=0)
    rhythm_variability = Column(Float, nullable=True)

    session = relationship("Session", back_populates="rhythmic_switching")


class EvalMirrorMappingReach(Base):
    __tablename__ = "eval_mirror_mapping_reach"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(Integer, ForeignKey("sessions.id"), unique=True, nullable=False)

    cue_zones = Column(JSON, nullable=False, default=list)
    response_zones = Column(JSON, nullable=False, default=list)
    successes = Column(JSON, nullable=False, default=list)
    wrong_side_count = Column(Integer, nullable=False, default=0)
    wrong_target_count = Column(Integer, nullable=False, default=0)
    timeouts = Column(Integer, nullable=False, default=0)
    reaction_times = Column(JSON, nullable=False, default=list)
    movement_times = Column(JSON, nullable=False, default=list)
    spatial_errors = Column(JSON, nullable=False, default=list)
    path_efficiencies = Column(JSON, nullable=False, default=list)

    session = relationship("Session", back_populates="mirror_mapping_reach")


_SESSION_COLUMNS = {
    "rapid_reach_score": "FLOAT",
    "continuous_tracking_score": "FLOAT",
    "moving_target_interception_score": "FLOAT",
    "adaptive_boundary_challenge_score": "FLOAT",
    "rhythmic_switching_score": "FLOAT",
    "mirror_mapping_reach_score": "FLOAT",
    "video_path": "VARCHAR(500)",
}


def ensure_schema():
    """Add new columns to existing SQLite databases without dropping legacy data."""
    if not os.path.exists(DATABASE_PATH):
        return
    with engine.begin() as conn:
        existing = {row[1] for row in conn.execute(text("PRAGMA table_info(sessions)"))}
        for column_name, column_type in _SESSION_COLUMNS.items():
            if column_name not in existing:
                conn.execute(text(f"ALTER TABLE sessions ADD COLUMN {column_name} {column_type}"))


def init_db():
    """Initialize database tables."""
    Base.metadata.create_all(bind=engine)
    ensure_schema()


def get_db():
    """Get database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
