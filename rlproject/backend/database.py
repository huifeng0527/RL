"""SQLAlchemy database models for rehabilitation evaluation system."""
from sqlalchemy import create_engine, Column, Integer, String, Float, Boolean, DateTime, ForeignKey, JSON, Text
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
    sprint_score = Column(Float, nullable=True)
    tracking_score = Column(Float, nullable=True)
    league_score = Column(Float, nullable=True)
    boundary_score = Column(Float, nullable=True)
    notes = Column(Text, nullable=True)  # 新增：评估备注
    video_path = Column(String(500), nullable=True)  # 评估录像路径

    patient = relationship("Patient", back_populates="sessions")
    sprint = relationship("EvalSprint", back_populates="session", uselist=False, cascade="all, delete-orphan")
    tracking = relationship("EvalTracking", back_populates="session", uselist=False, cascade="all, delete-orphan")
    league = relationship("EvalLeague", back_populates="session", uselist=False, cascade="all, delete-orphan")
    boundary = relationship("EvalBoundary", back_populates="session", uselist=False, cascade="all, delete-orphan")


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


def init_db():
    """Initialize database tables."""
    Base.metadata.create_all(bind=engine)


def get_db():
    """Get database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
