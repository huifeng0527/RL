"""FastAPI backend for rehabilitation evaluation system."""
from fastapi import FastAPI, Depends, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
from sqlalchemy import func
from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime
import json
import asyncio
import threading
import queue
import os

from database import (
    get_db, init_db, Patient, Session as EvalSession,
    EvalSprint, EvalTracking, EvalLeague, EvalBoundary,
    EvalRapidReach, EvalContinuousTracking, EvalMovingTargetInterception,
    EvalAdaptiveBoundaryChallenge, EvalRhythmicSwitching, EvalMirrorMappingReach
)
from eval_engine import EvalEngine, TaskProgress, EvalResult
from report_generator import ReportGenerator

# Singleton for scoring
_report_gen = ReportGenerator()


def _calculate_scores(results) -> dict:
    """Calculate normalized scores (0-100) using report_generator logic."""
    return _report_gen._calculate_scores(results)


# Initialize database
init_db()

# Main event loop reference (set on startup, used by threads to schedule async work)
_main_loop: asyncio.AbstractEventLoop = None

app = FastAPI(title="Rehab Evaluation System", version="1.0.0")


@app.on_event("startup")
async def _capture_loop():
    global _main_loop
    _main_loop = asyncio.get_running_loop()

cors_origins = os.getenv(
    "CORS_ORIGINS",
    "http://localhost:5173,http://localhost:3000,http://127.0.0.1:5173"
).split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[origin.strip() for origin in cors_origins if origin.strip()],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic models
class PatientCreate(BaseModel):
    name: str
    gender: Optional[str] = None
    birth_date: Optional[str] = None
    diagnosis: Optional[str] = None
    notes: Optional[str] = None


class PatientResponse(BaseModel):
    id: int
    name: str
    gender: Optional[str]
    birth_date: Optional[datetime]
    diagnosis: Optional[str]
    notes: Optional[str]
    created_at: datetime

    class Config:
        from_attributes = True


class SessionCreate(BaseModel):
    patient_id: int


class SessionResponse(BaseModel):
    id: int
    patient_id: int
    created_at: datetime
    total_score: Optional[float]
    notes: Optional[str]

    class Config:
        from_attributes = True


class SessionNotesUpdate(BaseModel):
    notes: str


class EvalResultCreate(BaseModel):
    rapid_reach: Optional[dict] = None
    continuous_tracking: Optional[dict] = None
    moving_target_interception: Optional[dict] = None
    adaptive_boundary_challenge: Optional[dict] = None
    rhythmic_switching: Optional[dict] = None
    mirror_mapping_reach: Optional[dict] = None
    legacy_league: Optional[dict] = None


def _parse_birth_date(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid birth_date format")


def _mean_score(values) -> Optional[float]:
    scores = [value for value in values if value is not None]
    if not scores:
        return None
    return round(sum(scores) / len(scores), 1)


def _score_value(session, new_attr: str, legacy_attr: Optional[str] = None) -> Optional[float]:
    value = getattr(session, new_attr, None)
    if value is not None or legacy_attr is None:
        return value
    return getattr(session, legacy_attr, None)


def _serialize_sprint(sprint: Optional[EvalSprint]) -> Optional[dict]:
    if not sprint:
        return None
    return {"catch_times": sprint.catch_times, "peak_vels": sprint.peak_vels}


def _serialize_tracking(tracking: Optional[EvalTracking]) -> Optional[dict]:
    if not tracking:
        return None
    return {"rmse_list": tracking.rmse_list, "jerk_list": tracking.jerk_list}


def _serialize_league(league: Optional[EvalLeague]) -> Optional[dict]:
    if not league:
        return None
    return {"is_caught": league.is_caught, "survival_time": league.survival_time, "dist_list": league.dist_list}


def _serialize_boundary(boundary: Optional[EvalBoundary]) -> Optional[dict]:
    if not boundary:
        return None
    return {
        "min_x": boundary.min_x,
        "max_x": boundary.max_x,
        "min_y": boundary.min_y,
        "max_y": boundary.max_y,
        "vel_list": boundary.vel_list,
    }


def _serialize_rapid_reach(result: Optional[EvalRapidReach]) -> Optional[dict]:
    if not result:
        return None
    return {
        "catch_times": result.catch_times,
        "peak_vels": result.peak_vels,
        "successes": result.successes,
        "target_positions": result.target_positions,
        "reaction_times": result.reaction_times,
        "movement_times": result.movement_times,
        "endpoint_errors": result.endpoint_errors,
    }


def _serialize_continuous_tracking(result: Optional[EvalContinuousTracking]) -> Optional[dict]:
    if not result:
        return None
    return {
        "rmse_list": result.rmse_list,
        "jerk_list": result.jerk_list,
        "mean_error": result.mean_error,
        "max_error": result.max_error,
        "target_loss_rate": result.target_loss_rate,
        "trajectory_names": result.trajectory_names,
    }


def _serialize_moving_target_interception(result: Optional[EvalMovingTargetInterception]) -> Optional[dict]:
    if not result:
        return None
    return {
        "total_trials": result.total_trials,
        "successes": result.successes,
        "timing_errors": result.timing_errors,
        "spatial_errors": result.spatial_errors,
        "early_count": result.early_count,
        "late_count": result.late_count,
        "reaction_times": result.reaction_times,
    }


def _serialize_adaptive_boundary_challenge(result: Optional[EvalAdaptiveBoundaryChallenge]) -> Optional[dict]:
    if not result:
        return None
    return {
        "reachable_radii": result.reachable_radii,
        "reachable_area": result.reachable_area,
        "directional_asymmetry": result.directional_asymmetry,
        "boundary_control_times": result.boundary_control_times,
        "boundary_violation_count": result.boundary_violation_count,
        "recovery_times": result.recovery_times,
        "min_x": result.min_x,
        "max_x": result.max_x,
        "min_y": result.min_y,
        "max_y": result.max_y,
        "vel_list": result.vel_list,
    }


def _serialize_rhythmic_switching(result: Optional[EvalRhythmicSwitching]) -> Optional[dict]:
    if not result:
        return None
    return {
        "beat_times": result.beat_times,
        "target_sequence": result.target_sequence,
        "response_times": result.response_times,
        "timing_errors": result.timing_errors,
        "correct_count": result.correct_count,
        "early_count": result.early_count,
        "late_count": result.late_count,
        "miss_count": result.miss_count,
        "rhythm_variability": result.rhythm_variability,
    }


def _serialize_mirror_mapping_reach(result: Optional[EvalMirrorMappingReach]) -> Optional[dict]:
    if not result:
        return None
    return {
        "cue_zones": result.cue_zones,
        "response_zones": result.response_zones,
        "successes": result.successes,
        "wrong_side_count": result.wrong_side_count,
        "wrong_target_count": result.wrong_target_count,
        "timeouts": result.timeouts,
        "reaction_times": result.reaction_times,
        "movement_times": result.movement_times,
        "spatial_errors": result.spatial_errors,
        "path_efficiencies": result.path_efficiencies,
    }


# Global eval engine state
eval_engine: Optional[EvalEngine] = None
eval_thread: Optional[threading.Thread] = None
eval_queue: queue.Queue = queue.Queue()


# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        """Broadcast message to all active WebSocket connections."""
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception:
                disconnected.append(connection)
        # Clean up disconnected
        for conn in disconnected:
            if conn in self.active_connections:
                self.active_connections.remove(conn)


manager = ConnectionManager()


# Routes - Patients
@app.get("/api/patients", response_model=List[PatientResponse])
def list_patients(db: Session = Depends(get_db)):
    """List all patients."""
    return db.query(Patient).order_by(Patient.created_at.desc()).all()


@app.post("/api/patients", response_model=PatientResponse)
def create_patient(patient: PatientCreate, db: Session = Depends(get_db)):
    """Create a new patient."""
    db_patient = Patient(
        name=patient.name,
        gender=patient.gender,
        birth_date=_parse_birth_date(patient.birth_date),
        diagnosis=patient.diagnosis,
        notes=patient.notes
    )
    db.add(db_patient)
    db.commit()
    db.refresh(db_patient)
    return db_patient


@app.get("/api/patients/{patient_id}", response_model=PatientResponse)
def get_patient(patient_id: int, db: Session = Depends(get_db)):
    """Get patient by ID."""
    patient = db.query(Patient).filter(Patient.id == patient_id).first()
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")
    return patient


@app.put("/api/patients/{patient_id}", response_model=PatientResponse)
def update_patient(patient_id: int, patient: PatientCreate, db: Session = Depends(get_db)):
    """Update patient."""
    db_patient = db.query(Patient).filter(Patient.id == patient_id).first()
    if not db_patient:
        raise HTTPException(status_code=404, detail="Patient not found")

    db_patient.name = patient.name
    db_patient.gender = patient.gender
    db_patient.birth_date = _parse_birth_date(patient.birth_date)
    db_patient.diagnosis = patient.diagnosis
    db_patient.notes = patient.notes

    db.commit()
    db.refresh(db_patient)
    return db_patient


@app.delete("/api/patients/{patient_id}")
def delete_patient(patient_id: int, db: Session = Depends(get_db)):
    """Delete patient."""
    patient = db.query(Patient).filter(Patient.id == patient_id).first()
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")
    db.delete(patient)
    db.commit()
    return {"message": "Patient deleted"}


# Routes - Sessions
@app.get("/api/sessions", response_model=List[SessionResponse])
def list_sessions(patient_id: Optional[int] = None, db: Session = Depends(get_db)):
    """List all sessions, optionally filtered by patient."""
    query = db.query(EvalSession)
    if patient_id:
        query = query.filter(EvalSession.patient_id == patient_id)
    return query.order_by(EvalSession.created_at.desc()).all()


@app.post("/api/sessions", response_model=SessionResponse)
def create_session(session: SessionCreate, db: Session = Depends(get_db)):
    """Create a new evaluation session."""
    # Verify patient exists
    patient = db.query(Patient).filter(Patient.id == session.patient_id).first()
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")

    db_session = EvalSession(patient_id=session.patient_id)
    db.add(db_session)
    db.commit()
    db.refresh(db_session)
    return db_session


@app.get("/api/sessions/{session_id}")
def get_session_detail(session_id: int, db: Session = Depends(get_db)):
    """Get session with full evaluation details."""
    session = db.query(EvalSession).filter(EvalSession.id == session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    rapid_reach = _serialize_rapid_reach(session.rapid_reach) or _serialize_sprint(session.sprint)
    continuous_tracking = _serialize_continuous_tracking(session.continuous_tracking) or _serialize_tracking(session.tracking)
    adaptive_boundary = _serialize_adaptive_boundary_challenge(session.adaptive_boundary_challenge) or _serialize_boundary(session.boundary)

    return {
        "id": session.id,
        "patient_id": session.patient_id,
        "created_at": session.created_at,
        "total_score": session.total_score,
        "rapid_reach_score": session.rapid_reach_score if session.rapid_reach_score is not None else session.sprint_score,
        "continuous_tracking_score": session.continuous_tracking_score if session.continuous_tracking_score is not None else session.tracking_score,
        "moving_target_interception_score": session.moving_target_interception_score,
        "adaptive_boundary_challenge_score": session.adaptive_boundary_challenge_score if session.adaptive_boundary_challenge_score is not None else session.boundary_score,
        "rhythmic_switching_score": session.rhythmic_switching_score,
        "mirror_mapping_reach_score": session.mirror_mapping_reach_score,
        "sprint_score": session.sprint_score,
        "tracking_score": session.tracking_score,
        "league_score": session.league_score,
        "boundary_score": session.boundary_score,
        "notes": session.notes,
        "video_path": session.video_path,
        "rapid_reach": rapid_reach,
        "continuous_tracking": continuous_tracking,
        "moving_target_interception": _serialize_moving_target_interception(session.moving_target_interception),
        "adaptive_boundary_challenge": adaptive_boundary,
        "rhythmic_switching": _serialize_rhythmic_switching(session.rhythmic_switching),
        "mirror_mapping_reach": _serialize_mirror_mapping_reach(session.mirror_mapping_reach),
        "legacy_league": _serialize_league(session.league),
        "sprint": _serialize_sprint(session.sprint),
        "tracking": _serialize_tracking(session.tracking),
        "league": _serialize_league(session.league),
        "boundary": _serialize_boundary(session.boundary),
    }


@app.patch("/api/sessions/{session_id}/notes")
def update_session_notes(session_id: int, notes_update: SessionNotesUpdate, db: Session = Depends(get_db)):
    """Update session notes."""
    db_session = db.query(EvalSession).filter(EvalSession.id == session_id).first()
    if not db_session:
        raise HTTPException(status_code=404, detail="Session not found")

    db_session.notes = notes_update.notes
    db.commit()
    db.refresh(db_session)
    return {"message": "Notes updated", "notes": db_session.notes}


@app.delete("/api/sessions/{session_id}")
def delete_session(session_id: int, db: Session = Depends(get_db)):
    """Delete a session and its associated evaluation video."""
    db_session = db.query(EvalSession).filter(EvalSession.id == session_id).first()
    if not db_session:
        raise HTTPException(status_code=404, detail="Session not found")

    # Delete associated video file if exists
    if db_session.video_path:
        try:
            if os.path.exists(db_session.video_path):
                os.remove(db_session.video_path)
                print(f"[API] Deleted video: {db_session.video_path}")
        except Exception as e:
            print(f"[API] Failed to delete video: {e}")

    db.delete(db_session)
    db.commit()
    return {"message": "Session deleted"}


# Routes - Evaluation
@app.post("/api/eval/start")
async def start_evaluation(session_id: int, db: Session = Depends(get_db)):
    """Start evaluation for a session."""
    global eval_engine, eval_thread

    if eval_thread and eval_thread.is_alive():
        raise HTTPException(status_code=400, detail="Evaluation already in progress")

    session = db.query(EvalSession).filter(EvalSession.id == session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    has_results = any([
        session.total_score is not None,
        session.rapid_reach,
        session.continuous_tracking,
        session.moving_target_interception,
        session.adaptive_boundary_challenge,
        session.rhythmic_switching,
        session.mirror_mapping_reach,
        session.sprint,
        session.tracking,
        session.league,
        session.boundary,
    ])
    if has_results:
        raise HTTPException(status_code=409, detail="Session already has evaluation results; create a new session to re-evaluate")

    def run_evaluation():
        global eval_engine, eval_thread

        def _broadcast(msg: dict):
            if _main_loop and _main_loop.is_running():
                asyncio.run_coroutine_threadsafe(manager.broadcast(msg), _main_loop)

        try:
            eval_engine = EvalEngine(simulate=False)
            if not eval_engine.connect():
                _broadcast({"type": "error", "message": "Failed to connect to hardware"})
                return

            def progress_callback(progress: TaskProgress):
                _broadcast({
                    "type": "progress",
                    "task": progress.current_task,
                    "task_index": progress.task_index,
                    "progress": progress.task_progress,
                    "message": progress.message,
                    "fps": getattr(eval_engine, '_current_fps', 0) if eval_engine else 0
                })

            def frame_broadcast_callback(frame_base64: str):
                _broadcast({
                    "type": "frame",
                    "data": frame_base64
                })

            eval_engine.set_progress_callback(progress_callback)
            eval_engine.set_frame_broadcast_callback(frame_broadcast_callback)

            result = eval_engine.run_all(session_id=session_id)

            db_local = next(get_db())
            try:
                db_session = db_local.query(EvalSession).filter(EvalSession.id == session_id).first()

                if result.rapid_reach:
                    db_local.add(EvalRapidReach(
                        session_id=session_id,
                        catch_times=result.rapid_reach.get('catch_times', []),
                        peak_vels=result.rapid_reach.get('peak_vels', []),
                        successes=result.rapid_reach.get('successes', []),
                        target_positions=result.rapid_reach.get('target_positions', []),
                        reaction_times=result.rapid_reach.get('reaction_times', []),
                        movement_times=result.rapid_reach.get('movement_times', []),
                        endpoint_errors=result.rapid_reach.get('endpoint_errors', []),
                    ))

                if result.continuous_tracking:
                    db_local.add(EvalContinuousTracking(
                        session_id=session_id,
                        rmse_list=result.continuous_tracking.get('rmse_list', []),
                        jerk_list=result.continuous_tracking.get('jerk_list', []),
                        mean_error=result.continuous_tracking.get('mean_error'),
                        max_error=result.continuous_tracking.get('max_error'),
                        target_loss_rate=result.continuous_tracking.get('target_loss_rate'),
                        trajectory_names=result.continuous_tracking.get('trajectory_names', []),
                    ))

                if result.moving_target_interception:
                    db_local.add(EvalMovingTargetInterception(
                        session_id=session_id,
                        total_trials=result.moving_target_interception.get('total_trials', 0),
                        successes=result.moving_target_interception.get('successes', []),
                        timing_errors=result.moving_target_interception.get('timing_errors', []),
                        spatial_errors=result.moving_target_interception.get('spatial_errors', []),
                        early_count=result.moving_target_interception.get('early_count', 0),
                        late_count=result.moving_target_interception.get('late_count', 0),
                        reaction_times=result.moving_target_interception.get('reaction_times', []),
                    ))

                if result.adaptive_boundary_challenge:
                    db_local.add(EvalAdaptiveBoundaryChallenge(
                        session_id=session_id,
                        reachable_radii=result.adaptive_boundary_challenge.get('reachable_radii', []),
                        reachable_area=result.adaptive_boundary_challenge.get('reachable_area'),
                        directional_asymmetry=result.adaptive_boundary_challenge.get('directional_asymmetry'),
                        boundary_control_times=result.adaptive_boundary_challenge.get('boundary_control_times', []),
                        boundary_violation_count=result.adaptive_boundary_challenge.get('boundary_violation_count', 0),
                        recovery_times=result.adaptive_boundary_challenge.get('recovery_times', []),
                        min_x=result.adaptive_boundary_challenge.get('min_x'),
                        max_x=result.adaptive_boundary_challenge.get('max_x'),
                        min_y=result.adaptive_boundary_challenge.get('min_y'),
                        max_y=result.adaptive_boundary_challenge.get('max_y'),
                        vel_list=result.adaptive_boundary_challenge.get('vel_list', []),
                    ))

                if result.rhythmic_switching:
                    db_local.add(EvalRhythmicSwitching(
                        session_id=session_id,
                        beat_times=result.rhythmic_switching.get('beat_times', []),
                        target_sequence=result.rhythmic_switching.get('target_sequence', []),
                        response_times=result.rhythmic_switching.get('response_times', []),
                        timing_errors=result.rhythmic_switching.get('timing_errors', []),
                        correct_count=result.rhythmic_switching.get('correct_count', 0),
                        early_count=result.rhythmic_switching.get('early_count', 0),
                        late_count=result.rhythmic_switching.get('late_count', 0),
                        miss_count=result.rhythmic_switching.get('miss_count', 0),
                        rhythm_variability=result.rhythmic_switching.get('rhythm_variability'),
                    ))

                if result.mirror_mapping_reach:
                    db_local.add(EvalMirrorMappingReach(
                        session_id=session_id,
                        cue_zones=result.mirror_mapping_reach.get('cue_zones', []),
                        response_zones=result.mirror_mapping_reach.get('response_zones', []),
                        successes=result.mirror_mapping_reach.get('successes', []),
                        wrong_side_count=result.mirror_mapping_reach.get('wrong_side_count', 0),
                        wrong_target_count=result.mirror_mapping_reach.get('wrong_target_count', 0),
                        timeouts=result.mirror_mapping_reach.get('timeouts', 0),
                        reaction_times=result.mirror_mapping_reach.get('reaction_times', []),
                        movement_times=result.mirror_mapping_reach.get('movement_times', []),
                        spatial_errors=result.mirror_mapping_reach.get('spatial_errors', []),
                        path_efficiencies=result.mirror_mapping_reach.get('path_efficiencies', []),
                    ))

                results = result.to_dict()
                scores = _calculate_scores(results)
                db_session.total_score = scores['total']
                db_session.rapid_reach_score = scores['rapid_reach']
                db_session.continuous_tracking_score = scores['continuous_tracking']
                db_session.moving_target_interception_score = scores['moving_target_interception']
                db_session.adaptive_boundary_challenge_score = scores['adaptive_boundary_challenge']
                db_session.rhythmic_switching_score = scores['rhythmic_switching']
                db_session.mirror_mapping_reach_score = scores['mirror_mapping_reach']
                db_session.sprint_score = scores['rapid_reach']
                db_session.tracking_score = scores['continuous_tracking']
                db_session.league_score = None
                db_session.boundary_score = scores['adaptive_boundary_challenge']
                db_local.commit()
            finally:
                db_local.close()

            # Broadcast video path if available
            video_path = eval_engine.get_video_path() if eval_engine else None
            _broadcast({"type": "complete", "scores": scores, "video_path": video_path})

            # Save video_path to session
            if video_path:
                db_session_local = next(get_db())
                try:
                    session_to_update = db_session_local.query(EvalSession).filter(EvalSession.id == session_id).first()
                    if session_to_update:
                        session_to_update.video_path = video_path
                        db_session_local.commit()
                finally:
                    db_session_local.close()

        except Exception as e:
            _broadcast({"type": "error", "message": str(e)})
        finally:
            if eval_engine:
                eval_engine.disconnect()
                eval_engine = None
            eval_thread = None

    eval_thread = threading.Thread(target=run_evaluation)
    eval_thread.start()

    return {"message": "Evaluation started", "session_id": session_id}


@app.post("/api/eval/stop")
async def stop_evaluation():
    """Stop the current evaluation."""
    global eval_engine
    if eval_engine:
        eval_engine.stop()
    return {"message": "Stop signal sent"}


@app.get("/api/eval/status")
async def get_eval_status():
    """Get current evaluation status."""
    return {
        "in_progress": eval_thread is not None and eval_thread.is_alive(),
        "engine_ready": eval_engine is not None
    }


# WebSocket for real-time updates
@app.websocket("/ws/eval")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            try:
                await asyncio.wait_for(websocket.receive_text(), timeout=60.0)
            except asyncio.TimeoutError:
                continue
    except WebSocketDisconnect:
        pass
    finally:
        manager.disconnect(websocket)


# Health check
@app.get("/api/health")
def health_check():
    return {"status": "healthy"}


# ========================================================
# Statistics API - Patient and Task Average Scores
# ========================================================

class PatientStatsResponse(BaseModel):
    patient_id: int
    patient_name: str
    session_count: int
    avg_total_score: Optional[float]
    avg_rapid_reach_score: Optional[float]
    avg_continuous_tracking_score: Optional[float]
    avg_moving_target_interception_score: Optional[float]
    avg_adaptive_boundary_challenge_score: Optional[float]
    avg_rhythmic_switching_score: Optional[float]
    avg_mirror_mapping_reach_score: Optional[float]
    avg_sprint_score: Optional[float]
    avg_tracking_score: Optional[float]
    avg_league_score: Optional[float]
    avg_boundary_score: Optional[float]


class TaskStatsResponse(BaseModel):
    task: str
    avg_score: Optional[float]
    min_score: Optional[float]
    max_score: Optional[float]
    session_count: int


@app.get("/api/stats/patients", response_model=List[PatientStatsResponse])
def get_patient_statistics(db: Session = Depends(get_db)):
    """Get average scores for each patient across all their sessions."""
    patients = db.query(Patient).all()
    result = []

    for patient in patients:
        sessions = db.query(EvalSession).filter(
            EvalSession.patient_id == patient.id,
            EvalSession.total_score.isnot(None)
        ).all()

        if not sessions:
            result.append(PatientStatsResponse(
                patient_id=patient.id,
                patient_name=patient.name,
                session_count=0,
                avg_total_score=None,
                avg_rapid_reach_score=None,
                avg_continuous_tracking_score=None,
                avg_moving_target_interception_score=None,
                avg_adaptive_boundary_challenge_score=None,
                avg_rhythmic_switching_score=None,
                avg_mirror_mapping_reach_score=None,
                avg_sprint_score=None,
                avg_tracking_score=None,
                avg_league_score=None,
                avg_boundary_score=None,
            ))
            continue

        n = len(sessions)
        result.append(PatientStatsResponse(
            patient_id=patient.id,
            patient_name=patient.name,
            session_count=n,
            avg_total_score=_mean_score(s.total_score for s in sessions),
            avg_rapid_reach_score=_mean_score(_score_value(s, 'rapid_reach_score', 'sprint_score') for s in sessions),
            avg_continuous_tracking_score=_mean_score(_score_value(s, 'continuous_tracking_score', 'tracking_score') for s in sessions),
            avg_moving_target_interception_score=_mean_score(s.moving_target_interception_score for s in sessions),
            avg_adaptive_boundary_challenge_score=_mean_score(_score_value(s, 'adaptive_boundary_challenge_score', 'boundary_score') for s in sessions),
            avg_rhythmic_switching_score=_mean_score(s.rhythmic_switching_score for s in sessions),
            avg_mirror_mapping_reach_score=_mean_score(s.mirror_mapping_reach_score for s in sessions),
            avg_sprint_score=_mean_score(s.sprint_score for s in sessions),
            avg_tracking_score=_mean_score(s.tracking_score for s in sessions),
            avg_league_score=_mean_score(s.league_score for s in sessions),
            avg_boundary_score=_mean_score(s.boundary_score for s in sessions),
        ))

    return result


@app.get("/api/stats/tasks", response_model=List[TaskStatsResponse])
def get_task_statistics(db: Session = Depends(get_db)):
    """Get average scores for each task across all completed sessions."""
    sessions = db.query(EvalSession).filter(EvalSession.total_score.isnot(None)).all()

    tasks = [
        ('rapid_reach', 'Rapid Reach', 'sprint_score'),
        ('continuous_tracking', 'Continuous Tracking', 'tracking_score'),
        ('moving_target_interception', 'Moving Target Interception', None),
        ('adaptive_boundary_challenge', 'Adaptive Boundary Challenge', 'boundary_score'),
        ('rhythmic_switching', 'Rhythmic Switching', None),
        ('mirror_mapping_reach', 'Mirror Mapping Reach', None),
    ]
    result = []

    for task_key, task_label, legacy_attr in tasks:
        score_attr = f'{task_key}_score'
        scores = [_score_value(s, score_attr, legacy_attr) for s in sessions]
        scores = [score for score in scores if score is not None]
        n = len(scores)

        if n == 0:
            result.append(TaskStatsResponse(
                task=task_label,
                avg_score=None,
                min_score=None,
                max_score=None,
                session_count=0,
            ))
        else:
            result.append(TaskStatsResponse(
                task=task_label,
                avg_score=round(sum(scores) / n, 1),
                min_score=round(min(scores), 1),
                max_score=round(max(scores), 1),
                session_count=n,
            ))

    return result


@app.get("/api/stats/patient/{patient_id}", response_model=PatientStatsResponse)
def get_patient_stats(patient_id: int, db: Session = Depends(get_db)):
    """Get detailed statistics for a specific patient."""
    patient = db.query(Patient).filter(Patient.id == patient_id).first()
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")

    sessions = db.query(EvalSession).filter(
        EvalSession.patient_id == patient_id,
        EvalSession.total_score.isnot(None)
    ).all()

    if not sessions:
        return PatientStatsResponse(
            patient_id=patient.id,
            patient_name=patient.name,
            session_count=0,
            avg_total_score=None,
            avg_rapid_reach_score=None,
            avg_continuous_tracking_score=None,
            avg_moving_target_interception_score=None,
            avg_adaptive_boundary_challenge_score=None,
            avg_rhythmic_switching_score=None,
            avg_mirror_mapping_reach_score=None,
            avg_sprint_score=None,
            avg_tracking_score=None,
            avg_league_score=None,
            avg_boundary_score=None,
        )

    n = len(sessions)
    return PatientStatsResponse(
        patient_id=patient.id,
        patient_name=patient.name,
        session_count=n,
        avg_total_score=_mean_score(s.total_score for s in sessions),
        avg_rapid_reach_score=_mean_score(_score_value(s, 'rapid_reach_score', 'sprint_score') for s in sessions),
        avg_continuous_tracking_score=_mean_score(_score_value(s, 'continuous_tracking_score', 'tracking_score') for s in sessions),
        avg_moving_target_interception_score=_mean_score(s.moving_target_interception_score for s in sessions),
        avg_adaptive_boundary_challenge_score=_mean_score(_score_value(s, 'adaptive_boundary_challenge_score', 'boundary_score') for s in sessions),
        avg_rhythmic_switching_score=_mean_score(s.rhythmic_switching_score for s in sessions),
        avg_mirror_mapping_reach_score=_mean_score(s.mirror_mapping_reach_score for s in sessions),
        avg_sprint_score=_mean_score(s.sprint_score for s in sessions),
        avg_tracking_score=_mean_score(s.tracking_score for s in sessions),
        avg_league_score=_mean_score(s.league_score for s in sessions),
        avg_boundary_score=_mean_score(s.boundary_score for s in sessions),
    )


# ========================================================
# Excel Export
# ========================================================
@app.get("/api/export/excel")
def export_excel(db: Session = Depends(get_db)):
    """Export all session data as an Excel file."""
    try:
        import openpyxl
    except ImportError:
        raise HTTPException(status_code=500, detail="openpyxl not installed. Run: pip install openpyxl")

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "M-HECS Sessions"

    # Header
    headers = [
        'Session ID', 'Patient Name', 'Date', 'Total Score',
        'Rapid Reach', 'Continuous Tracking', 'Moving Target Interception',
        'Adaptive Boundary Challenge', 'Rhythmic Switching', 'Mirror Mapping Reach',
        'Notes'
    ]
    ws.append(headers)

    sessions = db.query(EvalSession).join(Patient).order_by(EvalSession.created_at.desc()).all()

    for s in sessions:
        ws.append([
            s.id,
            s.patient.name,
            s.created_at.strftime('%Y-%m-%d %H:%M') if s.created_at else '',
            round(s.total_score, 1) if s.total_score is not None else '',
            round(_score_value(s, 'rapid_reach_score', 'sprint_score'), 1) if _score_value(s, 'rapid_reach_score', 'sprint_score') is not None else '',
            round(_score_value(s, 'continuous_tracking_score', 'tracking_score'), 1) if _score_value(s, 'continuous_tracking_score', 'tracking_score') is not None else '',
            round(s.moving_target_interception_score, 1) if s.moving_target_interception_score is not None else '',
            round(_score_value(s, 'adaptive_boundary_challenge_score', 'boundary_score'), 1) if _score_value(s, 'adaptive_boundary_challenge_score', 'boundary_score') is not None else '',
            round(s.rhythmic_switching_score, 1) if s.rhythmic_switching_score is not None else '',
            round(s.mirror_mapping_reach_score, 1) if s.mirror_mapping_reach_score is not None else '',
            s.notes or '',
        ])

    # Patient Summary sheet
    ws2 = wb.create_sheet("Patient Summary")
    ws2.append([
        'Patient Name', 'Session Count', 'Avg Total',
        'Avg Rapid Reach', 'Avg Continuous Tracking', 'Avg Moving Target Interception',
        'Avg Adaptive Boundary Challenge', 'Avg Rhythmic Switching', 'Avg Mirror Mapping Reach'
    ])

    patients = db.query(Patient).all()
    for patient in patients:
        sessions = db.query(EvalSession).filter(
            EvalSession.patient_id == patient.id,
            EvalSession.total_score.isnot(None)
        ).all()
        if not sessions:
            continue
        n = len(sessions)
        avg_total = round(sum(s.total_score for s in sessions) / n, 1)
        rapid_scores = [_score_value(s, 'rapid_reach_score', 'sprint_score') for s in sessions]
        tracking_scores = [_score_value(s, 'continuous_tracking_score', 'tracking_score') for s in sessions]
        interception_scores = [s.moving_target_interception_score for s in sessions]
        boundary_scores = [_score_value(s, 'adaptive_boundary_challenge_score', 'boundary_score') for s in sessions]
        rhythm_scores = [s.rhythmic_switching_score for s in sessions]
        mirror_scores = [s.mirror_mapping_reach_score for s in sessions]

        rapid_mean = _mean_score(rapid_scores)
        tracking_mean = _mean_score(tracking_scores)
        interception_mean = _mean_score(interception_scores)
        boundary_mean = _mean_score(boundary_scores)
        rhythm_mean = _mean_score(rhythm_scores)
        mirror_mean = _mean_score(mirror_scores)

        ws2.append([
            patient.name,
            n,
            avg_total,
            rapid_mean if rapid_mean is not None else '',
            tracking_mean if tracking_mean is not None else '',
            interception_mean if interception_mean is not None else '',
            boundary_mean if boundary_mean is not None else '',
            rhythm_mean if rhythm_mean is not None else '',
            mirror_mean if mirror_mean is not None else '',
        ])

    # Save file
    export_dir = os.path.join(os.path.dirname(__file__), 'exports')
    os.makedirs(export_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filepath = os.path.join(export_dir, f'mhecs_export_{timestamp}.xlsx')
    wb.save(filepath)

    return FileResponse(
        filepath,
        media_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        filename=f'mhecs_export_{timestamp}.xlsx'
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
