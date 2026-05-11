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
    EvalSprint, EvalTracking, EvalLeague, EvalBoundary
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

# CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "http://127.0.0.1:5173"],
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
    sprint: Optional[dict] = None
    tracking: Optional[dict] = None
    league: Optional[dict] = None
    boundary: Optional[dict] = None


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
        birth_date=datetime.fromisoformat(patient.birth_date) if patient.birth_date else None,
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
    if patient.birth_date:
        db_patient.birth_date = datetime.fromisoformat(patient.birth_date)
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

    return {
        "id": session.id,
        "patient_id": session.patient_id,
        "created_at": session.created_at,
        "total_score": session.total_score,
        "sprint_score": session.sprint_score,
        "tracking_score": session.tracking_score,
        "league_score": session.league_score,
        "boundary_score": session.boundary_score,
        "notes": session.notes,
        "sprint": session.sprint.__dict__ if session.sprint else None,
        "tracking": session.tracking.__dict__ if session.tracking else None,
        "league": session.league.__dict__ if session.league else None,
        "boundary": session.boundary.__dict__ if session.boundary else None,
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

    def run_evaluation():
        global eval_engine

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

                if result.sprint:
                    sprint = EvalSprint(
                        session_id=session_id,
                        catch_times=result.sprint['catch_times'],
                        peak_vels=result.sprint['peak_vels']
                    )
                    db_local.add(sprint)

                if result.tracking:
                    tracking = EvalTracking(
                        session_id=session_id,
                        rmse_list=result.tracking['rmse_list'],
                        jerk_list=result.tracking['jerk_list']
                    )
                    db_local.add(tracking)

                if result.league:
                    league = EvalLeague(
                        session_id=session_id,
                        is_caught=result.league['is_caught'],
                        survival_time=result.league['survival_time'],
                        dist_list=result.league['dist_list']
                    )
                    db_local.add(league)

                if result.boundary:
                    boundary = EvalBoundary(
                        session_id=session_id,
                        min_x=result.boundary['min_x'],
                        max_x=result.boundary['max_x'],
                        min_y=result.boundary['min_y'],
                        max_y=result.boundary['max_y'],
                        vel_list=result.boundary['vel_list']
                    )
                    db_local.add(boundary)

                # Build results dict for scoring
                results = {}
                if result.sprint:
                    results['sprint'] = result.sprint
                if result.tracking:
                    results['tracking'] = result.tracking
                if result.league:
                    results['league'] = result.league
                if result.boundary:
                    results['boundary'] = result.boundary

                scores = _calculate_scores(results)
                db_session.total_score = scores['total']
                db_session.sprint_score = scores['sprint']
                db_session.tracking_score = scores['tracking']
                db_session.league_score = scores['league']
                db_session.boundary_score = scores['boundary']
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
            data = await websocket.receive_text()
            # Handle incoming messages from client if needed
    except WebSocketDisconnect:
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
            avg_total_score=round(sum(s.total_score for s in sessions) / n, 1),
            avg_sprint_score=round(sum(s.sprint_score for s in sessions if s.sprint_score is not None) / n, 1) if any(s.sprint_score for s in sessions) else None,
            avg_tracking_score=round(sum(s.tracking_score for s in sessions if s.tracking_score is not None) / n, 1) if any(s.tracking_score for s in sessions) else None,
            avg_league_score=round(sum(s.league_score for s in sessions if s.league_score is not None) / n, 1) if any(s.league_score for s in sessions) else None,
            avg_boundary_score=round(sum(s.boundary_score for s in sessions if s.boundary_score is not None) / n, 1) if any(s.boundary_score for s in sessions) else None,
        ))

    return result


@app.get("/api/stats/tasks", response_model=List[TaskStatsResponse])
def get_task_statistics(db: Session = Depends(get_db)):
    """Get average scores for each task across all completed sessions."""
    sessions = db.query(EvalSession).filter(EvalSession.total_score.isnot(None)).all()

    task_names = ['sprint', 'tracking', 'league', 'boundary']
    result = []

    for task in task_names:
        score_attr = f'{task}_score'
        scores = [getattr(s, score_attr) for s in sessions if getattr(s, score_attr) is not None]
        n = len(scores)

        if n == 0:
            result.append(TaskStatsResponse(
                task=task.capitalize(),
                avg_score=None,
                min_score=None,
                max_score=None,
                session_count=0,
            ))
        else:
            result.append(TaskStatsResponse(
                task=task.capitalize(),
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
        avg_total_score=round(sum(s.total_score for s in sessions) / n, 1),
        avg_sprint_score=round(sum(s.sprint_score for s in sessions if s.sprint_score is not None) / n, 1) if any(s.sprint_score for s in sessions) else None,
        avg_tracking_score=round(sum(s.tracking_score for s in sessions if s.tracking_score is not None) / n, 1) if any(s.tracking_score for s in sessions) else None,
        avg_league_score=round(sum(s.league_score for s in sessions if s.league_score is not None) / n, 1) if any(s.league_score for s in sessions) else None,
        avg_boundary_score=round(sum(s.boundary_score for s in sessions if s.boundary_score is not None) / n, 1) if any(s.boundary_score for s in sessions) else None,
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
        'Session ID', 'Patient Name', 'Date',
        'Total Score', 'Sprint', 'Tracking', 'League', 'Boundary',
        'Notes'
    ]
    ws.append(headers)

    sessions = db.query(EvalSession).join(Patient).order_by(EvalSession.created_at.desc()).all()

    for s in sessions:
        ws.append([
            s.id,
            s.patient.name,
            s.created_at.strftime('%Y-%m-%d %H:%M') if s.created_at else '',
            round(s.total_score, 1) if s.total_score else '',
            round(s.sprint_score, 1) if s.sprint_score else '',
            round(s.tracking_score, 1) if s.tracking_score else '',
            round(s.league_score, 1) if s.league_score else '',
            round(s.boundary_score, 1) if s.boundary_score else '',
            s.notes or '',
        ])

    # Patient Summary sheet
    ws2 = wb.create_sheet("Patient Summary")
    ws2.append(['Patient Name', 'Session Count', 'Avg Total', 'Avg Sprint', 'Avg Tracking', 'Avg League', 'Avg Boundary'])

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
        sprint_scores = [s.sprint_score for s in sessions if s.sprint_score is not None]
        tracking_scores = [s.tracking_score for s in sessions if s.tracking_score is not None]
        league_scores = [s.league_score for s in sessions if s.league_score is not None]
        boundary_scores = [s.boundary_score for s in sessions if s.boundary_score is not None]

        ws2.append([
            patient.name,
            n,
            avg_total,
            round(sum(sprint_scores) / len(sprint_scores), 1) if sprint_scores else '',
            round(sum(tracking_scores) / len(tracking_scores), 1) if tracking_scores else '',
            round(sum(league_scores) / len(league_scores), 1) if league_scores else '',
            round(sum(boundary_scores) / len(boundary_scores), 1) if boundary_scores else '',
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
