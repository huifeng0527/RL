"""FastAPI backend for rehabilitation evaluation system."""
from fastapi import FastAPI, Depends, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime
import json
import asyncio
import threading
import queue

from database import (
    get_db, init_db, Patient, Session as EvalSession,
    EvalSprint, EvalTracking, EvalLeague, EvalBoundary
)
from eval_engine import EvalEngine, TaskProgress, EvalResult


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

    class Config:
        from_attributes = True


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
        "sprint": session.sprint.__dict__ if session.sprint else None,
        "tracking": session.tracking.__dict__ if session.tracking else None,
        "league": session.league.__dict__ if session.league else None,
        "boundary": session.boundary.__dict__ if session.boundary else None,
    }


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
                    "message": progress.message
                })

            def frame_broadcast_callback(frame_base64: str):
                _broadcast({
                    "type": "frame",
                    "data": frame_base64
                })

            eval_engine.set_progress_callback(progress_callback)
            eval_engine.set_frame_broadcast_callback(frame_broadcast_callback)

            result = eval_engine.run_all()

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

                total_score = 0
                if result.sprint and result.sprint['catch_times']:
                    avg_catch_time = sum(result.sprint['catch_times']) / len(result.sprint['catch_times'])
                    total_score += max(0, 25 - avg_catch_time * 2)
                if result.tracking and result.tracking['rmse_list']:
                    avg_rmse = sum(result.tracking['rmse_list']) / len(result.tracking['rmse_list'])
                    total_score += max(0, 30 - avg_rmse * 5)
                if result.league:
                    if not result.league['is_caught']:
                        total_score += 30
                    else:
                        total_score += result.league['survival_time']
                if result.boundary:
                    range_x = result.boundary['max_x'] - result.boundary['min_x']
                    range_y = result.boundary['max_y'] - result.boundary['min_y']
                    total_score += (range_x + range_y) * 2

                db_session.total_score = total_score
                db_local.commit()
            finally:
                db_local.close()

            _broadcast({"type": "complete", "total_score": total_score})

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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
