"""
HR Robot - FastAPI API Server (Phase 4)
REST API + WebSocket for real-time dashboard.
Runs in a background thread alongside the camera pipeline.
"""

import asyncio
import json
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware

from config.settings import API_HOST, API_PORT
from src.database.models import session_scope

# Will be set by start_api_server()
_runtime = None

app = FastAPI(title="HR Robot API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ========== Dashboard ==========

DASHBOARD_PATH = Path(__file__).parent / "dashboard.html"


@app.get("/", response_class=HTMLResponse)
async def serve_dashboard():
    """Serve the dashboard HTML page."""
    if DASHBOARD_PATH.exists():
        return FileResponse(DASHBOARD_PATH, media_type="text/html")
    return HTMLResponse("<h1>Dashboard not found</h1><p>Place dashboard.html in src/</p>")


# ========== REST API ==========

@app.get("/api/status")
async def get_status():
    """Real-time headcount & attendance status (from in-memory state)."""
    summary = _runtime.attendance.get_status_summary()
    # Convert datetime objects for JSON
    for m in summary.get("present_members", []):
        if isinstance(m.get("since"), datetime):
            m["since"] = m["since"].isoformat()
    summary["timestamp"] = datetime.utcnow().isoformat()
    return summary


@app.get("/api/members")
async def list_members():
    """List all registered members with embedding counts."""
    with session_scope() as session:
        faces = _runtime.repo.list_registered_faces(session)
    items = [
        {
            "member_id": info["member_id"],
            "full_name": name,
            "num_embeddings": info["num_embeddings"],
            "registered_at": info.get("registered_at"),
            "last_updated": info.get("last_updated"),
        }
        for name, info in faces.items()
    ]
    return {"total": len(items), "items": items}


@app.get("/api/attendance/today")
async def attendance_today(date: Optional[str] = Query(None, description="YYYY-MM-DD")):
    """Today's attendance logs (or specific date)."""
    if date:
        try:
            day = datetime.strptime(date, "%Y-%m-%d")
        except ValueError:
            return {"error": "Invalid date format. Use YYYY-MM-DD"}
    else:
        day = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)

    next_day = day + timedelta(days=1)

    with session_scope() as session:
        items, total = _runtime.repo.list_attendance_logs(
            session, start_dt=day, end_dt=next_day
        )
    return {"date": day.strftime("%Y-%m-%d"), "total": total, "items": items}


@app.get("/api/attendance/history")
async def attendance_history(
    start_date: Optional[str] = Query(None),
    end_date: Optional[str] = Query(None),
    member_id: Optional[int] = Query(None),
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
):
    """Historical attendance logs with filters."""
    now = datetime.utcnow()

    if start_date:
        try:
            start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        except ValueError:
            return {"error": "Invalid start_date"}
    else:
        start_dt = now - timedelta(days=7)

    if end_date:
        try:
            end_dt = datetime.strptime(end_date, "%Y-%m-%d") + timedelta(days=1)
        except ValueError:
            return {"error": "Invalid end_date"}
    else:
        end_dt = now + timedelta(days=1)

    with session_scope() as session:
        items, total = _runtime.repo.list_attendance_logs(
            session, start_dt=start_dt, end_dt=end_dt,
            member_id=member_id, limit=limit, offset=offset,
        )
    return {
        "start_date": start_dt.strftime("%Y-%m-%d"),
        "end_date": (end_dt - timedelta(days=1)).strftime("%Y-%m-%d"),
        "total": total,
        "items": items,
    }


@app.get("/api/stats")
async def get_stats(
    start_date: Optional[str] = Query(None),
    end_date: Optional[str] = Query(None),
):
    """KPI statistics for a date range (default: last 7 days)."""
    now = datetime.utcnow()

    if start_date:
        try:
            start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        except ValueError:
            return {"error": "Invalid start_date"}
    else:
        start_dt = now - timedelta(days=7)

    if end_date:
        try:
            end_dt = datetime.strptime(end_date, "%Y-%m-%d") + timedelta(days=1)
        except ValueError:
            return {"error": "Invalid end_date"}
    else:
        end_dt = now + timedelta(days=1)

    with session_scope() as session:
        stats = _runtime.repo.get_attendance_stats(session, start_dt, end_dt)

    stats["start_date"] = start_dt.strftime("%Y-%m-%d")
    stats["end_date"] = (end_dt - timedelta(days=1)).strftime("%Y-%m-%d")
    return stats


# ========== WebSocket ==========

@app.websocket("/ws/live")
async def websocket_live(ws: WebSocket):
    """Real-time status updates via WebSocket."""
    await ws.accept()
    _runtime.add_ws_client(ws)
    try:
        # Send initial status
        summary = _runtime.attendance.get_status_summary()
        summary["timestamp"] = datetime.utcnow().isoformat()
        summary["type"] = "status"
        for m in summary.get("present_members", []):
            if isinstance(m.get("since"), datetime):
                m["since"] = m["since"].isoformat()
        await ws.send_json(summary)

        # Keep connection alive, listen for pings
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        _runtime.remove_ws_client(ws)


# ========== Server Lifecycle ==========

_api_loop: Optional[asyncio.AbstractEventLoop] = None


async def _broadcast_to_clients(data: dict):
    """Send data to all connected WebSocket clients."""
    clients = _runtime.ws_clients
    for ws in clients:
        try:
            await ws.send_json(data)
        except Exception:
            _runtime.remove_ws_client(ws)


def broadcast_status(summary: dict):
    """
    Thread-safe broadcast from camera thread to WebSocket clients.
    Called by pipeline via on_status_update callback.
    """
    if _api_loop is None or not _runtime:
        return

    # Prepare data for JSON serialization
    data = {
        "type": "status",
        "timestamp": datetime.utcnow().isoformat(),
        "headcount": summary.get("headcount", 0),
        "known_count": summary.get("known_count", 0),
        "unknown_count": summary.get("unknown_count", 0),
        "is_overloaded": summary.get("is_overloaded", False),
        "new_checkins": summary.get("new_checkins", []),
        "new_checkouts": summary.get("new_checkouts", []),
        "overload_alert_triggered": summary.get("overload_alert_triggered", False),
    }

    # Get fresh present_members from attendance manager
    status = _runtime.attendance.get_status_summary()
    data["present_count"] = status.get("present_count", 0)
    data["max_capacity"] = status.get("max_capacity", 20)
    members = status.get("present_members", [])
    for m in members:
        if isinstance(m.get("since"), datetime):
            m["since"] = m["since"].isoformat()
    data["present_members"] = members

    # Convert datetime objects in checkins/checkouts
    for e in data.get("new_checkins", []):
        if isinstance(e.get("time"), datetime):
            e["time"] = e["time"].isoformat()
    for e in data.get("new_checkouts", []):
        if isinstance(e.get("time"), datetime):
            e["time"] = e["time"].isoformat()

    try:
        asyncio.run_coroutine_threadsafe(_broadcast_to_clients(data), _api_loop)
    except Exception:
        pass


def start_api_server(runtime):
    """Start FastAPI server in a background daemon thread."""
    global _runtime, _api_loop
    _runtime = runtime

    config = uvicorn.Config(
        app, host=API_HOST, port=API_PORT,
        log_level="warning",
        log_config=None,
    )
    server = uvicorn.Server(config)
    server.install_signal_handlers = lambda: None

    def _run():
        global _api_loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        _api_loop = loop
        loop.run_until_complete(server.serve())

    thread = threading.Thread(target=_run, daemon=True, name="api-server")
    thread.start()
    print(f"[API] Server started at http://{API_HOST}:{API_PORT}")
    print(f"[API] Dashboard: http://localhost:{API_PORT}/")
    return thread
