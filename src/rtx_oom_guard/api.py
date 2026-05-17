from fastapi import FastAPI, APIRouter, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from typing import Dict, Any
import json
from pathlib import Path

app = FastAPI(
    title="rtx_oom_guard API",
    description="Query simulated OOM risk and GPU memory telemetry",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Shared state

from rtx_oom_guard.scheduler.risk_model import OOMRiskModel
from rtx_oom_guard.profiler.allocator_logger import AllocatorLogger

_risk_model = OOMRiskModel(mode="rule")
_logger = AllocatorLogger()

# Schemas

class RiskRequest(BaseModel):
    fragmentation: float = Field(0.0, ge=0.0, le=1.0, description="Fragmentation ratio")
    utilisation: float = Field(0.0, ge=0.0, le=1.0, description="GPU memory utilisation")
    alloc_delta_mb: float = Field(0.0, description="Recent allocation delta in MB")

class RiskResponse(BaseModel):
    risk_score: float
    tier: str
    message: str

class MemoryResponse(BaseModel):
    current_allocated_mb: float
    current_reserved_mb: float
    free_estimate_mb: float
    current_frag: float
    cuda_available: bool

# GPU helpers

def _gpu_snapshot() -> Dict[str, Any]:
    try:
        import torch
        if torch.cuda.is_available():
            alloc = torch.cuda.memory_allocated() / (1024 ** 2)
            resv = torch.cuda.memory_reserved() / (1024 ** 2)
            return {
                "current_allocated_mb": round(alloc, 2),
                "current_reserved_mb": round(resv, 2),
                "free_estimate_mb": round(resv - alloc, 2),
                "current_frag": round(1 - alloc / resv, 4) if resv > 0 else 0.0,
                "cuda_available": True,
            }
    except (ImportError, Exception):
        pass
    return {
        "current_allocated_mb": 0.0,
        "current_reserved_mb": 0.0,
        "free_estimate_mb": 0.0,
        "current_frag": 0.0,
        "cuda_available": False,
    }

# API Router
api_router = APIRouter(prefix="/api")

@api_router.get("/health")
def health():
    """Liveness check."""
    return {"status": "ok", "service": "rtx_oom_guard-api"}

@api_router.get("/memory", response_model=MemoryResponse)
def get_memory():
    """Return current GPU memory state."""
    return _gpu_snapshot()

@api_router.post("/risk", response_model=RiskResponse)
def compute_risk(req: RiskRequest):
    """Compute OOM-risk score from memory statistics."""
    score = _risk_model.score(
        fragmentation=req.fragmentation,
        utilisation=req.utilisation,
        alloc_delta_mb=req.alloc_delta_mb,
    )

    if score >= 0.8:
        tier, msg = "ACT", "High OOM risk — consider clearing cache or reducing batch size"
    elif score >= 0.5:
        tier, msg = "WARN", "Elevated OOM risk — monitor closely"
    else:
        tier, msg = "SAFE", "OOM risk is low"

    return RiskResponse(risk_score=score, tier=tier, message=msg)

@api_router.get("/risk/history")
def risk_history():
    """Return all past risk evaluations."""
    return {"count": len(_risk_model.history), "entries": _risk_model.history}

@api_router.get("/telemetry")
def get_full_telemetry():
    """Real-time unified telemetry for the AeroGrid dashboard."""
    mem = _gpu_snapshot()
    telemetry = {
        **mem,
        "total_compactions": 0,
        "total_freed_mb": 0.0,
        "avg_latency_ms": 0.0,
        "compaction_history": []
    }

    live_path = Path("results/live_telemetry.json")
    if live_path.exists():
        import time
        for _ in range(3): # Retry loop for atomic read safety
            try:
                with open(live_path, "r") as f:
                    live_data = json.load(f)
                    telemetry["total_compactions"] = live_data.get("total_compactions", 0)
                    telemetry["total_freed_mb"] = live_data.get("total_freed_mb", 0.0)
                    telemetry["avg_latency_ms"] = live_data.get("avg_latency_ms", 0.0)
                    telemetry["compaction_history"] = live_data.get("compaction_history", [])
                    
                    # Core fix: The trainer process owns the CUDA allocation, not the API webserver.
                    # We must pull real VRAM snapshot metrics from the dashboard telemetry file.
                    if "current_allocated_mb" in live_data:
                        telemetry["current_allocated_mb"] = live_data["current_allocated_mb"]
                        telemetry["current_reserved_mb"] = live_data.get("current_reserved_mb", 0)
                        telemetry["free_estimate_mb"] = live_data.get("free_estimate_mb", 0)
                        telemetry["current_frag"] = live_data.get("current_frag", 0)
                        telemetry["cuda_available"] = True
                break # Success
            except (json.JSONDecodeError, OSError, IOError):
                time.sleep(0.01) # Short backoff and retry
    return telemetry

@api_router.get("/benchmarks")
def get_benchmark_results():
    """Fetch stored benchmark results for the dashboard comparison page."""
    results_dir = Path("results")
    if not results_dir.exists():
        return {"baseline": {}, "defrag": {}}
    try:
        base_path = results_dir / "baseline.json"
        defrag_path = results_dir / "defrag.json"
        base = json.loads(base_path.read_text()) if base_path.exists() else {}
        defrag = json.loads(defrag_path.read_text()) if defrag_path.exists() else {}
        return {"baseline": base, "defrag": defrag}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

app.include_router(api_router)

# Static Dashboard Mount (SPA Fallback)
# Resolves paths for standalone PyPI execution
DASHBOARD_PATH = Path(__file__).parent.parent / "dashboard" / "dist"

if DASHBOARD_PATH.exists():
    app.mount("/assets", StaticFiles(directory=DASHBOARD_PATH / "assets"), name="assets")

    # Serve index.html as a catch-all SPA router fallback
    @app.api_route("/{path_name:path}", methods=["GET"])
    def catch_all(path_name: str):
        file_path = DASHBOARD_PATH / path_name
        if file_path.is_file():
            return FileResponse(file_path)
        return FileResponse(DASHBOARD_PATH / "index.html")

