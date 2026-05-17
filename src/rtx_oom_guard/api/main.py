import json
import time
from pathlib import Path
from typing import Dict, Any, Optional

import structlog
from fastapi import FastAPI, APIRouter, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from prometheus_client import Counter, Gauge, Histogram, generate_latest, CONTENT_TYPE_LATEST

from rtx_oom_guard.scheduler.risk_model import OOMRiskModel
from rtx_oom_guard.profiler.allocator_logger import AllocatorLogger

# Structured Logging Configuration
structlog.configure(
    processors=[
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer(),
    ],
    wrapper_class=structlog.make_filtering_bound_logger(20), # INFO
    cache_logger_on_first_use=True,
)
logger = structlog.get_logger("rtx_oom_guard.api")

# Prometheus Metrics
METRIC_OOM_RISK = Gauge(
    "rtx_oom_guard_oom_risk_score", 
    "Current OOM risk score predicted by rtx-oom-guard",
    ["tier"]
)
METRIC_VRAM_ALLOCATED = Gauge(
    "rtx_oom_guard_vram_allocated_bytes",
    "Current GPU memory allocated by the trainer"
)
METRIC_VRAM_RESERVED = Gauge(
    "rtx_oom_guard_vram_reserved_bytes",
    "Current GPU memory reserved by the CUDA caching allocator"
)
METRIC_COMPACTIONS = Counter(
    "rtx_oom_guard_compactions_total",
    "Total number of proactive compactions triggered"
)
METRIC_COMPACTION_LATENCY = Histogram(
    "rtx_oom_guard_compaction_latency_seconds",
    "Latency of GPU memory compaction events",
    buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5]
)
METRIC_REQUEST_LATENCY = Histogram(
    "rtx_oom_guard_api_request_duration_seconds",
    "API request latency",
    ["method", "endpoint", "status"]
)

# APP Initialization
app = FastAPI(
    title="rtx-oom-guard API",
    description="Production-grade GPU memory telemetry and OOM forecasting surface",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def track_metrics_and_logs(request: Request, call_next):
    start_time = time.perf_counter()
    response = await call_next(request)
    duration = time.perf_counter() - start_time
    
    METRIC_REQUEST_LATENCY.labels(
        method=request.method,
        endpoint=request.url.path,
        status=response.status_code
    ).observe(duration)
    
    logger.info(
        "api_request",
        method=request.method,
        path=request.url.path,
        status=response.status_code,
        duration_ms=round(duration * 1000, 2),
    )
    return response

# Shared State
_risk_model = OOMRiskModel(mode="rule")
_logger = AllocatorLogger()

# Schemas
class RiskRequest(BaseModel):
    fragmentation: float = Field(0.0, ge=0.0, le=1.0)
    utilisation: float = Field(0.0, ge=0.0, le=1.0)
    alloc_delta_mb: float = 0.0

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

# Internal Helpers
def _gpu_snapshot() -> Dict[str, Any]:
    try:
        import torch
        if torch.cuda.is_available():
            alloc = torch.cuda.memory_allocated() / (1024 ** 2)
            resv = torch.cuda.memory_reserved() / (1024 ** 2)
            METRIC_VRAM_ALLOCATED.set(alloc * 1024 * 1024)
            METRIC_VRAM_RESERVED.set(resv * 1024 * 1024)
            return {
                "current_allocated_mb": round(alloc, 2),
                "current_reserved_mb": round(resv, 2),
                "free_estimate_mb": round(resv - alloc, 2),
                "current_frag": round(1 - alloc / resv, 4) if resv > 0 else 0.0,
                "cuda_available": True,
            }
    except Exception:
        pass
    return {
        "current_allocated_mb": 0.0,
        "current_reserved_mb": 0.0,
        "free_estimate_mb": 0.0,
        "current_frag": 0.0,
        "cuda_available": False,
    }

# API Routes
api_router = APIRouter(prefix="/api")

@api_router.get("/health")
def health():
    return {"status": "ok", "service": "rtx-oom-guard-api", "version": "2.0.0"}

@api_router.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@api_router.get("/memory", response_model=MemoryResponse)
def get_memory():
    return _gpu_snapshot()

@api_router.post("/risk", response_model=RiskResponse)
def compute_risk(req: RiskRequest):
    score = _risk_model.score(
        fragmentation=req.fragmentation,
        utilisation=req.utilisation,
        alloc_delta_mb=req.alloc_delta_mb,
    )

    if score >= 0.8:
        tier, msg = "ACT", "High OOM risk — compaction prioritized"
    elif score >= 0.5:
        tier, msg = "WARN", "Elevated OOM risk — monitor scheduler"
    else:
        tier, msg = "SAFE", "Nominal operations"

    METRIC_OOM_RISK.labels(tier=tier).set(score)
    logger.info("risk_calculated", score=score, tier=tier, frag=req.fragmentation)
    
    return RiskResponse(risk_score=score, tier=tier, message=msg)

@api_router.get("/telemetry")
def get_full_telemetry():
    telemetry = _gpu_snapshot()
    telemetry.update({
        "total_compactions": 0,
        "total_freed_mb": 0.0,
        "avg_latency_ms": 0.0,
        "compaction_history": []
    })

    live_path = Path("results/live_telemetry.json")
    if live_path.exists():
        try:
            with open(live_path, "r") as f:
                live_data = json.load(f)
                telemetry.update({k: live_data.get(k, telemetry.get(k)) for k in telemetry.keys()})
                
                # Sync metrics from live telemetry
                METRIC_COMPACTIONS._value.set(live_data.get("total_compactions", 0))
                if "current_allocated_mb" in live_data:
                    METRIC_VRAM_ALLOCATED.set(live_data["current_allocated_mb"] * 1024 * 1024)
                    METRIC_VRAM_RESERVED.set(live_data.get("current_reserved_mb", 0) * 1024 * 1024)
        except Exception as e:
            logger.error("telemetry_load_failed", error=str(e))
            
    return telemetry

app.include_router(api_router)

# Static Dashboard Mount
DASHBOARD_PATH = Path(__file__).parent.parent.parent.parent / "dashboard" / "dist"

if DASHBOARD_PATH.exists():
    app.mount("/assets", StaticFiles(directory=DASHBOARD_PATH / "assets"), name="assets")

    @app.api_route("/{path_name:path}", methods=["GET"])
    def catch_all(path_name: str):
        file_path = DASHBOARD_PATH / path_name
        if file_path.is_file():
            return FileResponse(file_path)
        return FileResponse(DASHBOARD_PATH / "index.html")
else:
    logger.warning("dashboard_not_found", path=str(DASHBOARD_PATH))
