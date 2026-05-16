// ============================================================================
// dataLoader.js — Production Telemetry Data Layer
// ============================================================================
// RULE: This module returns NUMBERS, not strings. All formatting is the
// responsibility of the UI components at render time. This prevents
// TypeError crashes from calling .toFixed() on string values.
// ============================================================================

// Fetch live telemetry (Real-time polling from rtx_oom_guard backend)
// Fetch live telemetry (Real-time polling from rtx_oom_guard backend)
export const fetchLiveTelemetry = async () => {
  try {
    const response = await fetch('/api/telemetry');
    if (!response.ok) throw new Error('Live telemetry not found');
    const data = await response.json();
    
    // Validate required top-level fields exist before accessing
    if (data.current_allocated_mb == null || data.current_reserved_mb == null) {
      throw new Error('Malformed telemetry: missing required fields');
    }

    return {
      currentAllocated: Number(data.current_allocated_mb) || 0,
      currentReserved:  Number(data.current_reserved_mb) || 0,
      currentFrag:      Number(data.current_frag) * 100 || 0,
      totalCompactions: Number(data.total_compactions) || 0,
      totalFreed:       Number(data.total_freed_mb) || 0,
      avgTime:          Number(data.avg_latency_ms) || 0,
      history: Array.isArray(data.compaction_history)
        ? data.compaction_history.map((c, idx) => ({
            id:            c.compaction_id ?? c.id ?? idx,
            freed:         Number(c.freed_mb ?? c.recoveredMb) || 0,
            fragReduction: Number(c.frag_reduction) * 100 || 0,
            elapsedMs:     Number(c.elapsed_ms ?? c.elapsedMs) || 0,
            timestamp:     c.timestamp || `sweep_${idx}`
          }))
        : []
    };
  } catch (e) {
    // Silent fail — returns null so App.jsx falls back to local synthesis
    return null;
  }
};

// Fetch baseline/static benchmark stats
export const fetchBenchmarkStats = async () => {
  try {
    const response = await fetch('/api/benchmarks');
    if (!response.ok) throw new Error('Benchmark data not found');
    
    const data = await response.json();
    const base = data.baseline || {};
    const defrag = data.defrag || {};
    
    return {
      baseline: {
        avgTime:  Number(base.avg_iteration_time) || 0,
        peakMem:  Number(base.peak_memory_mb) || 0,
        chart:    Array.isArray(base.memory_snapshots)
          ? base.memory_snapshots.map(s => ({
              iteration: s.iteration,
              frag: Number(s.frag) * 100 || 0
            }))
          : []
      },
      defrag: {
        avgTime:  Number(defrag.avg_iteration_time) || 0,
        peakMem:  Number(defrag.peak_memory_mb) || 0,
        chart:    Array.isArray(defrag.memory_snapshots)
          ? defrag.memory_snapshots.map(s => ({
              iteration: s.iteration,
              frag: Number(s.frag) * 100 || 0
            }))
          : []
      }
    };
  } catch (e) {
    return null;
  }
};

// Fetch High-Fidelity Simulation Modeling Results
export const fetchSimulatedModeling = async () => {
  try {
    const response = await fetch('/results/simulated_modeling_values.json');
    if (!response.ok) throw new Error('Simulation modeling results not found');
    const data = await response.json();
    return data;
  } catch (e) {
    console.error("Simulation data load failed:", e);
    // Returns a fallback mock if results file is missing (for dev/demo)
    return {
       hardware_profile: "RTX_3060_TIGHT (FALLBACK)",
       baseline: { oom_rate: 85, runs_to_oom: 12, stable_batch_size: 2, avg_frag: 0.45 },
       workflow: { oom_rate: 0, runs_to_oom: 100, stable_batch_size: 4, avg_frag: 0.12 },
       metrics: { oom_reduction_pct: 100, util_gain_pp: 8.5 }
    };
  }
};
