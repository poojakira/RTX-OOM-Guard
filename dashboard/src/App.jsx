import React, { useState, useEffect } from 'react';
import { Panel1_VramMap } from './components/Panel1_VramMap';
import { Panel2_GuardianCounter } from './components/Panel2_GuardianCounter';
import { Panel3_ShadowTimeline } from './components/Panel3_ShadowTimeline';
import { Panel4_AttentionHeatmap } from './components/Panel4_AttentionHeatmap';
import { Panel5_DdpChoreography } from './components/Panel5_DdpChoreography';
import { Panel6_TritonTrace } from './components/Panel6_TritonTrace';
import { Panel7_CompactionRay } from './components/Panel7_CompactionRay';
import { Panel8_TrendGraph } from './components/Panel8_TrendGraph';
import { Panel9_LatencyBarGraph } from './components/Panel9_LatencyBarGraph';
import { Panel10_CumulativeFreed } from './components/Panel10_CumulativeFreed';
import { Panel11_AllocationDist } from './components/Panel11_AllocationDist';
import { Panel12_SyncOverhead } from './components/Panel12_SyncOverhead';
import { Panel13_Benchmarking } from './components/Panel13_Benchmarking';
import { fetchLiveTelemetry, fetchSimulatedModeling } from './utils/dataLoader';
import { ErrorBoundary } from './components/ErrorBoundary';
import { Cpu, ShieldCheck, Server, History, AlignLeft, Layers, TerminalSquare, BarChart2 } from 'lucide-react';

// =============================================================================
// App.jsx — AeroGrid 6-Page Cinematic HUD Orchestrator
// =============================================================================
// STATE CONTRACT: Every value in liveState is a NUMBER (not a string).
// dataLoader.js guarantees numeric returns. Mock synthesis matches this.
// Components call .toFixed() at render time only.
// =============================================================================

function App() {
  const [activePage, setActivePage] = useState('mission_control');
  
  // ── Canonical Live State ──────────────────────────────────────────────────
  // Every field is a number. No strings. No nulls after first tick.
  const [liveState, setLiveState] = useState({
    fragPercent: 24.5,
    isCompacting: false,
    totalCompactions: 0,
    totalFreed: 0,
    avgTime: 0,
    lastCompactionTime: null,
    backendConnected: false,
    history: [],
    tickCount: 0,
    currentAllocated: 0,
    currentReserved: 0
  });

  const [benchmarkData, setBenchmarkData] = useState(null);

  const THRESHOLD = 80;

  useEffect(() => {
    let fallbackTickCount = 0;
    
    const interval = setInterval(() => {
      fetchLiveTelemetry().then(metrics => {
        setLiveState(prev => {
          // ── BACKEND CONNECTED: Real telemetry from rtx_oom_guard engine ──
          if (metrics) {
            const realFrag = metrics.currentFrag;
            const prevHistoryLen = prev.history ? prev.history.length : 0;
            const newHistoryLen = metrics.history ? metrics.history.length : 0;
            const isSweeping = newHistoryLen > prevHistoryLen || prev.isCompacting;

            let tick = isSweeping ? prev.tickCount + 1 : 0;
            let stillCompacting = isSweeping && tick < 10;

            return {
                ...prev,
                backendConnected: true,
                fragPercent: isNaN(realFrag) ? prev.fragPercent : realFrag,
                totalCompactions: metrics.totalCompactions || prev.totalCompactions,
                totalFreed: metrics.totalFreed || prev.totalFreed,
                avgTime: metrics.avgTime || prev.avgTime,
                isCompacting: stillCompacting,
                tickCount: stillCompacting ? tick : 0,
                history: metrics.history || [],
                currentAllocated: metrics.currentAllocated || prev.currentAllocated,
                currentReserved: metrics.currentReserved || prev.currentReserved,
                lastCompactionTime: (newHistoryLen > prevHistoryLen) ? new Date().toLocaleTimeString() : prev.lastCompactionTime
            };
          }

          // ── LOCAL SYNTHESIS: Simulated telemetry when backend is offline ──
          // Uses identical field names and numeric types as real telemetry.
          if (prev.isCompacting) {
              fallbackTickCount++;
              if (fallbackTickCount > 10) {
                 fallbackTickCount = 0;
                 const freedThisSweep = 64 + Math.random() * 128;
                 return {
                    ...prev,
                    isCompacting: false,
                    fragPercent: 12 + Math.random() * 5,
                    lastCompactionTime: new Date().toLocaleTimeString(),
                    totalCompactions: prev.totalCompactions + 1,
                    totalFreed: prev.totalFreed + freedThisSweep,
                    avgTime: 0.15 + Math.random() * 0.05
                 };
              }
              return prev;
          }

          const newFrag = prev.fragPercent + (Math.random() * 2.5);
          if (newFrag > THRESHOLD) {
              fallbackTickCount = 0;
              return {
                 ...prev,
                 backendConnected: false,
                 isCompacting: true,
                 fragPercent: newFrag,
                 currentAllocated: 6000 + Math.random() * 2000,
                 currentReserved: 12000
              };
          }
          return { ...prev, backendConnected: false, fragPercent: newFrag };
        });
      });
    }, 200);

    return () => clearInterval(interval);
  }, []);

  useEffect(() => {
    fetchSimulatedModeling().then(data => setBenchmarkData(data));
  }, []);

  const renderContent = () => {
     switch(activePage) {
        case 'mission_control':
            return (
                <div className="flex flex-col gap-8 h-full overflow-y-auto pr-4">
                    <div className="min-h-[400px]">
                        <Panel2_GuardianCounter totalPrevented={liveState.totalCompactions} msRecovered={liveState.totalFreed} />
                    </div>
                    <div className="min-h-[400px]">
                        <Panel10_CumulativeFreed totalFreedMB={liveState.totalFreed} history={liveState.history} />
                    </div>
                </div>
            );
        case 'vram_topology':
            return (
                <div className="flex flex-col gap-8 h-full">
                    <div className="flex-[0.7]">
                        <Panel1_VramMap fragPercent={liveState.fragPercent} isCompacting={liveState.isCompacting} />
                    </div>
                    <div className="flex-[0.3]">
                        <Panel11_AllocationDist fragPercent={liveState.fragPercent} />
                    </div>
                </div>
            );
        case 'shadow_timeline':
            return (
                <div className="flex flex-col gap-8 h-full">
                   <div className="flex-1">
                      <Panel3_ShadowTimeline currentFrag={liveState.fragPercent} thresholdLevel={THRESHOLD} />
                   </div>
                   <div className="flex-1">
                      <Panel8_TrendGraph currentFrag={liveState.fragPercent} thresholdLevel={THRESHOLD} history={liveState.history} />
                   </div>
                </div>
            );
        case 'attention_matrix':
            return (
                <div className="h-full">
                    <Panel4_AttentionHeatmap />
                </div>
            );
        case 'ddp_choreography':
            return (
                <div className="flex flex-col gap-8 h-full overflow-y-auto pr-4">
                   <div className="min-h-[400px]">
                      <Panel5_DdpChoreography isCompacting={liveState.isCompacting} />
                   </div>
                   <div className="min-h-[450px]">
                      <Panel12_SyncOverhead isCompacting={liveState.isCompacting} />
                   </div>
                </div>
            );
        case 'triton_inspector':
            return (
                <div className="flex flex-col gap-8 h-full">
                   <div className="h-48">
                      <Panel7_CompactionRay isCompacting={liveState.isCompacting} />
                   </div>
                   <div className="flex-1 flex gap-8">
                      <div className="flex-1">
                          <Panel6_TritonTrace isCompacting={liveState.isCompacting} lastCompactionTime={liveState.lastCompactionTime} />
                      </div>
                      <div className="flex-1">
                          <Panel9_LatencyBarGraph history={liveState.history} />
                      </div>
                   </div>
                </div>
            );
        case 'model_benchmarks':
            return (
                <div className="h-full">
                    <Panel13_Benchmarking data={benchmarkData} />
                </div>
            );
        default: return null;
     }
  };

  return (
    <>
      <div className={`absolute top-0 left-0 right-0 h-1 z-50 ${liveState.backendConnected ? 'bg-green' : 'bg-amber'}`} />
      
      <aside className="sidebar">
        <div className="logo-section">
          <div className="logo-icon">
             <Cpu size={18} color="#000" />
          </div>
          <div>
              <span className="text-xl font-bold tracking-tight text-white block">AeroGrid</span>
              <span className="text-[9px] text-green mono-metric uppercase font-bold tracking-widest block mt-1">GPU Defrag Engine v2.0.0</span>
          </div>
        </div>

        <nav className="flex-1">
           <div className="text-[10px] text-dim font-bold uppercase tracking-widest mb-4 ml-1">Core Metrics</div>
           
           <button onClick={() => setActivePage('mission_control')} className={`nav-btn ${activePage === 'mission_control' ? 'active' : ''}`}>
              <ShieldCheck size={16} /> Mission Control
           </button>
           
           <button onClick={() => setActivePage('vram_topology')} className={`nav-btn ${activePage === 'vram_topology' ? 'active' : ''}`}>
              <AlignLeft size={16} /> VRAM Topology
           </button>
           
           <button onClick={() => setActivePage('shadow_timeline')} className={`nav-btn ${activePage === 'shadow_timeline' ? 'active' : ''}`}>
              <History size={16} /> Shadow Forecast
           </button>
           
           <div className="mt-8 mb-4 text-[10px] text-dim font-bold uppercase tracking-widest ml-1">Deep Inspection</div>
           
           <button onClick={() => setActivePage('attention_matrix')} className={`nav-btn ${activePage === 'attention_matrix' ? 'active' : ''}`}>
               <Layers size={16} /> Scheduler Attention
           </button>

           <button onClick={() => setActivePage('ddp_choreography')} className={`nav-btn ${activePage === 'ddp_choreography' ? 'active' : ''}`}>
               <Server size={16} /> DDP Choreography
           </button>
           
           <button onClick={() => setActivePage('triton_inspector')} className={`nav-btn ${activePage === 'triton_inspector' ? 'active' : ''}`}>
               <TerminalSquare size={16} /> Triton Inspector
           </button>

           <div className="mt-8 mb-4 text-[10px] text-dim font-bold uppercase tracking-widest ml-1">Projections</div>

           <button onClick={() => setActivePage('model_benchmarks')} className={`nav-btn ${activePage === 'model_benchmarks' ? 'active' : ''}`}>
               <BarChart2 size={16} /> Modeling Benchmarks
           </button>
        </nav>
        
        <div className="mt-auto border-t border-glass-border pt-4">
            <div className={`flex items-center gap-2 text-[10px] font-bold uppercase mono-metric ${liveState.backendConnected ? 'text-green' : 'text-amber blink'}`}>
               <div className={`w-2 h-2 rounded-full ${liveState.backendConnected ? 'bg-green' : 'bg-amber'}`}></div>
               {liveState.backendConnected ? 'TELEMETRY LINK: OK' : 'LOCAL SYNTHESIS: ACTIVE'}
            </div>
            <div className="text-[9px] text-dim mono-metric mt-2 ml-4">POLLING_RATE: 200ms</div>
        </div>
      </aside>

      <main className="main-content">
        <header className="top-nav">
           <h2 className="text-xl font-medium tracking-wide">
              {activePage.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase())}
           </h2>
           <div className="text-[10px] uppercase font-bold tracking-widest text-dim flex gap-4">
              <span className="mono-metric">CUDA_VISIBLE_DEVICES="0,1,2,3"</span>
              <span className="text-white">NODE_001</span>
           </div>
        </header>

        <section className="page-container">
           <ErrorBoundary>
             {renderContent()}
           </ErrorBoundary>
        </section>
      </main>
    </>
  );
}

export default App;
