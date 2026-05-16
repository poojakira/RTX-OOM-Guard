import React, { useMemo } from 'react';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from 'recharts';

export const Panel9_LatencyBarGraph = ({ history }) => {
    // Map the actual history trace from rtx_oom_guard if available, else blank
    const data = useMemo(() => {
        if (!history || history.length === 0) {
            return Array.from({length: 10}).map((_,i) => ({ id: `SWEEP_${i}`, latency: 0 }));
        }
        
        return history.map((h, i) => ({
            id: `SWP_${h.id || i}`,
            latency: parseFloat(h.elapsedMs) || 0
        })).slice(-15); // Show last 15 sweeps
    }, [history]);

    return (
        <div className="hw-panel h-full w-full">
            <div className="hw-panel-header">
                <span className="panel-title">09/TRITON_LATENCY_PROFILE</span>
                <span className="text-amber">EXECUTION_MS</span>
            </div>
            
            <div className="flex-1 w-full h-full mt-4 font-mono text-[9px]">
                <ResponsiveContainer width="100%" height="85%">
                    <BarChart data={data} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
                        <XAxis dataKey="id" stroke="var(--text-dim)" tick={{fill: 'var(--text-dim)'}} tickLine={false} axisLine={false} />
                        <YAxis stroke="var(--text-dim)" tick={{fill: 'var(--text-dim)'}} tickLine={false} axisLine={false} />
                        <Tooltip 
                            cursor={{fill: 'rgba(255,255,255,0.05)'}}
                            contentStyle={{ backgroundColor: 'var(--bg-card)', border: '1px solid var(--glass-border)', fontFamily: 'JetBrains Mono, monospace', fontSize: '10px' }}
                            itemStyle={{ color: 'var(--hw-amber)' }}
                        />
                        <Bar dataKey="latency">
                            {data.map((entry, index) => (
                                <Cell key={`cell-${index}`} fill={entry.latency > 10 ? 'var(--hw-red)' : 'var(--hw-amber)'} />
                            ))}
                        </Bar>
                    </BarChart>
                </ResponsiveContainer>
            </div>
        </div>
    );
};
