import React from 'react';
import { AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';

export const IterationTimeChart = ({ data }) => {
  return (
    <ResponsiveContainer width="100%" height="100%">
      <AreaChart data={data} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
        <defs>
          <linearGradient id="colorBase" x1="0" y1="0" x2="0" y2="1">
            <stop offset="5%" stopColor="#ef4444" stopOpacity={0.1}/>
            <stop offset="95%" stopColor="#ef4444" stopOpacity={0}/>
          </linearGradient>
          <linearGradient id="colorDefrag" x1="0" y1="0" x2="0" y2="1">
            <stop offset="5%" stopColor="#3377ff" stopOpacity={0.1}/>
            <stop offset="95%" stopColor="#3377ff" stopOpacity={0}/>
          </linearGradient>
        </defs>
        <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.03)" vertical={false} />
        <XAxis 
          dataKey="iteration" 
          stroke="#4b4b4d" 
          tick={{fill: '#4b4b4d', fontSize: 10, fontFamily: 'JetBrains Mono'}}
          tickLine={false}
          axisLine={false}
        />
        <YAxis 
          stroke="#4b4b4d" 
          tick={{fill: '#4b4b4d', fontSize: 10, fontFamily: 'JetBrains Mono'}}
          tickLine={false}
          axisLine={false}
          tickFormatter={(value) => `${value}s`}
        />
        <Tooltip 
          contentStyle={{ backgroundColor: '#121214', border: '1px solid rgba(255,255,255,0.1)', borderRadius: '2px', fontFamily: 'JetBrains Mono', fontSize: '11px' }}
          itemStyle={{ color: '#fcfcfc' }}
          cursor={{stroke: 'rgba(255,255,255,0.1)'}}
        />
        
        <Area 
          type="monotone" 
          name="Baseline" 
          dataKey="baselineTime" 
          stroke="#ef4444" 
          strokeWidth={1}
          fillOpacity={1} 
          fill="url(#colorBase)" 
        />
        <Area 
          type="monotone" 
          name="rtx_oom_guard" 
          dataKey="defragTime" 
          stroke="#3377ff" 
          strokeWidth={2}
          fillOpacity={1} 
          fill="url(#colorDefrag)" 
        />
      </AreaChart>
    </ResponsiveContainer>
  );
};
