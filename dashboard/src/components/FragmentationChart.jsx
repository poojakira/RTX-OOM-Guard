import React from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, ReferenceLine } from 'recharts';

export const FragmentationChart = ({ data, timeline }) => {
  return (
    <ResponsiveContainer width="100%" height="100%">
      <LineChart data={data} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
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
          domain={[0, 100]}
          tickFormatter={(value) => `${value}%`}
        />
        <Tooltip 
          contentStyle={{ backgroundColor: '#121214', border: '1px solid rgba(255,255,255,0.1)', borderRadius: '2px', fontFamily: 'JetBrains Mono', fontSize: '11px' }}
          itemStyle={{ color: '#fcfcfc' }}
          cursor={{stroke: 'rgba(255,255,255,0.1)'}}
        />
        
        <Line 
          type="stepAfter" 
          name="Baseline" 
          dataKey="baselineFrag" 
          stroke="#ef4444" 
          strokeWidth={1}
          dot={false}
          activeDot={{ r: 4, stroke: '#ef4444', strokeWidth: 2, fill: '#000' }} 
        />
        <Line 
          type="stepAfter" 
          name="rtx_oom_guard" 
          dataKey="defragFrag" 
          stroke="#00ff88" 
          strokeWidth={2}
          dot={false}
          activeDot={{ r: 4, stroke: '#00ff88', strokeWidth: 2, fill: '#000' }} 
        />
        
        {timeline.map((event) => (
           <ReferenceLine 
             key={event.id}
             x={event.id * Math.floor(data.length / timeline.length)}
             stroke="rgba(0, 204, 255, 0.2)" 
             strokeDasharray="4 4"
             label={{ position: 'top', fill: '#00ccff', fontSize: 9, value: `CPT-${event.id}`, fontFamily: 'JetBrains Mono' }}
           />
        ))}
      </LineChart>
    </ResponsiveContainer>
  );
};
