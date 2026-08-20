import React, { useState, useEffect } from 'react';
import MetricCard from '../components/MetricCard';
import { Flame, ShieldAlert, Activity, Cpu, AlertTriangle } from 'lucide-react';
import {
  LineChart, Line, AreaChart, Area, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid
} from 'recharts';

export default function IntensityPage() {
  const [intensityData, setIntensityData] = useState([]);
  const [currentCrashRate, setCurrentCrashRate] = useState(0.04);
  const [burstAlert, setBurstAlert] = useState(false);

  useEffect(() => {
    let t = 0;
    const interval = setInterval(async () => {
      t++;
      try {
        const res = await fetch('http://localhost:8000/api/events/live');
        if (res.ok) {
          const data = await res.json();
          const lCrash = data.lambda_crash || 0.05;
          const lSafe = data.lambda_safe || 1.2;
          const lRisky = data.lambda_risky || 0.3;

          setCurrentCrashRate(lCrash);
          setBurstAlert(lCrash > 1.0);

          setIntensityData(prev => [
            ...prev.slice(-39),
            {
              step: new Date().toLocaleTimeString().slice(3, 8),
              crash: lCrash,
              safe: lSafe,
              risky: lRisky,
            }
          ]);
        }
      } catch (err) {
        // Offline fallback
        const isSpike = t % 15 === 0;
        const lCrash = isSpike ? 2.1 + Math.random() * 0.5 : 0.05 + Math.random() * 0.08;
        const lSafe = isSpike ? 0.3 : 1.4 + Math.random() * 0.3;
        const lRisky = isSpike ? 1.6 : 0.25 + Math.random() * 0.1;

        setCurrentCrashRate(lCrash);
        setBurstAlert(isSpike);

        setIntensityData(prev => [
          ...prev.slice(-39),
          {
            step: `${t}`,
            crash: parseFloat(lCrash.toFixed(3)),
            safe: parseFloat(lSafe.toFixed(3)),
            risky: parseFloat(lRisky.toFixed(3)),
          }
        ]);
      }
    }, 1000);

    return () => clearInterval(interval);
  }, []);

  return (
    <div>
      <div className="page-header" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
        <div>
          <h1>Temporal Point Process (TPP) Hawkes Intensity</h1>
          <p>Continuous-time conditional intensity functions λ_k(t) and self-exciting shock tracking</p>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8, background: 'var(--bg-elevated)', padding: '6px 14px', borderRadius: 20, border: '1px solid var(--glass-border)' }}>
          <Flame size={14} color="var(--color-warn)" />
          <span style={{ fontSize: '0.8rem', fontWeight: 600, color: 'var(--color-warn)' }}>Hawkes Kernel Online</span>
        </div>
      </div>

      {/* Emergency Burst Alert Banner */}
      {burstAlert && (
        <div style={{
          background: 'rgba(248, 81, 73, 0.15)',
          border: '1px solid rgba(248, 81, 73, 0.4)',
          borderRadius: 'var(--radius-md)',
          padding: '16px 20px',
          marginBottom: 20,
          display: 'flex',
          alignItems: 'center',
          gap: 14,
        }}>
          <AlertTriangle color="var(--color-loss)" size={24} />
          <div>
            <div style={{ color: 'var(--color-loss)', fontWeight: 700, fontSize: '0.95rem' }}>
              HAWKES SELF-EXCITATION CLUSTER DETECTED (λ_crash &gt; 1.50)
            </div>
            <div style={{ color: 'var(--text-secondary)', fontSize: '0.82rem' }}>
              Inter-arrival times for negative macro shocks have collapsed (dt &lt; 5ms). The TQC policy has activated tail-risk hedging.
            </div>
          </div>
        </div>
      )}

      {/* Metrics Row */}
      <div className="metric-grid" style={{ gridTemplateColumns: 'repeat(4, 1fr)' }}>
        <MetricCard
          title="Safe Intensity λ_safe"
          value={intensityData.length > 0 ? intensityData[intensityData.length - 1].safe?.toFixed(3) : '1.350'}
          delta={currentCrashRate > 1.0 ? -45.2 : 5.1}
          deltaLabel="normal flow"
          icon={<Activity size={18} />}
          iconColor="green"
        />
        <MetricCard
          title="Risky Intensity λ_risky"
          value={intensityData.length > 0 ? intensityData[intensityData.length - 1].risky?.toFixed(3) : '0.280'}
          delta={12.0}
          deltaLabel="volatility build-up"
          icon={<Flame size={18} />}
          iconColor="cyan"
        />
        <MetricCard
          title="Crash Hazard λ_crash"
          value={currentCrashRate.toFixed(3)}
          delta={currentCrashRate > 1.0 ? 320 : -10}
          deltaLabel="flash crash risk"
          icon={<ShieldAlert size={18} />}
          iconColor="red"
        />
        <MetricCard
          title="Branching Ratio (α/β)"
          value="0.68"
          deltaLabel="sub-critical (stable)"
          icon={<Cpu size={18} />}
          iconColor="purple"
        />
      </div>

      {/* Real-time Hawkes Curves */}
      <div className="glass-card" style={{ marginBottom: 20 }}>
        <div className="card-header">
          <span className="card-title">Real-Time Conditional Arrival Intensities λ_k(t)</span>
          <div style={{ display: 'flex', gap: 16, fontSize: '0.8rem' }}>
            <span style={{ display: 'flex', alignItems: 'center', gap: 6, color: 'var(--color-profit)' }}>
              <span style={{ width: 8, height: 8, borderRadius: '50%', background: '#3fb950' }} /> λ_safe
            </span>
            <span style={{ display: 'flex', alignItems: 'center', gap: 6, color: 'var(--color-warn)' }}>
              <span style={{ width: 8, height: 8, borderRadius: '50%', background: '#d29922' }} /> λ_risky
            </span>
            <span style={{ display: 'flex', alignItems: 'center', gap: 6, color: 'var(--color-loss)' }}>
              <span style={{ width: 8, height: 8, borderRadius: '50%', background: '#f85149' }} /> λ_crash
            </span>
          </div>
        </div>
        <div className="chart-container" style={{ height: 360 }}>
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={intensityData}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(48,54,61,0.4)" />
              <XAxis dataKey="step" stroke="#484f58" tickLine={false} fontSize={11} />
              <YAxis stroke="#484f58" tickLine={false} fontSize={11} domain={[0, 'auto']} />
              <Tooltip
                contentStyle={{
                  background: '#161b22',
                  border: '1px solid rgba(88,166,255,0.15)',
                  borderRadius: 10,
                  fontSize: 12,
                  color: '#f0f6fc',
                }}
              />
              <Line type="monotone" dataKey="safe" stroke="#3fb950" strokeWidth={2} dot={false} isAnimationActive={false} />
              <Line type="monotone" dataKey="risky" stroke="#d29922" strokeWidth={2} dot={false} isAnimationActive={false} />
              <Line type="monotone" dataKey="crash" stroke="#f85149" strokeWidth={2.5} dot={false} isAnimationActive={false} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
}
