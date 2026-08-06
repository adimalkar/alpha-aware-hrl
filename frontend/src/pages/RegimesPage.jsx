import { useState } from 'react';
import { mockRegimes } from '../utils/mockData';
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell,
} from 'recharts';

const REGIME_COLORS = ['#3fb950', '#d29922', '#f85149'];
const REGIME_LABELS = ['Safe', 'Risky', 'Crash'];

const tooltipStyle = {
  background: '#161b22',
  border: '1px solid rgba(88,166,255,0.15)',
  borderRadius: 10,
  fontSize: 12,
  color: '#f0f6fc',
};

export default function RegimesPage() {
  const [selected, setSelected] = useState(null);

  // Confidence chart data
  const confData = mockRegimes.map(r => ({
    chunk: r.chunkIdx,
    confidence: r.confidence,
    regime: r.regime,
  }));

  return (
    <div>
      <div className="page-header">
        <h1>LLM Regime Signals</h1>
        <p>TinyLlama-1.1B regime classifications from real FNSPID financial news</p>
      </div>

      {/* Regime Timeline */}
      <div className="glass-card" style={{ marginBottom: 18 }}>
        <div className="card-header">
          <span className="card-title">Regime Timeline — 362,400 Ticks</span>
          <div style={{ display: 'flex', gap: 16 }}>
            {REGIME_LABELS.map((label, i) => (
              <span key={label} style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: '0.8rem', color: 'var(--text-tertiary)' }}>
                <span style={{ width: 10, height: 10, borderRadius: 3, background: REGIME_COLORS[i], display: 'inline-block' }} />
                {label}
              </span>
            ))}
          </div>
        </div>
        <div className="regime-timeline" style={{ height: 48, borderRadius: 8 }}>
          {mockRegimes.map((r, i) => (
            <div
              key={i}
              className={`regime-block ${REGIME_LABELS[r.regime].toLowerCase()}`}
              style={{ flex: 1, cursor: 'pointer' }}
              onClick={() => setSelected(r)}
              title={`Chunk ${r.chunkIdx}: ${REGIME_LABELS[r.regime]} (${(r.confidence * 100).toFixed(0)}%)`}
            />
          ))}
        </div>
        <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: 8, fontSize: '0.75rem', color: 'var(--text-muted)' }}>
          <span>Tick 0</span>
          <span>Tick 180,000</span>
          <span>Tick 362,400</span>
        </div>
      </div>

      <div className="chart-grid">
        {/* Confidence Chart */}
        <div className="glass-card">
          <div className="card-header">
            <span className="card-title">LLM Confidence per Chunk</span>
          </div>
          <div className="chart-container">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={confData}>
                <XAxis dataKey="chunk" stroke="#484f58" tickLine={false} fontSize={11} />
                <YAxis stroke="#484f58" tickLine={false} fontSize={11} domain={[0, 1]} tickFormatter={v => `${(v * 100).toFixed(0)}%`} />
                <Tooltip contentStyle={tooltipStyle} formatter={(v) => [`${(v * 100).toFixed(1)}%`, 'Confidence']} labelFormatter={v => `Chunk ${v}`} />
                <Bar dataKey="confidence" radius={[3, 3, 0, 0]} animationDuration={800}>
                  {confData.map((entry, idx) => (
                    <Cell key={idx} fill={REGIME_COLORS[entry.regime]} fillOpacity={0.7} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Selected Detail Card */}
        <div className="glass-card">
          <div className="card-header">
            <span className="card-title">Selected Chunk Detail</span>
          </div>
          {selected ? (
            <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
                <span className={`badge ${REGIME_LABELS[selected.regime].toLowerCase()}`} style={{ fontSize: '0.85rem', padding: '6px 14px' }}>
                  {REGIME_LABELS[selected.regime]}
                </span>
                <span className="mono" style={{ color: 'var(--text-tertiary)', fontSize: '0.82rem' }}>
                  Chunk #{selected.chunkIdx}
                </span>
              </div>

              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12 }}>
                <div className="glass-card" style={{ padding: 16 }}>
                  <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.06em', marginBottom: 4 }}>Tick Range</div>
                  <div className="mono" style={{ color: 'var(--text-primary)', fontWeight: 600 }}>
                    {selected.tickStart.toLocaleString()} → {selected.tickEnd.toLocaleString()}
                  </div>
                </div>
                <div className="glass-card" style={{ padding: 16 }}>
                  <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.06em', marginBottom: 4 }}>Confidence</div>
                  <div style={{ color: 'var(--text-primary)', fontWeight: 700, fontSize: '1.4rem' }}>
                    {(selected.confidence * 100).toFixed(1)}%
                  </div>
                </div>
              </div>

              <div>
                <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.06em', marginBottom: 8 }}>News Context</div>
                <div style={{
                  background: 'var(--bg-primary)',
                  padding: 16,
                  borderRadius: 'var(--radius-md)',
                  border: '1px solid var(--glass-border)',
                  fontSize: '0.88rem',
                  lineHeight: 1.7,
                  color: 'var(--text-secondary)',
                  fontStyle: 'italic',
                }}>
                  "{selected.newsSnippet}"
                </div>
              </div>
            </div>
          ) : (
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: 200, color: 'var(--text-muted)', fontSize: '0.9rem' }}>
              Click a block in the timeline above to inspect details
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
