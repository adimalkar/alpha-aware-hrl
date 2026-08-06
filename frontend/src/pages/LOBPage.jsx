import { useMemo, useState } from 'react';
import { mockLOBDepth } from '../utils/mockData';

function interpolateColor(value, max) {
  const ratio = Math.min(value / max, 1);
  if (ratio < 0.33) {
    const t = ratio / 0.33;
    return `rgba(88, 166, 255, ${0.05 + t * 0.25})`;
  } else if (ratio < 0.66) {
    const t = (ratio - 0.33) / 0.33;
    return `rgba(57, 210, 192, ${0.3 + t * 0.35})`;
  } else {
    const t = (ratio - 0.66) / 0.34;
    return `rgba(63, 185, 80, ${0.5 + t * 0.5})`;
  }
}

export default function LOBPage() {
  const [view, setView] = useState('bid');
  const levels = 20;
  const ticks = 60;

  const maxVol = useMemo(() => {
    return Math.max(...mockLOBDepth.map(d => view === 'bid' ? d.bidVolume : d.askVolume));
  }, [view]);

  return (
    <div>
      <div className="page-header">
        <h1>Limit Order Book Heatmap</h1>
        <p>Visualize bid/ask depth across 20 price levels over 60 ticks from FI-2010</p>
      </div>

      <div className="glass-card">
        <div className="card-header">
          <span className="card-title">LOB Depth — {view === 'bid' ? 'Bid Side' : 'Ask Side'}</span>
          <div style={{ display: 'flex', gap: 8 }}>
            <button
              className={`btn ${view === 'bid' ? 'btn-primary' : 'btn-secondary'}`}
              style={{ padding: '6px 16px', fontSize: '0.8rem' }}
              onClick={() => setView('bid')}
            >
              Bid
            </button>
            <button
              className={`btn ${view === 'ask' ? 'btn-primary' : 'btn-secondary'}`}
              style={{ padding: '6px 16px', fontSize: '0.8rem' }}
              onClick={() => setView('ask')}
            >
              Ask
            </button>
          </div>
        </div>

        {/* Heatmap Grid */}
        <div style={{
          display: 'grid',
          gridTemplateColumns: `40px repeat(${ticks}, 1fr)`,
          gridTemplateRows: `repeat(${levels}, 20px) 28px`,
          gap: 1,
          borderRadius: 'var(--radius-md)',
          overflow: 'hidden',
        }}>
          {/* Y-axis labels + cells */}
          {Array.from({ length: levels }, (_, level) => {
            const rowCells = mockLOBDepth
              .filter(d => d.level === level)
              .sort((a, b) => a.tick - b.tick);

            return [
              <div
                key={`label-${level}`}
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'flex-end',
                  paddingRight: 6,
                  fontSize: '0.68rem',
                  color: 'var(--text-muted)',
                  fontFamily: 'var(--font-mono)',
                }}
              >
                L{level}
              </div>,
              ...rowCells.map(d => {
                const vol = view === 'bid' ? d.bidVolume : d.askVolume;
                return (
                  <div
                    key={`${d.level}-${d.tick}`}
                    className="heatmap-cell"
                    style={{
                      background: interpolateColor(vol, maxVol),
                      borderRadius: 1,
                    }}
                    title={`Level ${d.level}, Tick ${d.tick}: ${vol.toFixed(0)}`}
                  />
                );
              }),
            ];
          })}

          {/* X-axis labels */}
          <div />
          {Array.from({ length: ticks }, (_, t) => (
            <div
              key={`xtick-${t}`}
              style={{
                fontSize: '0.6rem',
                color: 'var(--text-muted)',
                textAlign: 'center',
                fontFamily: 'var(--font-mono)',
                paddingTop: 4,
              }}
            >
              {t % 10 === 0 ? t : ''}
            </div>
          ))}
        </div>

        {/* Legend */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginTop: 16, justifyContent: 'center' }}>
          <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Low Volume</span>
          <div style={{
            width: 200,
            height: 10,
            borderRadius: 5,
            background: 'linear-gradient(90deg, rgba(88,166,255,0.05), rgba(57,210,192,0.5), rgba(63,185,80,1))',
          }} />
          <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>High Volume</span>
        </div>
      </div>
    </div>
  );
}
