import { mockPortfolioHistory } from '../utils/mockData';
import {
  AreaChart, Area, LineChart, Line,
  XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid,
  ComposedChart, Bar,
} from 'recharts';

const tooltipStyle = {
  background: '#161b22',
  border: '1px solid rgba(88,166,255,0.15)',
  borderRadius: 10,
  fontSize: 12,
  color: '#f0f6fc',
};

export default function PortfolioPage() {
  const data = mockPortfolioHistory;

  // Compute drawdown from the HRL curve
  let peak = data[0].hrl;
  const drawdownData = data.map(d => {
    if (d.hrl > peak) peak = d.hrl;
    const dd = ((peak - d.hrl) / peak) * 100;
    return { tick: d.tick, drawdown: -dd };
  });

  // Compute per-tick returns
  const returnData = data.slice(1).map((d, i) => ({
    tick: d.tick,
    ret: ((d.hrl - data[i].hrl) / data[i].hrl) * 100,
  }));

  return (
    <div>
      <div className="page-header">
        <h1>Portfolio Analytics</h1>
        <p>Detailed PnL tracking, drawdown analysis, and return distribution</p>
      </div>

      {/* Equity Curve - Full Width */}
      <div className="glass-card" style={{ marginBottom: 18 }}>
        <div className="card-header">
          <span className="card-title">Equity Curve — All Strategies</span>
        </div>
        <div className="chart-container" style={{ height: 380 }}>
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={data.filter((_, i) => i % 3 === 0)}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(48,54,61,0.5)" />
              <XAxis dataKey="tick" stroke="#484f58" tickLine={false} fontSize={11} />
              <YAxis
                stroke="#484f58"
                tickLine={false}
                fontSize={11}
                tickFormatter={v => `$${(v / 1000).toFixed(0)}k`}
              />
              <Tooltip contentStyle={tooltipStyle} formatter={(v) => [`$${v.toFixed(0)}`]} labelFormatter={v => `Tick ${v}`} />
              <Line type="monotone" dataKey="hrl" stroke="#58a6ff" strokeWidth={2.5} dot={false} name="HRL Agent" />
              <Line type="monotone" dataKey="macd" stroke="#d29922" strokeWidth={1.5} dot={false} name="MACD" strokeDasharray="4 2" />
              <Line type="monotone" dataKey="bollinger" stroke="#f85149" strokeWidth={1.5} dot={false} name="Bollinger" strokeDasharray="4 2" />
              <Line type="monotone" dataKey="lstm" stroke="#bc8cff" strokeWidth={1.5} dot={false} name="LSTM" strokeDasharray="4 2" />
            </LineChart>
          </ResponsiveContainer>
        </div>
        <div style={{ display: 'flex', gap: 24, justifyContent: 'center', marginTop: 12 }}>
          {[
            { name: 'HRL Agent', color: '#58a6ff' },
            { name: 'MACD', color: '#d29922' },
            { name: 'Bollinger', color: '#f85149' },
            { name: 'LSTM', color: '#bc8cff' },
          ].map(item => (
            <span key={item.name} style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: '0.82rem', color: 'var(--text-tertiary)' }}>
              <span style={{ width: 12, height: 3, borderRadius: 2, background: item.color, display: 'inline-block' }} />
              {item.name}
            </span>
          ))}
        </div>
      </div>

      <div className="chart-grid">
        {/* Drawdown Chart */}
        <div className="glass-card">
          <div className="card-header">
            <span className="card-title">Drawdown Analysis</span>
          </div>
          <div className="chart-container">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={drawdownData.filter((_, i) => i % 3 === 0)}>
                <defs>
                  <linearGradient id="gradientRed" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor="#f85149" stopOpacity={0.4} />
                    <stop offset="100%" stopColor="#f85149" stopOpacity={0} />
                  </linearGradient>
                </defs>
                <XAxis dataKey="tick" stroke="#484f58" tickLine={false} fontSize={11} />
                <YAxis stroke="#484f58" tickLine={false} fontSize={11} tickFormatter={v => `${v.toFixed(1)}%`} />
                <Tooltip contentStyle={tooltipStyle} formatter={(v) => [`${v.toFixed(2)}%`, 'Drawdown']} labelFormatter={v => `Tick ${v}`} />
                <Area type="monotone" dataKey="drawdown" stroke="#f85149" strokeWidth={1.5} fill="url(#gradientRed)" dot={false} />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Return Distribution */}
        <div className="glass-card">
          <div className="card-header">
            <span className="card-title">Per-Tick Return Distribution</span>
          </div>
          <div className="chart-container">
            <ResponsiveContainer width="100%" height="100%">
              <ComposedChart data={returnData.filter((_, i) => i % 4 === 0)}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(48,54,61,0.4)" />
                <XAxis dataKey="tick" stroke="#484f58" tickLine={false} fontSize={11} />
                <YAxis stroke="#484f58" tickLine={false} fontSize={11} tickFormatter={v => `${v.toFixed(2)}%`} />
                <Tooltip contentStyle={tooltipStyle} formatter={(v) => [`${v.toFixed(3)}%`, 'Return']} labelFormatter={v => `Tick ${v}`} />
                <Bar dataKey="ret" fill="#58a6ff" fillOpacity={0.6} radius={[2, 2, 0, 0]} />
              </ComposedChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>
    </div>
  );
}
