import { mockBaselines, mockAblations } from '../utils/mockData';
import {
  RadarChart, Radar, PolarGrid, PolarAngleAxis, PolarRadiusAxis,
  ResponsiveContainer, Tooltip,
  BarChart, Bar, XAxis, YAxis, Cell, CartesianGrid,
} from 'recharts';

const tooltipStyle = {
  background: '#161b22',
  border: '1px solid rgba(88,166,255,0.15)',
  borderRadius: 10,
  fontSize: 12,
  color: '#f0f6fc',
};

const ABLATION_COLORS = ['#58a6ff', '#bc8cff', '#f85149', '#39d2c0', '#d29922'];

export default function BaselinesPage() {
  // Normalize metrics for radar chart (0-100 scale)
  const radarData = [
    { metric: 'Sharpe', ...Object.fromEntries(mockBaselines.map(b => [b.name, Math.max(0, (b.sharpe + 6) * 12)])) },
    { metric: 'Return', ...Object.fromEntries(mockBaselines.map(b => [b.name, Math.max(0, (b.returnPct + 70) * 0.6)])) },
    { metric: 'Win Rate', ...Object.fromEntries(mockBaselines.map(b => [b.name, 30 + Math.random() * 40])) },
    { metric: 'Low Drawdown', ...Object.fromEntries(mockBaselines.map(b => [b.name, Math.max(0, 100 - b.maxDrawdown * 0.01)])) },
    { metric: 'Low VaR', ...Object.fromEntries(mockBaselines.map(b => [b.name, Math.max(0, 100 - b.var95 * 500)])) },
  ];

  return (
    <div>
      <div className="page-header">
        <h1>Baseline Comparisons</h1>
        <p>Side-by-side comparison of HRL agent vs. traditional financial strategies and ablations</p>
      </div>

      {/* Comparison Table */}
      <div className="glass-card" style={{ marginBottom: 18 }}>
        <div className="card-header">
          <span className="card-title">Strategy Performance Table</span>
        </div>
        <table className="data-table">
          <thead>
            <tr>
              <th>Strategy</th>
              <th>Final Portfolio</th>
              <th>Return %</th>
              <th>Sharpe Ratio</th>
              <th>Max Drawdown %</th>
              <th>VaR (95%)</th>
              <th>CVaR (95%)</th>
            </tr>
          </thead>
          <tbody>
            {mockBaselines.map((b, i) => (
              <tr key={b.name} style={i === 0 ? { background: 'var(--accent-blue-glow)' } : {}}>
                <td style={{ fontWeight: i === 0 ? 700 : 400, color: i === 0 ? 'var(--accent-blue)' : 'var(--text-secondary)' }}>
                  {b.name}
                </td>
                <td className="mono">${b.finalPortfolio.toLocaleString()}</td>
                <td>
                  <span style={{ color: b.returnPct >= 0 ? 'var(--color-profit)' : 'var(--color-loss)', fontWeight: 600 }}>
                    {b.returnPct >= 0 ? '+' : ''}{b.returnPct.toFixed(2)}%
                  </span>
                </td>
                <td className="mono" style={{ color: b.sharpe >= 0 ? 'var(--color-profit)' : 'var(--color-loss)' }}>
                  {b.sharpe.toFixed(2)}
                </td>
                <td className="mono" style={{ color: 'var(--color-loss)' }}>{b.maxDrawdown.toFixed(1)}%</td>
                <td className="mono">{(b.var95 * 100).toFixed(2)}%</td>
                <td className="mono">{(b.cvar95 * 100).toFixed(2)}%</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className="chart-grid">
        {/* Radar Chart */}
        <div className="glass-card">
          <div className="card-header">
            <span className="card-title">Multi-Metric Radar</span>
          </div>
          <div className="chart-container">
            <ResponsiveContainer width="100%" height="100%">
              <RadarChart data={radarData}>
                <PolarGrid stroke="rgba(48,54,61,0.6)" />
                <PolarAngleAxis dataKey="metric" tick={{ fill: '#8b949e', fontSize: 11 }} />
                <PolarRadiusAxis tick={false} axisLine={false} />
                <Tooltip contentStyle={tooltipStyle} />
                <Radar name="HRL Agent" dataKey="Alpha-Aware HRL (Ours)" stroke="#58a6ff" fill="#58a6ff" fillOpacity={0.2} strokeWidth={2} />
                <Radar name="MACD" dataKey="MACD (Momentum)" stroke="#d29922" fill="transparent" strokeWidth={1.5} strokeDasharray="4 2" />
                <Radar name="LSTM" dataKey="Supervised LSTM" stroke="#bc8cff" fill="transparent" strokeWidth={1.5} strokeDasharray="4 2" />
              </RadarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Ablation Study */}
        <div className="glass-card">
          <div className="card-header">
            <span className="card-title">Ablation Study — Sharpe Ratio</span>
          </div>
          <div className="chart-container">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={mockAblations} layout="vertical">
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(48,54,61,0.4)" horizontal={false} />
                <XAxis type="number" stroke="#484f58" tickLine={false} fontSize={11} />
                <YAxis type="category" dataKey="variant" stroke="#484f58" tickLine={false} fontSize={11} width={140} />
                <Tooltip contentStyle={tooltipStyle} formatter={(v) => [v.toFixed(2), 'Sharpe']} />
                <Bar dataKey="sharpe" radius={[0, 6, 6, 0]} animationDuration={800}>
                  {mockAblations.map((_, idx) => (
                    <Cell key={idx} fill={ABLATION_COLORS[idx]} fillOpacity={0.75} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>
    </div>
  );
}
