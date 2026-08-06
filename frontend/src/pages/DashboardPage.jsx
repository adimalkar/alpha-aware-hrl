import MetricCard from '../components/MetricCard';
import { mockSummaryMetrics, mockPortfolioHistory, mockRegimes } from '../utils/mockData';
import {
  AreaChart, Area, XAxis, YAxis, Tooltip, ResponsiveContainer,
  PieChart, Pie, Cell,
} from 'recharts';
import { TrendingUp, Shield, Zap, Target, BarChart3, Activity } from 'lucide-react';

const REGIME_COLORS = ['var(--color-profit)', 'var(--color-warn)', 'var(--color-loss)'];
const REGIME_COLORS_RAW = ['#3fb950', '#d29922', '#f85149'];

export default function DashboardPage() {
  const m = mockSummaryMetrics;

  // Regime distribution for pie chart
  const regimeCounts = [0, 0, 0];
  mockRegimes.forEach(r => regimeCounts[r.regime]++);
  const pieData = [
    { name: 'Safe', value: regimeCounts[0] },
    { name: 'Risky', value: regimeCounts[1] },
    { name: 'Crash', value: regimeCounts[2] },
  ];

  // Mini equity data
  const equityData = mockPortfolioHistory.filter((_, i) => i % 5 === 0);

  return (
    <div>
      <div className="page-header">
        <h1>Dashboard</h1>
        <p>Alpha-Aware Hierarchical RL — Real-time system overview</p>
      </div>

      {/* KPI Row */}
      <div className="metric-grid">
        <MetricCard
          title="Sharpe Ratio"
          value={m.sharpe.toFixed(2)}
          delta={12.4}
          deltaLabel="vs baseline"
          icon={<TrendingUp size={18} />}
          iconColor="blue"
        />
        <MetricCard
          title="Total Return"
          value={`${m.totalReturn.toFixed(1)}%`}
          delta={m.totalReturn}
          icon={<Zap size={18} />}
          iconColor="green"
        />
        <MetricCard
          title="Max Drawdown"
          value={`${m.maxDrawdown.toFixed(1)}%`}
          delta={-m.maxDrawdown}
          icon={<Shield size={18} />}
          iconColor="red"
        />
        <MetricCard
          title="Win Rate"
          value={`${m.winRate.toFixed(1)}%`}
          delta={3.2}
          deltaLabel="vs LSTM"
          icon={<Target size={18} />}
          iconColor="cyan"
        />
        <MetricCard
          title="Total Trades"
          value={m.totalTrades.toLocaleString()}
          icon={<BarChart3 size={18} />}
          iconColor="purple"
        />
        <MetricCard
          title="CVaR (95%)"
          value={`${m.cvar95.toFixed(1)}%`}
          delta={-1.2}
          deltaLabel="tail risk"
          icon={<Activity size={18} />}
          iconColor="red"
        />
      </div>

      {/* Charts Row */}
      <div className="chart-grid">
        <div className="glass-card">
          <div className="card-header">
            <span className="card-title">Portfolio Equity Curve</span>
            <span className="badge running">● Live</span>
          </div>
          <div className="chart-container">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={equityData}>
                <defs>
                  <linearGradient id="gradientBlue" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor="#58a6ff" stopOpacity={0.3} />
                    <stop offset="100%" stopColor="#58a6ff" stopOpacity={0} />
                  </linearGradient>
                </defs>
                <XAxis
                  dataKey="tick"
                  stroke="#484f58"
                  tickLine={false}
                  axisLine={false}
                  fontSize={11}
                />
                <YAxis
                  stroke="#484f58"
                  tickLine={false}
                  axisLine={false}
                  fontSize={11}
                  tickFormatter={v => `$${(v / 1000).toFixed(0)}k`}
                  domain={['dataMin - 5000', 'dataMax + 5000']}
                />
                <Tooltip
                  contentStyle={{
                    background: '#161b22',
                    border: '1px solid rgba(88,166,255,0.15)',
                    borderRadius: 10,
                    fontSize: 12,
                    color: '#f0f6fc',
                  }}
                  formatter={(v) => [`$${v.toFixed(0)}`, 'Portfolio']}
                  labelFormatter={(v) => `Tick ${v}`}
                />
                <Area
                  type="monotone"
                  dataKey="hrl"
                  stroke="#58a6ff"
                  strokeWidth={2}
                  fill="url(#gradientBlue)"
                  dot={false}
                  animationDuration={1200}
                />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div className="glass-card">
          <div className="card-header">
            <span className="card-title">Market Regime Distribution</span>
          </div>
          <div className="chart-container" style={{ display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie
                  data={pieData}
                  cx="50%"
                  cy="50%"
                  innerRadius={70}
                  outerRadius={110}
                  paddingAngle={4}
                  dataKey="value"
                  animationDuration={1000}
                >
                  {pieData.map((_, idx) => (
                    <Cell key={idx} fill={REGIME_COLORS_RAW[idx]} />
                  ))}
                </Pie>
                <Tooltip
                  contentStyle={{
                    background: '#161b22',
                    border: '1px solid rgba(88,166,255,0.15)',
                    borderRadius: 10,
                    fontSize: 12,
                    color: '#f0f6fc',
                  }}
                />
              </PieChart>
            </ResponsiveContainer>
          </div>
          <div style={{ display: 'flex', justifyContent: 'center', gap: 20, marginTop: -12 }}>
            {['Safe', 'Risky', 'Crash'].map((label, i) => (
              <span key={label} style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: '0.82rem', color: 'var(--text-tertiary)' }}>
                <span style={{ width: 8, height: 8, borderRadius: '50%', background: REGIME_COLORS_RAW[i], display: 'inline-block' }} />
                {label}
              </span>
            ))}
          </div>
        </div>
      </div>

      {/* System Status */}
      <div className="glass-card" style={{ marginTop: 0 }}>
        <div className="card-header">
          <span className="card-title">System Status</span>
        </div>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 20 }}>
          {[
            { label: 'TQC Agent', status: 'online', detail: 'Trained · 1M steps' },
            { label: 'Mamba Extractor', status: 'online', detail: 'LSTM backend · d=128' },
            { label: 'LLM Analyst', status: 'online', detail: 'TinyLlama · Precomputed' },
            { label: 'FI-2010 Dataset', status: 'online', detail: '362,400 train · 31,937 test' },
          ].map(item => (
            <div key={item.label} style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
              <span className={`status-dot ${item.status}`} />
              <div>
                <div style={{ fontWeight: 600, color: 'var(--text-primary)', fontSize: '0.88rem' }}>{item.label}</div>
                <div style={{ fontSize: '0.75rem', color: 'var(--text-tertiary)' }}>{item.detail}</div>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
