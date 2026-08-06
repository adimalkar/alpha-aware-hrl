import { mockTrainingCurve } from '../utils/mockData';
import {
  LineChart, Line, AreaChart, Area,
  XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid,
} from 'recharts';
import MetricCard from '../components/MetricCard';
import { Activity, TrendingUp, Zap, Clock } from 'lucide-react';

const tooltipStyle = {
  background: '#161b22',
  border: '1px solid rgba(88,166,255,0.15)',
  borderRadius: 10,
  fontSize: 12,
  color: '#f0f6fc',
};

export default function TrainingPage() {
  const data = mockTrainingCurve;
  const latest = data[data.length - 1];

  return (
    <div>
      <div className="page-header">
        <h1>Training Metrics</h1>
        <p>Tensorboard-style training curves for the TQC agent on FI-2010</p>
      </div>

      <div className="metric-grid" style={{ gridTemplateColumns: 'repeat(4, 1fr)' }}>
        <MetricCard title="Avg Reward" value={latest.reward.toFixed(1)} delta={42.5} icon={<Zap size={18} />} iconColor="green" />
        <MetricCard title="Policy Loss" value={latest.loss.toFixed(3)} delta={-68.0} icon={<Activity size={18} />} iconColor="red" />
        <MetricCard title="Live Sharpe" value={latest.sharpe.toFixed(2)} delta={latest.sharpe > 0 ? 100 : -50} icon={<TrendingUp size={18} />} iconColor="blue" />
        <MetricCard title="Ep. Length" value={latest.episodeLength.toFixed(0)} icon={<Clock size={18} />} iconColor="purple" />
      </div>

      <div className="chart-grid">
        {/* Reward Curve */}
        <div className="glass-card">
          <div className="card-header">
            <span className="card-title">Episode Reward</span>
            <span className="badge running">● Training</span>
          </div>
          <div className="chart-container">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={data}>
                <defs>
                  <linearGradient id="gradGreen" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor="#3fb950" stopOpacity={0.3} />
                    <stop offset="100%" stopColor="#3fb950" stopOpacity={0} />
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(48,54,61,0.4)" />
                <XAxis dataKey="episode" stroke="#484f58" tickLine={false} fontSize={11} />
                <YAxis stroke="#484f58" tickLine={false} fontSize={11} />
                <Tooltip contentStyle={tooltipStyle} formatter={(v) => [v.toFixed(2), 'Reward']} labelFormatter={v => `Step ${v}`} />
                <Area type="monotone" dataKey="reward" stroke="#3fb950" strokeWidth={2} fill="url(#gradGreen)" dot={false} animationDuration={1200} />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Loss Curve */}
        <div className="glass-card">
          <div className="card-header">
            <span className="card-title">Critic Loss</span>
          </div>
          <div className="chart-container">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={data}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(48,54,61,0.4)" />
                <XAxis dataKey="episode" stroke="#484f58" tickLine={false} fontSize={11} />
                <YAxis stroke="#484f58" tickLine={false} fontSize={11} />
                <Tooltip contentStyle={tooltipStyle} formatter={(v) => [v.toFixed(4), 'Loss']} labelFormatter={v => `Step ${v}`} />
                <Line type="monotone" dataKey="loss" stroke="#f85149" strokeWidth={2} dot={false} animationDuration={1200} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Sharpe Over Time */}
        <div className="glass-card">
          <div className="card-header">
            <span className="card-title">Rolling Sharpe Ratio</span>
          </div>
          <div className="chart-container">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={data}>
                <defs>
                  <linearGradient id="gradPurple" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor="#bc8cff" stopOpacity={0.3} />
                    <stop offset="100%" stopColor="#bc8cff" stopOpacity={0} />
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(48,54,61,0.4)" />
                <XAxis dataKey="episode" stroke="#484f58" tickLine={false} fontSize={11} />
                <YAxis stroke="#484f58" tickLine={false} fontSize={11} />
                <Tooltip contentStyle={tooltipStyle} formatter={(v) => [v.toFixed(3), 'Sharpe']} labelFormatter={v => `Step ${v}`} />
                <Area type="monotone" dataKey="sharpe" stroke="#bc8cff" strokeWidth={2} fill="url(#gradPurple)" dot={false} animationDuration={1200} />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Episode Length */}
        <div className="glass-card">
          <div className="card-header">
            <span className="card-title">Episode Length</span>
          </div>
          <div className="chart-container">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={data}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(48,54,61,0.4)" />
                <XAxis dataKey="episode" stroke="#484f58" tickLine={false} fontSize={11} />
                <YAxis stroke="#484f58" tickLine={false} fontSize={11} />
                <Tooltip contentStyle={tooltipStyle} formatter={(v) => [v.toFixed(0), 'Steps']} labelFormatter={v => `Step ${v}`} />
                <Line type="monotone" dataKey="episodeLength" stroke="#39d2c0" strokeWidth={2} dot={false} animationDuration={1200} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>
    </div>
  );
}
