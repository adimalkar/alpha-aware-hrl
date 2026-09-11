import {
  BarChart, Bar, XAxis, YAxis, Cell, CartesianGrid,
  ResponsiveContainer, Tooltip,
} from 'recharts';
import { useApiData } from '../utils/api';
import NoData from '../components/NoData';

const tooltipStyle = {
  background: '#161b22',
  border: '1px solid rgba(88,166,255,0.15)',
  borderRadius: 10,
  fontSize: 12,
  color: '#f0f6fc',
};

const ABLATION_COLORS = ['#58a6ff', '#bc8cff', '#f85149', '#39d2c0', '#d29922'];

const RUN_ABLATION =
  'python scripts/run_ablations.py --symbol BTC/USD \\\n' +
  '    --arms lem lem_frozen gru mlp --seeds 0 1 2';

function num(v, digits = 4) {
  return typeof v === 'number' && Number.isFinite(v) ? v.toFixed(digits) : '—';
}

export default function BaselinesPage() {
  const { data, loading, error, noData, howToGenerate } = useApiData('/api/baselines');

  if (loading) {
    return (
      <div>
        <div className="page-header"><h1>Baseline Comparisons</h1></div>
        <NoData title="Loading…" detail="Fetching measured results from the API." />
      </div>
    );
  }

  if (error || noData) {
    return (
      <div>
        <div className="page-header">
          <h1>Baseline Comparisons</h1>
          <p>Agent vs. traditional strategies and feature-extractor ablations</p>
        </div>
        <NoData
          error={error}
          title="No comparative results yet"
          detail="No ablation or baseline run exists in this workspace. Previous results were withdrawn to experiments/INVALIDATED/ because they were produced on a leaking harness."
          command={howToGenerate ?? RUN_ABLATION}
        />
      </div>
    );
  }

  const ablations = data.ablations ?? [];
  const strategies = data.strategies ?? [];

  return (
    <div>
      <div className="page-header">
        <h1>Baseline Comparisons</h1>
        <p>Agent vs. traditional strategies and feature-extractor ablations</p>
      </div>

      {ablations.length > 0 && (
        <div className="glass-card" style={{ marginBottom: 18 }}>
          <div className="card-header">
            <span className="card-title">
              Ablation arms — mean over seeds, 95% CI
            </span>
          </div>
          <table className="data-table">
            <thead>
              <tr>
                <th>Arm</th>
                <th>Seeds</th>
                <th>Return % (mean)</th>
                <th>± CI95</th>
                <th>Sharpe / step</th>
                <th>Traded</th>
              </tr>
            </thead>
            <tbody>
              {ablations.map((a) => (
                <tr key={a.arm}>
                  <td style={{ fontWeight: 600 }}>{a.arm}</td>
                  <td>{a.n_seeds}</td>
                  <td>{num(a.return_pct_mean)}</td>
                  <td>{num(a.return_pct_ci95)}</td>
                  <td>{num(a.sharpe_mean)}</td>
                  <td>{a.arms_that_traded}/{a.n_seeds}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <p style={{ fontSize: 11.5, color: '#6e7681', marginTop: 10, lineHeight: 1.6 }}>
            <code>lem_frozen</code> reproduces the original configuration in which the
            encoder received no gradient. It is a control, not a competitor.
            Sharpe is per-step and deliberately not annualised — annualising
            tick returns by √N is how the withdrawn Sharpe of 365 was produced.
          </p>
        </div>
      )}

      {ablations.length > 0 && (
        <div className="glass-card" style={{ marginBottom: 18 }}>
          <div className="card-header">
            <span className="card-title">Mean return by arm (%)</span>
          </div>
          <ResponsiveContainer width="100%" height={260}>
            <BarChart data={ablations}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(240,246,252,0.06)" />
              <XAxis dataKey="arm" tick={{ fontSize: 11, fill: '#8b949e' }} />
              <YAxis tick={{ fontSize: 11, fill: '#8b949e' }} />
              <Tooltip contentStyle={tooltipStyle} />
              <Bar dataKey="return_pct_mean" radius={[6, 6, 0, 0]}>
                {ablations.map((_, i) => (
                  <Cell key={i} fill={ABLATION_COLORS[i % ABLATION_COLORS.length]} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}

      {strategies.length > 0 && (
        <div className="glass-card">
          <div className="card-header">
            <span className="card-title">Traditional strategy baselines</span>
          </div>
          <table className="data-table">
            <thead>
              <tr>
                <th>Strategy</th>
                <th>Final Portfolio</th>
                <th>Return %</th>
                <th>Sharpe</th>
                <th>Max DD %</th>
              </tr>
            </thead>
            <tbody>
              {strategies.map((s, i) => (
                <tr key={i}>
                  <td style={{ fontWeight: 600 }}>{s.Name ?? s.name}</td>
                  <td>{num(s['Final Portfolio'], 2)}</td>
                  <td>{num(s['Return %'], 4)}</td>
                  <td>{num(s['Sharpe Ratio'], 4)}</td>
                  <td>{num(s['Max Drawdown %'], 4)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
