import MetricCard from '../components/MetricCard';
import NoData from '../components/NoData';
import { useApiData } from '../utils/api';
import { TrendingUp, Shield, Zap, Target, Activity } from 'lucide-react';

const TRAIN_CMD =
  'python scripts/train_rl_agent.py --symbol BTC/USD --encoder lem \\\n' +
  '    --pretrained checkpoints/event_encoder_thp.pt';

function fmt(v, digits = 4, suffix = '') {
  return typeof v === 'number' && Number.isFinite(v)
    ? `${v.toFixed(digits)}${suffix}`
    : '—';
}

export default function DashboardPage() {
  const { data, loading, error, noData, howToGenerate } =
    useApiData('/api/metrics/summary', { pollMs: 15000 });

  const header = (
    <div className="page-header">
      <h1>Dashboard</h1>
      <p>Alpha-Aware Hierarchical RL — measured results only</p>
    </div>
  );

  if (loading) {
    return <div>{header}<NoData title="Loading…" detail="Fetching the latest evaluation." /></div>;
  }

  if (error || noData) {
    return (
      <div>
        {header}
        <NoData
          error={error}
          title="No evaluation has been run"
          detail="This dashboard shows only measured results. Previous figures were withdrawn to experiments/INVALIDATED/ — the headline Sharpe of 365.38 came from a fallback branch that fired because portfolio tracking returned nothing."
          command={howToGenerate ?? TRAIN_CMD}
        />
      </div>
    );
  }

  const warn = data.warning;

  return (
    <div>
      {header}

      {warn && (
        <div
          className="glass-card"
          style={{
            marginBottom: 16,
            padding: '12px 16px',
            borderLeft: '3px solid #d29922',
            fontSize: 12.5,
            color: '#d29922',
          }}
        >
          {warn}
        </div>
      )}

      <div className="metric-grid">
        <MetricCard
          title="Sharpe (per step)"
          value={fmt(data.sharpePerStep)}
          icon={<TrendingUp size={18} />}
        />
        <MetricCard
          title="Total Return"
          value={fmt(data.totalReturnPct, 4, '%')}
          icon={<Activity size={18} />}
        />
        <MetricCard
          title="Max Drawdown"
          value={fmt(data.maxDrawdownPct, 4, '%')}
          icon={<Shield size={18} />}
        />
        <MetricCard
          title="Win Rate"
          value={fmt(data.winRatePct, 2, '%')}
          icon={<Target size={18} />}
        />
        <MetricCard
          title="VaR 95%"
          value={fmt(data.var95, 6)}
          icon={<Zap size={18} />}
        />
        <MetricCard
          title="CVaR 95%"
          value={fmt(data.cvar95, 6)}
          icon={<Zap size={18} />}
        />
        <MetricCard
          title="Eval Steps"
          value={data.evalSteps ?? '—'}
          icon={<Activity size={18} />}
        />
        <MetricCard
          title="Position Changes"
          value={data.positionChanges ?? '—'}
          icon={<Activity size={18} />}
        />
      </div>

      <div className="glass-card" style={{ marginTop: 18 }}>
        <div className="card-header">
          <span className="card-title">Measurement context</span>
        </div>
        <table className="data-table">
          <tbody>
            <tr>
              <td>Annualised</td>
              <td>
                {String(data.annualised)}
                {data.annualised === false && (
                  <span style={{ color: '#6e7681', marginLeft: 8 }}>
                    — Sharpe is per-step. Annualising tick returns by √N produced the withdrawn 365.38.
                  </span>
                )}
              </td>
            </tr>
            <tr><td>Source</td><td><code>{data.source}</code></td></tr>
            {data.provenance && (
              <>
                <tr><td>Commit</td><td><code>{String(data.provenance.git_commit).slice(0, 12)}</code></td></tr>
                <tr>
                  <td>Working tree</td>
                  <td style={{ color: data.provenance.git_dirty ? '#d29922' : '#3fb950' }}>
                    {data.provenance.git_dirty ? 'dirty — not reproducible from this commit alone' : 'clean'}
                  </td>
                </tr>
                <tr><td>Seed</td><td>{data.provenance.seed ?? '—'}</td></tr>
                <tr><td>Run at (UTC)</td><td>{data.provenance.utc ?? '—'}</td></tr>
              </>
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}
