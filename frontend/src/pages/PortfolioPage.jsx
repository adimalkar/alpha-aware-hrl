import NoData from '../components/NoData';

export default function PortfolioPage() {
  return (
    <div>
      <div className="page-header">
        <h1>Portfolio</h1>
        <p>Equity curve over the evaluation window</p>
      </div>
      <NoData
        title="No equity curve is recorded"
        detail={
          'This page previously plotted mockPortfolioHistory — 500 points of ' +
          'Math.sin(i/40) + Math.random(), generated in the browser and drawn as ' +
          'if it were a measured equity curve. Nothing currently persists the ' +
          'per-step equity series, so there is nothing honest to draw here yet. ' +
          'Evaluation records summary metrics only; the curve itself needs to be ' +
          'written out by the eval loop first.'
        }
        command={'python scripts/train_rl_agent.py --symbol BTC/USD --encoder lem'}
      />
    </div>
  );
}
