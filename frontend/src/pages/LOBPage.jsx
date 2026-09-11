import NoData from '../components/NoData';

export default function LOBPage() {
  return (
    <div>
      <div className="page-header">
        <h1>Order Book Depth</h1>
        <p>Level-2 depth over time</p>
      </div>
      <NoData
        title="No order book snapshots are served"
        detail={
          'This page previously rendered mockLOBDepth — 20 levels x 60 ticks of ' +
          'Math.sin(tick/8 + level) + Math.random(), drawn as a live depth heatmap. ' +
          'Real Level-2 snapshots are collected to data/live_market/<symbol>/*.npz ' +
          'but no endpoint currently serves them to the dashboard.'
        }
        command={'python scripts/fetch_live_market_data.py --symbols BTC/USD'}
      />
    </div>
  );
}
