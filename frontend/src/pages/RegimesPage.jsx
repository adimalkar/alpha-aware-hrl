import NoData from '../components/NoData';

export default function RegimesPage() {
  return (
    <div>
      <div className="page-header">
        <h1>Macro Regimes</h1>
        <p>Regime classification over time</p>
      </div>
      <NoData
        title="The regime signal is not a live component"
        detail={
          'This page previously rendered mockRegimes — random regime assignments ' +
          'paired with three hardcoded news snippets. The real regime artefacts are ' +
          'degenerate: train_regimes.npy is the constant 2 across all 362,400 rows ' +
          'and test_regimes.npy the constant 2 across 31,937, while train_confidences ' +
          'holds 8 distinct values in blocks of exactly 50,000 — one LLM call per ' +
          'chunk, broadcast. The one-hot of a constant is an intercept, so the ' +
          'component is excluded from training and from any claim until it produces ' +
          'a signal that varies.'
        }
      />
    </div>
  );
}
