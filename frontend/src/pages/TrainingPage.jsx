import NoData from '../components/NoData';

export default function TrainingPage() {
  return (
    <div>
      <div className="page-header">
        <h1>Training</h1>
        <p>Learning curves</p>
      </div>
      <NoData
        title="No training curve is served"
        detail={
          'This page previously plotted mockTrainingCurve — a fabricated learning ' +
          'curve of the form reward = -200 + i*2.5 + sin(i/10) + noise, which rose ' +
          'monotonically by construction regardless of whether any training had ' +
          'occurred. Real curves are written to TensorBoard during training.'
        }
        command={'tensorboard --logdir experiments/rl_run/tensorboard'}
      />
    </div>
  );
}
