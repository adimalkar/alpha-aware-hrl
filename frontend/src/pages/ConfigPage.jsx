import { useState } from 'react';
import { Play, Square, Terminal } from 'lucide-react';

export default function ConfigPage() {
  const [config, setConfig] = useState({
    timesteps: 1000000,
    nEnvs: 4,
    learningRate: 0.0003,
    batchSize: 256,
    gamma: 0.99,
    bufferSize: 100000,
    seed: 42,
    vecEnv: 'dummy',
    quantiles: 25,
    topQuantilesToDrop: 2,
  });

  const [isRunning, setIsRunning] = useState(false);
  const [logs, setLogs] = useState([
    '[System] Ready. Configure hyperparameters and click Launch.',
  ]);

  const handleChange = (key, value) => {
    setConfig(prev => ({ ...prev, [key]: value }));
  };

  const handleLaunch = () => {
    setIsRunning(true);
    setLogs(prev => [
      ...prev,
      `[${new Date().toLocaleTimeString()}] Launching training...`,
      `  --timesteps ${config.timesteps}`,
      `  --n-envs ${config.nEnvs}`,
      `  --seed ${config.seed}`,
      `  --vec-env ${config.vecEnv}`,
      `[${new Date().toLocaleTimeString()}] Initializing TQC Agent...`,
      `[${new Date().toLocaleTimeString()}] Loading FI-2010 data...`,
    ]);

    // Simulate training progress
    let step = 0;
    const interval = setInterval(() => {
      step += Math.floor(config.timesteps / 20);
      if (step >= config.timesteps) {
        step = config.timesteps;
        clearInterval(interval);
        setIsRunning(false);
        setLogs(prev => [...prev, `[${new Date().toLocaleTimeString()}] ✅ Training complete! Model saved.`]);
      } else {
        const reward = (-200 + step * 0.0005 + (Math.random() - 0.5) * 40).toFixed(1);
        setLogs(prev => [...prev, `[Step ${step.toLocaleString()}] reward=${reward}  loss=${(2.5 - step * 0.000002).toFixed(4)}`]);
      }
    }, 1500);
  };

  const handleStop = () => {
    setIsRunning(false);
    setLogs(prev => [...prev, `[${new Date().toLocaleTimeString()}] ⚠ Training interrupted by user.`]);
  };

  const cmd = `python scripts/run_cluster_training.py --timesteps ${config.timesteps} --n-envs ${config.nEnvs} --seed ${config.seed} --vec-env ${config.vecEnv}`;

  return (
    <div>
      <div className="page-header">
        <h1>Model Configuration & Launcher</h1>
        <p>Configure training hyperparameters and launch cluster execution</p>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '380px 1fr', gap: 18 }}>
        {/* Config Panel */}
        <div className="glass-card" style={{ alignSelf: 'start' }}>
          <div className="card-header">
            <span className="card-title">Hyperparameters</span>
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0 16px' }}>
            <div className="form-group">
              <label className="form-label">Timesteps</label>
              <input type="number" className="form-input" value={config.timesteps}
                onChange={e => handleChange('timesteps', parseInt(e.target.value))} />
            </div>
            <div className="form-group">
              <label className="form-label">Num Envs</label>
              <input type="number" className="form-input" value={config.nEnvs}
                onChange={e => handleChange('nEnvs', parseInt(e.target.value))} />
            </div>
            <div className="form-group">
              <label className="form-label">Learning Rate</label>
              <input type="number" step="0.0001" className="form-input" value={config.learningRate}
                onChange={e => handleChange('learningRate', parseFloat(e.target.value))} />
            </div>
            <div className="form-group">
              <label className="form-label">Batch Size</label>
              <input type="number" className="form-input" value={config.batchSize}
                onChange={e => handleChange('batchSize', parseInt(e.target.value))} />
            </div>
            <div className="form-group">
              <label className="form-label">Gamma</label>
              <input type="number" step="0.01" className="form-input" value={config.gamma}
                onChange={e => handleChange('gamma', parseFloat(e.target.value))} />
            </div>
            <div className="form-group">
              <label className="form-label">Buffer Size</label>
              <input type="number" className="form-input" value={config.bufferSize}
                onChange={e => handleChange('bufferSize', parseInt(e.target.value))} />
            </div>
            <div className="form-group">
              <label className="form-label">Seed</label>
              <input type="number" className="form-input" value={config.seed}
                onChange={e => handleChange('seed', parseInt(e.target.value))} />
            </div>
            <div className="form-group">
              <label className="form-label">Vec Env</label>
              <select className="form-select" value={config.vecEnv}
                onChange={e => handleChange('vecEnv', e.target.value)}>
                <option value="dummy">DummyVecEnv</option>
                <option value="subproc">SubprocVecEnv</option>
              </select>
            </div>
            <div className="form-group">
              <label className="form-label">Quantiles</label>
              <input type="number" className="form-input" value={config.quantiles}
                onChange={e => handleChange('quantiles', parseInt(e.target.value))} />
            </div>
            <div className="form-group">
              <label className="form-label">Drop Quantiles</label>
              <input type="number" className="form-input" value={config.topQuantilesToDrop}
                onChange={e => handleChange('topQuantilesToDrop', parseInt(e.target.value))} />
            </div>
          </div>

          <div style={{ display: 'flex', gap: 10, marginTop: 8 }}>
            <button className="btn btn-primary" onClick={handleLaunch} disabled={isRunning} style={{ flex: 1 }}>
              <Play size={16} /> Launch Training
            </button>
            <button className="btn btn-danger" onClick={handleStop} disabled={!isRunning}>
              <Square size={16} /> Stop
            </button>
          </div>
        </div>

        {/* Terminal Output */}
        <div className="glass-card" style={{ display: 'flex', flexDirection: 'column' }}>
          <div className="card-header">
            <span className="card-title" style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
              <Terminal size={14} /> Training Output
            </span>
            {isRunning && <span className="badge running pulse">● Running</span>}
          </div>

          {/* Command preview */}
          <div style={{
            background: 'var(--bg-primary)',
            border: '1px solid var(--glass-border)',
            borderRadius: 'var(--radius-sm)',
            padding: '10px 14px',
            marginBottom: 12,
            fontFamily: 'var(--font-mono)',
            fontSize: '0.78rem',
            color: 'var(--accent-cyan)',
            wordBreak: 'break-all',
          }}>
            $ {cmd}
          </div>

          {/* Log viewer */}
          <div style={{
            flex: 1,
            minHeight: 420,
            maxHeight: 500,
            background: 'var(--bg-primary)',
            border: '1px solid var(--glass-border)',
            borderRadius: 'var(--radius-md)',
            padding: 14,
            overflowY: 'auto',
            fontFamily: 'var(--font-mono)',
            fontSize: '0.78rem',
            lineHeight: 1.8,
          }}>
            {logs.map((line, i) => (
              <div key={i} style={{
                color: line.includes('✅') ? 'var(--color-profit)'
                  : line.includes('⚠') ? 'var(--color-warn)'
                  : line.includes('[Step') ? 'var(--text-tertiary)'
                  : 'var(--text-secondary)',
              }}>
                {line}
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
