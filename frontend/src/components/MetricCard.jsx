export default function MetricCard({ title, value, delta, deltaLabel, icon, iconColor = 'blue' }) {
  const isPositive = delta != null && delta >= 0;

  return (
    <div className="glass-card">
      <div className="card-header">
        <span className="card-title">{title}</span>
        {icon && (
          <div className={`card-icon ${iconColor}`}>
            {icon}
          </div>
        )}
      </div>
      <div className="card-value">{value}</div>
      {delta != null && (
        <span className={`card-delta ${isPositive ? 'positive' : 'negative'}`}>
          {isPositive ? '↑' : '↓'} {Math.abs(delta).toFixed(2)}%
          {deltaLabel && <span style={{ marginLeft: 4, fontWeight: 400 }}>{deltaLabel}</span>}
        </span>
      )}
    </div>
  );
}
