/**
 * Explicit empty state.
 *
 * Shown when the backend has no result for a panel. The dashboard previously
 * filled these panels with Math.random() mock data, so a page with nothing
 * behind it was indistinguishable from a page showing real measurements.
 */
export default function NoData({ title, detail, command, error }) {
  const isError = Boolean(error);
  return (
    <div
      className="glass-card"
      style={{
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        gap: 10,
        padding: '44px 24px',
        textAlign: 'center',
        minHeight: 200,
      }}
    >
      <div
        style={{
          fontSize: 13,
          fontWeight: 600,
          letterSpacing: '0.02em',
          color: isError ? '#f85149' : '#8b949e',
        }}
      >
        {isError ? 'Could not reach the API' : title ?? 'No data yet'}
      </div>

      <div style={{ fontSize: 12, color: '#6e7681', maxWidth: 460, lineHeight: 1.6 }}>
        {isError
          ? error
          : detail ?? 'This panel has no measured result behind it yet.'}
      </div>

      {!isError && command && (
        <code
          style={{
            marginTop: 6,
            padding: '9px 13px',
            background: 'rgba(88,166,255,0.07)',
            border: '1px solid rgba(88,166,255,0.18)',
            borderRadius: 8,
            fontSize: 11.5,
            color: '#58a6ff',
            fontFamily: 'ui-monospace, SFMono-Regular, Menlo, monospace',
            whiteSpace: 'pre-wrap',
            textAlign: 'left',
          }}
        >
          {command}
        </code>
      )}
    </div>
  );
}
