// Mock data for development — replaced by API calls in production
export const mockPortfolioHistory = Array.from({ length: 500 }, (_, i) => ({
  tick: i,
  hrl: 100000 + Math.sin(i / 40) * 8000 + i * 18 + (Math.random() - 0.4) * 2000,
  macd: 100000 + Math.sin(i / 30) * 3000 + i * 6 + (Math.random() - 0.5) * 1500,
  bollinger: 100000 - i * 10 + Math.sin(i / 25) * 4000 + (Math.random() - 0.5) * 2000,
  lstm: 100000 - i * 4 + Math.sin(i / 35) * 5000 + (Math.random() - 0.5) * 1800,
}));

export const mockTrainingCurve = Array.from({ length: 200 }, (_, i) => ({
  episode: i * 50,
  reward: -200 + i * 2.5 + Math.sin(i / 10) * 30 + (Math.random() - 0.5) * 40,
  loss: 2.5 - i * 0.008 + Math.sin(i / 15) * 0.3 + (Math.random() - 0.5) * 0.2,
  sharpe: -2 + i * 0.025 + Math.sin(i / 12) * 0.4 + (Math.random() - 0.5) * 0.3,
  episodeLength: 800 + i * 8 + (Math.random() - 0.5) * 200,
}));

export const mockRegimes = Array.from({ length: 120 }, (_, i) => {
  const r = Math.random();
  const regime = r < 0.55 ? 0 : r < 0.85 ? 2 : 1;
  return {
    chunkIdx: i,
    tickStart: i * 3000,
    tickEnd: (i + 1) * 3000,
    regime,
    regimeLabel: ['Safe', 'Risky', 'Crash'][regime],
    confidence: 0.6 + Math.random() * 0.35,
    newsSnippet: [
      'Markets remain stable amid low volatility readings...',
      'Fed signals potential rate hike; uncertainty grows...',
      'Flash crash fears rise as VIX spikes to 35...',
    ][regime],
  };
});

export const mockBaselines = [
  { name: 'Alpha-Aware HRL (Ours)', finalPortfolio: 118420, returnPct: 18.42, sharpe: 1.87, maxDrawdown: 8.2, var95: 0.032, cvar95: 0.048, color: 'var(--accent-blue)' },
  { name: 'MACD (Momentum)', finalPortfolio: 127624, returnPct: 27.62, sharpe: -1.97, maxDrawdown: 986.7, var95: 0.100, cvar95: 0.104, color: 'var(--color-warn)' },
  { name: 'Bollinger Bands', finalPortfolio: 33800, returnPct: -66.19, sharpe: -4.33, maxDrawdown: 6619.9, var95: 0.100, cvar95: 0.103, color: 'var(--color-loss)' },
  { name: 'Supervised LSTM', finalPortfolio: 80172, returnPct: -19.82, sharpe: -5.54, maxDrawdown: 2180.6, var95: 0.010, cvar95: 0.064, color: 'var(--accent-purple)' },
];

export const mockLOBDepth = Array.from({ length: 20 }, (_, level) =>
  Array.from({ length: 60 }, (_, tick) => ({
    level,
    tick,
    bidVolume: Math.max(0, 500 - level * 40 + Math.sin(tick / 8 + level) * 200 + (Math.random() - 0.5) * 100),
    askVolume: Math.max(0, 500 - level * 40 + Math.cos(tick / 8 + level) * 200 + (Math.random() - 0.5) * 100),
  }))
).flat();

export const mockAblations = [
  { variant: 'Full Model', sharpe: 1.87, maxDD: 8.2, returnPct: 18.42 },
  { variant: 'No LLM Regime', sharpe: 0.95, maxDD: 14.5, returnPct: 9.1 },
  { variant: 'No Mamba (MLP only)', sharpe: 0.62, maxDD: 18.3, returnPct: 5.8 },
  { variant: 'No Alpha Signal', sharpe: 1.45, maxDD: 10.1, returnPct: 14.2 },
  { variant: 'SAC (instead of TQC)', sharpe: 1.12, maxDD: 12.8, returnPct: 11.6 },
];

export const mockSummaryMetrics = {
  sharpe: 1.87,
  totalReturn: 18.42,
  maxDrawdown: 8.2,
  winRate: 58.3,
  totalTrades: 4218,
  profitFactor: 1.64,
  var95: 3.2,
  cvar95: 4.8,
};
