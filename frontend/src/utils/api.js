/**
 * API access for the dashboard.
 *
 * This module replaces src/utils/mockData.js, which generated portfolio
 * curves, training curves, regime timelines, LOB depth, baseline tables and
 * summary metrics client-side with Math.random(). Six of the nine pages
 * rendered that synthetic data as if it were measured output — including a
 * 6619.9% max drawdown, which is definitionally impossible.
 *
 * Nothing here invents data. When the backend has no result, the caller gets
 * `noData` and renders an explicit empty state telling the user which command
 * produces it.
 */

import { useCallback, useEffect, useState } from 'react';

export const API_BASE =
  import.meta.env.VITE_API_BASE ?? 'http://localhost:8000';

export async function apiGet(path) {
  const res = await fetch(`${API_BASE}${path}`);
  if (res.status === 404) {
    const body = await res.json().catch(() => ({}));
    const err = new Error(body.detail ?? 'No data available');
    err.noData = true;
    err.howToGenerate = body.how_to_generate ?? null;
    throw err;
  }
  if (!res.ok) {
    throw new Error(`${path} failed: ${res.status} ${res.statusText}`);
  }
  return res.json();
}

/**
 * Fetch `path`, distinguishing three states: loading, no-data, and error.
 * `noData` is a normal outcome (the experiment has not been run), not a bug,
 * and must be rendered differently from a real failure.
 */
export function useApiData(path, { pollMs = 0 } = {}) {
  const [state, setState] = useState({
    data: null,
    loading: true,
    error: null,
    noData: false,
    howToGenerate: null,
  });

  const load = useCallback(async () => {
    try {
      const data = await apiGet(path);
      setState({ data, loading: false, error: null, noData: false, howToGenerate: null });
    } catch (err) {
      setState({
        data: null,
        loading: false,
        error: err.noData ? null : err.message,
        noData: Boolean(err.noData),
        howToGenerate: err.howToGenerate ?? null,
      });
    }
  }, [path]);

  useEffect(() => {
    let cancelled = false;
    const run = () => { if (!cancelled) load(); };
    run();
    if (pollMs > 0) {
      const id = setInterval(run, pollMs);
      return () => { cancelled = true; clearInterval(id); };
    }
    return () => { cancelled = true; };
  }, [load, pollMs]);

  return { ...state, reload: load };
}
