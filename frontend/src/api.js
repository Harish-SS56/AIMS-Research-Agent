// Use environment variable for API base, fallback to /api for local dev
const BASE = import.meta.env.VITE_API_URL || '/api'

// Demo mode flag - set when backend is unavailable
let demoMode = false

async function request(path, options = {}) {
  try {
    const res = await fetch(`${BASE}${path}`, {
      headers: { 'Content-Type': 'application/json', ...options.headers },
      ...options,
    })
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }))
      throw new Error(err.detail || 'Request failed')
    }
    return res.json()
  } catch (err) {
    // If backend unavailable, enable demo mode
    if (err.message.includes('Failed to fetch') || err.message.includes('NetworkError')) {
      demoMode = true
      throw new Error('Backend unavailable - running in demo mode. See README for local setup.')
    }
    throw err
  }
}

export const isDemoMode = () => demoMode

export const api = {
  health:        ()        => request('/health'),
  getConfigs:    ()        => request('/configs'),
  getStats:      ()        => request('/stats'),
  getPapers:     ()        => request('/papers'),
  getAblation:   ()        => request('/ablation'),
  runResearch:   (body)    => request('/research', {
    method: 'POST',
    body: JSON.stringify(body),
    signal: AbortSignal.timeout(180_000), // 3-minute timeout
  }),
}
