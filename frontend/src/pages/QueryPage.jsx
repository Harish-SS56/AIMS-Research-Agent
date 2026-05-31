import { useState, useEffect } from 'react'
import {
  Search, ChevronDown, Loader2, Clock, Zap,
  CheckCircle, AlertCircle, ExternalLink, BookOpen,
  ChevronRight, RotateCcw, Cpu,
} from 'lucide-react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { api } from '../api.js'

const EXAMPLES = [
  'What is the ReAct framework and how does it combine reasoning with acting?',
  'What is Self-RAG and how does it improve retrieval-augmented generation?',
  'Compare ReAct and Reflexion — how do their approaches differ?',
  'What are the main architectural patterns used in LLM agent systems?',
  'How do LLM agents handle multi-step tool use and planning?',
]

const CONFIG_META = {
  full_agent:   { label: 'Full Agent',    color: 'navy',  desc: 'All components enabled' },
  baseline:     { label: 'Baseline',      color: 'slate', desc: 'Single-pass, no planning' },
  no_planner:   { label: 'No Planner',    color: 'slate', desc: 'Skip query decomposition' },
  no_reranker:  { label: 'No Reranker',   color: 'slate', desc: 'Skip LLM reranking' },
  no_reflector: { label: 'No Reflector',  color: 'slate', desc: 'Single iteration only' },
  no_hybrid:    { label: 'No Hybrid',     color: 'slate', desc: 'Semantic search only' },
  no_verifier:  { label: 'No Verifier',   color: 'slate', desc: 'Skip citation verification' },
}

// ── Helpers ──────────────────────────────────────────────────────────────────

function AccBadge({ score }) {
  const pct = Math.round((score / 5) * 100)
  const color =
    score >= 4.5 ? 'bg-emerald-100 text-emerald-700 border-emerald-200' :
    score >= 3.5 ? 'bg-amber-100  text-amber-700  border-amber-200'  :
                   'bg-red-100    text-red-700    border-red-200'
  return (
    <span className={`badge border ${color}`}>
      {score}/5
    </span>
  )
}

function MetricPill({ icon: Icon, label, value }) {
  return (
    <div className="flex items-center gap-1.5 text-slate-600">
      <Icon size={13} className="text-slate-400" />
      <span className="text-xs">{label}</span>
      <span className="text-xs font-semibold text-slate-800">{value}</span>
    </div>
  )
}

function CitationChip({ id }) {
  return (
    <a
      href={`https://arxiv.org/abs/${id}`}
      target="_blank"
      rel="noreferrer"
      className="inline-flex items-center gap-1 text-xs font-medium
                 bg-navy-50 text-navy-600 border border-navy-100
                 rounded-full px-2.5 py-0.5 hover:bg-navy-100 transition-colors"
    >
      arXiv:{id}
      <ExternalLink size={10} />
    </a>
  )
}

// ── Main Component ────────────────────────────────────────────────────────────

export default function QueryPage() {
  const [query, setQuery]           = useState('')
  const [configName, setConfigName] = useState('full_agent')
  const [configs, setConfigs]       = useState({})
  const [result, setResult]         = useState(null)
  const [error, setError]           = useState(null)
  const [loading, setLoading]       = useState(false)
  const [elapsed, setElapsed]       = useState(0)
  const [traceOpen, setTraceOpen]   = useState(false)

  useEffect(() => { api.getConfigs().then(setConfigs).catch(() => {}) }, [])

  // Live elapsed timer while loading
  useEffect(() => {
    if (!loading) return
    setElapsed(0)
    const id = setInterval(() => setElapsed(s => s + 1), 1000)
    return () => clearInterval(id)
  }, [loading])

  async function handleSubmit(e) {
    e?.preventDefault()
    if (!query.trim() || loading) return
    setLoading(true)
    setResult(null)
    setError(null)
    setTraceOpen(false)
    try {
      const data = await api.runResearch({ query: query.trim(), config_name: configName })
      setResult(data)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="max-w-4xl mx-auto px-4 sm:px-6 py-8">

      {/* Page heading */}
      <div className="mb-7">
        <h1 className="text-xl font-semibold text-slate-900">Research Query</h1>
        <p className="text-sm text-slate-500 mt-0.5">
          Ask a research question about LLM agents. The agent will retrieve,
          reason, and synthesize an answer from the corpus.
        </p>
      </div>

      {/* Query form card */}
      <div className="card p-5 mb-6">
        <form onSubmit={handleSubmit} className="space-y-4">

          {/* Text area */}
          <div>
            <label className="block text-xs font-semibold uppercase tracking-widest text-slate-400 mb-1.5">
              Question
            </label>
            <div className="relative">
              <textarea
                value={query}
                onChange={e => setQuery(e.target.value)}
                onKeyDown={e => { if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) handleSubmit() }}
                rows={3}
                placeholder="e.g. What is the ReAct framework and how does it combine reasoning with acting?"
                className="w-full resize-none rounded-lg border border-slate-200 px-4 py-3 text-sm
                           text-slate-800 placeholder:text-slate-400 outline-none
                           focus:border-navy-400 focus:ring-2 focus:ring-navy-100
                           transition-colors duration-150"
              />
              <span className="absolute bottom-2.5 right-3 text-[10px] text-slate-400">
                Ctrl+Enter to submit
              </span>
            </div>
          </div>

          {/* Examples */}
          <div>
            <label className="block text-xs font-semibold uppercase tracking-widest text-slate-400 mb-1.5">
              Examples
            </label>
            <div className="flex flex-wrap gap-1.5">
              {EXAMPLES.map(ex => (
                <button
                  key={ex}
                  type="button"
                  onClick={() => setQuery(ex)}
                  className="text-xs border border-slate-200 rounded-full px-3 py-1
                             text-slate-600 hover:border-navy-300 hover:text-navy-600
                             hover:bg-navy-50 transition-colors truncate max-w-[280px]"
                >
                  {ex}
                </button>
              ))}
            </div>
          </div>

          {/* Config + Submit row */}
          <div className="flex items-end gap-3 flex-wrap">
            <div className="flex-1 min-w-[180px]">
              <label className="block text-xs font-semibold uppercase tracking-widest text-slate-400 mb-1.5">
                Agent Configuration
              </label>
              <div className="relative">
                <select
                  value={configName}
                  onChange={e => setConfigName(e.target.value)}
                  className="w-full appearance-none rounded-lg border border-slate-200
                             bg-white px-3 py-2.5 text-sm text-slate-700 outline-none
                             focus:border-navy-400 focus:ring-2 focus:ring-navy-100
                             transition-colors duration-150 pr-8"
                >
                  {Object.keys(configs).map(name => (
                    <option key={name} value={name}>
                      {CONFIG_META[name]?.label ?? name} — {CONFIG_META[name]?.desc ?? ''}
                    </option>
                  ))}
                </select>
                <ChevronDown size={14} className="pointer-events-none absolute right-2.5 top-3.5 text-slate-400" />
              </div>
            </div>

            <button
              type="submit"
              disabled={loading || !query.trim()}
              className="btn-accent"
            >
              {loading
                ? <><Loader2 size={15} className="animate-spin" /> Researching…</>
                : <><Search size={15} /> Run Research</>
              }
            </button>

            {result && (
              <button
                type="button"
                onClick={() => { setResult(null); setQuery(''); setError(null) }}
                className="btn-secondary text-sm"
              >
                <RotateCcw size={14} /> Reset
              </button>
            )}
          </div>
        </form>
      </div>

      {/* Loading state */}
      {loading && (
        <div className="card p-8 text-center">
          <div className="w-12 h-12 rounded-full bg-amber-50 border-4 border-amber-200
                          flex items-center justify-center mx-auto mb-4 animate-pulse">
            <Cpu size={22} className="text-amber-500" />
          </div>
          <p className="text-sm font-medium text-slate-700">Agent is researching…</p>
          <p className="text-xs text-slate-500 mt-1">
            Planning → Retrieval → Reflection → Synthesis
          </p>
          <div className="mt-4 inline-flex items-center gap-1.5 bg-slate-100 rounded-full px-3 py-1.5">
            <Clock size={12} className="text-slate-500" />
            <span className="text-xs font-mono text-slate-600">{elapsed}s elapsed</span>
          </div>
          <p className="text-[11px] text-slate-400 mt-3">
            Typically 30–90 seconds depending on configuration
          </p>
        </div>
      )}

      {/* Error state */}
      {error && !loading && (
        <div className="card border-red-100 p-5 flex gap-3">
          <AlertCircle size={18} className="text-red-500 flex-shrink-0 mt-0.5" />
          <div>
            <p className="text-sm font-medium text-red-700">Research failed</p>
            <p className="text-xs text-red-500 mt-0.5">{error}</p>
            <p className="text-xs text-slate-500 mt-2">
              Make sure the FastAPI backend is running: <code className="bg-slate-100 px-1 rounded">uvicorn app.api:app --reload --port 8000</code>
            </p>
          </div>
        </div>
      )}

      {/* Results */}
      {result && !loading && (
        <div className="space-y-4">

          {/* Summary metrics bar */}
          <div className="card px-5 py-3.5 flex items-center flex-wrap gap-x-6 gap-y-2">
            <div className="flex items-center gap-2">
              <CheckCircle size={15} className="text-emerald-500" />
              <span className="text-xs font-medium text-slate-700">
                {CONFIG_META[result.config_name]?.label ?? result.config_name}
              </span>
            </div>
            <MetricPill icon={Zap}   label="Confidence"  value={(result.confidence * 100).toFixed(0) + '%'} />
            <MetricPill icon={RotateCcw} label="Iterations" value={result.iterations} />
            <MetricPill icon={Search}    label="Tool calls" value={result.tool_calls} />
            <MetricPill icon={Clock}     label="Latency"    value={result.latency_seconds.toFixed(1) + 's'} />
            <div className="ml-auto">
              {result.accuracy && <AccBadge score={result.accuracy} />}
            </div>
          </div>

          {/* Answer */}
          <div className="card p-6">
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-sm font-semibold text-slate-900 uppercase tracking-wider flex items-center gap-2">
                <span className="w-1 h-4 bg-amber-500 rounded-full inline-block" />
                Answer
              </h2>
            </div>
            <div className="prose-answer text-sm">
              <ReactMarkdown remarkPlugins={[remarkGfm]}>
                {result.answer}
              </ReactMarkdown>
            </div>
          </div>

          {/* Citations */}
          {result.citations?.length > 0 && (
            <div className="card px-5 py-4">
              <h3 className="text-xs font-semibold uppercase tracking-widest text-slate-400 mb-3 flex items-center gap-2">
                <BookOpen size={12} />
                Citations
              </h3>
              <div className="flex flex-wrap gap-2">
                {result.citations.map(id => (
                  <CitationChip key={id} id={id} />
                ))}
              </div>
            </div>
          )}

          {result.citations?.length === 0 && (
            <div className="card px-5 py-3.5 flex items-center gap-2 text-slate-500">
              <AlertCircle size={14} className="text-amber-400" />
              <span className="text-xs">
                No citations — the citation verifier may have filtered all references.
                Try the <strong>No Verifier</strong> configuration for retained citations.
              </span>
            </div>
          )}
        </div>
      )}
    </div>
  )
}
