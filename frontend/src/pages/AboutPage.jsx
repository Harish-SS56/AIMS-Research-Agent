import { FlaskConical, BookOpen, BarChart2, Search, ExternalLink, GitBranch } from 'lucide-react'
import { useState, useEffect } from 'react'
import { api } from '../api'

const PIPELINE = [
  { label: 'Planner',          color: 'bg-navy-100 text-navy-700',    desc: 'Decomposes the query into sub-questions to guide retrieval.' },
  { label: 'Retriever',        color: 'bg-amber-100 text-amber-700',  desc: 'Hybrid BM25 + semantic search with RRF fusion over 33,175 chunks.' },
  { label: 'Reranker',         color: 'bg-purple-100 text-purple-700', desc: 'LLM-based reranking of top-k candidates for relevance.' },
  { label: 'Reader',           color: 'bg-sky-100 text-sky-700',       desc: 'Reads and summarises the top passages.' },
  { label: 'Reflector',        color: 'bg-emerald-100 text-emerald-700', desc: 'Evaluates answer completeness; triggers another retrieval loop if needed.' },
  { label: 'Synthesizer',      color: 'bg-orange-100 text-orange-700', desc: 'Combines evidence into a coherent, structured answer.' },
  { label: 'Citation Verifier', color: 'bg-rose-100 text-rose-700',   desc: 'Filters citations to only include directly grounding references.' },
]

function Step({ idx, label, color, desc }) {
  return (
    <div className="flex gap-3">
      <div className="flex flex-col items-center">
        <div className={`w-7 h-7 rounded-full flex items-center justify-center text-xs font-bold flex-shrink-0 ${color}`}>
          {idx + 1}
        </div>
        {idx < PIPELINE.length - 1 && (
          <div className="w-px flex-1 bg-slate-100 my-1" />
        )}
      </div>
      <div className="pb-4">
        <p className={`text-xs font-semibold inline-block px-2 py-0.5 rounded ${color}`}>{label}</p>
        <p className="text-xs text-slate-500 mt-1 leading-relaxed">{desc}</p>
      </div>
    </div>
  )
}

export default function AboutPage() {
  const [stats, setStats] = useState(null)
  useEffect(() => {
    api.getStats().then(setStats).catch(() => {})
  }, [])

  return (
    <div className="max-w-3xl mx-auto px-4 sm:px-6 py-8 space-y-6">

      {/* Heading */}
      <div>
        <h1 className="text-xl font-semibold text-slate-900">About</h1>
        <p className="text-sm text-slate-500 mt-0.5">
          AIMS-DTU Research Intern 2026 — Round 2 Assignment
        </p>
      </div>

      {/* Overview card */}
      <div className="card p-6 space-y-3">
        <div className="flex items-center gap-2 mb-1">
          <div className="w-7 h-7 rounded bg-amber-500 flex items-center justify-center">
            <FlaskConical size={14} className="text-white" />
          </div>
          <h2 className="font-semibold text-slate-800">AIMS Research Agent</h2>
        </div>
        <p className="text-sm text-slate-600 leading-relaxed">
          A multi-agent pipeline for research question answering over a curated corpus of {stats ? stats.num_papers : 574} arXiv papers
          on LLM agents, retrieval-augmented generation, reasoning, and tool use.
          The system implements a 7-stage pipeline with hybrid retrieval, iterative reflection,
          and citation verification.
        </p>
        <div className="grid grid-cols-3 gap-3 pt-2">
          {[
            { label: 'Papers',  value: stats ? String(stats.num_papers) : '574',    icon: BookOpen },
            { label: 'Chunks',  value: stats ? stats.num_chunks.toLocaleString() : '33,175', icon: GitBranch },
            { label: 'Configs', value: '7', icon: BarChart2 },
          ].map(({ label, value, icon: Icon }) => (
            <div key={label} className="bg-slate-50 rounded-lg p-3 text-center">
              <Icon size={16} className="mx-auto text-slate-400 mb-1" />
              <div className="text-lg font-bold text-slate-900">{value}</div>
              <div className="text-[11px] text-slate-500">{label}</div>
            </div>
          ))}
        </div>
      </div>

      {/* Pipeline card */}
      <div className="card p-6">
        <h2 className="font-semibold text-slate-800 mb-4 text-sm flex items-center gap-2">
          <Search size={14} className="text-amber-500" />
          Agent Pipeline
        </h2>
        <div>
          {PIPELINE.map((step, i) => (
            <Step key={step.label} idx={i} {...step} />
          ))}
        </div>
      </div>

      {/* Tech stack */}
      <div className="card p-6">
        <h2 className="font-semibold text-slate-800 mb-3 text-sm">Tech Stack</h2>
        <div className="grid grid-cols-2 sm:grid-cols-3 gap-2">
          {[
            ['Azure OpenAI', 'GPT-4o + text-embedding-3-large'],
            ['ChromaDB', 'Vector store for semantic search'],
            ['BM25', 'Keyword retrieval via rank-bm25'],
            ['FastAPI', 'REST backend, port 8000'],
            ['React 18 + Vite', 'Frontend dev server'],
            ['Tailwind CSS', 'DTU-inspired color scheme'],
          ].map(([name, desc]) => (
            <div key={name} className="bg-slate-50 rounded-lg p-3">
              <p className="text-xs font-semibold text-slate-700">{name}</p>
              <p className="text-[11px] text-slate-500 mt-0.5 leading-snug">{desc}</p>
            </div>
          ))}
        </div>
      </div>

      {/* Links */}
      <div className="card p-5 flex flex-wrap gap-3">
        <a
          href="https://arxiv.org"
          target="_blank"
          rel="noreferrer"
          className="btn-secondary text-xs"
        >
          <ExternalLink size={12} /> arXiv
        </a>
        <a
          href="https://dtu.ac.in"
          target="_blank"
          rel="noreferrer"
          className="btn-secondary text-xs"
        >
          <ExternalLink size={12} /> DTU
        </a>
      </div>
    </div>
  )
}
