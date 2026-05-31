import { useState, useEffect, useMemo } from 'react'
import { Search, ExternalLink, Tag, Calendar, ChevronDown, BookOpen } from 'lucide-react'
import { api } from '../api.js'

const CATEGORY_COLOR = {
  'Agent Systems':  'bg-navy-100 text-navy-700 border-navy-200',
  'RAG':            'bg-amber-100 text-amber-700 border-amber-200',
  'Reasoning':      'bg-purple-100 text-purple-700 border-purple-200',
  'Tool Use':       'bg-emerald-100 text-emerald-700 border-emerald-200',
  'Planning':       'bg-sky-100 text-sky-700 border-sky-200',
  'Memory':         'bg-rose-100 text-rose-700 border-rose-200',
  'Evaluation':     'bg-orange-100 text-orange-700 border-orange-200',
  'Multimodal':     'bg-fuchsia-100 text-fuchsia-700 border-fuchsia-200',
}

function catColor(cat) {
  return CATEGORY_COLOR[cat] ?? 'bg-slate-100 text-slate-600 border-slate-200'
}

function PaperCard({ paper }) {
  const [expanded, setExpanded] = useState(false)
  const abs = paper.abstract ?? ''
  const short = abs.length > 220 ? abs.slice(0, 220) + '…' : abs

  return (
    <div className="card p-5 hover:shadow-card-hover transition-shadow duration-150">
      {/* Title row */}
      <div className="flex items-start justify-between gap-3 mb-2">
        <h3 className="font-semibold text-slate-900 text-sm leading-snug line-clamp-2 flex-1">
          {paper.title}
        </h3>
        {paper.arxiv_id && (
          <a
            href={`https://arxiv.org/abs/${paper.arxiv_id}`}
            target="_blank"
            rel="noreferrer"
            className="flex-shrink-0 text-navy-500 hover:text-amber-500 transition-colors"
            title="View on arXiv"
          >
            <ExternalLink size={14} />
          </a>
        )}
      </div>

      {/* Meta row */}
      <div className="flex flex-wrap items-center gap-x-3 gap-y-1 mb-3">
        {paper.authors && (
          <span className="text-xs text-slate-500 truncate max-w-[260px]">
            {Array.isArray(paper.authors)
              ? paper.authors.slice(0, 3).join(', ') + (paper.authors.length > 3 ? ' et al.' : '')
              : paper.authors}
          </span>
        )}
        {paper.year && (
          <span className="flex items-center gap-0.5 text-xs text-slate-400">
            <Calendar size={11} />
            {paper.year}
          </span>
        )}
      </div>

      {/* Category badges */}
      {paper.categories?.length > 0 && (
        <div className="flex flex-wrap gap-1.5 mb-3">
          {paper.categories.map(c => (
            <span key={c} className={`badge border text-[10px] ${catColor(c)}`}>{c}</span>
          ))}
        </div>
      )}

      {/* Abstract */}
      {abs && (
        <div>
          <p className="text-xs text-slate-600 leading-relaxed">
            {expanded ? abs : short}
          </p>
          {abs.length > 220 && (
            <button
              onClick={() => setExpanded(e => !e)}
              className="mt-1 text-[11px] text-navy-500 hover:text-navy-700 flex items-center gap-0.5"
            >
              {expanded ? 'Show less' : 'Read more'}
              <ChevronDown size={11} className={`transition-transform ${expanded ? 'rotate-180' : ''}`} />
            </button>
          )}
        </div>
      )}

      {/* ArXiv ID */}
      {paper.arxiv_id && (
        <div className="mt-3 pt-3 border-t border-slate-50">
          <span className="text-[10px] font-mono text-slate-400">arXiv:{paper.arxiv_id}</span>
        </div>
      )}
    </div>
  )
}

export default function PapersPage() {
  const [papers, setPapers]     = useState([])
  const [loading, setLoading]   = useState(true)
  const [search, setSearch]     = useState('')
  const [category, setCategory] = useState('All')

  useEffect(() => {
    api.getPapers()
      .then(data => setPapers(Array.isArray(data) ? data : data.papers ?? []))
      .catch(() => setPapers([]))
      .finally(() => setLoading(false))
  }, [])

  const categories = useMemo(() => {
    const set = new Set()
    papers.forEach(p => p.categories?.forEach(c => set.add(c)))
    return ['All', ...Array.from(set).sort()]
  }, [papers])

  const filtered = useMemo(() => {
    const q = search.toLowerCase()
    return papers.filter(p => {
      const matchCat = category === 'All' || p.categories?.includes(category)
      if (!matchCat) return false
      if (!q) return true
      const haystack = [
        p.title, p.arxiv_id,
        Array.isArray(p.authors) ? p.authors.join(' ') : p.authors,
        p.abstract,
      ].filter(Boolean).join(' ').toLowerCase()
      return haystack.includes(q)
    })
  }, [papers, search, category])

  return (
    <div className="max-w-5xl mx-auto px-4 sm:px-6 py-8">

      {/* Heading */}
      <div className="mb-6">
        <h1 className="text-xl font-semibold text-slate-900">Corpus</h1>
        <p className="text-sm text-slate-500 mt-0.5">
          {papers.length} arXiv papers on LLM agents, RAG, reasoning, and tool use.
        </p>
      </div>

      {/* Search + filter bar */}
      <div className="flex flex-col sm:flex-row gap-3 mb-6">
        <div className="relative flex-1">
          <Search size={14} className="absolute left-3 top-2.5 text-slate-400 pointer-events-none" />
          <input
            value={search}
            onChange={e => setSearch(e.target.value)}
            placeholder="Search by title, author, arxiv ID…"
            className="w-full pl-9 pr-4 py-2.5 text-sm border border-slate-200 rounded-lg
                       outline-none focus:border-navy-400 focus:ring-2 focus:ring-navy-100
                       bg-white text-slate-700 placeholder:text-slate-400"
          />
        </div>
        <div className="relative">
          <Tag size={13} className="absolute left-3 top-3 text-slate-400 pointer-events-none" />
          <select
            value={category}
            onChange={e => setCategory(e.target.value)}
            className="appearance-none pl-8 pr-8 py-2.5 text-sm border border-slate-200
                       rounded-lg outline-none focus:border-navy-400 focus:ring-2 focus:ring-navy-100
                       bg-white text-slate-700"
          >
            {categories.map(c => (
              <option key={c} value={c}>{c}</option>
            ))}
          </select>
          <ChevronDown size={13} className="pointer-events-none absolute right-2.5 top-3 text-slate-400" />
        </div>
      </div>

      {/* Results count */}
      <div className="flex items-center gap-2 mb-4">
        <BookOpen size={13} className="text-slate-400" />
        <span className="text-xs text-slate-500">
          {loading ? 'Loading…' : `${filtered.length} paper${filtered.length !== 1 ? 's' : ''}`}
          {search || category !== 'All' ? ` matching filters` : ''}
        </span>
      </div>

      {/* Grid */}
      {loading ? (
        <div className="grid sm:grid-cols-2 gap-4">
          {Array.from({ length: 6 }).map((_, i) => (
            <div key={i} className="card p-5 space-y-3 animate-pulse">
              <div className="h-4 bg-slate-100 rounded w-4/5" />
              <div className="h-3 bg-slate-100 rounded w-1/2" />
              <div className="space-y-1.5">
                <div className="h-3 bg-slate-100 rounded" />
                <div className="h-3 bg-slate-100 rounded w-5/6" />
              </div>
            </div>
          ))}
        </div>
      ) : filtered.length === 0 ? (
        <div className="text-center py-16 text-slate-400">
          <BookOpen size={40} className="mx-auto mb-3 opacity-30" />
          <p className="text-sm">No papers match your search.</p>
        </div>
      ) : (
        <div className="grid sm:grid-cols-2 gap-4">
          {filtered.map((p, i) => (
            <PaperCard key={p.arxiv_id ?? i} paper={p} />
          ))}
        </div>
      )}
    </div>
  )
}
