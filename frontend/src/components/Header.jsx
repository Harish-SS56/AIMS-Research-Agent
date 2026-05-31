import { Menu, X, FlaskConical } from 'lucide-react'
import { useState, useEffect } from 'react'

export default function Header({ sidebarOpen, setSidebarOpen }) {
  const [numPapers, setNumPapers] = useState(null)

  useEffect(() => {
    fetch('/api/stats')
      .then(r => r.json())
      .then(d => setNumPapers(d.num_papers))  // num_papers = indexed arXiv papers only
      .catch(() => {})
  }, [])

  return (
    <header className="fixed top-0 left-0 right-0 z-40 h-14 bg-navy-900 flex items-center px-4 gap-4 select-none">
      {/* Mobile toggle */}
      <button
        className="lg:hidden text-white/70 hover:text-white p-1"
        onClick={() => setSidebarOpen(o => !o)}
        aria-label="Toggle sidebar"
      >
        {sidebarOpen ? <X size={20} /> : <Menu size={20} />}
      </button>

      {/* Logo + wordmark */}
      <div className="flex items-center gap-2.5">
        <div className="w-7 h-7 rounded bg-amber-500 flex items-center justify-center flex-shrink-0">
          <FlaskConical size={15} className="text-white" />
        </div>
        <div className="leading-tight">
          <div className="text-white font-semibold text-sm tracking-wide">
            AIMS Research Agent
          </div>
          <div className="text-white/50 text-[10px] tracking-wider uppercase">
            Delhi Technological University · 2026
          </div>
        </div>
      </div>

      {/* Spacer */}
      <div className="flex-1" />

      {/* Status pill */}
      <div className="hidden sm:flex items-center gap-1.5 bg-white/10 rounded-full px-3 py-1">
        <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" />
        <span className="text-white/70 text-xs font-medium">API Live</span>
      </div>

      {/* DTU badge */}
      <div className="hidden md:block text-white/30 text-xs border-l border-white/10 pl-4 ml-1">
        arXiv · {numPapers ?? '…'} papers · GPT-4o
      </div>
    </header>
  )
}
