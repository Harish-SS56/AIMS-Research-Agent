import { NavLink } from 'react-router-dom'
import {
  Search, BarChart2, BookOpen, Info, ChevronRight,
} from 'lucide-react'

const NAV = [
  { to: '/',          icon: Search,    label: 'Research Query' },
  { to: '/ablation',  icon: BarChart2, label: 'Ablation Study' },
  { to: '/papers',    icon: BookOpen,  label: 'Corpus' },
  { to: '/about',     icon: Info,      label: 'About' },
]

const CONFIG_LABELS = {
  full_agent:   'Full Agent',
  baseline:     'Baseline',
  no_planner:   'No Planner',
  no_reranker:  'No Reranker',
  no_reflector: 'No Reflector',
  no_hybrid:    'No Hybrid',
  no_verifier:  'No Verifier',
}

export default function Sidebar({ open, onClose, stats }) {
  return (
    <>
      {/* Mobile backdrop */}
      {open && (
        <div
          className="fixed inset-0 z-30 bg-black/30 lg:hidden"
          onClick={onClose}
        />
      )}

      <aside
        className={[
          'fixed top-14 bottom-0 left-0 z-30 w-56 bg-white border-r border-slate-100',
          'flex flex-col overflow-y-auto transition-transform duration-200',
          open ? 'translate-x-0' : '-translate-x-full lg:translate-x-0',
        ].join(' ')}
      >
        {/* Navigation */}
        <nav className="flex-1 py-4 px-2 space-y-0.5">
          <p className="px-3 mb-2 text-[10px] font-semibold uppercase tracking-widest text-slate-400">
            Navigation
          </p>
          {NAV.map(({ to, icon: Icon, label }) => (
            <NavLink
              key={to}
              to={to}
              end={to === '/'}
              onClick={onClose}
              className={({ isActive }) =>
                [
                  'flex items-center gap-3 px-3 py-2 rounded-lg text-sm font-medium transition-colors duration-100',
                  isActive
                    ? 'bg-amber-50 text-amber-700 border-l-2 border-amber-500 pl-[10px]'
                    : 'text-slate-600 hover:bg-slate-50 hover:text-slate-900',
                ].join(' ')
              }
            >
              <Icon size={16} className="flex-shrink-0" />
              {label}
            </NavLink>
          ))}
        </nav>

        {/* Stats card */}
        {stats && (
          <div className="mx-3 mb-4 p-3 rounded-lg bg-navy-50 border border-navy-100">
            <p className="text-[10px] font-semibold uppercase tracking-widest text-navy-400 mb-2">
              Corpus
            </p>
            <div className="space-y-1.5">
              <Stat label="Papers" value={stats.num_papers} title="Indexed arXiv papers" />
              <Stat label="Chunks" value={stats.num_chunks} />
              <Stat
                label="Configs Evaluated"
                value={`${stats.configs_evaluated}/${stats.total_configs}`}
              />
            </div>
          </div>
        )}

        {/* Footer */}
        <div className="px-4 py-3 border-t border-slate-100">
          <p className="text-[10px] text-slate-400 leading-snug">
            AIMS-DTU Research Intern 2026<br />
            Round 2 Assignment
          </p>
        </div>
      </aside>
    </>
  )
}

function Stat({ label, value }) {
  return (
    <div className="flex items-center justify-between">
      <span className="text-xs text-slate-500">{label}</span>
      <span className="text-xs font-semibold text-navy-700">{value}</span>
    </div>
  )
}
