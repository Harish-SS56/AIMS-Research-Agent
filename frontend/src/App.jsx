import { useState, useEffect } from 'react'
import { Routes, Route } from 'react-router-dom'
import Header from './components/Header.jsx'
import Sidebar from './components/Sidebar.jsx'
import QueryPage from './pages/QueryPage.jsx'
import AblationPage from './pages/AblationPage.jsx'
import PapersPage from './pages/PapersPage.jsx'
import AboutPage from './pages/AboutPage.jsx'
import { api } from './api.js'

export default function App() {
  const [sidebarOpen, setSidebarOpen] = useState(false)
  const [stats, setStats] = useState(null)

  useEffect(() => {
    api.getStats().then(setStats).catch(() => {})
  }, [])

  return (
    <div className="min-h-screen bg-slate-50">
      <Header sidebarOpen={sidebarOpen} setSidebarOpen={setSidebarOpen} />

      <div className="flex pt-14 min-h-screen">
        <Sidebar
          open={sidebarOpen}
          onClose={() => setSidebarOpen(false)}
          stats={stats}
        />

        {/* Main content — offset by sidebar width on large screens */}
        <main className="flex-1 lg:ml-56 min-h-screen">
          <Routes>
            <Route path="/"          element={<QueryPage />} />
            <Route path="/ablation"  element={<AblationPage />} />
            <Route path="/papers"    element={<PapersPage />} />
            <Route path="/about"     element={<AboutPage />} />
          </Routes>
        </main>
      </div>
    </div>
  )
}
