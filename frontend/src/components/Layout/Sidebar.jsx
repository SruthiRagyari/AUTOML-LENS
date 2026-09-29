import React from 'react'
import { NavLink } from 'react-router-dom'
import { FiGrid, FiPlus, FiClock, FiBook, FiInfo, FiZap } from 'react-icons/fi'

const LINKS = [
  { to: '/dashboard', icon: <FiGrid />, label: 'Dashboard' },
  { to: '/dashboard?new=1', icon: <FiPlus />, label: 'New Experiment' },
  { to: '/history', icon: <FiClock />, label: 'History' },
  { to: '/methodology', icon: <FiBook />, label: 'Methodology' },
  { to: '/about', icon: <FiInfo />, label: 'About' },
]

export default function Sidebar() {
  return (
    <div className="sidebar">
      <div className="sidebar-logo">
        <h1>🔬 AutoML-Lens</h1>
        <span>LLM-Powered AutoML</span>
      </div>
      <nav className="sidebar-nav">
        {LINKS.map(({ to, icon, label }) => (
          <NavLink
            key={to}
            to={to.split('?')[0]}
            className={({ isActive }) => `nav-link${isActive ? ' active' : ''}`}
          >
            {icon}
            {label}
          </NavLink>
        ))}
      </nav>
      <div className="sidebar-footer">
        <p>B.Tech Final Year Project</p>
        <span className="version-badge">v1.0.0</span>
      </div>
    </div>
  )
}
