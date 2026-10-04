import React, { useState, useEffect } from 'react'
import { NavLink, Link, useNavigate } from 'react-router-dom'
import {
  FiGrid,
  FiDatabase,
  FiCpu,
  FiTarget,
  FiFileText,
  FiMessageSquare,
  FiPlus,
  FiMenu,
  FiX
} from 'react-icons/fi'

export default function Navbar() {
  const [scrolled, setScrolled] = useState(false)
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false)
  const navigate = useNavigate()

  useEffect(() => {
    const handleScroll = () => {
      setScrolled(window.scrollY > 20)
    }
    window.addEventListener('scroll', handleScroll, { passive: true })
    return () => window.removeEventListener('scroll', handleScroll)
  }, [])

  const navLinks = [
    { to: '/', label: 'Home' },
    { to: '/dashboard', label: 'Experiments' },
    { to: '/datasets', label: 'Datasets' },
    { to: '/models', label: 'Models' },
    { to: '/predictions', label: 'Predictions' },
    { to: '/reports', label: 'Reports' },
    { to: '/assistant', label: 'AI Assistant' },
  ]

  return (
    <header className={`netflix-navbar ${scrolled ? 'scrolled' : ''}`}>
      <div className="nav-left">
        <Link to="/" className="netflix-brand">
          <span className="netflix-brand-logo">AUTOML-LENS</span>
        </Link>

        <nav className="nav-links">
          {navLinks.map((item) => (
            <NavLink
              key={item.to}
              to={item.to}
              className={({ isActive }) => `nav-link-item ${isActive ? 'active' : ''}`}
            >
              {item.label}
            </NavLink>
          ))}
        </nav>
      </div>

      <div className="nav-right">
        <div className="system-status-pill" title="FastAPI backend connected and operating">
          <span className="status-dot-pulse"></span>
          <span>System Online</span>
        </div>

        <button
          className="btn-netflix-primary"
          onClick={() => navigate('/dashboard?new=1')}
          title="Create a new AutoML experiment"
          type="button"
        >
          <FiPlus />
          <span>New Experiment</span>
        </button>

        <button
          className="mobile-nav-toggle"
          onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
          aria-label="Toggle Navigation Menu"
          type="button"
        >
          {mobileMenuOpen ? <FiX /> : <FiMenu />}
        </button>
      </div>

      {mobileMenuOpen && (
        <div className="mobile-menu-drawer">
          {navLinks.map((item) => (
            <Link
              key={item.to}
              to={item.to}
              className="mobile-nav-item"
              onClick={() => setMobileMenuOpen(false)}
            >
              <span>{item.label}</span>
            </Link>
          ))}
          <Link
            to="/dashboard?new=1"
            className="btn-netflix-primary"
            style={{ width: '100%', marginTop: '16px' }}
            onClick={() => setMobileMenuOpen(false)}
          >
            <FiPlus /> New Experiment
          </Link>
        </div>
      )}
    </header>
  )
}
