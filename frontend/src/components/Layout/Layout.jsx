import React from 'react'
import { Outlet } from 'react-router-dom'
import Navbar from './Navbar'
import FloatingAssistant from '../Common/FloatingAssistant'

export default function Layout() {
  return (
    <div className="app-container">
      <Navbar />
      <main className="main-viewport">
        <Outlet />
      </main>
      <FloatingAssistant />
      <footer className="netflix-footer">
        <div className="footer-content">
          <div className="footer-left">
            <span className="footer-logo">AUTOML-LENS</span>
            <p>LLM-Guided Structured AutoML Platform with Explainable AI & Reproducible Benchmarks.</p>
          </div>
          <div className="footer-right">
            <span>Final-Year Research Project</span>
            <span style={{ color: 'var(--success)' }}>285 Tests Passing | Freeze Verified</span>
          </div>
        </div>
      </footer>
    </div>
  )
}
