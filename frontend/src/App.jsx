import React from 'react'
import { BrowserRouter, Routes, Route } from 'react-router-dom'
import Landing from './pages/Landing'
import Dashboard from './pages/Dashboard'
import Experiment from './pages/Experiment'
import History from './pages/History'
import Methodology from './pages/Methodology'
import About from './pages/About'
import Layout from './components/Layout/Layout'

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Landing />} />
        <Route element={<Layout />}>
          <Route path="/dashboard" element={<Dashboard />} />
          <Route path="/experiment/:id" element={<Experiment />} />
          <Route path="/history" element={<History />} />
          <Route path="/methodology" element={<Methodology />} />
          <Route path="/about" element={<About />} />
        </Route>
      </Routes>
    </BrowserRouter>
  )
}
