import React from 'react'
import { BrowserRouter, Routes, Route } from 'react-router-dom'
import Landing from './pages/Landing'
import Dashboard from './pages/Dashboard'
import Datasets from './pages/Datasets'
import Models from './pages/Models'
import Predictions from './pages/Predictions'
import Reports from './pages/Reports'
import Assistant from './pages/Assistant'
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
          <Route path="/experiments" element={<Dashboard />} />
          <Route path="/datasets" element={<Datasets />} />
          <Route path="/models" element={<Models />} />
          <Route path="/predictions" element={<Predictions />} />
          <Route path="/reports" element={<Reports />} />
          <Route path="/assistant" element={<Assistant />} />
          <Route path="/experiment/:id" element={<Experiment />} />
          <Route path="/history" element={<History />} />
          <Route path="/methodology" element={<Methodology />} />
          <Route path="/about" element={<About />} />
        </Route>
      </Routes>
    </BrowserRouter>
  )
}
