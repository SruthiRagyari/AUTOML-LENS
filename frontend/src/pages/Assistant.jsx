import React, { useState, useEffect, useRef } from 'react'
import {
  FiMessageSquare,
  FiSend,
  FiTrash2,
  FiRotateCcw,
  FiCpu,
  FiInfo,
  FiCheck
} from 'react-icons/fi'
import ReactMarkdown from 'react-markdown'
import { api } from '../services/api'
import CodeBlock from '../components/Common/CodeBlock'
import LoadingSpinner from '../components/Common/LoadingSpinner'

const SAMPLE_PROMPTS = [
  'What is quantum computing?',
  'Explain Python decorators with a code example',
  'Write a Python function to reverse a string',
  'What is the difference between AI and ML?',
  'Give me a recipe for pasta',
  'How does AutoML-Lens prevent data leakage in holdout evaluation?',
  'Explain how Optuna Bayesian optimization works',
]

export default function Assistant() {
  const [messages, setMessages] = useState([
    {
      role: 'assistant',
      content:
        'Hello! I am your general-purpose AI assistant. I can help you with anything — programming, machine learning, mathematics, interview prep, science, cooking recipes, or explaining your AutoML-Lens experiments. Ask me whatever you like!',
    },
  ])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [contextMode, setContextMode] = useState('general')
  const [experiments, setExperiments] = useState([])
  const [selectedExpId, setSelectedExpId] = useState('')
  const [activeProvider, setActiveProvider] = useState('Active Provider')
  const messagesEndRef = useRef(null)

  useEffect(() => {
    loadExperiments()
  }, [])

  useEffect(() => {
    scrollToBottom()
  }, [messages])

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }

  const loadExperiments = async () => {
    try {
      const res = await api.listExperiments()
      setExperiments(res.data?.experiments || res.data || [])
      if (res.data?.length > 0) setSelectedExpId(String(res.data[0].id))
    } catch (err) {
      console.error('Failed to load experiments:', err)
    }
  }

  const handleSend = async (textToSend) => {
    const text = typeof textToSend === 'string' ? textToSend : input
    if (!text.trim() || loading) return

    const newMessages = [...messages, { role: 'user', content: text.trim() }]
    setMessages(newMessages)
    setInput('')
    setLoading(true)

    try {
      const history = newMessages
        .slice(-10)
        .map((m) => ({ role: m.role, content: m.content }))

      const expId = contextMode === 'project' && selectedExpId ? Number(selectedExpId) : null
      const res = await api.chat(text.trim(), expId, contextMode, history)

      setActiveProvider(res.data.provider || 'AI Provider')
      setMessages((prev) => [
        ...prev,
        {
          role: 'assistant',
          content: res.data.response || 'No response returned.',
          is_fallback: res.data.is_fallback,
          context_mode: res.data.context_mode,
        },
      ])
    } catch (err) {
      setMessages((prev) => [
        ...prev,
        {
          role: 'assistant',
          content: 'I apologize, but I encountered an error communicating with the AI service. Please verify your backend server.',
          error: true,
        },
      ])
    } finally {
      setLoading(false)
    }
  }

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSend()
    }
  }

  const clearChat = () => {
    setMessages([
      {
        role: 'assistant',
        content: 'Conversation cleared. How can I help you now?',
      },
    ])
  }

  return (
    <div className="page-container">
      {/* Top Header / Mode Switcher */}
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '20px', flexWrap: 'wrap', gap: '16px' }}>
        <div>
          <h1 style={{ fontSize: '28px', fontWeight: '900', color: '#ffffff', display: 'flex', alignItems: 'center', gap: '10px' }}>
            <FiCpu style={{ color: 'var(--netflix-red)' }} />
            <span>AI Assistant</span>
          </h1>
          <p style={{ color: 'var(--text-secondary)', fontSize: '14px' }}>
            General-purpose AI assistant powered by Gemini / OpenAI with optional AutoML project context
          </p>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
          <div style={{ display: 'flex', background: '#111', padding: '3px', borderRadius: 'var(--radius-sm)', border: '1px solid var(--border)' }}>
            <button
              className={`btn-netflix-ghost ${contextMode === 'general' ? 'active' : ''}`}
              onClick={() => setContextMode('general')}
              style={{
                padding: '6px 14px',
                fontSize: '13px',
                fontWeight: '700',
                background: contextMode === 'general' ? 'var(--netflix-red)' : 'transparent',
                color: '#fff',
                borderRadius: '4px',
              }}
            >
              GENERAL
            </button>
            <button
              className={`btn-netflix-ghost ${contextMode === 'project' ? 'active' : ''}`}
              onClick={() => setContextMode('project')}
              style={{
                padding: '6px 14px',
                fontSize: '13px',
                fontWeight: '700',
                background: contextMode === 'project' ? 'var(--netflix-red)' : 'transparent',
                color: '#fff',
                borderRadius: '4px',
              }}
            >
              PROJECT CONTEXT
            </button>
          </div>

          {contextMode === 'project' && experiments.length > 0 && (
            <select
              className="netflix-select"
              value={selectedExpId}
              onChange={(e) => setSelectedExpId(e.target.value)}
              style={{ width: '200px', padding: '6px 10px', fontSize: '13px' }}
            >
              {experiments.map((e) => (
                <option key={e.id} value={e.id}>
                  Exp #{e.id}: {e.name || `Experiment ${e.id}`}
                </option>
              ))}
            </select>
          )}

          <button
            className="btn-netflix-ghost"
            onClick={clearChat}
            title="Clear Conversation"
            style={{ padding: '8px', fontSize: '16px' }}
          >
            <FiTrash2 />
          </button>
        </div>
      </div>

      {/* Suggestion Chips */}
      <div style={{ display: 'flex', gap: '8px', overflowX: 'auto', paddingBottom: '12px', marginBottom: '16px' }}>
        {SAMPLE_PROMPTS.map((prompt) => (
          <button
            key={prompt}
            className="btn-netflix-secondary"
            style={{ fontSize: '12px', padding: '6px 14px', whiteSpace: 'nowrap' }}
            onClick={() => handleSend(prompt)}
          >
            {prompt}
          </button>
        ))}
      </div>

      {/* Chat Messages Container */}
      <div className="chat-container">
        <div className="chat-messages">
          {messages.map((m, idx) => (
            <div key={idx} className={`message-row ${m.role}`}>
              <div className={`message-avatar ${m.role}`}>
                {m.role === 'user' ? 'U' : 'AI'}
              </div>
              <div className={`message-bubble ${m.role}`}>
                <ReactMarkdown
                  components={{
                    code({ node, inline, className, children, ...props }) {
                      const match = /language-(\w+)/.exec(className || '')
                      return !inline ? (
                        <CodeBlock
                          code={String(children).replace(/\n$/, '')}
                          language={match ? match[1] : ''}
                        />
                      ) : (
                        <code className="inline-code" {...props}>
                          {children}
                        </code>
                      )
                    },
                  }}
                >
                  {m.content}
                </ReactMarkdown>

                {m.is_fallback && (
                  <div style={{ fontSize: '11px', color: 'var(--warning)', marginTop: '8px', borderTop: '1px solid rgba(255,255,255,0.08)', paddingTop: '6px' }}>
                    Note: Running in honest deterministic fallback mode (no LLM API key configured).
                  </div>
                )}
              </div>
            </div>
          ))}
          {loading && (
            <div className="message-row assistant">
              <div className="message-avatar assistant">AI</div>
              <div className="message-bubble assistant">
                <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                  <LoadingSpinner size="sm" />
                  <span>Thinking...</span>
                </div>
              </div>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>

        {/* Input Bar */}
        <div className="chat-input-bar">
          <textarea
            className="netflix-textarea"
            rows="2"
            placeholder={
              contextMode === 'project'
                ? 'Ask about this experiment (e.g. why was this model chosen?)...'
                : 'Ask anything (coding, math, ML theory, recipes, interviews)...'
            }
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            disabled={loading}
            style={{ resize: 'none' }}
          />
          <button
            className="btn-netflix-primary"
            onClick={() => handleSend()}
            disabled={loading || !input.trim()}
            style={{ height: '56px', padding: '0 24px' }}
          >
            <FiSend />
          </button>
        </div>
      </div>
    </div>
  )
}
