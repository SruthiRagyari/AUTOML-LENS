import React, { useState, useRef, useEffect } from 'react'
import { FiMessageSquare, FiX, FiSend, FiCpu, FiExternalLink } from 'react-icons/fi'
import { Link } from 'react-router-dom'
import ReactMarkdown from 'react-markdown'
import { api } from '../../services/api'
import CodeBlock from './CodeBlock'

export default function FloatingAssistant() {
  const [isOpen, setIsOpen] = useState(false)
  const [messages, setMessages] = useState([
    {
      role: 'assistant',
      content: 'Hello! I am the AutoML-Lens AI Assistant. I can help with general programming, ML theory, math, or your AutoML experiments. How can I help you today?',
    },
  ])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const messagesEndRef = useRef(null)

  useEffect(() => {
    if (isOpen && messagesEndRef.current) {
      messagesEndRef.current.scrollIntoView({ behavior: 'smooth' })
    }
  }, [messages, isOpen])

  const handleSend = async (e) => {
    if (e) e.preventDefault()
    if (!input.trim() || loading) return

    const userText = input.trim()
    const newMessages = [...messages, { role: 'user', content: userText }]
    setMessages(newMessages)
    setInput('')
    setLoading(true)

    try {
      const history = newMessages
        .slice(-10)
        .map((m) => ({ role: m.role, content: m.content }))

      const res = await api.chat(userText, null, 'general', history)
      setMessages((prev) => [
        ...prev,
        {
          role: 'assistant',
          content: res.data.response || 'No response returned.',
          is_fallback: res.data.is_fallback,
        },
      ])
    } catch (err) {
      setMessages((prev) => [
        ...prev,
        {
          role: 'assistant',
          content: 'Sorry, I encountered an error connecting to the AI assistant service. Please check your backend connection.',
          error: true,
        },
      ])
    } finally {
      setLoading(false)
    }
  }

  return (
    <>
      <button
        className="floating-ai-fab"
        onClick={() => setIsOpen(!isOpen)}
        aria-label="Toggle AI Assistant"
        title="Open AI Assistant"
        type="button"
      >
        {isOpen ? <FiX /> : <FiMessageSquare />}
      </button>

      {isOpen && (
        <div className="floating-chat-window">
          <div className="floating-chat-header">
            <div className="floating-chat-title">
              <FiCpu style={{ color: 'var(--netflix-red)' }} />
              <span>AI Assistant</span>
            </div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <Link
                to="/assistant"
                title="Open Full Screen Assistant"
                style={{ color: 'var(--text-secondary)' }}
                onClick={() => setIsOpen(false)}
              >
                <FiExternalLink />
              </Link>
              <button
                className="btn-netflix-ghost"
                onClick={() => setIsOpen(false)}
                style={{ padding: '4px', fontSize: '16px' }}
                type="button"
              >
                <FiX />
              </button>
            </div>
          </div>

          <div className="floating-chat-messages">
            {messages.map((m, idx) => (
              <div key={idx} className={`message-row ${m.role}`}>
                <div className={`message-avatar ${m.role}`}>
                  {m.role === 'user' ? 'U' : 'AI'}
                </div>
                <div className={`message-bubble ${m.role}`}>
                  {m.role === 'assistant' ? (
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
                            <code className="inline-code" {...props}>{children}</code>
                          )
                        },
                      }}
                    >
                      {m.content}
                    </ReactMarkdown>
                  ) : (
                    <span>{m.content}</span>
                  )}
                </div>
              </div>
            ))}
            {loading && (
              <div className="message-row assistant">
                <div className="message-avatar assistant">AI</div>
                <div className="message-bubble assistant">
                  <span>Thinking...</span>
                </div>
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>

          <form onSubmit={handleSend} className="floating-chat-input">
            <input
              type="text"
              placeholder="Ask me anything..."
              value={input}
              onChange={(e) => setInput(e.target.value)}
              disabled={loading}
              className="netflix-input"
              style={{ fontSize: '13px', padding: '8px 12px' }}
            />
            <button
              type="submit"
              disabled={loading || !input.trim()}
              className="btn-netflix-primary"
              style={{ padding: '8px 14px' }}
            >
              <FiSend />
            </button>
          </form>
        </div>
      )}
    </>
  )
}
