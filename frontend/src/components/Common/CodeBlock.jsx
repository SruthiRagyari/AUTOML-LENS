import React, { useState } from 'react'
import { FiCopy, FiCheck } from 'react-icons/fi'

export default function CodeBlock({ code, language, inline }) {
  const [copied, setCopied] = useState(false)

  if (inline) {
    return <code className="inline-code">{code}</code>
  }

  const handleCopy = () => {
    navigator.clipboard.writeText(code)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  return (
    <div className="code-block-wrapper">
      <div className="code-block-header">
        <span className="code-language">{language || 'code'}</span>
        <button
          className="code-copy-btn"
          onClick={handleCopy}
          type="button"
          title="Copy code to clipboard"
        >
          {copied ? <><FiCheck /> Copied!</> : <><FiCopy /> Copy</>}
        </button>
      </div>
      <pre className="code-content">
        <code>{code}</code>
      </pre>
    </div>
  )
}
