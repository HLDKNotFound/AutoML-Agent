import React, { useState } from 'react';
import { Key, X, Check, ExternalLink } from 'lucide-react';

export default function ApiKeyModal({ isOpen, onClose, onSaveKey, currentKey }) {
  const [keyInput, setKeyInput] = useState(currentKey || '');

  if (!isOpen) return null;

  const handleSave = () => {
    onSaveKey(keyInput);
    onClose();
  };

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div
        className="glass-panel"
        style={{
          width: '100%',
          maxWidth: '480px',
          padding: '28px',
          backgroundColor: '#0f172a',
          borderRadius: 'var(--radius-xl)',
          border: '1px solid var(--border-active)',
          boxShadow: 'var(--glow-cyan)'
        }}
        onClick={(e) => e.stopPropagation()}
      >
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
            <div style={{
              width: '36px',
              height: '36px',
              borderRadius: '8px',
              backgroundColor: 'rgba(56, 189, 248, 0.15)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center'
            }}>
              <Key size={18} color="#38bdf8" />
            </div>
            <h3 style={{ fontSize: '1.1rem', fontWeight: 800, color: 'var(--text-primary)' }}>
              Gemini 2.5 Flash API Key
            </h3>
          </div>
          <button onClick={onClose} style={{ background: 'none', border: 'none', color: 'var(--text-muted)', cursor: 'pointer' }}>
            <X size={18} />
          </button>
        </div>

        <p style={{ fontSize: '0.84rem', color: 'var(--text-secondary)', marginBottom: '18px', lineHeight: '1.4' }}>
          Connect your Google Gemini API key to empower the agents with Gemini 2.5 Flash LLM reasoning through LangChain. 
          If not provided, the pipeline executes seamlessly using the built-in deterministic Senior Data Scientist heuristics engine.
        </p>

        <div style={{ marginBottom: '20px' }}>
          <label style={{ display: 'block', fontSize: '0.75rem', fontWeight: 700, color: 'var(--text-muted)', textTransform: 'uppercase', marginBottom: '8px' }}>
            API Key
          </label>
          <input
            type="password"
            placeholder="AIzaSy..."
            value={keyInput}
            onChange={(e) => setKeyInput(e.target.value)}
            style={{
              width: '100%',
              padding: '10px 14px',
              backgroundColor: 'var(--bg-secondary)',
              border: '1px solid var(--border-subtle)',
              borderRadius: 'var(--radius-md)',
              color: 'var(--text-primary)',
              fontSize: '0.88rem',
              fontFamily: 'var(--font-mono)'
            }}
          />
        </div>

        <div style={{ display: 'flex', justifyContent: 'flex-end', gap: '10px' }}>
          <button onClick={onClose} className="btn btn-secondary">
            Cancel
          </button>
          <button onClick={handleSave} className="btn btn-primary">
            <Check size={14} /> Save Configuration
          </button>
        </div>
      </div>
    </div>
  );
}
