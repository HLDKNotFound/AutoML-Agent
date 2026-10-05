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

        {keyInput.trim().startsWith('AQ.') && (
          <div style={{
            padding: '10px 14px',
            backgroundColor: 'rgba(244, 63, 94, 0.12)',
            border: '1px solid rgba(244, 63, 94, 0.35)',
            borderRadius: 'var(--radius-md)',
            marginBottom: '16px',
            fontSize: '0.78rem',
            color: '#fda4af',
            lineHeight: '1.4'
          }}>
            <strong>⚠️ Invalid Key Format:</strong> The entered key appears to be an OAuth access token (starts with <code>AQ.</code>). Google AI Studio Gemini API keys start with <code>AIzaSy...</code>. Using an OAuth token here will cause a 401 UNAUTHENTICATED error.
          </div>
        )}

        <div style={{ marginBottom: '20px' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
            <label style={{ fontSize: '0.75rem', fontWeight: 700, color: 'var(--text-muted)', textTransform: 'uppercase' }}>
              Gemini API Key (Google AI Studio)
            </label>
            <a
              href="https://aistudio.google.com/app/apikey"
              target="_blank"
              rel="noopener noreferrer"
              style={{ fontSize: '0.72rem', color: '#38bdf8', textDecoration: 'none', display: 'flex', alignItems: 'center', gap: '4px' }}
            >
              Get Free Key <ExternalLink size={11} />
            </a>
          </div>
          <input
            type="password"
            placeholder="AIzaSy..."
            value={keyInput}
            onChange={(e) => setKeyInput(e.target.value)}
            style={{
              width: '100%',
              padding: '10px 14px',
              backgroundColor: 'var(--bg-secondary)',
              border: keyInput.trim().startsWith('AQ.') ? '1px solid #f43f5e' : '1px solid var(--border-subtle)',
              borderRadius: 'var(--radius-md)',
              color: 'var(--text-primary)',
              fontSize: '0.88rem',
              fontFamily: 'var(--font-mono)'
            }}
          />
        </div>

        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: '10px' }}>
          {currentKey ? (
            <button
              onClick={() => {
                setKeyInput('');
                onSaveKey('');
                onClose();
              }}
              className="btn btn-secondary"
              style={{ color: '#fb7185', borderColor: 'rgba(251, 113, 133, 0.3)', fontSize: '0.78rem' }}
            >
              Clear Key (Use Offline Mode)
            </button>
          ) : <div />}

          <div style={{ display: 'flex', gap: '10px' }}>
            <button onClick={onClose} className="btn btn-secondary">
              Cancel
            </button>
            <button
              onClick={handleSave}
              className="btn btn-primary"
              disabled={keyInput.trim().startsWith('AQ.')}
            >
              <Check size={14} /> Save Configuration
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
