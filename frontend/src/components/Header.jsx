import React from 'react';
import { Bot, Key, RotateCcw, Database } from 'lucide-react';

export default function Header({
  filename,
  onOpenKeyModal,
  onResetSession,
  hasKey,
  workflowStatus,
  activeStepId,
  activeStepName
}) {
  return (
    <header style={{
      height: '64px',
      borderBottom: '1px solid var(--border-subtle)',
      backgroundColor: 'rgba(13, 18, 29, 0.8)',
      backdropFilter: 'blur(10px)',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'space-between',
      padding: '0 28px',
      position: 'sticky',
      top: 0,
      zIndex: 100
    }}>
      {/* Active Dataset Display & Live Running Badge */}
      <div style={{ display: 'flex', alignItems: 'center', gap: '14px' }}>
        {filename ? (
          <div style={{
            display: 'flex',
            alignItems: 'center',
            gap: '8px',
            backgroundColor: 'rgba(255, 255, 255, 0.05)',
            padding: '6px 12px',
            borderRadius: 'var(--radius-sm)',
            border: '1px solid var(--border-subtle)'
          }}>
            <Database size={15} color="#38bdf8" />
            <span style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>Dataset:</span>
            <span style={{ fontSize: '0.82rem', fontWeight: 600, color: 'var(--text-primary)' }}>
              {filename}
            </span>
          </div>
        ) : (
          <span style={{ fontSize: '0.84rem', color: 'var(--text-muted)' }}>
            No dataset loaded. Upload a CSV or select a sample dataset.
          </span>
        )}

        {workflowStatus === 'running' && (
          <div style={{
            display: 'flex',
            alignItems: 'center',
            gap: '8px',
            padding: '5px 12px',
            borderRadius: '9999px',
            backgroundColor: 'rgba(56, 189, 248, 0.15)',
            border: '1px solid rgba(56, 189, 248, 0.45)',
            color: '#38bdf8',
            fontSize: '0.78rem',
            fontWeight: 700
          }}>
            <span className="radar-dot" />
            Stage {activeStepId || '•'}: {activeStepName || 'Running'}
          </div>
        )}
      </div>

      {/* Right Controls */}
      <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
        {/* LLM Status Pill */}
        <div style={{
          display: 'flex',
          alignItems: 'center',
          gap: '6px',
          padding: '5px 12px',
          borderRadius: '9999px',
          backgroundColor: hasKey ? 'rgba(16, 185, 129, 0.12)' : 'rgba(56, 189, 248, 0.12)',
          border: `1px solid ${hasKey ? 'rgba(16, 185, 129, 0.3)' : 'rgba(56, 189, 248, 0.3)'}`
        }}>
          <Bot size={14} color={hasKey ? '#34d399' : '#38bdf8'} />
          <span style={{ fontSize: '0.75rem', fontWeight: 600, color: hasKey ? '#34d399' : '#38bdf8' }}>
            Gemini 2.5 Flash {hasKey ? '(API Active)' : '(Auto Heuristics Fallback)'}
          </span>
        </div>

        {/* Configure API Key */}
        <button
          onClick={onOpenKeyModal}
          className="btn btn-secondary"
          style={{ padding: '6px 12px', fontSize: '0.78rem' }}
          title="Configure Gemini API Key"
        >
          <Key size={14} />
          API Key
        </button>

        {/* Reset Session */}
        <button
          onClick={onResetSession}
          className="btn btn-secondary"
          style={{ padding: '6px 12px', fontSize: '0.78rem' }}
          title="Start fresh session"
        >
          <RotateCcw size={14} />
          New Run
        </button>
      </div>
    </header>
  );
}
