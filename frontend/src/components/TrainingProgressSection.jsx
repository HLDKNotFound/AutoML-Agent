import React from 'react';
import { Activity, Terminal, CheckCircle2, Loader2, XCircle } from 'lucide-react';

export default function TrainingProgressSection({ events, workflowStatus, modelResults }) {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
      {/* Header status */}
      <div className="glass-panel" style={{ padding: '24px' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '12px' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
            <Activity size={20} color="#38bdf8" />
            <h3 style={{ fontSize: '1.1rem', fontWeight: 700, color: 'var(--text-primary)' }}>
              Two-Stage Coarse-to-Fine Hyperparameter Optimization
            </h3>
          </div>
          <span className={`badge badge-${workflowStatus}`}>
            {workflowStatus.toUpperCase()}
          </span>
        </div>
        <p style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>
          Stage 1 executes broad random exploration over logarithmic and uniform parameter distributions. 
          Stage 2 executes Bayesian Optimization (Optuna TPE) explicitly refining promising parameter regions on full training data.
        </p>
      </div>

      {/* Real-time Streaming Logs Console */}
      <div className="glass-panel" style={{ padding: '20px' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '12px' }}>
          <Terminal size={16} color="#38bdf8" />
          <span style={{ fontSize: '0.85rem', fontWeight: 700, color: 'var(--text-primary)' }}>
            Real-Time Execution Logs
          </span>
        </div>

        <div style={{
          height: '350px',
          overflowY: 'auto',
          backgroundColor: '#040711',
          borderRadius: 'var(--radius-md)',
          padding: '16px',
          fontFamily: 'var(--font-mono)',
          fontSize: '0.78rem',
          display: 'flex',
          flexDirection: 'column',
          gap: '8px',
          border: '1px solid var(--border-subtle)'
        }}>
          {events.length === 0 ? (
            <span style={{ color: 'var(--text-muted)' }}>Waiting for pipeline execution events...</span>
          ) : (
            events.map((ev, idx) => {
              let icon;
              if (ev.status === 'completed') icon = <CheckCircle2 size={13} color="#34d399" />;
              else if (ev.status === 'running') icon = <Loader2 size={13} color="#38bdf8" className="animate-spin" />;
              else if (ev.status === 'failed') icon = <XCircle size={13} color="#fb7185" />;
              else icon = <span style={{ width: '13px' }}>•</span>;

              return (
                <div key={idx} style={{ display: 'flex', alignItems: 'flex-start', gap: '8px', lineHeight: '1.4' }}>
                  <span style={{ color: 'var(--text-muted)', minWidth: '60px' }}>[{ev.timestamp}]</span>
                  <div style={{ marginTop: '2px' }}>{icon}</div>
                  <span style={{ color: '#38bdf8', fontWeight: 600, minWidth: '150px' }}>
                    [{ev.step_name}]
                  </span>
                  <span style={{ color: ev.status === 'failed' ? '#fb7185' : 'var(--text-primary)' }}>
                    {ev.message}
                  </span>
                </div>
              );
            })
          )}
        </div>
      </div>
    </div>
  );
}
