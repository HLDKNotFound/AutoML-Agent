import React from 'react';
import { Compass, Cpu, Layers, GitBranch, Sparkles, Check, Brain } from 'lucide-react';

export default function MLStrategySection({ 
  problemType, 
  primaryMetric, 
  candidateModels, 
  agentDecisions 
}) {
  if (!candidateModels || candidateModels.length === 0) {
    return (
      <div className="glass-panel" style={{ padding: '36px', textAlign: 'center' }}>
        <p style={{ color: 'var(--text-muted)' }}>ML Strategy and model candidates will be selected after validation passes.</p>
      </div>
    );
  }

  const families = [...new Set(candidateModels.map((m) => m.model_family))];

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
      {/* Strategy Summary Banner */}
      <div className="glass-panel" style={{ padding: '24px' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '14px', marginBottom: '16px' }}>
          <div>
            <span style={{ fontSize: '0.75rem', fontWeight: 700, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
              Architectural Strategy
            </span>
            <h3 style={{ fontSize: '1.25rem', fontWeight: 800, color: 'var(--text-primary)', marginTop: '2px' }}>
              Problem: {problemType.replace('_', ' ').toUpperCase()}
            </h3>
          </div>

          <div style={{ display: 'flex', gap: '10px' }}>
            <div className="stat-card" style={{ padding: '8px 16px', alignItems: 'center' }}>
              <span className="stat-label">Primary Metric</span>
              <span style={{ fontSize: '1.0rem', fontWeight: 700, color: '#38bdf8' }}>
                {primaryMetric.toUpperCase()}
              </span>
            </div>
            <div className="stat-card" style={{ padding: '8px 16px', alignItems: 'center' }}>
              <span className="stat-label">Model Families</span>
              <span style={{ fontSize: '1.0rem', fontWeight: 700, color: '#c084fc' }}>
                {families.length} Families
              </span>
            </div>
          </div>
        </div>

        <p style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', lineHeight: '1.5' }}>
          The Strategy Agent has dynamically configured <strong>5 competitive ML candidate models</strong> across <strong>{families.length} distinct families</strong>. 
          To prevent data leakage and respect inductive biases, <strong>each candidate has an independent, tailored preprocessing pipeline</strong>.
        </p>
      </div>

      {/* 5 Candidate Models with Model-Specific Preprocessing */}
      <div>
        <h3 style={{ fontSize: '1.05rem', fontWeight: 700, color: 'var(--text-primary)', marginBottom: '14px' }}>
          Candidate Models & Model-Specific Pipelines ({candidateModels.length} Models)
        </h3>

        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(320px, 1fr))', gap: '16px' }}>
          {candidateModels.map((cand, idx) => {
            const prep = cand.preprocessing_strategy || {};
            return (
              <div key={idx} className="glass-panel" style={{ padding: '20px', display: 'flex', flexDirection: 'column', gap: '12px' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
                  <div>
                    <h4 style={{ fontSize: '1.0rem', fontWeight: 700, color: 'var(--text-primary)' }}>
                      {cand.model_name}
                    </h4>
                    <span className="badge badge-family" style={{ marginTop: '4px' }}>
                      {cand.model_family}
                    </span>
                  </div>
                  <span style={{ fontSize: '0.72rem', color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}>
                    Candidate #{idx + 1}
                  </span>
                </div>

                <p style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', lineHeight: '1.4' }}>
                  {cand.reason}
                </p>

                {/* Preprocessing Pipeline Specs */}
                <div style={{
                  padding: '12px',
                  backgroundColor: 'rgba(15, 23, 42, 0.6)',
                  borderRadius: 'var(--radius-md)',
                  border: '1px solid var(--border-subtle)',
                  display: 'flex',
                  flexDirection: 'column',
                  gap: '6px'
                }}>
                  <div style={{ fontSize: '0.72rem', fontWeight: 700, color: '#38bdf8', textTransform: 'uppercase', letterSpacing: '0.04em' }}>
                    Dedicated Preprocessing
                  </div>
                  <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '6px', fontSize: '0.75rem' }}>
                    <div>
                      <span style={{ color: 'var(--text-muted)' }}>Numeric Scaler: </span>
                      <strong style={{ color: prep.numeric_scaler ? '#34d399' : '#94a3b8' }}>
                        {prep.numeric_scaler || 'None (Scale Invariant)'}
                      </strong>
                    </div>
                    <div>
                      <span style={{ color: 'var(--text-muted)' }}>Num Imputer: </span>
                      <strong style={{ color: 'var(--text-primary)' }}>{prep.numeric_imputer || 'median'}</strong>
                    </div>
                    <div>
                      <span style={{ color: 'var(--text-muted)' }}>Cat Encoder: </span>
                      <strong style={{ color: 'var(--text-primary)' }}>{prep.categorical_encoder || 'one_hot'}</strong>
                    </div>
                    <div>
                      <span style={{ color: 'var(--text-muted)' }}>Cat Imputer: </span>
                      <strong style={{ color: 'var(--text-primary)' }}>{prep.categorical_imputer || 'most_frequent'}</strong>
                    </div>
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* Observability: Senior Agent Decision Cards */}
      {agentDecisions && agentDecisions.length > 0 && (
        <div className="glass-panel" style={{ padding: '24px' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '16px' }}>
            <Brain size={18} color="#38bdf8" />
            <h3 style={{ fontSize: '1.05rem', fontWeight: 700, color: 'var(--text-primary)' }}>
              Agent Decisions & Observability Audit
            </h3>
          </div>

          <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
            {agentDecisions.map((dec, idx) => (
              <div
                key={idx}
                style={{
                  padding: '14px 16px',
                  backgroundColor: 'rgba(15, 23, 42, 0.6)',
                  borderRadius: 'var(--radius-md)',
                  border: '1px solid var(--border-subtle)',
                  display: 'flex',
                  flexDirection: 'column',
                  gap: '6px'
                }}
              >
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <span style={{ fontSize: '0.85rem', fontWeight: 700, color: '#38bdf8' }}>
                    {dec.agent}
                  </span>
                  <span className="badge badge-completed">
                    {dec.decision}
                  </span>
                </div>
                <p style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>
                  <strong>Reasoning:</strong> {dec.reason}
                </p>
                <div style={{ display: 'flex', gap: '16px', fontSize: '0.72rem', color: 'var(--text-muted)', marginTop: '4px' }}>
                  <span>Input: <em>{dec.input_statistics}</em></span>
                  <span>Output: <em>{dec.output_configuration}</em></span>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
