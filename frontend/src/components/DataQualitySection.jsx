import React from 'react';
import { 
  ShieldCheck, 
  ShieldAlert, 
  AlertCircle, 
  CheckCircle2, 
  XCircle, 
  RefreshCw, 
  BarChart3, 
  Check, 
  HelpCircle,
  Brain
} from 'lucide-react';

export default function DataQualitySection({ 
  profile, 
  qualityReport, 
  featureSelection, 
  plots 
}) {
  if (!qualityReport && !profile) return (
    <div className="glass-panel" style={{ padding: '36px', textAlign: 'center' }}>
      <p style={{ color: 'var(--text-muted)' }}>Profiling and Quality analysis will appear once the pipeline runs.</p>
    </div>
  );

  const problems = qualityReport?.problems || [];
  const warnings = qualityReport?.warnings || [];
  const isValid = qualityReport?.valid ?? true;
  const iteration = qualityReport?.iteration || 1;

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
      {/* Validation Loop Status Banner */}
      <div className="glass-panel" style={{
        padding: '20px 24px',
        backgroundColor: isValid ? 'rgba(16, 185, 129, 0.08)' : 'rgba(244, 63, 94, 0.08)',
        borderLeft: `4px solid ${isValid ? '#10b981' : '#f43f5e'}`
      }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '10px' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
            {isValid ? (
              <div style={{
                width: '40px',
                height: '40px',
                borderRadius: '50%',
                backgroundColor: 'rgba(16, 185, 129, 0.15)',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center'
              }}>
                <ShieldCheck size={22} color="#34d399" />
              </div>
            ) : (
              <div style={{
                width: '40px',
                height: '40px',
                borderRadius: '50%',
                backgroundColor: 'rgba(244, 63, 94, 0.15)',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center'
              }}>
                <ShieldAlert size={22} color="#fb7185" />
              </div>
            )}
            <div>
              <h3 style={{ fontSize: '1.05rem', fontWeight: 700, color: 'var(--text-primary)' }}>
                {isValid ? 'Data Quality Audit Passed' : 'Data Quality Issues Detected'}
              </h3>
              <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>
                Validation Loop Iteration: <strong>{iteration} of 3</strong>
              </p>
            </div>
          </div>

          <span className={`badge ${isValid ? 'badge-completed' : 'badge-failed'}`}>
            {isValid ? 'VALID FOR ML' : 'REMEDIATING'}
          </span>
        </div>

        {qualityReport?.reasoning && (
          <div style={{
            marginTop: '14px',
            padding: '12px 16px',
            backgroundColor: 'rgba(15, 23, 42, 0.6)',
            borderRadius: 'var(--radius-md)',
            display: 'flex',
            gap: '10px',
            alignItems: 'flex-start'
          }}>
            <Brain size={16} color="#38bdf8" style={{ marginTop: '2px', flexShrink: 0 }} />
            <p style={{ fontSize: '0.82rem', color: 'var(--text-secondary)', lineHeight: '1.5' }}>
              <strong>LLM Data Analyst Assessment:</strong> {qualityReport.reasoning}
            </p>
          </div>
        )}
      </div>

      {/* Matplotlib Visualization Charts */}
      {plots && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(360px, 1fr))', gap: '16px' }}>
          {plots.target_distribution && (
            <div className="glass-panel" style={{ padding: '16px', display: 'flex', flexDirection: 'column' }}>
              <img
                src={plots.target_distribution}
                alt="Target Distribution"
                style={{ width: '100%', borderRadius: 'var(--radius-md)', objectFit: 'contain' }}
              />
            </div>
          )}

          {plots.missing_values && (
            <div className="glass-panel" style={{ padding: '16px', display: 'flex', flexDirection: 'column' }}>
              <img
                src={plots.missing_values}
                alt="Missing Values Audit"
                style={{ width: '100%', borderRadius: 'var(--radius-md)', objectFit: 'contain' }}
              />
            </div>
          )}

          {plots.correlation_matrix && (
            <div className="glass-panel" style={{ padding: '16px', display: 'flex', flexDirection: 'column', gridColumn: 'span 1' }}>
              <img
                src={plots.correlation_matrix}
                alt="Correlation Heatmap"
                style={{ width: '100%', borderRadius: 'var(--radius-md)', objectFit: 'contain' }}
              />
            </div>
          )}
        </div>
      )}

      {/* Detected Problems List */}
      <div className="glass-panel" style={{ padding: '24px' }}>
        <h3 style={{ fontSize: '1.05rem', fontWeight: 700, color: 'var(--text-primary)', marginBottom: '14px' }}>
          Audit Findings ({problems.length} Items Identified)
        </h3>

        {problems.length === 0 ? (
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', color: '#34d399', fontSize: '0.88rem' }}>
            <CheckCircle2 size={16} />
            <span>No critical data hygiene issues or target leakage detected.</span>
          </div>
        ) : (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
            {problems.map((prob, idx) => (
              <div
                key={idx}
                style={{
                  padding: '14px',
                  backgroundColor: 'rgba(15, 23, 42, 0.6)',
                  borderRadius: 'var(--radius-md)',
                  borderLeft: `3px solid ${prob.severity === 'critical' ? '#f43f5e' : '#f59e0b'}`
                }}
              >
                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '6px' }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                    <span className={`badge ${prob.severity === 'critical' ? 'badge-failed' : 'badge-running'}`}>
                      {prob.severity.toUpperCase()}
                    </span>
                    <span style={{ fontSize: '0.85rem', fontWeight: 700, color: 'var(--text-primary)' }}>
                      {prob.category.replace('_', ' ').toUpperCase()} {prob.feature ? `• ${prob.feature}` : ''}
                    </span>
                  </div>
                </div>
                <p style={{ fontSize: '0.82rem', color: 'var(--text-secondary)', marginBottom: '6px' }}>
                  {prob.description}
                </p>
                <p style={{ fontSize: '0.78rem', color: '#38bdf8' }}>
                  <strong>Recommendation:</strong> {prob.suggestion}
                </p>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Feature Selection Decisions */}
      {featureSelection && (
        <div className="glass-panel" style={{ padding: '24px' }}>
          <h3 style={{ fontSize: '1.05rem', fontWeight: 700, color: 'var(--text-primary)', marginBottom: '14px' }}>
            Automated Feature Selection
          </h3>

          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: '20px' }}>
            {/* Retained Features */}
            <div>
              <div style={{ display: 'flex', alignItems: 'center', gap: '6px', marginBottom: '10px' }}>
                <CheckCircle2 size={15} color="#34d399" />
                <span style={{ fontSize: '0.85rem', fontWeight: 700, color: '#34d399' }}>
                  Retained Features ({featureSelection.selected_features?.length || 0})
                </span>
              </div>
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: '6px' }}>
                {featureSelection.selected_features?.map((f) => (
                  <span key={f} className="badge badge-completed">
                    {f}
                  </span>
                ))}
              </div>
            </div>

            {/* Dropped Features */}
            <div>
              <div style={{ display: 'flex', alignItems: 'center', gap: '6px', marginBottom: '10px' }}>
                <XCircle size={15} color="#fb7185" />
                <span style={{ fontSize: '0.85rem', fontWeight: 700, color: '#fb7185' }}>
                  Pruned Problem Features ({featureSelection.dropped_features?.length || 0})
                </span>
              </div>
              {featureSelection.dropped_features?.length === 0 ? (
                <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>None dropped.</span>
              ) : (
                <div style={{ display: 'flex', flexWrap: 'wrap', gap: '6px' }}>
                  {featureSelection.dropped_features?.map((f) => (
                    <span key={f} className="badge badge-failed" title={featureSelection.reasons?.[f] || 'Pruned'}>
                      {f}
                    </span>
                  ))}
                </div>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
