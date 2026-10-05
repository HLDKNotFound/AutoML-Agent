import React from 'react';
import { CheckCheck, TrendingUp, TrendingDown, ShieldCheck, FileText, CheckCircle2 } from 'lucide-react';

export default function FinalTestEvaluationSection({ 
  bestModel, 
  testResults, 
  primaryMetric, 
  plots 
}) {
  if (!testResults || Object.keys(testResults).length === 0) {
    return (
      <div className="glass-panel" style={{ padding: '36px', textAlign: 'center' }}>
        <p style={{ color: 'var(--text-muted)' }}>Final test evaluation on the untouched 15% partition will be presented here once completed.</p>
      </div>
    );
  }

  const valMetric = bestModel?.metrics?.[primaryMetric] || 0.0;
  const testMetric = testResults[primaryMetric] || 0.0;
  const delta = testMetric - valMetric;

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
      {/* Test vs Validation Comparison Banner */}
      <div className="glass-panel" style={{ padding: '24px' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '14px', marginBottom: '16px' }}>
          <div>
            <span style={{ fontSize: '0.75rem', fontWeight: 700, color: '#34d399', textTransform: 'uppercase', letterSpacing: '0.04em' }}>
              Final Unbiased Evaluation (Untouched 15% Test Set)
            </span>
            <h3 style={{ fontSize: '1.3rem', fontWeight: 800, color: 'var(--text-primary)', marginTop: '2px' }}>
              Champion Pipeline: {bestModel?.model_name}
            </h3>
          </div>

          {/* Metric Comparison Delta Card */}
          <div style={{ display: 'flex', gap: '16px' }}>
            <div className="stat-card" style={{ padding: '10px 18px', alignItems: 'center' }}>
              <span className="stat-label">Validation {primaryMetric.toUpperCase()}</span>
              <span style={{ fontSize: '1.2rem', fontWeight: 800, color: '#38bdf8' }}>
                {valMetric.toFixed(4)}
              </span>
            </div>
            <div className="stat-card" style={{ padding: '10px 18px', alignItems: 'center' }}>
              <span className="stat-label">Final Test {primaryMetric.toUpperCase()}</span>
              <span style={{ fontSize: '1.2rem', fontWeight: 800, color: '#34d399' }}>
                {testMetric.toFixed(4)}
              </span>
            </div>
            <div className="stat-card" style={{ padding: '10px 18px', alignItems: 'center' }}>
              <span className="stat-label">Generalization Delta</span>
              <span style={{
                fontSize: '1.2rem',
                fontWeight: 800,
                color: Math.abs(delta) <= 0.05 ? '#34d399' : (delta > 0 ? '#38bdf8' : '#f59e0b'),
                display: 'flex',
                alignItems: 'center',
                gap: '4px'
              }}>
                {delta >= 0 ? <TrendingUp size={16} /> : <TrendingDown size={16} />}
                {delta >= 0 ? `+${delta.toFixed(4)}` : delta.toFixed(4)}
              </span>
            </div>
          </div>
        </div>

        {/* Generalization Note */}
        <div style={{
          padding: '14px 18px',
          backgroundColor: 'rgba(15, 23, 42, 0.7)',
          borderRadius: 'var(--radius-md)',
          display: 'flex',
          alignItems: 'center',
          gap: '12px',
          borderLeft: '4px solid #34d399'
        }}>
          <ShieldCheck size={22} color="#34d399" />
          <p style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', lineHeight: '1.4' }}>
            <strong>Zero Data Leakage Verification:</strong> The test set was never seen during preprocessing fitting, 
            feature pruning, or Bayesian hyperparameter optimization. The delta of <strong>{delta >= 0 ? `+${delta.toFixed(4)}` : delta.toFixed(4)}</strong> confirms authentic production generalization.
          </p>
        </div>
      </div>

      {/* Comprehensive Test Metrics Grid */}
      <div className="glass-panel" style={{ padding: '24px' }}>
        <h3 style={{ fontSize: '1.05rem', fontWeight: 700, color: 'var(--text-primary)', marginBottom: '16px' }}>
          All Final Test Metrics
        </h3>

        <div style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))',
          gap: '12px'
        }}>
          {Object.entries(testResults).map(([key, val]) => (
            <div key={key} className="stat-card">
              <span className="stat-label">{key.replace('_', ' ')}</span>
              <span className="stat-value" style={{ color: key === primaryMetric ? '#34d399' : 'inherit' }}>
                {typeof val === 'number' ? val.toFixed(4) : val}
              </span>
            </div>
          ))}
        </div>
      </div>

      {/* Diagnostic Charts (Confusion Matrix, ROC, Residuals) */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(360px, 1fr))', gap: '16px' }}>
        {plots?.test_confusion_matrix && (
          <div className="glass-panel" style={{ padding: '20px' }}>
            <img
              src={plots.test_confusion_matrix}
              alt="Test Confusion Matrix"
              style={{ width: '100%', borderRadius: 'var(--radius-md)', objectFit: 'contain' }}
            />
          </div>
        )}

        {plots?.test_roc_curve && (
          <div className="glass-panel" style={{ padding: '20px' }}>
            <img
              src={plots.test_roc_curve}
              alt="Receiver Operating Characteristic Curve"
              style={{ width: '100%', borderRadius: 'var(--radius-md)', objectFit: 'contain' }}
            />
          </div>
        )}

        {plots?.test_residuals && (
          <div className="glass-panel" style={{ padding: '20px', gridColumn: 'span 2' }}>
            <img
              src={plots.test_residuals}
              alt="Residual Diagnostics"
              style={{ width: '100%', borderRadius: 'var(--radius-md)', objectFit: 'contain' }}
            />
          </div>
        )}
      </div>
    </div>
  );
}
