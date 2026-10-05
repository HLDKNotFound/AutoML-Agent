import React from 'react';
import { Award, Clock, Zap, CheckCircle2, ChevronRight, BarChart3 } from 'lucide-react';

export default function ValidationComparisonSection({ 
  modelResults, 
  bestModel, 
  primaryMetric, 
  plots 
}) {
  if (!modelResults || modelResults.length === 0) {
    return (
      <div className="glass-panel" style={{ padding: '36px', textAlign: 'center' }}>
        <p style={{ color: 'var(--text-muted)' }}>Validation results will be populated once model tuning finishes.</p>
      </div>
    );
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
      {/* Champion Model Banner */}
      {bestModel && bestModel.model_name && (
        <div className="glass-panel" style={{
          padding: '24px',
          background: 'linear-gradient(135deg, rgba(14, 165, 233, 0.12) 0%, rgba(99, 102, 241, 0.12) 100%)',
          border: '1px solid rgba(56, 189, 248, 0.35)',
          boxShadow: 'var(--glow-cyan)'
        }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '12px' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '14px' }}>
              <div style={{
                width: '46px',
                height: '46px',
                borderRadius: '12px',
                background: 'linear-gradient(135deg, #0284c7 0%, #6366f1 100%)',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                boxShadow: '0 4px 14px rgba(37, 99, 235, 0.4)'
              }}>
                <Award size={24} color="#ffffff" />
              </div>
              <div>
                <span style={{ fontSize: '0.75rem', fontWeight: 700, color: '#38bdf8', textTransform: 'uppercase', letterSpacing: '0.04em' }}>
                  Validation Champion Model
                </span>
                <h3 style={{ fontSize: '1.35rem', fontWeight: 800, color: 'var(--text-primary)' }}>
                  {bestModel.model_name}
                </h3>
              </div>
            </div>

            <div style={{ display: 'flex', gap: '16px' }}>
              <div className="stat-card" style={{ padding: '8px 16px', alignItems: 'center' }}>
                <span className="stat-label">Validation {primaryMetric.toUpperCase()}</span>
                <span style={{ fontSize: '1.25rem', fontWeight: 800, color: '#34d399' }}>
                  {bestModel.metrics?.[primaryMetric]?.toFixed(4) || 'N/A'}
                </span>
              </div>
              <div className="stat-card" style={{ padding: '8px 16px', alignItems: 'center' }}>
                <span className="stat-label">Inference Latency</span>
                <span style={{ fontSize: '1.25rem', fontWeight: 800, color: '#38bdf8' }}>
                  {bestModel.inference_time_ms} ms
                </span>
              </div>
            </div>
          </div>

          <div style={{ marginTop: '16px', fontSize: '0.82rem', color: 'var(--text-secondary)' }}>
            <strong>Optimal Hyperparameters: </strong>
            <code style={{ fontFamily: 'var(--font-mono)', color: '#38bdf8', backgroundColor: 'rgba(0,0,0,0.3)', padding: '2px 6px', borderRadius: '4px' }}>
              {JSON.stringify(bestModel.best_hyperparameters)}
            </code>
          </div>
        </div>
      )}

      {/* Comparison Table */}
      <div className="glass-panel" style={{ padding: '24px' }}>
        <h3 style={{ fontSize: '1.05rem', fontWeight: 700, color: 'var(--text-primary)', marginBottom: '14px' }}>
          Candidate Model Validation Comparison Table
        </h3>

        <div className="data-table-container">
          <table className="data-table">
            <thead>
              <tr>
                <th>Model</th>
                <th>Model Family</th>
                <th>Validation {primaryMetric.toUpperCase()}</th>
                <th>Training Time</th>
                <th>Latency (ms)</th>
                <th>Model Size</th>
                <th>Status</th>
              </tr>
            </thead>
            <tbody>
              {modelResults.map((res, idx) => {
                const isChampion = bestModel?.model_name === res.model_name;
                return (
                  <tr key={idx} style={{ backgroundColor: isChampion ? 'rgba(56, 189, 248, 0.08)' : 'transparent' }}>
                    <td>
                      <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                        {isChampion && <Award size={14} color="#38bdf8" />}
                        <strong style={{ color: isChampion ? '#38bdf8' : 'inherit' }}>
                          {res.model_name}
                        </strong>
                      </div>
                    </td>
                    <td>
                      <span className="badge badge-family">
                        {res.model_family}
                      </span>
                    </td>
                    <td style={{ fontWeight: 700, color: isChampion ? '#34d399' : 'inherit' }}>
                      {res.metrics?.[primaryMetric]?.toFixed(4) ?? 'N/A'}
                    </td>
                    <td>{res.train_time_sec?.toFixed(2)}s</td>
                    <td>{res.inference_time_ms} ms</td>
                    <td>{res.model_size_kb} KB</td>
                    <td>
                      <span className="badge badge-completed">
                        <CheckCircle2 size={11} /> {res.status.toUpperCase()}
                      </span>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>

      {/* Comparison Chart */}
      {plots?.model_comparison && (
        <div className="glass-panel" style={{ padding: '20px' }}>
          <img
            src={plots.model_comparison}
            alt="Candidate Model Comparison Chart"
            style={{ width: '100%', borderRadius: 'var(--radius-md)', objectFit: 'contain' }}
          />
        </div>
      )}
    </div>
  );
}
