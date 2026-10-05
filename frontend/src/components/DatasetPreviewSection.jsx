import React from 'react';
import { Play, Check, AlertTriangle, Layers, Database } from 'lucide-react';

export default function DatasetPreviewSection({
  summary,
  targetColumn,
  onSelectTarget,
  selectedFeatures,
  onToggleFeature,
  onSelectAllFeatures,
  onDeselectAllFeatures,
  onStartPipeline,
  isStarting
}) {
  if (!summary) return null;

  const columns = summary.columns || [];
  const preview = summary.preview || [];

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
      {/* Overview Metric Cards */}
      <div style={{
        display: 'grid',
        gridTemplateColumns: 'repeat(auto-fit, minmax(160px, 1fr))',
        gap: '12px'
      }}>
        <div className="stat-card">
          <span className="stat-label">Total Rows</span>
          <span className="stat-value">{summary.total_rows.toLocaleString()}</span>
        </div>
        <div className="stat-card">
          <span className="stat-label">Total Columns</span>
          <span className="stat-value">{summary.total_cols}</span>
        </div>
        <div className="stat-card">
          <span className="stat-label">Memory Footprint</span>
          <span className="stat-value">{summary.memory_usage_kb} KB</span>
        </div>
        <div className="stat-card">
          <span className="stat-label">Selected Features (X)</span>
          <span className="stat-value" style={{ color: '#38bdf8' }}>{selectedFeatures.length}</span>
        </div>
        <div className="stat-card">
          <span className="stat-label">Target (Y)</span>
          <span className="stat-value" style={{ color: '#34d399', fontSize: '1.15rem' }}>
            {targetColumn || 'Not Selected'}
          </span>
        </div>
      </div>

      {/* Target Column Selection Card */}
      <div className="glass-panel" style={{ padding: '24px' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px' }}>
          <div>
            <h3 style={{ fontSize: '1.05rem', fontWeight: 700, color: 'var(--text-primary)' }}>
              1. Select Target Column (Y)
            </h3>
            <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>
              Choose the column you wish to predict. Classification or regression will be auto-detected.
            </p>
          </div>
          {targetColumn && (
            <span className="badge badge-completed">
              <Check size={12} /> Target Configured: {targetColumn}
            </span>
          )}
        </div>

        <select
          value={targetColumn}
          onChange={(e) => onSelectTarget(e.target.value)}
          style={{
            width: '100%',
            maxWidth: '400px',
            padding: '10px 14px',
            backgroundColor: 'var(--bg-secondary)',
            border: '1px solid var(--border-subtle)',
            borderRadius: 'var(--radius-md)',
            color: 'var(--text-primary)',
            fontSize: '0.9rem',
            fontFamily: 'var(--font-sans)',
            cursor: 'pointer'
          }}
        >
          <option value="" disabled>-- Select Target Variable --</option>
          {columns.map((col) => (
            <option key={col} value={col}>
              {col} ({summary.dtypes[col]}, {summary.unique_counts[col]} unique values)
            </option>
          ))}
        </select>
      </div>

      {/* Feature Columns Selection Card */}
      <div className="glass-panel" style={{ padding: '24px' }}>
        <div style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          flexWrap: 'wrap',
          gap: '12px',
          marginBottom: '16px'
        }}>
          <div>
            <h3 style={{ fontSize: '1.05rem', fontWeight: 700, color: 'var(--text-primary)' }}>
              2. Select Input Features (X)
            </h3>
            <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>
              Choose candidate predictor features. Target column ({targetColumn}) is automatically isolated.
            </p>
          </div>

          <div style={{ display: 'flex', gap: '8px' }}>
            <button onClick={onSelectAllFeatures} className="btn btn-secondary" style={{ padding: '5px 12px', fontSize: '0.75rem' }}>
              Select All
            </button>
            <button onClick={onDeselectAllFeatures} className="btn btn-secondary" style={{ padding: '5px 12px', fontSize: '0.75rem' }}>
              Clear All
            </button>
          </div>
        </div>

        {/* Feature Chips */}
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: '8px' }}>
          {columns.filter((c) => c !== targetColumn).map((col) => {
            const isSelected = selectedFeatures.includes(col);
            return (
              <div
                key={col}
                className={`chip ${isSelected ? 'selected' : ''}`}
                onClick={() => onToggleFeature(col)}
              >
                <span>{col}</span>
                <span style={{ fontSize: '0.7rem', opacity: 0.6 }}>
                  ({summary.dtypes[col]})
                </span>
                {isSelected && <Check size={12} />}
              </div>
            );
          })}
        </div>
      </div>

      {/* Dataset Preview Table */}
      <div className="glass-panel" style={{ padding: '24px' }}>
        <h3 style={{ fontSize: '1.05rem', fontWeight: 700, color: 'var(--text-primary)', marginBottom: '14px' }}>
          Dataset Preview (First 10 Rows)
        </h3>

        <div className="data-table-container">
          <table className="data-table">
            <thead>
              <tr>
                {columns.map((col) => (
                  <th key={col}>
                    <div style={{ display: 'flex', flexDirection: 'column', gap: '2px' }}>
                      <span style={{ color: col === targetColumn ? '#34d399' : 'inherit' }}>
                        {col} {col === targetColumn ? '(Target)' : ''}
                      </span>
                      <span style={{ fontSize: '0.65rem', color: 'var(--text-muted)' }}>
                        {summary.dtypes[col]} • {summary.missing_counts[col]} nulls
                      </span>
                    </div>
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {preview.map((row, rIdx) => (
                <tr key={rIdx}>
                  {columns.map((col) => (
                    <td key={col} style={{ color: col === targetColumn ? '#34d399' : 'inherit' }}>
                      {String(row[col])}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Start Automation CTA */}
      <div style={{
        display: 'flex',
        justifyContent: 'flex-end',
        alignItems: 'center',
        gap: '14px',
        padding: '16px 0'
      }}>
        {!targetColumn && (
          <div style={{ display: 'flex', alignItems: 'center', gap: '6px', color: '#f59e0b', fontSize: '0.85rem' }}>
            <AlertTriangle size={15} />
            <span>Please select a Target Column (Y) to proceed.</span>
          </div>
        )}

        <button
          onClick={onStartPipeline}
          disabled={!targetColumn || selectedFeatures.length === 0 || isStarting}
          className="btn btn-accent"
          style={{ padding: '12px 28px', fontSize: '0.95rem' }}
        >
          <Play size={16} />
          {isStarting ? 'Initiating LangGraph Workflow...' : 'Run Automated ML Pipeline'}
        </button>
      </div>
    </div>
  );
}
