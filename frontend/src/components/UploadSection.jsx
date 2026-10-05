import React, { useState } from 'react';
import { UploadCloud, FileText, Sparkles, Check, ArrowRight } from 'lucide-react';

export default function UploadSection({ onUploadFile, onLoadSample, sampleDatasets, loading }) {
  const [dragOver, setDragOver] = useState(false);

  const handleFileDrop = (e) => {
    e.preventDefault();
    setDragOver(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      onUploadFile(e.dataTransfer.files[0]);
    }
  };

  const handleFileSelect = (e) => {
    if (e.target.files && e.target.files[0]) {
      onUploadFile(e.target.files[0]);
    }
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '30px' }}>
      {/* Hero Intro */}
      <div>
        <h2 style={{ fontSize: '1.6rem', fontWeight: 800, color: 'var(--text-primary)', marginBottom: '8px' }}>
          Automated Machine Learning Studio
        </h2>
        <p style={{ color: 'var(--text-secondary)', fontSize: '0.92rem', maxWidth: '800px' }}>
          Upload any tabular dataset. The agent will orchestrate automated data quality analysis, 
          leakage detection, multi-model family selection, model-specific preprocessing, 
          coarse-to-fine Bayesian hyperparameter optimization, and untouched test set evaluation.
        </p>
      </div>

      {/* Upload Zone */}
      <div
        onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
        onDragLeave={() => setDragOver(false)}
        onDrop={handleFileDrop}
        className="glass-panel"
        style={{
          padding: '48px 24px',
          textAlign: 'center',
          border: dragOver ? '2px dashed var(--accent-cyan)' : '2px dashed var(--border-subtle)',
          backgroundColor: dragOver ? 'rgba(56, 189, 248, 0.05)' : 'var(--bg-card)',
          cursor: 'pointer',
          borderRadius: 'var(--radius-xl)'
        }}
        onClick={() => document.getElementById('csv-file-input').click()}
      >
        <input
          id="csv-file-input"
          type="file"
          accept=".csv"
          onChange={handleFileSelect}
          style={{ display: 'none' }}
        />

        <div style={{
          width: '64px',
          height: '64px',
          borderRadius: '50%',
          backgroundColor: 'rgba(56, 189, 248, 0.1)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          margin: '0 auto 16px auto',
          boxShadow: 'var(--glow-cyan)'
        }}>
          <UploadCloud size={32} color="#38bdf8" />
        </div>

        <h3 style={{ fontSize: '1.1rem', fontWeight: 700, marginBottom: '6px' }}>
          Upload your CSV dataset
        </h3>
        <p style={{ fontSize: '0.85rem', color: 'var(--text-muted)', marginBottom: '20px' }}>
          Drag and drop your file here, or click to browse from your device
        </p>

        <button className="btn btn-primary" disabled={loading}>
          {loading ? 'Reading Dataset...' : 'Browse CSV Files'}
        </button>
      </div>

      {/* Pre-packaged Sample Datasets for 1-Click Demo */}
      <div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '14px' }}>
          <Sparkles size={16} color="#fbbf24" />
          <h3 style={{ fontSize: '1.0rem', fontWeight: 700, color: 'var(--text-primary)' }}>
            Or explore with instant sample datasets
          </h3>
        </div>

        <div style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))',
          gap: '16px'
        }}>
          {sampleDatasets.map((sample) => (
            <div
              key={sample.name}
              className="glass-panel"
              style={{
                padding: '20px',
                display: 'flex',
                flexDirection: 'column',
                justifyContent: 'space-between',
                gap: '14px'
              }}
            >
              <div>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '8px' }}>
                  <h4 style={{ fontSize: '0.95rem', fontWeight: 700, color: 'var(--text-primary)' }}>
                    {sample.title}
                  </h4>
                  <span className="badge badge-family">
                    {sample.type}
                  </span>
                </div>
                <p style={{ fontSize: '0.82rem', color: 'var(--text-muted)', lineHeight: '1.4' }}>
                  {sample.description}
                </p>
                <div style={{
                  display: 'flex',
                  gap: '12px',
                  marginTop: '12px',
                  fontSize: '0.75rem',
                  color: 'var(--text-secondary)'
                }}>
                  <span>Rows: <strong>{sample.rows}</strong></span>
                  <span>Target: <strong>{sample.target}</strong></span>
                </div>
              </div>

              <button
                onClick={(e) => {
                  e.stopPropagation();
                  onLoadSample(sample.name);
                }}
                disabled={loading}
                className="btn btn-secondary"
                style={{ width: '100%', justifyContent: 'space-between' }}
              >
                <span>Load Dataset</span>
                <ArrowRight size={14} />
              </button>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
