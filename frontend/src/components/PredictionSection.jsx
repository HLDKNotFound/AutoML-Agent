import React, { useState } from 'react';
import { Download, UploadCloud, CheckCircle2, AlertTriangle, FileSpreadsheet, ArrowDown } from 'lucide-react';

export default function PredictionSection({ 
  sessionId, 
  onDownloadModel, 
  onUploadPredictionFile, 
  predictionResult, 
  isPredicting 
}) {
  const [dragOver, setDragOver] = useState(false);

  const handleDrop = (e) => {
    e.preventDefault();
    setDragOver(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      onUploadPredictionFile(e.dataTransfer.files[0]);
    }
  };

  const handleFileSelect = (e) => {
    if (e.target.files && e.target.files[0]) {
      onUploadPredictionFile(e.target.files[0]);
    }
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
      {/* Download Trained Model Artifact Card */}
      <div className="glass-panel" style={{ padding: '24px' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '16px' }}>
          <div>
            <span style={{ fontSize: '0.75rem', fontWeight: 700, color: '#38bdf8', textTransform: 'uppercase', letterSpacing: '0.04em' }}>
              Production Deployment Package
            </span>
            <h3 style={{ fontSize: '1.2rem', fontWeight: 800, color: 'var(--text-primary)', marginTop: '2px' }}>
              Download Trained Pipeline Bundle (.ZIP)
            </h3>
            <p style={{ fontSize: '0.82rem', color: 'var(--text-secondary)', marginTop: '4px' }}>
              Contains <code>model.joblib</code> (complete Preprocessing + Estimator), <code>metadata.json</code>, 
              <code>feature_schema.json</code>, and <code>training_report.json</code>.
            </p>
          </div>

          <button onClick={onDownloadModel} className="btn btn-primary" style={{ padding: '12px 22px' }}>
            <Download size={16} />
            Download Model Package
          </button>
        </div>
      </div>

      {/* Batch Inference Studio */}
      <div className="glass-panel" style={{ padding: '24px' }}>
        <h3 style={{ fontSize: '1.15rem', fontWeight: 800, color: 'var(--text-primary)', marginBottom: '6px' }}>
          Batch Prediction Studio
        </h3>
        <p style={{ fontSize: '0.84rem', color: 'var(--text-secondary)', marginBottom: '20px' }}>
          Upload an unlabeled CSV containing the required features. The pipeline will apply the exact fitted transformations and return predicted outputs formatted as <code>id,predict</code>.
        </p>

        {/* Upload Zone */}
        <div
          onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
          onDragLeave={() => setDragOver(false)}
          onDrop={handleDrop}
          style={{
            padding: '36px 20px',
            textAlign: 'center',
            border: dragOver ? '2px dashed var(--accent-cyan)' : '2px dashed var(--border-subtle)',
            backgroundColor: dragOver ? 'rgba(56, 189, 248, 0.05)' : 'rgba(15, 23, 42, 0.6)',
            borderRadius: 'var(--radius-lg)',
            cursor: 'pointer'
          }}
          onClick={() => document.getElementById('pred-file-input').click()}
        >
          <input
            id="pred-file-input"
            type="file"
            accept=".csv"
            onChange={handleFileSelect}
            style={{ display: 'none' }}
          />

          <div style={{
            width: '52px',
            height: '52px',
            borderRadius: '50%',
            backgroundColor: 'rgba(56, 189, 248, 0.1)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            margin: '0 auto 12px auto'
          }}>
            <FileSpreadsheet size={26} color="#38bdf8" />
          </div>

          <h4 style={{ fontSize: '1.0rem', fontWeight: 700, marginBottom: '4px' }}>
            {isPredicting ? 'Executing Pipeline Inference...' : 'Upload Unlabeled CSV for Prediction'}
          </h4>
          <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>
            Drag & drop or click to select CSV (e.g. <code>data/churn_unlabeled.csv</code>)
          </p>
        </div>

        {/* Prediction Results Preview */}
        {predictionResult && (
          <div style={{ marginTop: '24px', display: 'flex', flexDirection: 'column', gap: '16px' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                <CheckCircle2 size={18} color="#34d399" />
                <span style={{ fontSize: '0.95rem', fontWeight: 700, color: 'var(--text-primary)' }}>
                  Inference Complete ({predictionResult.total_rows} records predicted)
                </span>
              </div>

              <a
                href={predictionResult.download_url}
                download
                className="btn btn-accent"
                style={{ padding: '8px 16px', fontSize: '0.82rem' }}
              >
                <ArrowDown size={14} />
                Download Predictions (CSV)
              </a>
            </div>

            <div className="data-table-container">
              <table className="data-table">
                <thead>
                  <tr>
                    {predictionResult.preview?.[0] && Object.keys(predictionResult.preview[0]).map((key) => (
                      <th key={key}>{key.toUpperCase()}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {predictionResult.preview?.map((row, idx) => (
                    <tr key={idx}>
                      {Object.entries(row).map(([k, v], cellIdx) => (
                        <td
                          key={k}
                          style={{
                            color: cellIdx === 0 ? 'var(--text-secondary)' : cellIdx === 1 ? '#34d399' : '#38bdf8',
                            fontWeight: cellIdx === 1 ? 700 : 400
                          }}
                        >
                          {String(v)}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
