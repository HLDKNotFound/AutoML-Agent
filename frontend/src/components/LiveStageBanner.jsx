import React, { useEffect, useState } from 'react';
import { Activity, Terminal, CheckCircle2, AlertCircle, ArrowRight } from 'lucide-react';

export default function LiveStageBanner({
  workflowStatus,
  activeStepId,
  activeStepName,
  activeMessage,
  steps = [],
  startTime,
  onViewLogs,
  onNavigateToStage
}) {
  const [elapsedSeconds, setElapsedSeconds] = useState(0);

  useEffect(() => {
    if (workflowStatus !== 'running' || !startTime) return;
    const interval = setInterval(() => {
      setElapsedSeconds(Math.max(0, Math.floor((Date.now() - startTime) / 1000)));
    }, 1000);
    return () => clearInterval(interval);
  }, [workflowStatus, startTime]);

  if (workflowStatus === 'pending') return null;

  const completedSteps = steps.filter((s) => s.status === 'completed').length;
  const progressPercent = Math.min(100, Math.round((completedSteps / 13) * 100));

  const formatTime = (secs) => {
    const mins = Math.floor(secs / 60);
    const rem = secs % 60;
    return `${mins.toString().padStart(2, '0')}:${rem.toString().padStart(2, '0')}`;
  };

  const isRunning = workflowStatus === 'running';
  const isCompleted = workflowStatus === 'completed';
  const isFailed = workflowStatus === 'failed';

  return (
    <div
      className={`glass-panel ${isRunning ? 'stage-pulse' : ''}`}
      style={{
        padding: '16px 20px',
        marginBottom: '24px',
        border: isRunning
          ? '1px solid rgba(56, 189, 248, 0.5)'
          : isCompleted
          ? '1px solid rgba(16, 185, 129, 0.5)'
          : '1px solid rgba(244, 63, 94, 0.6)',
        background: isRunning
          ? 'linear-gradient(135deg, rgba(13, 27, 42, 0.9) 0%, rgba(15, 23, 42, 0.95) 100%)'
          : isCompleted
          ? 'linear-gradient(135deg, rgba(6, 78, 59, 0.3) 0%, rgba(15, 23, 42, 0.95) 100%)'
          : 'linear-gradient(135deg, rgba(88, 28, 42, 0.85) 0%, rgba(30, 10, 18, 0.95) 100%)',
        position: 'relative',
        overflow: 'hidden',
        boxShadow: isFailed ? '0 8px 32px rgba(244, 63, 94, 0.25)' : 'none'
      }}
    >
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: '16px', flexWrap: 'wrap' }}>
        {/* Left: Active Stage Info */}
        <div style={{ display: 'flex', alignItems: 'flex-start', gap: '14px', flex: 1, minWidth: '280px' }}>
          <div style={{
            width: '42px',
            height: '42px',
            borderRadius: '10px',
            backgroundColor: isRunning ? 'rgba(56, 189, 248, 0.15)' : isCompleted ? 'rgba(16, 185, 129, 0.15)' : 'rgba(244, 63, 94, 0.25)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            flexShrink: 0
          }}>
            {isRunning && <span className="radar-dot" />}
            {isCompleted && <CheckCircle2 size={22} color="#34d399" />}
            {isFailed && <AlertCircle size={24} color="#f43f5e" />}
          </div>

          <div style={{ display: 'flex', flexDirection: 'column', gap: '4px', flex: 1 }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <span style={{
                fontSize: '0.72rem',
                fontWeight: 800,
                textTransform: 'uppercase',
                letterSpacing: '0.06em',
                color: isRunning ? '#38bdf8' : isCompleted ? '#34d399' : '#fb7185'
              }}>
                {isRunning ? `STAGE ${activeStepId || '•'} OF 13: ${activeStepName || 'Processing'}` : isCompleted ? 'PIPELINE COMPLETE (13/13)' : `EXECUTION ERROR (STAGE ${activeStepId || '•'})`}
              </span>
              {isRunning && (
                <span style={{
                  fontSize: '0.7rem',
                  padding: '2px 8px',
                  borderRadius: '9999px',
                  backgroundColor: 'rgba(56, 189, 248, 0.2)',
                  color: '#38bdf8',
                  fontWeight: 600
                }}>
                  {formatTime(elapsedSeconds)}
                </span>
              )}
            </div>

            <span style={{
              fontSize: '0.88rem',
              color: isFailed ? '#fecdd3' : 'var(--text-primary)',
              fontWeight: isFailed ? 600 : 500,
              lineHeight: '1.4'
            }}>
              {activeMessage || (isRunning ? 'Executing ML pipeline stage...' : isCompleted ? 'Model trained, evaluated, and ready for inference.' : 'Pipeline encountered an unexpected runtime failure.')}
            </span>

            {isFailed && (
              <div style={{
                marginTop: '6px',
                padding: '8px 12px',
                borderRadius: '6px',
                backgroundColor: 'rgba(0, 0, 0, 0.4)',
                border: '1px solid rgba(244, 63, 94, 0.3)',
                fontSize: '0.75rem',
                color: '#fda4af',
                fontFamily: 'monospace',
                wordBreak: 'break-word',
                maxHeight: '120px',
                overflowY: 'auto'
              }}>
                ⚠️ Failure report logged. Check terminal logs or reconfigure pipeline settings.
              </div>
            )}
          </div>
        </div>

        {/* Right: Progress bar & Actions */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '14px' }}>
          {/* Progress bar */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: '4px', minWidth: '130px' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.72rem', color: 'var(--text-muted)' }}>
              <span>{isFailed ? 'Failed at' : 'Progress'}</span>
              <span style={{ fontWeight: 600, color: isFailed ? '#fb7185' : 'var(--text-primary)' }}>{progressPercent}%</span>
            </div>
            <div style={{
              height: '6px',
              backgroundColor: 'rgba(255, 255, 255, 0.08)',
              borderRadius: '9999px',
              overflow: 'hidden'
            }}>
              <div style={{
                height: '100%',
                width: `${progressPercent}%`,
                backgroundColor: isFailed ? '#fb7185' : isCompleted ? '#34d399' : '#38bdf8',
                borderRadius: '9999px',
                transition: 'width 0.4s ease'
              }} />
            </div>
          </div>

          {/* Action button */}
          <button
            onClick={onViewLogs}
            className="btn btn-secondary"
            style={{
              padding: '7px 12px',
              fontSize: '0.75rem',
              gap: '6px',
              backgroundColor: isFailed ? 'rgba(244, 63, 94, 0.15)' : 'rgba(56, 189, 248, 0.1)',
              border: isFailed ? '1px solid rgba(244, 63, 94, 0.4)' : '1px solid rgba(56, 189, 248, 0.3)',
              color: isFailed ? '#fb7185' : '#38bdf8'
            }}
          >
            <Terminal size={14} />
            Live Logs
          </button>

          {activeStepId && onNavigateToStage && !isFailed && (
            <button
              onClick={() => onNavigateToStage(activeStepId)}
              className="btn btn-secondary"
              style={{ padding: '7px 12px', fontSize: '0.75rem', gap: '6px' }}
            >
              Inspect Stage <ArrowRight size={13} />
            </button>
          )}
        </div>
      </div>
    </div>
  );
}
