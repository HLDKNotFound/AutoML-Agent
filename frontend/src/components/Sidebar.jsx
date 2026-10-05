import React from 'react';
import { 
  CheckCircle2, 
  Loader2, 
  Circle, 
  XCircle, 
  Upload, 
  Sliders, 
  BarChart3, 
  ShieldAlert, 
  Filter, 
  Compass, 
  Cpu, 
  Layers, 
  Activity, 
  Award, 
  CheckCheck, 
  Download, 
  Sparkles 
} from 'lucide-react';

const STEP_ICONS = {
  1: Upload,
  2: Sliders,
  3: BarChart3,
  4: ShieldAlert,
  5: Filter,
  6: Compass,
  7: Cpu,
  8: Layers,
  9: Activity,
  10: Award,
  11: CheckCheck,
  12: Download,
  13: Sparkles
};

export default function Sidebar({ steps, currentStep, onSelectStep, workflowStatus }) {
  return (
    <aside style={{
      width: '280px',
      minWidth: '280px',
      backgroundColor: 'var(--bg-secondary)',
      borderRight: '1px solid var(--border-subtle)',
      display: 'flex',
      flexDirection: 'column',
      height: '100vh',
      position: 'sticky',
      top: 0
    }}>
      {/* Brand Header */}
      <div style={{
        padding: '24px 20px',
        borderBottom: '1px solid var(--border-subtle)',
        display: 'flex',
        alignItems: 'center',
        gap: '12px'
      }}>
        <div style={{
          width: '38px',
          height: '38px',
          borderRadius: '10px',
          background: 'linear-gradient(135deg, #0ea5e9 0%, #6366f1 100%)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          boxShadow: 'var(--glow-cyan)'
        }}>
          <Sparkles size={20} color="#ffffff" />
        </div>
        <div>
          <h1 style={{ fontSize: '0.95rem', fontWeight: 800, letterSpacing: '-0.02em', color: 'var(--text-primary)' }}>
            ML Automation Agent
          </h1>
          <p style={{ fontSize: '0.72rem', color: 'var(--text-muted)', fontWeight: 500 }}>
            LangGraph • Scikit-Learn • Gemini
          </p>
        </div>
      </div>

      {/* Pipeline Steps Tracker */}
      <div style={{
        flex: 1,
        overflowY: 'auto',
        padding: '16px 12px',
        display: 'flex',
        flexDirection: 'column',
        gap: '4px'
      }}>
        <div style={{
          padding: '6px 10px 10px 10px',
          fontSize: '0.72rem',
          fontWeight: 700,
          textTransform: 'uppercase',
          letterSpacing: '0.06em',
          color: 'var(--text-muted)'
        }}>
          Automated Pipeline Steps
        </div>

        {steps.map((step) => {
          const StepIcon = STEP_ICONS[step.id] || Circle;
          const isActive = currentStep === step.id;

          const isRunning = step.status === 'running';
          let statusIcon;
          if (step.status === 'completed') {
            statusIcon = <CheckCircle2 size={15} color="#34d399" />;
          } else if (isRunning) {
            statusIcon = <Loader2 size={15} color="#38bdf8" className="animate-spin" />;
          } else if (step.status === 'failed') {
            statusIcon = <XCircle size={15} color="#fb7185" />;
          } else {
            statusIcon = <Circle size={13} color="#475569" />;
          }

          return (
            <button
              key={step.id}
              onClick={() => onSelectStep(step.id)}
              style={{
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'space-between',
                padding: '9px 12px',
                borderRadius: 'var(--radius-md)',
                backgroundColor: isRunning 
                  ? 'rgba(56, 189, 248, 0.16)' 
                  : isActive 
                  ? 'rgba(56, 189, 248, 0.08)' 
                  : 'transparent',
                border: isRunning 
                  ? '1px solid #38bdf8' 
                  : isActive 
                  ? '1px solid var(--border-active)' 
                  : '1px solid transparent',
                boxShadow: isRunning ? '0 0 12px rgba(56, 189, 248, 0.35)' : 'none',
                color: isRunning || isActive ? 'var(--text-primary)' : 'var(--text-secondary)',
                cursor: 'pointer',
                textAlign: 'left',
                width: '100%',
                transition: 'all 0.15s ease'
              }}
            >
              <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                <div style={{
                  color: isRunning || isActive ? '#38bdf8' : 'var(--text-muted)',
                  display: 'flex',
                  alignItems: 'center'
                }}>
                  <StepIcon size={16} />
                </div>
                <div style={{ display: 'flex', flexDirection: 'column' }}>
                  <span style={{ fontSize: '0.8rem', fontWeight: isRunning || isActive ? 600 : 500 }}>
                    {step.id}. {step.name}
                  </span>
                  {isRunning && (
                    <span style={{ fontSize: '0.68rem', color: '#38bdf8', fontWeight: 700, letterSpacing: '0.04em' }}>
                      IN PROGRESS...
                    </span>
                  )}
                </div>
              </div>
              <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                {isRunning && (
                  <span style={{
                    fontSize: '0.65rem',
                    padding: '1px 6px',
                    borderRadius: '4px',
                    backgroundColor: '#38bdf8',
                    color: '#040711',
                    fontWeight: 800,
                    letterSpacing: '0.04em'
                  }}>
                    ACTIVE
                  </span>
                )}
                {statusIcon}
              </div>
            </button>
          );
        })}
      </div>

      {/* Global Status Footer */}
      <div style={{
        padding: '16px',
        borderTop: '1px solid var(--border-subtle)',
        backgroundColor: 'rgba(7, 9, 14, 0.6)'
      }}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
          <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Status</span>
          <span className={`badge badge-${workflowStatus}`}>
            {workflowStatus.toUpperCase()}
          </span>
        </div>
      </div>
    </aside>
  );
}
