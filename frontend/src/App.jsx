import React, { useState, useEffect, useRef } from 'react';
import Sidebar from './components/Sidebar';
import Header from './components/Header';
import UploadSection from './components/UploadSection';
import DatasetPreviewSection from './components/DatasetPreviewSection';
import DataQualitySection from './components/DataQualitySection';
import MLStrategySection from './components/MLStrategySection';
import TrainingProgressSection from './components/TrainingProgressSection';
import ValidationComparisonSection from './components/ValidationComparisonSection';
import FinalTestEvaluationSection from './components/FinalTestEvaluationSection';
import PredictionSection from './components/PredictionSection';
import ApiKeyModal from './components/ApiKeyModal';

const INITIAL_STEPS = [
  { id: 1, name: 'Upload Dataset', status: 'pending' },
  { id: 2, name: 'Features & Target', status: 'pending' },
  { id: 3, name: 'Dataset Analysis', status: 'pending' },
  { id: 4, name: 'Data Quality Loop', status: 'pending' },
  { id: 5, name: 'Feature Selection', status: 'pending' },
  { id: 6, name: 'ML Strategy', status: 'pending' },
  { id: 7, name: 'Preprocessing', status: 'pending' },
  { id: 8, name: 'Model Training', status: 'pending' },
  { id: 9, name: 'Hyperparameter Tuning', status: 'pending' },
  { id: 10, name: 'Validation Evaluation', status: 'pending' },
  { id: 11, name: 'Final Test Evaluation', status: 'pending' },
  { id: 12, name: 'Download Model', status: 'pending' },
  { id: 13, name: 'Prediction Studio', status: 'pending' }
];

export default function App() {
  const [steps, setSteps] = useState(INITIAL_STEPS);
  const [currentStep, setCurrentStep] = useState(1);
  const [sessionId, setSessionId] = useState('');
  const [filename, setFilename] = useState('');
  const [summary, setSummary] = useState(null);
  const [targetColumn, setTargetColumn] = useState('');
  const [selectedFeatures, setSelectedFeatures] = useState([]);
  const [sampleDatasets, setSampleDatasets] = useState([]);
  
  const [workflowStatus, setWorkflowStatus] = useState('pending'); // pending, running, completed, failed
  const [events, setEvents] = useState([]);
  
  // Pipeline Data from Backend
  const [profile, setProfile] = useState(null);
  const [qualityReport, setQualityReport] = useState(null);
  const [featureSelection, setFeatureSelection] = useState(null);
  const [problemType, setProblemType] = useState('');
  const [primaryMetric, setPrimaryMetric] = useState('');
  const [candidateModels, setCandidateModels] = useState([]);
  const [modelResults, setModelResults] = useState([]);
  const [bestModel, setBestModel] = useState({});
  const [testResults, setTestResults] = useState({});
  const [plots, setPlots] = useState({});
  const [agentDecisions, setAgentDecisions] = useState([]);
  
  // Prediction State
  const [predictionResult, setPredictionResult] = useState(null);
  const [isPredicting, setIsPredicting] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [isStarting, setIsStarting] = useState(false);

  // API Key Modal State
  const [isKeyModalOpen, setIsKeyModalOpen] = useState(false);
  const [geminiApiKey, setGeminiApiKey] = useState('');

  const eventSourceRef = useRef(null);

  // Load sample datasets on mount
  useEffect(() => {
    fetch('/api/sample-datasets')
      .then((res) => res.json())
      .then((data) => {
        if (data.samples) setSampleDatasets(data.samples);
      })
      .catch((err) => console.error('Error loading sample datasets:', err));
  }, []);

  const updateStepStatus = (stepId, status) => {
    setSteps((prev) =>
      prev.map((s) => (s.id === stepId ? { ...s, status } : s))
    );
  };

  const handleUploadFile = async (file) => {
    setIsLoading(true);
    const formData = new FormData();
    formData.append('file', file);

    try {
      const res = await fetch('/api/upload', {
        method: 'POST',
        body: formData
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || 'Upload failed');

      setSessionId(data.session_id);
      setFilename(file.name);
      setSummary(data.summary);
      
      // Auto select default target and all other features
      const cols = data.summary.columns;
      const lastCol = cols[cols.length - 1];
      setTargetColumn(lastCol);
      setSelectedFeatures(cols.filter((c) => c !== lastCol));

      updateStepStatus(1, 'completed');
      updateStepStatus(2, 'running');
      setCurrentStep(2);
    } catch (err) {
      alert(`Error uploading file: ${err.message}`);
    } finally {
      setIsLoading(false);
    }
  };

  const handleLoadSample = async (sampleName) => {
    setIsLoading(true);
    try {
      const res = await fetch('/api/load-sample', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ sample_name: sampleName })
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || 'Failed to load sample');

      setSessionId(data.session_id);
      setFilename(sampleName);
      setSummary(data.summary);

      // Pick target based on dataset type
      const cols = data.summary.columns;
      let target = cols[cols.length - 1];
      if (sampleName.includes('churn') && cols.includes('churn')) target = 'churn';
      if (sampleName.includes('housing') && cols.includes('MedHouseVal')) target = 'MedHouseVal';
      if (sampleName.includes('wine') && cols.includes('target')) target = 'target';

      setTargetColumn(target);
      setSelectedFeatures(cols.filter((c) => c !== target));

      updateStepStatus(1, 'completed');
      updateStepStatus(2, 'running');
      setCurrentStep(2);
    } catch (err) {
      alert(`Error loading sample: ${err.message}`);
    } finally {
      setIsLoading(false);
    }
  };

  const handleToggleFeature = (col) => {
    setSelectedFeatures((prev) =>
      prev.includes(col) ? prev.filter((c) => c !== col) : [...prev, col]
    );
  };

  const handleSelectAllFeatures = () => {
    if (!summary) return;
    setSelectedFeatures(summary.columns.filter((c) => c !== targetColumn));
  };

  const handleDeselectAllFeatures = () => {
    setSelectedFeatures([]);
  };

  const fetchSessionSnapshot = async (sId) => {
    try {
      const res = await fetch(`/api/session/${sId}`);
      if (!res.ok) return;
      const data = await res.json();
      
      if (data.profile) setProfile(data.profile);
      if (data.data_quality_report) setQualityReport(data.data_quality_report);
      if (data.feature_selection) setFeatureSelection(data.feature_selection);
      if (data.problem_type) setProblemType(data.problem_type);
      if (data.primary_metric) setPrimaryMetric(data.primary_metric);
      if (data.candidate_models) setCandidateModels(data.candidate_models);
      if (data.model_results) setModelResults(data.model_results);
      if (data.best_model) setBestModel(data.best_model);
      if (data.test_results) setTestResults(data.test_results);
      if (data.plots) setPlots(data.plots);
      if (data.agent_decisions) setAgentDecisions(data.agent_decisions);
      if (data.status) setWorkflowStatus(data.status);
    } catch (err) {
      console.error('Failed to sync session state:', err);
    }
  };

  const handleStartPipeline = async () => {
    if (!sessionId || !targetColumn) return;
    setIsStarting(true);
    setWorkflowStatus('running');
    updateStepStatus(2, 'completed');

    try {
      const res = await fetch('/api/start-pipeline', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          session_id: sessionId,
          target_column: targetColumn,
          selected_features: selectedFeatures,
          gemini_api_key: geminiApiKey
        })
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || 'Failed to start pipeline');

      // Connect SSE Stream
      if (eventSourceRef.current) {
        eventSourceRef.current.close();
      }

      const es = new EventSource(`/api/stream/${sessionId}`);
      eventSourceRef.current = es;

      es.onmessage = (event) => {
        try {
          const ev = JSON.parse(event.data);
          setEvents((prev) => [...prev, ev]);

          if (ev.step_id && ev.step_id > 0) {
            updateStepStatus(ev.step_id, ev.status);
            if (ev.status === 'running') {
              setCurrentStep(ev.step_id);
            }
          }

          // Trigger state sync on node completion
          fetchSessionSnapshot(sessionId);

          if (ev.step_id === 13 || ev.status === 'completed') {
            setWorkflowStatus('completed');
            setSteps((prev) => prev.map((s) => ({ ...s, status: 'completed' })));
            setCurrentStep(10); // Show validation/champion view
            es.close();
          } else if (ev.status === 'failed') {
            setWorkflowStatus('failed');
            es.close();
          }
        } catch (e) {
          console.warn('Non-JSON SSE event:', event.data);
        }
      };

      es.onerror = () => {
        es.close();
        fetchSessionSnapshot(sessionId);
      };
    } catch (err) {
      alert(`Error starting pipeline: ${err.message}`);
      setWorkflowStatus('failed');
    } finally {
      setIsStarting(false);
    }
  };

  const handleDownloadModel = () => {
    if (!sessionId) return;
    window.location.href = `/api/models/${sessionId}/download`;
  };

  const handleUploadPredictionFile = async (file) => {
    if (!sessionId) return;
    setIsPredicting(true);
    const formData = new FormData();
    formData.append('file', file);

    try {
      const res = await fetch(`/api/predict/${sessionId}`, {
        method: 'POST',
        body: formData
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || 'Prediction failed');
      setPredictionResult(data);
      updateStepStatus(13, 'completed');
    } catch (err) {
      alert(`Prediction failed: ${err.message}`);
    } finally {
      setIsPredicting(false);
    }
  };

  const handleResetSession = () => {
    if (eventSourceRef.current) {
      eventSourceRef.current.close();
    }
    setSteps(INITIAL_STEPS);
    setCurrentStep(1);
    setSessionId('');
    setFilename('');
    setSummary(null);
    setTargetColumn('');
    setSelectedFeatures([]);
    setWorkflowStatus('pending');
    setEvents([]);
    setProfile(null);
    setQualityReport(null);
    setFeatureSelection(null);
    setCandidateModels([]);
    setModelResults([]);
    setBestModel({});
    setTestResults({});
    setPlots({});
    setAgentDecisions([]);
    setPredictionResult(null);
  };

  return (
    <div style={{ display: 'flex', minHeight: '100vh', backgroundColor: 'var(--bg-primary)' }}>
      {/* 13-Step Progress Tracker Sidebar */}
      <Sidebar
        steps={steps}
        currentStep={currentStep}
        onSelectStep={setCurrentStep}
        workflowStatus={workflowStatus}
      />

      {/* Main Workspace Area */}
      <div style={{ flex: 1, display: 'flex', flexDirection: 'column', minWidth: 0 }}>
        <Header
          filename={filename}
          onOpenKeyModal={() => setIsKeyModalOpen(true)}
          onResetSession={handleResetSession}
          hasKey={Boolean(geminiApiKey)}
        />

        <main style={{ flex: 1, padding: '32px', overflowY: 'auto' }}>
          {/* Step 1: Upload or Sample selection */}
          {currentStep === 1 && (
            <UploadSection
              onUploadFile={handleUploadFile}
              onLoadSample={handleLoadSample}
              sampleDatasets={sampleDatasets}
              loading={isLoading}
            />
          )}

          {/* Step 2: Features, Target & Preview */}
          {currentStep === 2 && summary && (
            <DatasetPreviewSection
              summary={summary}
              targetColumn={targetColumn}
              onSelectTarget={setTargetColumn}
              selectedFeatures={selectedFeatures}
              onToggleFeature={handleToggleFeature}
              onSelectAllFeatures={handleSelectAllFeatures}
              onDeselectAllFeatures={handleDeselectAllFeatures}
              onStartPipeline={handleStartPipeline}
              isStarting={isStarting}
            />
          )}

          {/* Step 3 & 4: Data Quality & Profiling */}
          {(currentStep === 3 || currentStep === 4 || currentStep === 5) && (
            <DataQualitySection
              profile={profile}
              qualityReport={qualityReport}
              featureSelection={featureSelection}
              plots={plots}
            />
          )}

          {/* Step 6, 7 & 8: ML Strategy & Model Preprocessing */}
          {(currentStep === 6 || currentStep === 7 || currentStep === 8) && (
            <MLStrategySection
              problemType={problemType}
              primaryMetric={primaryMetric}
              candidateModels={candidateModels}
              agentDecisions={agentDecisions}
            />
          )}

          {/* Step 9: Coarse-to-Fine Tuning Progress */}
          {currentStep === 9 && (
            <TrainingProgressSection
              events={events}
              workflowStatus={workflowStatus}
              modelResults={modelResults}
            />
          )}

          {/* Step 10: Validation Evaluation & Champion Selection */}
          {currentStep === 10 && (
            <ValidationComparisonSection
              modelResults={modelResults}
              bestModel={bestModel}
              primaryMetric={primaryMetric}
              plots={plots}
            />
          )}

          {/* Step 11: Final Test Evaluation (Untouched 15%) */}
          {currentStep === 11 && (
            <FinalTestEvaluationSection
              bestModel={bestModel}
              testResults={testResults}
              primaryMetric={primaryMetric}
              plots={plots}
            />
          )}

          {/* Step 12 & 13: Model Download & Batch Prediction Studio */}
          {(currentStep === 12 || currentStep === 13) && (
            <PredictionSection
              sessionId={sessionId}
              onDownloadModel={handleDownloadModel}
              onUploadPredictionFile={handleUploadPredictionFile}
              predictionResult={predictionResult}
              isPredicting={isPredicting}
            />
          )}
        </main>
      </div>

      {/* Gemini API Key Configuration Modal */}
      <ApiKeyModal
        isOpen={isKeyModalOpen}
        onClose={() => setIsKeyModalOpen(false)}
        onSaveKey={setGeminiApiKey}
        currentKey={geminiApiKey}
      />
    </div>
  );
}
