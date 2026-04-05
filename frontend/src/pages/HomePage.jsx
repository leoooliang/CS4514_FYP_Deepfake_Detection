import React, { useState } from 'react';
import { AlertCircle } from 'lucide-react';
import FileUpload from '../components/FileUpload';
import ProgressBar from '../components/ProgressBar';
import ResultsDashboard from '../components/ResultsDashboard';
import HistoryDashboard from '../components/HistoryDashboard';
import ErrorBoundary from '../components/ErrorBoundary';
import { detectDeepfake } from '../services/api';
import { useSession } from '../hooks/useSession';

const HomePage = () => {
  const { sessionId } = useSession();
  const [file, setFile] = useState(null);
  const [resetKey, setResetKey] = useState(0);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [progress, setProgress] = useState(0);
  const [results, setResults] = useState([]);
  const [error, setError] = useState(null);
  const [refreshHistory, setRefreshHistory] = useState(0);

  const handleFileSelect = (selectedFile) => {
    setFile(selectedFile);
    setResults([]);
    setError(null);
  };

  const handleFileRemove = () => {
    setFile(null);
    setResults([]);
    setError(null);
  };

  const handleAnalyze = async () => {
    if (!file) return;

    setIsAnalyzing(true);
    setProgress(0);
    setError(null);
    setResults([]);

    try {
      const data = await detectDeepfake(file, (uploadProgress) => {
        setProgress(uploadProgress);
      });

      setProgress(100);

      setTimeout(() => {
        const normalizedResults = Array.isArray(data)
          ? data
          : (data?.results && Array.isArray(data.results) ? data.results : [data]);

        setResults(normalizedResults.filter(Boolean));
        setIsAnalyzing(false);
        setRefreshHistory(prev => prev + 1);
      }, 500);

    } catch (err) {
      console.error('Analysis error:', err);
      setError(err.message || 'Analysis failed. Please try again.');
      setIsAnalyzing(false);
      setProgress(0);
    }
  };

  const handleReset = () => {
    setFile(null);
    setResults([]);
    setError(null);
    setProgress(0);
    setResetKey(prev => prev + 1);
    window.scrollTo({ top: 0, behavior: 'smooth' });
  };

  return (
    <div className="py-12">
      <div className="container mx-auto px-4 max-w-5xl">
        {/* Hero */}
        <div className="text-center mb-14">
          <h2 className="text-5xl md:text-6xl font-extrabold mb-5 tracking-tight leading-none text-white">
            Detect Deepfakes
          </h2>
          
          <p className="text-lg text-gray-400 max-w-xl mx-auto leading-relaxed">
            Upload images, videos, or audio files to check if they are Deepfakes. It only takes a few seconds to analyze.
          </p>
        </div>

        <div className="space-y-8">
          {/* File Upload */}
          <div>
            <FileUpload
              key={resetKey}
              onFileSelect={handleFileSelect}
              onFileRemove={handleFileRemove}
              disabled={isAnalyzing}
            />
          </div>

          {/* Analyze Button */}
          {file && results.length === 0 && !isAnalyzing && (
            <div className="flex justify-center">
              <button
                onClick={handleAnalyze}
                className="btn-primary text-base px-10 py-3.5"
              >
                Analyze for Deepfakes
              </button>
            </div>
          )}

          {/* Progress Bar */}
          {isAnalyzing && (
            <div>
              <ProgressBar progress={progress} status="Analyzing..." />
            </div>
          )}

          {/* Error */}
          {error && (
            <div className="card bg-red-500/10 border-red-500/15">
              <div className="flex items-start gap-3">
                <AlertCircle className="w-5 h-5 text-red-400 flex-shrink-0 mt-0.5" />
                <div>
                  <h3 className="text-base font-semibold text-red-400 mb-1">
                    Analysis Failed
                  </h3>
                  <p className="text-sm text-gray-300">{error}</p>
                  <button
                    onClick={handleReset}
                    className="mt-3 btn-secondary"
                  >
                    Try Again
                  </button>
                </div>
              </div>
            </div>
          )}

          {/* Results */}
          {results.length > 0 && !isAnalyzing && (
            <>
              <div className="space-y-6">
                {results.map((r, idx) => (
                  <ResultsDashboard key={r?.id ?? idx} result={r} />
                ))}
              </div>
              
              <div className="flex justify-center">
                <button
                  onClick={handleReset}
                  className="btn-secondary"
                >
                  Analyze Another File
                </button>
              </div>
            </>
          )}
        </div>
      </div>

      {/* History */}
      <div className="container mx-auto px-4 max-w-7xl mt-16">
        <ErrorBoundary showRetry={true}>
          <HistoryDashboard key={refreshHistory} />
        </ErrorBoundary>
      </div>
    </div>
  );
};

export default HomePage;
