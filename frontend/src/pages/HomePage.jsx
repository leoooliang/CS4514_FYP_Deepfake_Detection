/**
 * ============================================================================
 * HomePage - Main Application Page
 * ============================================================================
 * Main page with file upload and analysis workflow
 * 
 * Author: Senior Full-Stack Engineer
 * Date: 2026-01-28
 * ============================================================================
 */

import React, { useState } from 'react';
import { motion } from 'framer-motion';
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

  // Handle file selection
  const handleFileSelect = (selectedFile) => {
    setFile(selectedFile);
    setResults([]);
    setError(null);
  };

  // Handle file removal
  const handleFileRemove = () => {
    setFile(null);
    setResults([]);
    setError(null);
  };

  // Handle analysis
  const handleAnalyze = async () => {
    if (!file) return;

    setIsAnalyzing(true);
    setProgress(0);
    setError(null);
    setResults([]);

    try {
      // Simulate progress during upload
      const data = await detectDeepfake(file, (uploadProgress) => {
        setProgress(uploadProgress);
      });

      // Complete progress
      setProgress(100);

      // Show results
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

  // Reset for new analysis
  const handleReset = () => {
    setFile(null);
    setResults([]);
    setError(null);
    setProgress(0);
    // Force FileUpload remount so previews / internal state are cleared instantly.
    setResetKey(prev => prev + 1);
    window.scrollTo({ top: 0, behavior: 'smooth' });
  };

  return (
    <div className="min-h-screen py-12">
      <div className="container mx-auto px-4 max-w-5xl">
        {/* Hero Section */}
        <motion.div
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          className="text-center mb-16 relative"
        >
          <div className="absolute -top-20 left-1/2 -translate-x-1/2 w-96 h-96 bg-cyber-blue/10 rounded-full blur-[120px] pointer-events-none" />
          
          <motion.div
            initial={{ scale: 0.9, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            transition={{ duration: 0.5 }}
            className="inline-block mb-4 px-4 py-1.5 rounded-full bg-white/5 border border-white/10 backdrop-blur-md"
          >
            <span className="text-xs font-bold tracking-[0.2em] text-cyber-blue uppercase">AI-Powered Detection</span>
          </motion.div>

          <h2 className="text-6xl md:text-7xl font-black mb-6 tracking-tight leading-none">
            <span className="text-white">Detect </span>
            <span className="text-gradient-cyber">Deepfakes</span>
          </h2>
          
          <p className="text-xl text-gray-400 max-w-2xl mx-auto leading-relaxed font-medium">
            Upload images, videos, or audio files to check if they are Deepfakes! <br /> It only takes few seconds to analyze!
          </p>
        </motion.div>

        {/* Main Content */}
        <div className="space-y-8">
          {/* File Upload */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.1 }}
          >
            <FileUpload
              key={resetKey}
              onFileSelect={handleFileSelect}
              onFileRemove={handleFileRemove}
              disabled={isAnalyzing}
            />
          </motion.div>

          {/* Analyze Button */}
          {file && results.length === 0 && !isAnalyzing && (
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              className="flex justify-center"
            >
              <button
                onClick={handleAnalyze}
                className="btn-primary text-lg px-12 py-4"
              >
                Analyze for Deepfakes
              </button>
            </motion.div>
          )}

          {/* Progress Bar */}
          {isAnalyzing && (
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
            >
              <ProgressBar progress={progress} status="Analyzing..." />
            </motion.div>
          )}

          {/* Error Message */}
          {error && (
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              className="card bg-red-500/10 border-red-500/30"
            >
              <div className="flex items-start gap-3">
                <AlertCircle className="w-6 h-6 text-red-400 flex-shrink-0 mt-0.5" />
                <div>
                  <h3 className="text-lg font-semibold text-red-400 mb-1">
                    Analysis Failed
                  </h3>
                  <p className="text-gray-300">{error}</p>
                  <button
                    onClick={handleReset}
                    className="mt-4 btn-secondary"
                  >
                    Try Again
                  </button>
                </div>
              </div>
            </motion.div>
          )}

          {/* Results Dashboard */}
          {results.length > 0 && !isAnalyzing && (
            <>
              <div className="space-y-8">
                {results.map((r, idx) => (
                  <ResultsDashboard key={r?.id ?? idx} result={r} />
                ))}
              </div>
              
              {/* New Analysis Button */}
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                className="flex justify-center"
              >
                <button
                  onClick={handleReset}
                  className="btn-secondary"
                >
                  Analyze Another File
                </button>
              </motion.div>
            </>
          )}
        </div>
      </div>

      {/* History Dashboard Section */}
      <div className="container mx-auto px-4 max-w-7xl mt-20">
        <ErrorBoundary showRetry={true}>
          <HistoryDashboard key={refreshHistory} />
        </ErrorBoundary>
      </div>
    </div>
  );
};

export default HomePage;
