/**
 * ============================================================================
 * ResultsDashboard Component - Display Detection Results
 * ============================================================================
 * Comprehensive results display with:
 *   - Prediction verdict with confidence
 *   - Probability distribution
 *   - Processing metrics
 *   - Detailed metadata
 *   - Visual indicators
 * 
 * Author: Senior Full-Stack Engineer
 * Date: 2026-01-28
 * ============================================================================
 */

import React from 'react';
import { motion } from 'framer-motion';
import {
  AlertTriangle,
  CheckCircle,
  Info,
  Clock,
  BarChart3
} from 'lucide-react';

const ResultsDashboard = ({ result }) => {
  if (!result) return null;

  // Standardized field access with backward compatibility
  const classification = result.classification ?? result.prediction;
  const isDeepfake =
    classification?.toLowerCase() === 'fake' ||
    classification?.toLowerCase() === 'deepfake' ||
    result.is_fake === true;

  // Get deepfake probability (confidence is now deepfake probability from backend)
  const rawConfidence =
    result.confidence ??
    result.confidence_score ??
    result.detection_score ??
    result?.probabilities?.deepfake;

  // Normalize to [0,1] range
  const deepfakeProbability = (() => {
    const n = Number(rawConfidence);
    if (!Number.isFinite(n)) return 0;
    return n > 1 ? n / 100 : n;
  })();

  // Get processing time (standardized field name)
  const processingTime =
    result.processing_time_seconds ??
    result.processing_time ??
    result.processing_duration ??
    0;

  // Animation variants
  const containerVariants = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: {
        staggerChildren: 0.1
      }
    }
  };

  const itemVariants = {
    hidden: { opacity: 0, y: 20 },
    visible: { opacity: 1, y: 0 }
  };

  return (
    <motion.div
      variants={containerVariants}
      initial="hidden"
      animate="visible"
      className="space-y-6"
    >
      {/* Main Verdict Card */}
      <motion.div
        variants={itemVariants}
        className={`
          card text-center py-10 px-6 relative overflow-hidden
          ${isDeepfake ? 'border-red-500/30' : 'border-green-500/30'}
        `}
      >
        {/* Decorative background elements */}
        <div className={`absolute top-0 left-0 w-full h-1 ${isDeepfake ? 'bg-red-500' : 'bg-green-500'} opacity-50`} />
        <div className={`absolute -right-20 -top-20 w-64 h-64 rounded-full blur-[100px] opacity-10 ${isDeepfake ? 'bg-red-500' : 'bg-green-500'}`} />
        <div className={`absolute -left-20 -bottom-20 w-64 h-64 rounded-full blur-[100px] opacity-10 ${isDeepfake ? 'bg-red-500' : 'bg-green-500'}`} />

        {/* Icon */}
        <motion.div
          className="flex justify-center mb-6 relative z-10"
          initial={{ scale: 0, rotate: -180 }}
          animate={{ scale: 1, rotate: 0 }}
          transition={{ type: 'spring', stiffness: 200, damping: 15, delay: 0.2 }}
        >
          {isDeepfake ? (
            <div className="p-6 rounded-2xl bg-red-500/10 border border-red-500/20 shadow-[0_0_30px_rgba(239,68,68,0.2)]">
              <AlertTriangle className="w-20 h-20 text-red-500" />
            </div>
          ) : (
            <div className="p-6 rounded-2xl bg-green-500/10 border border-green-500/20 shadow-[0_0_30px_rgba(34,197,94,0.2)]">
              <CheckCircle className="w-20 h-20 text-green-500" />
            </div>
          )}
        </motion.div>

        {/* Verdict */}
        <div className="relative z-10">
          <h2 className="text-5xl font-black mb-3 tracking-tighter">
            <span className={isDeepfake ? 'text-red-500' : 'text-green-500'}>
              {isDeepfake ? 'DEEPFAKE DETECTED' : 'REAL CONTENT'}
            </span>
          </h2>
        </div>
      </motion.div>

      {/* Deepfake Score */}
      <motion.div variants={itemVariants} className="card border-white/5 bg-white/[0.02]">
        <div className="flex items-center gap-3 mb-6">
          <div className="p-2 rounded-lg bg-cyber-blue/10">
            <BarChart3 className="w-5 h-5 text-cyber-blue" />
          </div>
          <h3 className="text-xl font-bold tracking-tight">Analysis Score</h3>
        </div>

        <div className="space-y-6">
          <div className="relative pt-2">
            <div className="flex justify-between items-center mb-3">
              <span className="text-sm font-bold text-gray-400 uppercase tracking-wider">Deepfake Probability</span>
              <span className={`text-3xl font-black ${(deepfakeProbability * 100) > 50 ? 'text-red-500' : 'text-green-500'}`}>
                {(deepfakeProbability * 100).toFixed(1)}%
              </span>
            </div>
            
            <div className="h-6 bg-white/5 rounded-xl p-1 border border-white/10 relative overflow-hidden">
              {/* Background markers */}

              <motion.div
                className={`h-full rounded-lg relative z-10 ${
                  deepfakeProbability < 0.5 
                    ? 'bg-gradient-to-r from-green-600 to-green-400 shadow-[0_0_15px_rgba(34,197,94,0.4)]' 
                    : 'bg-gradient-to-r from-red-600 to-red-400 shadow-[0_0_15px_rgba(239,68,68,0.4)]'
                }`}
                initial={{ width: 0 }}
                animate={{ width: `${deepfakeProbability * 100}%` }}
                transition={{ duration: 1.2, ease: [0.22, 1, 0.36, 1] }}
              />
            </div>
            
            <div className="flex justify-between mt-3 px-1">
              <div className="flex flex-col items-start">
                <span className="text-[10px] font-bold text-gray-500 uppercase tracking-tighter">Minimum</span>
                <span className="text-xs font-bold text-green-500/80">0 (REAL)</span>
              </div>
              <div className="flex flex-col items-center">
                <div className="w-px h-2 bg-white/20 mb-1" />
                <span className="text-[10px] font-bold text-gray-600 uppercase">Threshold</span>
              </div>
              <div className="flex flex-col items-end">
                <span className="text-[10px] font-bold text-gray-500 uppercase tracking-tighter">Maximum</span>
                <span className="text-xs font-bold text-red-500/80">100 (DEEPFAKE)</span>
              </div>
            </div>
          </div>
        </div>
      </motion.div>

      {/* Processing Time */}
      <motion.div variants={itemVariants} className="card border-white/5 bg-white/[0.02] py-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="p-2 rounded-lg bg-cyber-blue/10">
              <Clock className="w-5 h-5 text-cyber-blue" />
            </div>
            <span className="text-sm font-medium text-gray-400">Analysis completed in</span>
          </div>
          <div className="flex items-baseline gap-1">
            <span className="text-2xl font-bold text-white">{processingTime?.toFixed(2) ?? 'N/A'}</span>
            <span className="text-xs font-bold text-gray-500 uppercase">seconds</span>
          </div>
        </div>
      </motion.div>

      {/* Disclaimer */}
      <motion.div variants={itemVariants} className="card bg-yellow-500/10 border-yellow-500/30">
        <div className="flex gap-3">
          <Info className="w-5 h-5 text-yellow-400 flex-shrink-0 mt-0.5" />
          <div className="text-sm text-gray-300">
            <p className="font-semibold text-yellow-400 mb-1">Disclaimer</p>
            <p>
              This analysis is provided for informational purposes only. 
              The model's predictions should be used as a tool to aid human judgment, 
              not as a definitive determination. Detection results may not be 100% accurate and may contain false positives or false negatives.
            </p>
          </div>
        </div>
      </motion.div>
    </motion.div>
  );
};

export default ResultsDashboard;
