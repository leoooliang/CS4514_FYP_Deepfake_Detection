/**
 * ============================================================================
 * ProgressBar Component - Animated Progress Indicator
 * ============================================================================
 * Beautiful progress bar with:
 *   - Smooth animations
 *   - Status messages
 *   - Percentage display
 *   - Glowing effects
 * 
 * Author: Senior Full-Stack Engineer
 * Date: 2026-01-28
 * ============================================================================
 */

import React from 'react';
import { motion } from 'framer-motion';
import { Loader2 } from 'lucide-react';

const ProgressBar = ({ 
  progress = 0, 
  status = 'Processing...', 
  showPercentage = true,
  className = '' 
}) => {
  return (
    <div className={`w-full ${className}`}>
      {/* Status and Percentage */}
      <div className="flex justify-between items-center mb-3">
        <div className="flex items-center gap-2">
          <Loader2 className="w-5 h-5 text-cyber-blue animate-spin" />
          <span className="text-sm font-medium text-gray-300">{status}</span>
        </div>
        {showPercentage && (
          <span className="text-sm font-bold text-cyber-blue">
            {Math.round(progress)}%
          </span>
        )}
      </div>

      {/* Progress Bar Container */}
      <div className="relative w-full h-3 bg-white/10 rounded-full overflow-hidden">
        {/* Progress Fill */}
        <motion.div
          className="absolute top-0 left-0 h-full bg-gradient-to-r from-cyber-blue via-cyber-purple to-cyber-pink rounded-full"
          style={{
            boxShadow: '0 0 20px rgba(0, 217, 255, 0.5)'
          }}
          initial={{ width: '0%' }}
          animate={{ width: `${progress}%` }}
          transition={{
            duration: 0.5,
            ease: 'easeOut'
          }}
        />

        {/* Shimmer Effect */}
        <motion.div
          className="absolute top-0 left-0 h-full w-full"
          style={{
            background: 'linear-gradient(90deg, transparent, rgba(255,255,255,0.3), transparent)',
          }}
          animate={{
            x: ['-100%', '200%']
          }}
          transition={{
            repeat: Infinity,
            duration: 1.5,
            ease: 'linear'
          }}
        />
      </div>

      {/* Stages (optional - for multi-stage processing) */}
      {progress > 0 && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          className="mt-3 text-xs text-gray-500 text-center"
        >
          {progress < 30 && 'Uploading file...'}
          {progress >= 30 && progress < 60 && 'Preprocessing data...'}
          {progress >= 60 && progress < 90 && 'Running AI analysis...'}
          {progress >= 90 && 'Finalizing results...'}
        </motion.div>
      )}
    </div>
  );
};

export default ProgressBar;
