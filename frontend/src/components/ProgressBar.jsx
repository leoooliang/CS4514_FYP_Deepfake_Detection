import React from 'react';
import { Loader2 } from 'lucide-react';

const ProgressBar = ({ 
  progress = 0, 
  status = 'Processing...', 
  showPercentage = true,
  className = '' 
}) => {
  return (
    <div className={`w-full ${className}`}>
      <div className="flex justify-between items-center mb-3">
        <div className="flex items-center gap-2">
          <Loader2 className="w-4 h-4 text-primary animate-spin" />
          <span className="text-sm font-medium text-gray-300">{status}</span>
        </div>
        {showPercentage && (
          <span className="text-sm font-semibold text-primary">
            {Math.round(progress)}%
          </span>
        )}
      </div>

      <div className="relative w-full h-2.5 bg-white/10 rounded-full overflow-hidden">
        <div
          className="absolute top-0 left-0 h-full bg-primary rounded-full"
          style={{ width: `${progress}%` }}
        />
      </div>

      {progress > 0 && (
        <div className="mt-2.5 text-xs text-gray-500 text-center">
          {progress < 30 && 'Uploading file...'}
          {progress >= 30 && progress < 60 && 'Preprocessing data...'}
          {progress >= 60 && progress < 90 && 'Running analysis...'}
          {progress >= 90 && 'Finalizing results...'}
        </div>
      )}
    </div>
  );
};

export default ProgressBar;
