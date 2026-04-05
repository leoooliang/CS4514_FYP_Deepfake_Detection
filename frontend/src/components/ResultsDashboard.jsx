import React from 'react';
import {
  AlertTriangle,
  CheckCircle,
  Info,
  Clock,
  BarChart3
} from 'lucide-react';

const ResultsDashboard = ({ result }) => {
  if (!result) return null;

  const classification = result.classification ?? result.prediction;
  const isDeepfake =
    classification?.toLowerCase() === 'fake' ||
    classification?.toLowerCase() === 'deepfake' ||
    result.is_deepfake === true;

  const rawConfidence =
    result.confidence ??
    result.confidence_score ??
    result.detection_score ??
    result?.probabilities?.deepfake;

  const deepfakeProbability = (() => {
    const n = Number(rawConfidence);
    if (!Number.isFinite(n)) return 0;
    return n > 1 ? n / 100 : n;
  })();

  const processingTime =
    result.processing_time_seconds ??
    result.processing_time ??
    result.processing_duration ??
    0;

  return (
    <div className="space-y-5">
      {/* Verdict */}
      <div className="card text-center py-10 px-6 relative overflow-hidden">
        <div className="flex justify-center mb-5 relative z-10">
          {isDeepfake ? (
            <div className="p-5 rounded-xl bg-red-500/10 border border-red-500/15">
              <AlertTriangle className="w-16 h-16 text-red-500" />
            </div>
          ) : (
            <div className="p-5 rounded-xl bg-green-500/10 border border-green-500/15">
              <CheckCircle className="w-16 h-16 text-green-500" />
            </div>
          )}
        </div>

        <div className="relative z-10">
          <h2 className="text-4xl font-extrabold mb-2 tracking-tight">
            <span className={isDeepfake ? 'text-red-500' : 'text-green-500'}>
              {isDeepfake ? 'DEEPFAKE DETECTED' : 'REAL CONTENT'}
            </span>
          </h2>
        </div>
      </div>

      {/* Score */}
      <div className="card">
        <div className="flex items-center gap-3 mb-5">
          <div className="p-2 rounded-lg bg-primary/10 border border-primary/15">
            <BarChart3 className="w-4 h-4 text-primary" />
          </div>
          <h3 className="text-lg font-bold tracking-tight">Analysis Score</h3>
        </div>

        <div className="space-y-5">
          <div className="relative pt-1">
            <div className="flex justify-between items-center mb-3">
              <span className="text-sm font-medium text-gray-400">Deepfake Probability</span>
              <span className={`text-2xl font-bold ${(deepfakeProbability * 100) > 50 ? 'text-red-500' : 'text-green-500'}`}>
                {(deepfakeProbability * 100).toFixed(1)}%
              </span>
            </div>
            
            <div className="h-5 bg-white/5 rounded-lg p-0.5 border border-white/10 relative overflow-hidden">
              <div
                className={`h-full rounded-md relative z-10 ${
                  deepfakeProbability < 0.5 ? 'bg-green-500' : 'bg-red-500'
                }`}
                style={{ width: `${deepfakeProbability * 100}%` }}
              />
            </div>
            
            <div className="flex justify-between mt-2.5 px-0.5">
              <div className="flex flex-col items-start">
                <span className="text-[10px] text-gray-500">Min</span>
                <span className="text-xs font-medium text-green-500/70">0 (REAL)</span>
              </div>
              <div className="flex flex-col items-center">
                <div className="w-px h-2 bg-white/15 mb-0.5" />
                <span className="text-[10px] text-gray-600">Threshold</span>
              </div>
              <div className="flex flex-col items-end">
                <span className="text-[10px] text-gray-500">Max</span>
                <span className="text-xs font-medium text-red-500/70">100 (DEEPFAKE)</span>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Processing Time */}
      <div className="card py-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="p-2 rounded-lg bg-primary/10 border border-primary/15">
              <Clock className="w-4 h-4 text-primary" />
            </div>
            <span className="text-sm text-gray-400">Analysis completed in</span>
          </div>
          <div className="flex items-baseline gap-1">
            <span className="text-xl font-bold text-white">{processingTime?.toFixed(2) ?? 'N/A'}</span>
            <span className="text-xs text-gray-500">seconds</span>
          </div>
        </div>
      </div>

      {/* Disclaimer */}
      <div className="card bg-yellow-500/5 border-yellow-500/10">
        <div className="flex gap-3">
          <Info className="w-4 h-4 text-yellow-400 flex-shrink-0 mt-0.5" />
          <div className="text-sm text-gray-300">
            <p className="font-medium text-yellow-400 mb-1">Disclaimer</p>
            <p className="leading-relaxed">
              This analysis is provided for informational purposes only. 
              The model's predictions should be used as a tool to aid human judgment, 
              not as a definitive determination. Detection results may not be 100% accurate and may contain false positives or false negatives.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ResultsDashboard;
