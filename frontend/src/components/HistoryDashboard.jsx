/**
 * ============================================================================
 * HistoryDashboard Component - Display Scan History
 * ============================================================================
 * Displays user's scan history with media previews and results.
 * 
 * Features:
 *   - Fetches history from backend
 *   - Displays media previews (image/video/audio)
 *   - Shows prediction results
 *   - Responsive grid layout
 *   - Loading and error states
 * 
 * Author: Senior Full-Stack Engineer
 * Date: 2026-03-25
 * ============================================================================
 */

import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import {
  History,
  AlertTriangle,
  CheckCircle,
  Clock,
  Image as ImageIcon,
  Video as VideoIcon,
  Music,
  RefreshCw,
  AlertCircle
} from 'lucide-react';
import { fetchScanHistory, API_BASE_URL } from '../services/api';
import MediaPreviewModal from './MediaPreviewModal';

const HistoryDashboard = () => {
  const [history, setHistory] = useState([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);
  const [selectedRecord, setSelectedRecord] = useState(null);
  const [currentPage, setCurrentPage] = useState(1);
  const HISTORY_ITEMS_PER_PAGE = 10;

  const loadHistory = async () => {
    setIsLoading(true);
    setError(null);
    
    try {
      const data = await fetchScanHistory(100);
      setHistory(data || []);
      setCurrentPage(1);
    } catch (err) {
      console.error('Failed to load history:', err);
      // Set empty array on error to prevent crashes
      setHistory([]);
      setError(err.message || 'Failed to load scan history. Make sure the server is running.');
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    loadHistory();
  }, []);

  const getMediaUrl = (mediaPath) => {
    if (!mediaPath) return null;
    return `${API_BASE_URL}${mediaPath}`;
  };

  const getMediaTypeFromPath = (mediaPath) => {
    const path = mediaPath?.toLowerCase?.() || '';
    const ext = path.split('.').pop();

    const imageExts = ['jpg', 'jpeg', 'png', 'bmp', 'webp'];
    const videoExts = ['mp4', 'avi', 'mov', 'mkv', 'webm'];
    const audioExts = ['mp3', 'wav', 'flac', 'ogg', 'm4a'];

    if (imageExts.includes(ext)) return 'image';
    if (videoExts.includes(ext)) return 'video';
    if (audioExts.includes(ext)) return 'audio';

    return null;
  };

  const getFileTypeIcon = (fileType) => {
    switch (fileType?.toLowerCase()) {
      case 'image':
        return <ImageIcon className="w-5 h-5 text-cyber-blue" />;
      case 'video':
        return <VideoIcon className="w-5 h-5 text-cyber-purple" />;
      case 'audio':
        return <Music className="w-5 h-5 text-cyber-pink" />;
      default:
        return <ImageIcon className="w-5 h-5 text-gray-400" />;
    }
  };

  const formatDate = (timestamp) => {
    if (!timestamp) return 'N/A';

    const timezonePattern = /(Z|[+-]\d{2}:\d{2})$/;
    const normalizedTimestamp =
      typeof timestamp === 'string' && !timezonePattern.test(timestamp)
        ? `${timestamp}Z`
        : timestamp;
    const date = new Date(normalizedTimestamp);

    if (Number.isNaN(date.getTime())) return 'N/A';

    return date.toLocaleString('en-US', {
      month: 'short',
      day: 'numeric',
      year: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
      timeZone: Intl.DateTimeFormat().resolvedOptions().timeZone
    });
  };

  const formatFileSize = (bytes) => {
    if (typeof bytes !== 'number' || Number.isNaN(bytes) || bytes < 0) return 'N/A';
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(2)} KB`;
    return `${(bytes / 1024 / 1024).toFixed(2)} MB`;
  };

  const resetPreview = () => {
    setSelectedRecord(null);
  };

  const renderMediaPreview = (record) => {
    const mediaUrl = getMediaUrl(record.media_path);
    
    const fileType =
      record.file_type?.toLowerCase() ||
      getMediaTypeFromPath(record.media_path);

    if (!mediaUrl) {
      return (
        <div className="w-full h-32 bg-black/20 rounded-lg border border-white/10 flex items-center justify-center">
          {getFileTypeIcon(fileType)}
        </div>
      );
    }

    switch (fileType) {
      case 'image':
        return (
          <img
            src={mediaUrl}
            alt="Scanned media"
            className="object-cover w-full h-32 rounded-lg"
            onError={(e) => {
              e.target.style.display = 'none';
              e.target.parentElement.innerHTML = '<div class="w-full h-32 bg-black/20 rounded-lg border border-white/10 flex items-center justify-center"><div class="text-gray-500 text-sm">Preview unavailable</div></div>';
            }}
          />
        );
      case 'video':
        return (
          <video
            src={mediaUrl}
            className="object-cover w-full h-32 rounded-lg"
            muted
            onError={(e) => {
              e.target.style.display = 'none';
              e.target.parentElement.innerHTML = '<div class="w-full h-32 bg-black/20 rounded-lg border border-white/10 flex items-center justify-center"><div class="text-gray-500 text-sm">Preview unavailable</div></div>';
            }}
          />
        );
      case 'audio':
        return (
          <div className="w-full h-32 bg-black/20 rounded-lg border border-white/10 flex flex-col items-center justify-center px-4 py-3">
            <Music className="w-8 h-8 text-cyber-pink mb-2" />
            <audio
              src={mediaUrl}
              controls
              className="w-full max-w-[180px]"
              onError={(e) => (e.target.style.display = 'none')}
            />
          </div>
        );
      default:
        return (
          <div className="w-full h-32 bg-black/20 rounded-lg border border-white/10 flex items-center justify-center">
            <span className="text-gray-500 text-sm">No preview available</span>
          </div>
        );
    }
  };

  if (isLoading) {
    return (
      <div className="w-full py-20">
        <div className="text-center">
          <motion.div
            animate={{ rotate: 360 }}
            transition={{ duration: 1, repeat: Infinity, ease: 'linear' }}
            className="inline-block mb-4"
          >
            <RefreshCw className="w-12 h-12 text-cyber-blue" />
          </motion.div>
          <p className="text-gray-400 text-lg">Loading your scan history...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="card bg-yellow-500/10 border-yellow-500/30"
      >
        <div className="flex items-start gap-3">
          <AlertCircle className="w-6 h-6 text-yellow-400 flex-shrink-0 mt-0.5" />
          <div>
            <h3 className="text-lg font-semibold text-yellow-400 mb-1">
              Unable to Load Scan History
            </h3>
            <p className="text-gray-300 mb-2">{error}</p>
            <p className="text-sm text-gray-400 mb-4">
              Your scan history will be available once the server is running.
            </p>
            <button
              onClick={loadHistory}
              className="btn-secondary flex items-center gap-2"
            >
              <RefreshCw className="w-4 h-4" />
              Retry Connection
            </button>
          </div>
        </div>
      </motion.div>
    );
  }

  if (history.length === 0) {
    return (
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="card text-center py-16"
      >
        <History className="w-16 h-16 text-gray-600 mx-auto mb-4" />
        <h3 className="text-xl font-semibold text-gray-400 mb-2">
          No history found
        </h3>
        <p className="text-gray-500">
          Upload your first file to start building your detection history.
        </p>
      </motion.div>
    );
  }

  const totalPages = Math.max(1, Math.ceil(history.length / HISTORY_ITEMS_PER_PAGE));
  const pageStart = (currentPage - 1) * HISTORY_ITEMS_PER_PAGE;
  const pageEnd = pageStart + HISTORY_ITEMS_PER_PAGE;
  const paginatedHistory = history.slice(pageStart, pageEnd);
  const selectedMediaType =
    selectedRecord?.file_type?.toLowerCase() || getMediaTypeFromPath(selectedRecord?.media_path);
  const selectedMediaUrl = selectedRecord ? getMediaUrl(selectedRecord.media_path) : null;

  return (
    <div className="w-full">
      <motion.div
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        className="flex items-center justify-between mb-8"
      >
        <div className="flex items-center gap-3">
          <div className="p-3 rounded-xl bg-cyber-blue/10 border border-cyber-blue/20">
            <History className="w-6 h-6 text-cyber-blue" />
          </div>
          <div>
            <h2 className="text-3xl font-black tracking-tight text-white">
              Detection History
            </h2>
            <p className="text-gray-400 text-sm">
              {history.length} {history.length === 1 ? 'scan' : 'scans'} in this session
            </p>
          </div>
        </div>
        <button
          onClick={loadHistory}
          className="btn-secondary flex items-center gap-2"
        >
          <RefreshCw className="w-4 h-4" />
          Refresh
        </button>
      </motion.div>

      <div className="space-y-4">
        {paginatedHistory.map((record, index) => {
          const isDeepfake =
            record.classification?.toLowerCase() === 'fake' ||
            record.classification?.toLowerCase() === 'deepfake';

          const detectionScore = record.detection_score;
          const detectionPercent = (() => {
            if (typeof detectionScore !== 'number' || Number.isNaN(detectionScore)) {
              return null;
            }
            return detectionScore <= 1 ? detectionScore * 100 : detectionScore;
          })();
          
          return (
            <motion.div
              key={record.id || index}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: index * 0.03 }}
              className={`
                card-hover overflow-hidden
                ${isDeepfake ? 'border-red-500/20 hover:border-red-500/40' : 'border-green-500/20 hover:border-green-500/40'}
                transition-all duration-300
              `}
              onClick={() => {
                setSelectedRecord(record);
              }}
            >
              <div className="flex flex-col md:flex-row gap-4">
                {/* Media Preview Section */}
                <div className="md:w-48 flex-shrink-0">
                  {renderMediaPreview(record)}
                </div>

                {/* Details Section */}
                <div className="flex-1 flex flex-col justify-between min-w-0">
                  {/* File Info Row */}
                  <div className="mb-3">
                    <div className="flex items-center gap-2 mb-2">
                      {getFileTypeIcon(record.file_type)}
                      <span className="text-base font-semibold text-white truncate">
                        {record.file_name}
                      </span>
                    </div>
                    <div className="flex items-center gap-3 text-xs text-gray-500">
                      <span className="capitalize">{record.file_type}</span>
                      <span>•</span>
                      <span>
                        {formatFileSize(record.file_size)}
                      </span>
                      <span>•</span>
                      <span className="flex items-center gap-1">
                        <Clock className="w-3 h-3" />
                        {formatDate(record.timestamp)}
                      </span>
                    </div>
                  </div>

                  {/* Detection Result Row */}
                  <div className="flex items-center justify-between gap-4">
                    <div className={`
                      flex items-center gap-3 px-4 py-2.5 rounded-lg flex-1
                      ${isDeepfake ? 'bg-red-500/10 border border-red-500/20' : 'bg-green-500/10 border border-green-500/20'}
                    `}>
                      {isDeepfake ? (
                        <AlertTriangle className="w-5 h-5 text-red-500 flex-shrink-0" />
                      ) : (
                        <CheckCircle className="w-5 h-5 text-green-500 flex-shrink-0" />
                      )}
                      <div className="flex-1">
                        <div className={`text-sm font-bold ${isDeepfake ? 'text-red-400' : 'text-green-400'}`}>
                          {isDeepfake ? 'Deepfake Detected' : 'Real Media'}
                        </div>
                        <div className="text-xs text-gray-400">
                          Confidence: {detectionPercent === null ? 'N/A' : `${detectionPercent.toFixed(1)}%`}
                        </div>
                      </div>
                    </div>

                    {/* Processing Time Badge */}
                    {(record.processing_duration || record.processing_time) && (
                      <div className="text-xs text-gray-500 bg-white/5 px-3 py-2 rounded-lg border border-white/10">
                        <div className="font-semibold text-gray-400">Processing Time</div>
                        <div className="text-white font-mono">
                          {(record.processing_duration || record.processing_time).toFixed(2)}s
                        </div>
                      </div>
                    )}
                  </div>
                </div>
              </div>
            </motion.div>
          );
        })}
      </div>

      {totalPages > 1 && (
        <div className="mt-8 flex items-center justify-between gap-3">
          <p className="text-sm text-gray-400">
            Showing {pageStart + 1}-{Math.min(pageEnd, history.length)} of {history.length}
          </p>
          <div className="flex items-center gap-2">
            <button
              type="button"
              onClick={() => setCurrentPage((prev) => Math.max(1, prev - 1))}
              disabled={currentPage === 1}
              className="btn-secondary disabled:opacity-50 disabled:cursor-not-allowed"
            >
              Previous
            </button>
            <span className="text-sm text-gray-300 px-2">
              Page {currentPage} / {totalPages}
            </span>
            <button
              type="button"
              onClick={() => setCurrentPage((prev) => Math.min(totalPages, prev + 1))}
              disabled={currentPage === totalPages}
              className="btn-secondary disabled:opacity-50 disabled:cursor-not-allowed"
            >
              Next
            </button>
          </div>
        </div>
      )}

      <MediaPreviewModal
        isOpen={Boolean(selectedRecord)}
        onClose={resetPreview}
        mediaUrl={selectedMediaUrl}
        mediaType={selectedMediaType}
        title={selectedRecord?.file_name}
      />
    </div>
  );
};

export default HistoryDashboard;
