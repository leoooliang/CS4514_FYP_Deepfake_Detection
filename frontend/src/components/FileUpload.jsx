/**
 * ============================================================================
 * FileUpload Component - Drag & Drop File Upload Zone
 * ============================================================================
 * Beautiful drag-and-drop file upload component with:
 *   - Visual feedback
 *   - File validation
 *   - Preview support
 *   - Animation effects
 * 
 * Author: Senior Full-Stack Engineer
 * Date: 2026-01-28
 * ============================================================================
 */

import React, { useCallback, useState, useEffect } from 'react';
import { useDropzone } from 'react-dropzone';
import { motion } from 'framer-motion';
import { Upload, File, X, Image, Video, Music } from 'lucide-react';
import { getFileType, formatFileSize, validateFile } from '../services/api';
import MediaPreviewModal from './MediaPreviewModal';

const FileUpload = ({ onFileSelect, onFileRemove, disabled = false }) => {
  const [selectedFile, setSelectedFile] = useState(null);
  const [error, setError] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [isPreviewOpen, setIsPreviewOpen] = useState(false);

  // Handle file drop
  const onDrop = useCallback((acceptedFiles) => {
    setError(null);
    
    if (acceptedFiles.length === 0) return;
    
    const file = acceptedFiles[0];
    
    // Validate file
    const validation = validateFile(file);
    if (!validation.valid) {
      setError(validation.error);
      return;
    }
    
    // Create preview URL
    const url = URL.createObjectURL(file);
    setPreviewUrl(url);
    
    // Set selected file
    setSelectedFile(file);
    onFileSelect(file);
  }, [onFileSelect]);

  // Configure dropzone
  const {
    getRootProps,
    getInputProps,
    isDragActive,
    isDragReject
  } = useDropzone({
    onDrop,
    accept: {
      'image/*': ['.jpg', '.jpeg', '.png', '.bmp', '.webp'],
      'video/*': ['.mp4', '.avi', '.mov', '.mkv', '.webm'],
      'audio/*': ['.mp3', '.wav', '.flac', '.ogg', '.m4a']
    },
    maxFiles: 1,
    disabled
  });

  // Remove selected file
  const handleRemove = () => {
    if (previewUrl) {
      URL.revokeObjectURL(previewUrl);
    }
    setSelectedFile(null);
    setPreviewUrl(null);
    setError(null);
    onFileRemove();
  };

  // Cleanup preview URL on unmount
  useEffect(() => {
    return () => {
      if (previewUrl) {
        URL.revokeObjectURL(previewUrl);
      }
    };
  }, [previewUrl]);

  // Get icon based on file type
  const getFileIcon = (file) => {
    const type = getFileType(file);
    switch (type) {
      case 'image':
        return <Image className="w-12 h-12 text-cyber-blue" />;
      case 'video':
        return <Video className="w-12 h-12 text-cyber-purple" />;
      case 'audio':
        return <Music className="w-12 h-12 text-cyber-pink" />;
      default:
        return <File className="w-12 h-12 text-gray-400" />;
    }
  };

  return (
    <div className="w-full">
      {!selectedFile ? (
        /* Upload Zone */
        <motion.div
          {...getRootProps()}
          className={`
            relative overflow-hidden rounded-2xl border-2 border-dashed 
            transition-all duration-300 cursor-pointer
            ${isDragActive && !isDragReject ? 'border-cyber-blue bg-cyber-blue/10 glow-cyan' : ''}
            ${isDragReject ? 'border-red-500 bg-red-500/10' : ''}
            ${!isDragActive && !isDragReject ? 'border-white/20 bg-white/5 hover:border-cyber-blue/50 hover:bg-white/10' : ''}
            ${disabled ? 'opacity-50 cursor-not-allowed' : ''}
          `}
          whileHover={{ scale: disabled ? 1 : 1.02 }}
          whileTap={{ scale: disabled ? 1 : 0.98 }}
        >
          <input {...getInputProps()} />
          
          <div className="py-16 px-8 text-center">
            {/* Upload Icon */}
            <motion.div
              className="flex justify-center mb-6"
              animate={isDragActive ? { scale: [1, 1.2, 1] } : {}}
              transition={{ repeat: isDragActive ? Infinity : 0, duration: 1 }}
            >
              <div className={`
                p-6 rounded-full 
                ${isDragActive ? 'bg-cyber-blue/20' : 'bg-white/10'}
                transition-all duration-300
              `}>
                <Upload className={`
                  w-16 h-16 
                  ${isDragActive ? 'text-cyber-blue' : 'text-gray-400'}
                  transition-colors duration-300
                `} />
              </div>
            </motion.div>

            {/* Text */}
            <h3 className="text-2xl font-bold mb-2 text-gradient-cyber">
              {isDragActive ? 'Drop your file here' : 'Upload Media for Analysis'}
            </h3>
            
            <p className="text-gray-400 mb-6">
              {isDragReject
                ? 'Unsupported file type'
                : 'Drag & drop or click to select a file'
              }
            </p>

            {/* Supported Formats */}
            <div className="flex flex-wrap justify-center gap-3 mb-6">
              <div className="badge bg-cyber-blue/20 text-cyber-blue border-cyber-blue/30">
                <Image className="w-4 h-4 mr-1" />
                Images
              </div>
              <div className="badge bg-cyber-purple/20 text-cyber-purple border-cyber-purple/30">
                <Video className="w-4 h-4 mr-1" />
                Videos
              </div>
              <div className="badge bg-cyber-pink/20 text-cyber-pink border-cyber-pink/30">
                <Music className="w-4 h-4 mr-1" />
                Audio
              </div>
            </div>

            {/* File Size Limit */}
            <p className="text-sm text-gray-500">
              Maximum file size: 100 MB
            </p>
          </div>

          {/* Scanning Animation Line (when active) */}
          {isDragActive && (
            <div className="absolute inset-0 overflow-hidden pointer-events-none">
              <div className="scanning-line absolute inset-x-0 h-1 bg-gradient-to-r from-transparent via-cyber-blue to-transparent opacity-50" />
            </div>
          )}
        </motion.div>
      ) : (
        /* Selected File Preview */
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="card glow-cyan"
        >
          {/* Media Preview */}
          {previewUrl && (
            <div
              className="mb-6 rounded-xl overflow-hidden bg-black/30 cursor-pointer"
              onClick={() => setIsPreviewOpen(true)}
              title="Click to enlarge preview"
            >
              {getFileType(selectedFile) === 'image' && (
                <img
                  src={previewUrl}
                  alt={selectedFile.name}
                  className="w-full h-auto max-h-96 object-contain"
                />
              )}
              {getFileType(selectedFile) === 'video' && (
                <video
                  src={previewUrl}
                  controls
                  className="w-full h-auto max-h-96"
                >
                  Your browser does not support video playback.
                </video>
              )}
              {getFileType(selectedFile) === 'audio' && (
                <div className="p-8 flex flex-col items-center justify-center">
                  <Music className="w-20 h-20 text-cyber-pink mb-4" />
                  <audio
                    src={previewUrl}
                    controls
                    className="w-full max-w-md"
                  >
                    Your browser does not support audio playback.
                  </audio>
                </div>
              )}
            </div>
          )}

          <div className="flex items-center gap-6">
            {/* File Icon */}
            <div className="flex-shrink-0">
              {getFileIcon(selectedFile)}
            </div>

            {/* File Info */}
            <div className="flex-1 min-w-0">
              <h4 className="text-lg font-semibold text-white truncate mb-1">
                {selectedFile.name}
              </h4>
              <div className="flex items-center gap-4 text-sm text-gray-400">
                <span className="capitalize">{getFileType(selectedFile)}</span>
                <span>•</span>
                <span>{formatFileSize(selectedFile.size)}</span>
              </div>
            </div>

            {/* Remove Button */}
            {!disabled && (
              <button
                onClick={handleRemove}
                className="flex-shrink-0 p-2 rounded-lg glass-hover 
                         text-gray-400 hover:text-red-400 transition-colors"
                title="Remove file"
              >
                <X className="w-6 h-6" />
              </button>
            )}
          </div>
        </motion.div>
      )}

      {/* Error Message */}
      {error && (
        <motion.div
          initial={{ opacity: 0, y: -10 }}
          animate={{ opacity: 1, y: 0 }}
          className="mt-4 p-4 rounded-lg bg-red-500/20 border border-red-500/30"
        >
          <p className="text-red-400 text-sm">{error}</p>
        </motion.div>
      )}

      <MediaPreviewModal
        isOpen={isPreviewOpen}
        onClose={() => setIsPreviewOpen(false)}
        mediaUrl={previewUrl}
        mediaType={selectedFile ? getFileType(selectedFile) : null}
        title={selectedFile?.name}
      />
    </div>
  );
};

export default FileUpload;
