import React, { useCallback, useState, useEffect } from 'react';
import { useDropzone } from 'react-dropzone';
import { Upload, File, X, Image, Video, Music } from 'lucide-react';
import { getFileType, formatFileSize, validateFile } from '../services/api';
import MediaPreviewModal from './MediaPreviewModal';

const FileUpload = ({ onFileSelect, onFileRemove, disabled = false }) => {
  const [selectedFile, setSelectedFile] = useState(null);
  const [error, setError] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [isPreviewOpen, setIsPreviewOpen] = useState(false);

  const onDrop = useCallback((acceptedFiles) => {
    setError(null);
    
    if (acceptedFiles.length === 0) return;
    
    const file = acceptedFiles[0];
    
    const validation = validateFile(file);
    if (!validation.valid) {
      setError(validation.error);
      return;
    }
    
    const url = URL.createObjectURL(file);
    setPreviewUrl(url);
    
    setSelectedFile(file);
    onFileSelect(file);
  }, [onFileSelect]);

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

  const handleRemove = () => {
    if (previewUrl) {
      URL.revokeObjectURL(previewUrl);
    }
    setSelectedFile(null);
    setPreviewUrl(null);
    setError(null);
    onFileRemove();
  };

  useEffect(() => {
    return () => {
      if (previewUrl) {
        URL.revokeObjectURL(previewUrl);
      }
    };
  }, [previewUrl]);

  const getFileIcon = (file) => {
    const type = getFileType(file);
    switch (type) {
      case 'image':
        return <Image className="w-10 h-10 text-primary" />;
      case 'video':
        return <Video className="w-10 h-10 text-accent-violet" />;
      case 'audio':
        return <Music className="w-10 h-10 text-accent-rose" />;
      default:
        return <File className="w-10 h-10 text-gray-400" />;
    }
  };

  return (
    <div className="w-full">
      {!selectedFile ? (
        <div
          {...getRootProps()}
          className={`
            relative overflow-hidden rounded-xl border-2 border-dashed 
            transition-all duration-200 cursor-pointer
            ${isDragActive && !isDragReject ? 'border-primary/50 bg-primary/5' : ''}
            ${isDragReject ? 'border-red-400/40 bg-red-500/5' : ''}
            ${!isDragActive && !isDragReject ? 'border-white/15 bg-white/[0.02] hover:border-white/25 hover:bg-white/[0.04]' : ''}
            ${disabled ? 'opacity-50 cursor-not-allowed' : ''}
          `}
        >
          <input {...getInputProps()} />
          
          <div className="py-14 px-8 text-center">
            <div className="flex justify-center mb-5">
              <div className={`
                p-5 rounded-2xl
                ${isDragActive ? 'bg-primary/10' : 'bg-white/5'}
                transition-all duration-200
              `}>
                <Upload className={`
                  w-12 h-12 
                  ${isDragActive ? 'text-primary' : 'text-gray-500'}
                  transition-colors duration-200
                `} />
              </div>
            </div>

            <h3 className="text-xl font-semibold mb-2 text-white">
              {isDragActive ? 'Drop your file here' : 'Upload Media for Analysis'}
            </h3>
            
            <p className="text-gray-500 mb-5 text-sm">
              {isDragReject
                ? 'Unsupported file type'
                : 'Drag & drop or click to select a file'
              }
            </p>

            <div className="flex flex-wrap justify-center gap-2 mb-5">
              <div className="badge bg-primary/10 text-primary border-primary/15">
                <Image className="w-3.5 h-3.5 mr-1.5" />
                Images
              </div>
              <div className="badge bg-accent-violet/10 text-accent-violet border-accent-violet/15">
                <Video className="w-3.5 h-3.5 mr-1.5" />
                Videos
              </div>
              <div className="badge bg-accent-rose/10 text-accent-rose border-accent-rose/15">
                <Music className="w-3.5 h-3.5 mr-1.5" />
                Audio
              </div>
            </div>

            <p className="text-xs text-gray-600">
              Maximum file size: 100 MB
            </p>
          </div>

        </div>
      ) : (
        <div className="card">
          {previewUrl && (
            <div
              className="mb-5 rounded-lg overflow-hidden bg-black/30 cursor-pointer"
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
                  <Music className="w-16 h-16 text-accent-rose mb-4" />
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

          <div className="flex items-center gap-5">
            <div className="flex-shrink-0">
              {getFileIcon(selectedFile)}
            </div>

            <div className="flex-1 min-w-0">
              <h4 className="text-base font-semibold text-white truncate mb-0.5">
                {selectedFile.name}
              </h4>
              <div className="flex items-center gap-3 text-sm text-gray-500">
                <span className="capitalize">{getFileType(selectedFile)}</span>
                <span>&middot;</span>
                <span>{formatFileSize(selectedFile.size)}</span>
              </div>
            </div>

            {!disabled && (
              <button
                onClick={handleRemove}
                className="flex-shrink-0 p-2 rounded-lg glass-hover 
                         text-gray-400 hover:text-red-400 transition-colors"
                title="Remove file"
              >
                <X className="w-5 h-5" />
              </button>
            )}
          </div>
        </div>
      )}

      {error && (
        <div className="mt-3 p-3 rounded-lg bg-red-500/10 border border-red-500/15">
          <p className="text-red-400 text-sm">{error}</p>
        </div>
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
