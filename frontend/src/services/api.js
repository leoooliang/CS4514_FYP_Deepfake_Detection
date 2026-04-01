/**
 * ============================================================================
 * API Service - HTTP Client for Backend Communication
 * ============================================================================
 * Centralized API service using Axios for all backend requests.
 * 
 * Features:
 *   - Axios instance with base configuration
 *   - Request/response interceptors
 *   - Error handling
 *   - Type-specific detection functions
 * 
 * Author: Senior Full-Stack Engineer
 * Date: 2026-01-28
 * ============================================================================
 */

import axios from 'axios';
import { getSessionId } from '../hooks/useSession';

// =============================================================================
// Axios Instance Configuration
// =============================================================================

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

export { API_BASE_URL };

const apiClient = axios.create({
  baseURL: `${API_BASE_URL}/api/v1`,
  timeout: 120000, // 2 minutes (video processing can take time)
  headers: {
    'Content-Type': 'application/json',
  },
});

// =============================================================================
// Request Interceptor
// =============================================================================

apiClient.interceptors.request.use(
  (config) => {
    // Add authentication token if available
    const token = localStorage.getItem('auth_token');
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    
    console.log(`[API Request] ${config.method.toUpperCase()} ${config.url}`);
    return config;
  },
  (error) => {
    console.error('[API Request Error]', error);
    return Promise.reject(error);
  }
);

// =============================================================================
// Response Interceptor
// =============================================================================

apiClient.interceptors.response.use(
  (response) => {
    console.log(`[API Response] ${response.status} ${response.config.url}`);
    return response;
  },
  (error) => {
    // Enhanced error handling
    if (error.response) {
      // Server responded with error status
      console.error('[API Error Response]', {
        status: error.response.status,
        data: error.response.data,
        url: error.config?.url,
      });
      
      // Extract error message from different response formats
      const errorData = error.response.data;
      let errorMessage = 'An error occurred.';
      
      if (errorData) {
        if (typeof errorData === 'string') {
          errorMessage = errorData;
        } else if (errorData.detail) {
          // FastAPI validation error format
          if (typeof errorData.detail === 'string') {
            errorMessage = errorData.detail;
          } else if (Array.isArray(errorData.detail)) {
            // Multiple validation errors
            errorMessage = errorData.detail.map(err => err.msg).join(', ');
          }
        } else if (errorData.message) {
          errorMessage = errorData.message;
        } else if (errorData.error) {
          errorMessage = errorData.error;
        }
      }
      
      // Handle specific status codes
      switch (error.response.status) {
        case 400:
          // Bad request
          error.message = errorMessage || 'Invalid request. Please check your input.';
          break;
        case 401:
          // Unauthorized - clear token and redirect
          localStorage.removeItem('auth_token');
          error.message = 'Unauthorized. Please log in again.';
          break;
        case 413:
          // File too large
          error.message = 'File too large. Please upload a smaller file.';
          break;
        case 422:
          // Validation error
          error.message = errorMessage || 'Invalid data submitted.';
          break;
        case 500:
          // Server error
          error.message = 'Server error. Please try again later.';
          break;
        case 503:
          // Service unavailable
          error.message = 'Service temporarily unavailable. Please try again later.';
          break;
        default:
          error.message = errorMessage;
      }
    } else if (error.request) {
      // Request made but no response received
      console.error('[API No Response]', error.request);
      error.message = 'No response from server. Please check your connection and ensure the backend is running.';
    } else {
      // Error in request setup
      console.error('[API Request Setup Error]', error.message);
      error.message = error.message || 'Request failed. Please try again.';
    }
    
    return Promise.reject(error);
  }
);

// =============================================================================
// Type Definitions (JSDoc)
// =============================================================================

/**
 * @typedef {Object} PredictionResponse
 * @property {string} prediction - Prediction result: 'real' or 'deepfake'
 * @property {boolean} is_fake - Boolean flag: True if deepfake, False if real
 * @property {number} confidence - Confidence score (0-1)
 * @property {Object<string, number>} probabilities - Probability distribution: {real: 0.15, deepfake: 0.85}
 * @property {number} processing_time_seconds - Total processing time in seconds
 * @property {number} inference_time_ms - Model inference time in milliseconds
 * @property {Object} [metadata] - Additional metadata about the detection
 * @property {string} [record_id] - Database record ID (UUID)
 */

/**
 * @typedef {Object} DetectionRecord
 * @property {string} id - Unique identifier (UUID)
 * @property {string} file_name - Original filename
 * @property {string} file_type - Media type: 'image', 'audio', or 'video'
 * @property {number} file_size - File size in bytes
 * @property {number} detection_score - Confidence score (0-1)
 * @property {string} classification - Classification: 'Real' or 'Fake'
 * @property {string} model_version - Model version identifier
 * @property {string} timestamp - ISO 8601 timestamp
 * @property {number} processing_duration - Processing time in seconds
 * @property {string} [session_id] - Session ID for tracking
 * @property {string} [media_path] - URL path to saved media file
 */

/**
 * @typedef {Object} HistoryResponse
 * @property {number} total - Total number of records
 * @property {DetectionRecord[]} records - List of detection records
 */

// =============================================================================
// API Functions
// =============================================================================

/**
 * Detect deepfake in an image
 * @param {File} imageFile - Image file to analyze
 * @param {Function} onProgress - Progress callback (optional)
 * @returns {Promise<PredictionResponse>} API response with prediction results
 */
export const detectImageDeepfake = async (imageFile, onProgress = null) => {
  const formData = new FormData();
  formData.append('file', imageFile);
  formData.append('session_id', getSessionId());
  
  const config = {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
  };
  
  if (onProgress) {
    config.onUploadProgress = (progressEvent) => {
      const percentCompleted = Math.round(
        (progressEvent.loaded * 100) / progressEvent.total
      );
      onProgress(percentCompleted);
    };
  }
  
  const response = await apiClient.post('/predict/image', formData, config);
  return response.data;
};

/**
 * Detect deepfake in a video
 * @param {File} videoFile - Video file to analyze
 * @param {Function} onProgress - Progress callback (optional)
 * @returns {Promise<PredictionResponse>} API response with prediction results
 */
export const detectVideoDeepfake = async (videoFile, onProgress = null) => {
  const formData = new FormData();
  formData.append('file', videoFile);
  formData.append('session_id', getSessionId());
  
  const config = {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
    timeout: 300000, // 5 minutes for video processing
  };
  
  if (onProgress) {
    config.onUploadProgress = (progressEvent) => {
      const percentCompleted = Math.round(
        (progressEvent.loaded * 100) / progressEvent.total
      );
      onProgress(percentCompleted);
    };
  }
  
  const response = await apiClient.post('/predict/video', formData, config);
  return response.data;
};

/**
 * Detect deepfake in audio
 * @param {File} audioFile - Audio file to analyze
 * @param {Function} onProgress - Progress callback (optional)
 * @returns {Promise<PredictionResponse>} API response with prediction results
 */
export const detectAudioDeepfake = async (audioFile, onProgress = null) => {
  const formData = new FormData();
  formData.append('file', audioFile);
  formData.append('session_id', getSessionId());
  
  const config = {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
  };
  
  if (onProgress) {
    config.onUploadProgress = (progressEvent) => {
      const percentCompleted = Math.round(
        (progressEvent.loaded * 100) / progressEvent.total
      );
      onProgress(percentCompleted);
    };
  }
  
  const response = await apiClient.post('/predict/audio', formData, config);
  return response.data;
};

/**
 * Check API health status
 * @returns {Promise} Health check response
 */
export const checkHealth = async () => {
  const response = await axios.get(`${API_BASE_URL}/health`);
  return response.data;
};

/**
 * Get API root information
 * @returns {Promise} API info
 */
export const getApiInfo = async () => {
  const response = await axios.get(`${API_BASE_URL}/`);
  return response.data;
};

/**
 * Fetch scan history for the current session
 * @param {number} limit - Max records to retrieve (default: 100)
 * @returns {Promise<DetectionRecord[]>} Array of history records
 */
export const fetchScanHistory = async (limit = 100) => {
  try {
    const session_id = getSessionId();

    // If session_id doesn't exist yet, there can't be any history for this anonymous session.
    if (!session_id) return [];

    // Backend endpoint: /api/v1/telemetry/history?session_id=...
    const cappedLimit = Math.max(1, Math.min(Number(limit) || 100, 100));
    const url = `${API_BASE_URL}/api/v1/telemetry/history?session_id=${encodeURIComponent(session_id)}&limit=${cappedLimit}`;
    const response = await axios.get(url);
    
    // Handle different response formats
    const data = response.data;
    
    // If response.data is already an array, return it
    if (Array.isArray(data)) {
      return data;
    }
    
    // If response.data has a records property that's an array, return it (standard format)
    if (data && Array.isArray(data.records)) {
      return data.records;
    }
    
    // If response.data has a history property that's an array, return it
    if (data && Array.isArray(data.history)) {
      return data.history;
    }
    
    // If response.data has a data property that's an array, return it
    if (data && Array.isArray(data.data)) {
      return data.data;
    }
    
    // If none of the above, return empty array to prevent crashes
    console.warn('[API] Unexpected response format for scan history:', data);
    return [];
  } catch (error) {
    console.error('[API] Failed to fetch scan history:', error);
    throw error;
  }
};

// =============================================================================
// Unified Detection Function
// =============================================================================

/**
 * Detect deepfake based on file type
 * @param {File} file - File to analyze
 * @param {Function} onProgress - Progress callback
 * @returns {Promise} Detection results
 */
export const detectDeepfake = async (file, onProgress = null) => {
  const fileType = getFileType(file);
  
  switch (fileType) {
    case 'image':
      return await detectImageDeepfake(file, onProgress);
    case 'video':
      return await detectVideoDeepfake(file, onProgress);
    case 'audio':
      return await detectAudioDeepfake(file, onProgress);
    default:
      throw new Error(`Unsupported file type: ${file.type}`);
  }
};

// =============================================================================
// Helper Functions
// =============================================================================

/**
 * Determine file type from MIME type
 * @param {File} file - File object
 * @returns {string} File type category (image, video, audio)
 */
export const getFileType = (file) => {
  const mimeType = file.type.toLowerCase();
  const extension = file.name.split('.').pop().toLowerCase();
  const imageExts = ['jpg', 'jpeg', 'png', 'bmp', 'webp'];
  const videoExts = ['mp4', 'avi', 'mov', 'mkv', 'webm'];
  const audioExts = ['mp3', 'wav', 'flac', 'ogg', 'm4a'];
  
  if (mimeType.startsWith('image/') && imageExts.includes(extension)) {
    return 'image';
  } else if (mimeType.startsWith('video/') && videoExts.includes(extension)) {
    return 'video';
  } else if (mimeType.startsWith('audio/') && audioExts.includes(extension)) {
    return 'audio';
  }
  
  // Fallback to extension only (for browsers/devices with missing MIME info)
  
  if (imageExts.includes(extension)) return 'image';
  if (videoExts.includes(extension)) return 'video';
  if (audioExts.includes(extension)) return 'audio';
  
  return 'unknown';
};

/**
 * Format file size for display
 * @param {number} bytes - File size in bytes
 * @returns {string} Formatted file size
 */
export const formatFileSize = (bytes) => {
  if (bytes === 0) return '0 Bytes';
  
  const k = 1024;
  const sizes = ['Bytes', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  
  return Math.round(bytes / Math.pow(k, i) * 100) / 100 + ' ' + sizes[i];
};

/**
 * Validate file before upload
 * @param {File} file - File to validate
 * @returns {Object} Validation result {valid: boolean, error: string}
 */
export const validateFile = (file) => {
  const MAX_SIZE = 100 * 1024 * 1024; // 100 MB
  const fileType = getFileType(file);
  
  if (fileType === 'unknown') {
    return {
      valid: false,
      error: 'Unsupported file type. Please upload an image, video, or audio file.'
    };
  }
  
  if (file.size > MAX_SIZE) {
    return {
      valid: false,
      error: `File too large. Maximum size is ${formatFileSize(MAX_SIZE)}.`
    };
  }
  
  return { valid: true, error: null };
};

export default apiClient;
