// Axios client and helpers for the backend API.

import axios from 'axios';
import { getSessionId } from '../hooks/useSession';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

export { API_BASE_URL };

const apiClient = axios.create({
  baseURL: `${API_BASE_URL}/api/v1`,
  timeout: 120000, // 2 min — uploads / video can be slow
  headers: {
    'Content-Type': 'application/json',
  },
});

apiClient.interceptors.request.use(
  (config) => {
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

apiClient.interceptors.response.use(
  (response) => {
    console.log(`[API Response] ${response.status} ${response.config.url}`);
    return response;
  },
  (error) => {
    if (error.response) {
      console.error('[API Error Response]', {
        status: error.response.status,
        data: error.response.data,
        url: error.config?.url,
      });
      
      const errorData = error.response.data;
      let errorMessage = 'An error occurred.';
      
      if (errorData) {
        if (typeof errorData === 'string') {
          errorMessage = errorData;
        } else if (errorData.detail) {
          if (typeof errorData.detail === 'string') {
            errorMessage = errorData.detail;
          } else if (Array.isArray(errorData.detail)) {
            errorMessage = errorData.detail.map(err => err.msg).join(', ');
          }
        } else if (errorData.message) {
          errorMessage = errorData.message;
        } else if (errorData.error) {
          errorMessage = errorData.error;
        }
      }
      
      switch (error.response.status) {
        case 400:
          error.message = errorMessage || 'Invalid request. Please check your input.';
          break;
        case 401:
          localStorage.removeItem('auth_token');
          error.message = 'Unauthorized. Please log in again.';
          break;
        case 413:
          error.message = 'File too large. Please upload a smaller file.';
          break;
        case 422:
          error.message = errorMessage || 'Invalid data submitted.';
          break;
        case 500:
          error.message = 'Server error. Please try again later.';
          break;
        case 503:
          error.message = 'Service temporarily unavailable. Please try again later.';
          break;
        default:
          error.message = errorMessage;
      }
    } else if (error.request) {
      console.error('[API No Response]', error.request);
      error.message = 'No response from server. Please check your connection and ensure the backend is running.';
    } else {
      console.error('[API Request Setup Error]', error.message);
      error.message = error.message || 'Request failed. Please try again.';
    }
    
    return Promise.reject(error);
  }
);

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

export const detectVideoDeepfake = async (videoFile, onProgress = null) => {
  const formData = new FormData();
  formData.append('file', videoFile);
  formData.append('session_id', getSessionId());
  
  const config = {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
    timeout: 300000, // 5 min for long videos
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

export const checkHealth = async () => {
  const response = await axios.get(`${API_BASE_URL}/health`);
  return response.data;
};

export const getApiInfo = async () => {
  const response = await axios.get(`${API_BASE_URL}/`);
  return response.data;
};

// History rows for this browser session (telemetry endpoint).
export const fetchScanHistory = async (limit = 100) => {
  try {
    const session_id = getSessionId();

    if (!session_id) return [];

    const cappedLimit = Math.max(1, Math.min(Number(limit) || 100, 100));
    const url = `${API_BASE_URL}/api/v1/telemetry/history?session_id=${encodeURIComponent(session_id)}&limit=${cappedLimit}`;
    const response = await axios.get(url);
    
    const data = response.data;
    
    if (Array.isArray(data)) {
      return data;
    }
    
    if (data && Array.isArray(data.records)) {
      return data.records;
    }
    
    if (data && Array.isArray(data.history)) {
      return data.history;
    }
    
    if (data && Array.isArray(data.data)) {
      return data.data;
    }
    
    console.warn('[API] Unexpected response format for scan history:', data);
    return [];
  } catch (error) {
    console.error('[API] Failed to fetch scan history:', error);
    throw error;
  }
};

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

// image / video / audio from MIME + extension; extension-only fallback if MIME is wrong
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
  
  if (imageExts.includes(extension)) return 'image';
  if (videoExts.includes(extension)) return 'video';
  if (audioExts.includes(extension)) return 'audio';
  
  return 'unknown';
};

export const formatFileSize = (bytes) => {
  if (bytes === 0) return '0 Bytes';
  
  const k = 1024;
  const sizes = ['Bytes', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  
  return Math.round(bytes / Math.pow(k, i) * 100) / 100 + ' ' + sizes[i];
};

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
