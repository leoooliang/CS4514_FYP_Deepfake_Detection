/**
 * Utility functions for formatting data
 */

/**
 * Format confidence score as percentage
 * @param {number} confidence - Confidence value (0-1)
 * @returns {string} Formatted percentage
 */
export const formatConfidence = (confidence) => {
  return `${(confidence * 100).toFixed(1)}%`;
};

/**
 * Format processing time
 * @param {number} seconds - Time in seconds
 * @returns {string} Formatted time string
 */
export const formatProcessingTime = (seconds) => {
  if (seconds < 1) {
    return `${(seconds * 1000).toFixed(0)}ms`;
  }
  return `${seconds.toFixed(2)}s`;
};

/**
 * Format timestamp to readable date
 * @param {number} timestamp - Unix timestamp
 * @returns {string} Formatted date string
 */
export const formatTimestamp = (timestamp) => {
  const date = new Date(timestamp * 1000);
  return date.toLocaleString();
};

/**
 * Get color class based on confidence level
 * @param {number} confidence - Confidence value (0-1)
 * @returns {string} Tailwind color class
 */
export const getConfidenceColor = (confidence) => {
  if (confidence >= 0.9) return 'text-green-400';
  if (confidence >= 0.7) return 'text-yellow-400';
  return 'text-red-400';
};
