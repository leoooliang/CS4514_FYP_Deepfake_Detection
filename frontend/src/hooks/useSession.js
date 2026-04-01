/**
 * ============================================================================
 * useSession Hook - Anonymous Session Management
 * ============================================================================
 * Manages anonymous session IDs for tracking user activity across the app.
 * Uses localStorage for persistence and crypto.randomUUID() for generation.
 * 
 * Features:
 *   - Automatic session ID generation
 *   - localStorage persistence
 *   - React Hook interface
 * 
 * Author: Senior Full-Stack Engineer
 * Date: 2026-03-25
 * ============================================================================
 */

import { useState, useEffect } from 'react';

const SESSION_STORAGE_KEY = 'deepfake_session_id';

/**
 * Generate a new session ID using crypto.randomUUID()
 * @returns {string} New UUID session ID
 */
const generateSessionId = () => {
  return crypto.randomUUID();
};

/**
 * Get the current session ID from localStorage or generate a new one
 * @returns {string} Session ID
 */
export const getSessionId = () => {
  let sessionId = localStorage.getItem(SESSION_STORAGE_KEY);
  
  if (!sessionId) {
    sessionId = generateSessionId();
    localStorage.setItem(SESSION_STORAGE_KEY, sessionId);
    console.log('[Session] New session created:', sessionId);
  } else {
    console.log('[Session] Existing session loaded:', sessionId);
  }
  
  return sessionId;
};

/**
 * Clear the current session (useful for testing or logout)
 */
export const clearSession = () => {
  localStorage.removeItem(SESSION_STORAGE_KEY);
  console.log('[Session] Session cleared');
};

/**
 * React Hook for session management
 * @returns {Object} { sessionId, clearSession }
 */
export const useSession = () => {
  const [sessionId, setSessionId] = useState(() => getSessionId());
  
  useEffect(() => {
    const id = getSessionId();
    setSessionId(id);
  }, []);
  
  const handleClearSession = () => {
    clearSession();
    const newId = generateSessionId();
    localStorage.setItem(SESSION_STORAGE_KEY, newId);
    setSessionId(newId);
  };
  
  return {
    sessionId,
    clearSession: handleClearSession
  };
};

export default useSession;
