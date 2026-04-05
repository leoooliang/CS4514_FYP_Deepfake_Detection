// Anonymous session id stored in localStorage for telemetry/history.

import { useState, useEffect } from 'react';

const SESSION_STORAGE_KEY = 'deepfake_session_id';

const generateSessionId = () => {
  return crypto.randomUUID();
};

// Reads existing id or creates one and saves it.
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

export const clearSession = () => {
  localStorage.removeItem(SESSION_STORAGE_KEY);
  console.log('[Session] Session cleared');
};

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
