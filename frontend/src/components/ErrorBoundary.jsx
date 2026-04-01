/**
 * ============================================================================
 * ErrorBoundary Component - Catch React Rendering Errors
 * ============================================================================
 * Prevents the entire app from crashing when a component throws an error.
 * 
 * Author: Senior Full-Stack Engineer
 * Date: 2026-03-25
 * ============================================================================
 */

import React from 'react';
import { AlertCircle } from 'lucide-react';

class ErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error) {
    return { hasError: true, error };
  }

  componentDidCatch(error, errorInfo) {
    console.error('ErrorBoundary caught an error:', error, errorInfo);
  }

  render() {
    if (this.state.hasError) {
      return (
        <div className="card bg-red-500/10 border-red-500/30">
          <div className="flex items-start gap-3">
            <AlertCircle className="w-6 h-6 text-red-400 flex-shrink-0 mt-0.5" />
            <div>
              <h3 className="text-lg font-semibold text-red-400 mb-1">
                Something went wrong
              </h3>
              <p className="text-gray-300 text-sm">
                {this.state.error?.message || 'An unexpected error occurred'}
              </p>
              {this.props.showRetry && (
                <button
                  onClick={() => this.setState({ hasError: false, error: null })}
                  className="mt-4 btn-secondary text-sm"
                >
                  Try Again
                </button>
              )}
            </div>
          </div>
        </div>
      );
    }

    return this.props.children;
  }
}

export default ErrorBoundary;
