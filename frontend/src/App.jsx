/**
 * ============================================================================
 * App Component - Main Application Root
 * ============================================================================
 * Root component that sets up routing and global layout
 * 
 * Author: Senior Full-Stack Engineer
 * Date: 2026-01-28
 * ============================================================================
 */

import React from 'react';
import { BrowserRouter as Router } from 'react-router-dom';
import Header from './components/Header';
import HomePage from './pages/HomePage';

function App() {
  return (
    <Router>
      <div className="min-h-screen gradient-cyber text-white">
        {/* Animated Background Pattern */}
        <div className="fixed inset-0 opacity-[0.015] pointer-events-none">
          <div className="absolute inset-0" style={{
            backgroundImage: `
              repeating-linear-gradient(0deg, transparent, transparent 2px, rgba(0, 217, 255, 0.1) 2px, rgba(0, 217, 255, 0.1) 4px),
              repeating-linear-gradient(90deg, transparent, transparent 2px, rgba(0, 217, 255, 0.1) 2px, rgba(0, 217, 255, 0.1) 4px)
            `,
            backgroundSize: '50px 50px'
          }} />
        </div>

        {/* Main Content */}
        <div className="relative z-10">
          <Header />
          <HomePage />
        </div>

        {/* Footer */}
        <footer className="relative z-10 border-t border-white/10 bg-cyber-dark/50 backdrop-blur-xl mt-20">
          <div className="container mx-auto px-4 py-8 text-center">
            <p className="text-gray-400 text-sm">
              © 2026 Deepfake Detection System. 
            </p>
            <p className="text-gray-500 text-xs mt-2">
              For educational and research purposes. Use responsibly.
            </p>
          </div>
        </footer>
      </div>
    </Router>
  );
}

export default App;
