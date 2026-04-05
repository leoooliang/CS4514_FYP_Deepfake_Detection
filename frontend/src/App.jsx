import React from 'react';
import { BrowserRouter as Router } from 'react-router-dom';
import Header from './components/Header';
import HomePage from './pages/HomePage';

function App() {
  return (
    <Router>
      <div className="min-h-screen flex flex-col bg-surface text-white">
        <Header />
        <main className="flex-1 relative z-10">
          <HomePage />
        </main>

        <footer className="relative z-10 border-t border-white/10 bg-surface/50 backdrop-blur-xl">
          <div className="container mx-auto px-4 py-8 text-center">
            <p className="text-gray-400 text-sm">
              &copy; 2026 Deepfake Detection System. Developed by LIANG Wai Ching
            </p>
            <p className="text-gray-500 text-xs mt-2">
              Developed as a Final Year Project for the Department of Computer Science at the City University of Hong Kong (2025-2026).
            </p>
          </div>
        </footer>
      </div>
    </Router>
  );
}

export default App;
