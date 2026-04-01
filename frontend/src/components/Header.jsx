/**
 * ============================================================================
 * Header Component - Application Header with Branding
 * ============================================================================
 * Beautiful header with cybersecurity theme
 * 
 * Author: Senior Full-Stack Engineer
 * Date: 2026-01-28
 * ============================================================================
 */

import React from 'react';
import { motion } from 'framer-motion';
import { Shield, Github, Info } from 'lucide-react';

const Header = () => {
  return (
    <header className="relative border-b border-white/10 bg-cyber-dark/50 backdrop-blur-xl">
      <div className="container mx-auto px-4 py-6">
        <div className="flex items-center justify-between">
          {/* Logo and Title */}
          <motion.button
            type="button"
            onClick={() => window.location.assign('/')}
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            className="flex items-center gap-4 text-left"
            title="Go to homepage"
          >
            <div className="relative">
              <motion.div
                className="p-3 rounded-xl bg-gradient-to-br from-cyber-blue/20 to-cyber-purple/20 border border-cyber-blue/30"
                animate={{
                  boxShadow: [
                    '0 0 20px rgba(0, 217, 255, 0.3)',
                    '0 0 40px rgba(0, 217, 255, 0.5)',
                    '0 0 20px rgba(0, 217, 255, 0.3)',
                  ]
                }}
                transition={{
                  duration: 2,
                  repeat: Infinity,
                  ease: 'easeInOut'
                }}
              >
                <Shield className="w-8 h-8 text-cyber-blue" />
              </motion.div>
            </div>

            <div>
              <h1 className="text-2xl font-bold text-gradient-cyber">
                Deepfake Detection System
              </h1>
              <p className="text-sm text-gray-400">
                Detecting Fake Image, Video, and Audio
              </p>
            </div>
          </motion.button>

          {/* Navigation Links */}
          <motion.div
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            className="flex items-center gap-3"
          >
            <a
              href="https://github.com"
              target="_blank"
              rel="noopener noreferrer"
              className="p-2 rounded-lg glass-hover transition-all"
              title="GitHub Repository"
            >
              <Github className="w-5 h-5 text-gray-400 hover:text-white transition-colors" />
            </a>

            <button
              className="p-2 rounded-lg glass-hover transition-all"
              title="About"
            >
              <Info className="w-5 h-5 text-gray-400 hover:text-white transition-colors" />
            </button>
          </motion.div>
        </div>
      </div>

      {/* Gradient Line */}
      <div className="absolute bottom-0 left-0 right-0 h-[2px] gradient-cyber-border" />
    </header>
  );
};

export default Header;
