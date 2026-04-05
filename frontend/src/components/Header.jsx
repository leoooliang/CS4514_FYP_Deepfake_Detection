import React, { useState } from 'react';
import { Fingerprint, Github, Info } from 'lucide-react';
import AboutModal from './AboutModal';

const Header = () => {
  const [aboutOpen, setAboutOpen] = useState(false);

  return (
    <header className="border-b border-white/10 bg-surface/60 backdrop-blur-xl">
      <div className="container mx-auto px-4 py-5">
        <div className="flex items-center justify-between">
          <button
            type="button"
            onClick={() => window.location.assign('/')}
            className="flex items-center gap-3 text-left"
            title="Go to homepage"
          >
            <div className="p-2.5 rounded-lg bg-primary/10 border border-primary/20">
              <Fingerprint className="w-7 h-7 text-primary" />
            </div>

            <div>
              <h1 className="text-xl font-bold text-white tracking-tight">
                Deepfake Detection System
              </h1>
              <p className="text-xs text-gray-500">
                Detecting Deepfakes in Image, Video, and Audio
              </p>
            </div>
          </button>

          <div className="flex items-center gap-2">
            <a
              href="https://github.com/leoooliang/CS4514_FYP_Deepfake_Detection"
              target="_blank"
              rel="noopener noreferrer"
              className="p-2 rounded-lg glass-hover"
              title="GitHub Repository"
            >
              <Github className="w-5 h-5 text-gray-400 hover:text-white transition-colors" />
            </a>

            <button
              onClick={() => setAboutOpen(true)}
              className="p-2 rounded-lg glass-hover"
              title="About"
            >
              <Info className="w-5 h-5 text-gray-400 hover:text-white transition-colors" />
            </button>
          </div>
        </div>
      </div>

      <AboutModal isOpen={aboutOpen} onClose={() => setAboutOpen(false)} />
    </header>
  );
};

export default Header;
