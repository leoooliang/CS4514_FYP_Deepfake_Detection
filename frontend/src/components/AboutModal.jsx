import React, { useEffect } from 'react';
import { createPortal } from 'react-dom';
import {
  X,
  Image as ImageIcon,
  AudioLines,
  Video,
} from 'lucide-react';

const AboutModal = ({ isOpen, onClose }) => {
  useEffect(() => {
    if (!isOpen) return;
    const onKey = (e) => e.key === 'Escape' && onClose();
    window.addEventListener('keydown', onKey);
    document.body.style.overflow = 'hidden';
    return () => {
      window.removeEventListener('keydown', onKey);
      document.body.style.overflow = '';
    };
  }, [isOpen, onClose]);

  if (!isOpen) return null;

  return createPortal(
    <div
      className="fixed inset-0 z-[100] flex items-start justify-center overflow-y-auto py-10 px-4"
      role="presentation"
    >
      <div
        className="fixed inset-0 bg-black/60 backdrop-blur-sm"
        onClick={onClose}
        aria-hidden="true"
      />

      <div
        role="dialog"
        aria-modal="true"
        aria-labelledby="about-modal-title"
        className="relative z-[1] w-full max-w-4xl rounded-xl border border-white/10 bg-surface/95 backdrop-blur-xl shadow-2xl shadow-black/30"
      >
        <button
          type="button"
          onClick={onClose}
          className="absolute top-4 right-4 p-1.5 rounded-lg text-gray-400 hover:text-white hover:bg-white/10"
          aria-label="Close dialog"
        >
          <X className="w-5 h-5" />
        </button>

        <div className="p-8 sm:p-10 space-y-8">
          <div className="text-center space-y-3">
            <h2
              id="about-modal-title"
              className="text-2xl sm:text-3xl font-bold text-white tracking-tight"
            >
              Detection of Deepfake Images, Videos, and Audio
            </h2>
            <p className="text-sm text-gray-500">
              Final Year Project &mdash; Department of Computer Science, City
              University of Hong Kong
            </p>
          </div>

          <section className="space-y-3" aria-labelledby="about-mission-heading">
            <h3
              id="about-mission-heading"
              className="text-base font-semibold text-white"
            >
              Project mission
            </h3>
            <p className="text-sm text-gray-300 leading-relaxed">
              "Deepfake" is a highly realistic multimedia with swapped or AI-generated human faces or voices. 
              The rapid advancement in artificial intelligence technologies has led to the development of 
              highly realistic deepfake multimedia materials. This has caused an increase in misinformation 
              and led to a loss of trust among people. Existing techniques have been reported to be inaccurate 
              and ineffective in detecting deepfakes. 
              <br />
              <br />
              Our mission is to build a fast, user-friendly, and reliable deepfake detection system for 
              the improved detection of deepfake images, videos, and audio,
              to help users make more informed judgements.
            </p>
          </section>

          <section
            className="space-y-4"
            aria-labelledby="about-tech-heading"
          >
            <h3
              id="about-tech-heading"
              className="text-base font-semibold text-white"
            >
              Core technology (3 Detection Modules)
            </h3>

            <div className="rounded-lg border border-white/10 bg-white/[0.02] p-5 space-y-3">
              <div className="flex items-center gap-3">
                <div className="p-1.5 rounded-md bg-primary/10">
                  <ImageIcon className="w-4 h-4 text-primary" />
                </div>
                <h4 className="font-semibold text-white text-sm">Image Detection Module</h4>
              </div>
              <p className="text-sm text-gray-400 leading-relaxed">
                A dual-stream detection network runs two complementary analyses in parallel:
                <br />
                <br />
                - Stream 1: uses a fine-tuned CLIP model (a large vision model trained to relate images and text) to spot
                unnatural spatial patterns in the picture.
                <br />
                - Stream 2: applies SRM (Spatial Rich Model) filter and uses a EfficientNet model, which emphasise the 
                high-frequency details (fine edges and texture) where generative tools often leave faint synthesis noise.
                <br />
                <br />
                These streams are fused to produce a final deepfake detection score.
              </p>
            </div>

            <div className="rounded-lg border border-white/10 bg-white/[0.02] p-5 space-y-3">
              <div className="flex items-center gap-3">
                <div className="p-1.5 rounded-md bg-accent-violet/10">
                  <AudioLines className="w-4 h-4 text-accent-violet" />
                </div>
                <h4 className="font-semibold text-white text-sm">Audio Detection Module</h4>
              </div>
              <p className="text-sm text-gray-400 leading-relaxed">
                A dual-stream detection network analyzes audio signals through two parallel feature extractions:
                <br />
                <br />
                - Stream 1: Transforms audio into Log-Mel Spectrograms and processes them through a ResNet-18 model to capture 
                high-level frequency artifacts, such as "robot textures" and unnatural spectral frequencies.
                <br />
                - Stream 2: Uses Linear Frequency Cepstral Coefficients (LFCC) and a parallel ResNet-18 model to capture 
                low-level synthetic vocoder artifacts, such as unnatural hissing or popping noises.
                <br />
                <br />
                These streams are combined via Attention-Based Fusion and passed through a 2-layer Bidirectional GRU 
                to recognize rhythmic disorders and temporal inconsistencies in speech patterns.
              </p>
            </div>

            <div className="rounded-lg border border-white/10 bg-white/[0.02] p-5 space-y-3">
              <div className="flex items-center gap-3">
                <div className="p-1.5 rounded-md bg-accent-rose/10">
                  <Video className="w-4 h-4 text-accent-rose" />
                </div>
                <h4 className="font-semibold text-white text-sm">Video Detection Module</h4>
              </div>
              <p className="text-sm text-gray-400 leading-relaxed">
                A tri-stream detection network analyzes video through three parallel analysis:
                <br />
                <br />
                - Stream 1: Employs the Image Detection Module to analyze individual frames for spatial inconsistencies and high-frequency synthesis noise.
                <br />
                - Stream 2: Employs the Audio Detection Module to identify vocal synthesis artifacts within the video's soundtrack.
                <br />
                - Stream 3: Uses a vision model and CNN model with Multi-Head Attention and a BiLSTM model to evaluate the alignment of audio and visual streams, 
                specifically verifying lip-sync consistency.
                <br />
                <br />
                The three streams are merged through Concatenation and a Multi-Layer Perceptron (MLP) to project a final binary classification of "Real" or "Deepfake".
              </p>
            </div>
          </section>

          <footer
            className="pt-4 border-t border-white/10 space-y-2"
            aria-labelledby="about-credits-heading"
          >
            <p className="text-sm text-gray-500 leading-relaxed">
              Developed by LIANG Wai Ching at
              the Department of Computer Science, City University of Hong Kong.
            </p>
          </footer>
        </div>
      </div>
    </div>,
    document.body
  );
};

export default AboutModal;
