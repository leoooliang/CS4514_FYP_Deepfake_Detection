import React, { useEffect, useRef, useState } from 'react';

const MediaPreviewModal = ({
  isOpen,
  onClose,
  mediaUrl,
  mediaType,
  title
}) => {
  const [previewZoom, setPreviewZoom] = useState(1);
  const [previewOffset, setPreviewOffset] = useState({ x: 0, y: 0 });
  const [isDraggingImage, setIsDraggingImage] = useState(false);
  const [dragStart, setDragStart] = useState({ x: 0, y: 0 });
  const previewCanvasRef = useRef(null);

  const MIN_ZOOM = 1;
  const MAX_ZOOM = 3;
  const ZOOM_STEP = 0.1;

  useEffect(() => {
    if (!isOpen) return undefined;

    const previousOverflow = document.body.style.overflow;
    document.body.style.overflow = 'hidden';

    return () => {
      document.body.style.overflow = previousOverflow;
    };
  }, [isOpen]);

  useEffect(() => {
    if (!isOpen) {
      setPreviewZoom(1);
      setPreviewOffset({ x: 0, y: 0 });
      setIsDraggingImage(false);
      setDragStart({ x: 0, y: 0 });
    }
  }, [isOpen]);

  const updateZoom = (nextZoom, cursorPoint = null) => {
    const previousZoom = previewZoom;
    const clampedZoom = Math.min(MAX_ZOOM, Math.max(MIN_ZOOM, nextZoom));
    const normalizedZoom = Number(clampedZoom.toFixed(2));

    if (cursorPoint && normalizedZoom !== previousZoom) {
      const zoomRatio = normalizedZoom / previousZoom;
      const offsetX = cursorPoint.x - (cursorPoint.x - previewOffset.x) * zoomRatio;
      const offsetY = cursorPoint.y - (cursorPoint.y - previewOffset.y) * zoomRatio;
      setPreviewOffset({ x: offsetX, y: offsetY });
    }

    if (normalizedZoom <= 1) {
      setPreviewOffset({ x: 0, y: 0 });
      setIsDraggingImage(false);
    }

    setPreviewZoom(normalizedZoom);
  };

  if (!isOpen) return null;

  return (
    <div
      className="fixed inset-0 z-50 bg-black/70 backdrop-blur-sm flex items-center justify-center p-4"
      onClick={onClose}
      onWheel={(e) => {
        if (mediaType !== 'image') return;
        e.preventDefault();

        const canvas = previewCanvasRef.current;
        const rect = canvas?.getBoundingClientRect();
        const cursorPoint =
          rect
            ? {
                x: e.clientX - (rect.left + rect.width / 2),
                y: e.clientY - (rect.top + rect.height / 2)
              }
            : { x: 0, y: 0 };

        const direction = e.deltaY > 0 ? -1 : 1;
        updateZoom(previewZoom + direction * ZOOM_STEP, cursorPoint);
      }}
    >
      <div
        className="w-full max-w-4xl rounded-xl border border-white/20 bg-black/80 p-4 md:p-6"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-white font-semibold truncate mr-4">{title || 'Media Preview'}</h3>
          <div className="flex items-center gap-2">
            {mediaType === 'image' && (
              <>
                <button
                  type="button"
                  className="btn-secondary"
                  onClick={() => updateZoom(previewZoom - ZOOM_STEP)}
                >
                  -
                </button>
                <span className="text-sm text-gray-300 min-w-[64px] text-center">
                  {(previewZoom * 100).toFixed(0)}%
                </span>
                <button
                  type="button"
                  className="btn-secondary"
                  onClick={() => updateZoom(previewZoom + ZOOM_STEP)}
                >
                  +
                </button>
              </>
            )}
            <button
              type="button"
              className="btn-secondary"
              onClick={onClose}
            >
              Close
            </button>
          </div>
        </div>

        <div className="rounded-lg bg-black/40 border border-white/10 overflow-hidden flex items-center justify-center min-h-[260px] max-h-[70vh]">
          {mediaType === 'image' && mediaUrl ? (
            <div
              ref={previewCanvasRef}
              className={`w-full h-full min-h-[260px] max-h-[70vh] overflow-hidden flex items-center justify-center select-none ${previewZoom > 1 ? 'cursor-grab' : 'cursor-default'} ${isDraggingImage ? 'cursor-grabbing' : ''}`}
              onMouseDown={(e) => {
                if (previewZoom <= 1) return;
                e.preventDefault();
                setIsDraggingImage(true);
                setDragStart({
                  x: e.clientX - previewOffset.x,
                  y: e.clientY - previewOffset.y
                });
              }}
              onMouseMove={(e) => {
                if (!isDraggingImage || previewZoom <= 1) return;
                e.preventDefault();
                setPreviewOffset({
                  x: e.clientX - dragStart.x,
                  y: e.clientY - dragStart.y
                });
              }}
              onMouseUp={() => setIsDraggingImage(false)}
              onMouseLeave={() => setIsDraggingImage(false)}
            >
              <img
                src={mediaUrl}
                alt={title || 'Selected media'}
                className="max-h-[70vh] w-auto object-contain transition-transform duration-100 pointer-events-none"
                style={{
                  transform: `translate(${previewOffset.x}px, ${previewOffset.y}px) scale(${previewZoom})`,
                  transformOrigin: 'center center'
                }}
              />
            </div>
          ) : mediaType === 'video' && mediaUrl ? (
            <video
              src={mediaUrl}
              controls
              className="max-h-[70vh] w-full object-contain"
            />
          ) : mediaType === 'audio' && mediaUrl ? (
            <audio
              src={mediaUrl}
              controls
              className="w-full max-w-xl"
            />
          ) : (
            <span className="text-gray-400 text-sm">Preview unavailable</span>
          )}
        </div>
      </div>
    </div>
  );
};

export default MediaPreviewModal;
