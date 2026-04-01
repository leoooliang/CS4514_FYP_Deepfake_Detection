"""
============================================================================
Custom Exceptions for Deepfake Detection Backend
============================================================================
This module defines custom exceptions for fail-fast validation in the
deepfake detection pipeline.

These exceptions are raised during preprocessing when media does not meet
minimum quality requirements (e.g., no face detected, no voice detected).

Author: Senior ML Engineer
Date: 2026-03-25
============================================================================
"""


class NoFaceDetectedError(Exception):
    """
    Raised when no face is detected in an image or video.
    
    This exception provides a user-friendly message by default while
    optionally storing technical details for logging purposes.
    
    Example:
        >>> if boxes is None or len(boxes) == 0:
        ...     raise NoFaceDetectedError()
        >>> # Or with custom message:
        ...     raise NoFaceDetectedError("Custom user message")
    """
    
    def __init__(self, user_message: str = None, technical_details: str = None):
        """
        Initialize the exception with user-friendly and technical messages.
        
        Args:
            user_message: Optional custom user-friendly message.
                         Defaults to "No face detected in the provided image."
            technical_details: Optional technical details for logging (not shown to user)
        """
        self.user_message = user_message or "No face detected in the provided image."
        self.technical_details = technical_details
        super().__init__(self.user_message)


class NoVoiceDetectedError(Exception):
    """
    Raised when no voice or speech is detected in an audio file.
    
    This exception provides a user-friendly message by default while
    optionally storing technical details for logging purposes.
    
    Example:
        >>> if total_active_duration < min_duration_threshold:
        ...     raise NoVoiceDetectedError()
        >>> # Or with custom message:
        ...     raise NoVoiceDetectedError("Custom user message")
    """
    
    def __init__(self, user_message: str = None, technical_details: str = None):
        """
        Initialize the exception with user-friendly and technical messages.
        
        Args:
            user_message: Optional custom user-friendly message.
                         Defaults to "No voice detected in the provided audio."
            technical_details: Optional technical details for logging (not shown to user)
        """
        self.user_message = user_message or "No voice detected in the provided audio."
        self.technical_details = technical_details
        super().__init__(self.user_message)
