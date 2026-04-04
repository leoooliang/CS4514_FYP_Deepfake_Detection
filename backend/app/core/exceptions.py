"""
Custom exceptions for fail-fast validation in the detection pipeline.

Exceptions:
- NoFaceDetectedError
- NoVoiceDetectedError
"""


class NoFaceDetectedError(Exception):
    """Raised when no face is detected in an image or video."""

    def __init__(self, user_message: str = None, technical_details: str = None):
        self.user_message = user_message or "No face detected in the provided image."
        self.technical_details = technical_details
        super().__init__(self.user_message)


class NoVoiceDetectedError(Exception):
    """Raised when no voice / speech is detected in an audio file."""

    def __init__(self, user_message: str = None, technical_details: str = None):
        self.user_message = user_message or "No voice detected in the provided audio."
        self.technical_details = technical_details
        super().__init__(self.user_message)
