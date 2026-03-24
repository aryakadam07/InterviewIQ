"""
Audio Processing Module
Extracts audio from video and transcribes speech using OpenAI Whisper.
"""

import os
import tempfile
import logging
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class AudioSegment:
    start: float
    end: float
    text: str
    confidence: float = 1.0


@dataclass
class AudioAnalysisResult:
    transcript: str = ""
    segments: list = field(default_factory=list)
    language: str = "en"
    duration: float = 0.0
    words_per_minute: float = 0.0
    silence_ratio: float = 0.0
    audio_extracted: bool = False
    transcription_success: bool = False
    error_message: str = ""


class AudioProcessor:
    """Handles audio extraction from video and speech-to-text transcription."""

    def __init__(self, model_size: str = "base"):
        """
        Args:
            model_size: Whisper model size ('tiny', 'base', 'small', 'medium', 'large')
                        'base' is a good balance of speed vs accuracy.
        """
        self.model_size = model_size
        self._whisper_model = None

    def _load_whisper(self):
        """Lazy-load Whisper model."""
        if self._whisper_model is None:
            try:
                import whisper
                logger.info(f"Loading Whisper model: {self.model_size}")
                self._whisper_model = whisper.load_model(self.model_size)
            except ImportError:
                logger.error("Whisper not installed. Run: pip install openai-whisper")
                raise
        return self._whisper_model

    def extract_audio(self, video_path: str, output_path: Optional[str] = None) -> str:
        """
        Extract audio track from video file.
        
        Args:
            video_path: Path to input video
            output_path: Optional path for extracted audio (.wav)
        
        Returns:
            Path to extracted audio file
        """
        try:
            from moviepy import VideoFileClip
        except ImportError:
            try:
                from moviepy.editor import VideoFileClip  # fallback for v1.x
            except ImportError:
                raise ImportError("moviepy required: pip install moviepy")

        if output_path is None:
            tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            output_path = tmp.name
            tmp.close()

        try:
            with VideoFileClip(video_path) as clip:
                if clip.audio is None:
                    raise ValueError("Video has no audio track")
                clip.audio.write_audiofile(
                    output_path,
                    fps=16000,
                    nbytes=2,
                    codec="pcm_s16le",
                    logger=None,
                )
            logger.info(f"Audio extracted to: {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"Audio extraction failed: {e}")
            raise

    def transcribe(self, audio_path: str) -> AudioAnalysisResult:
        """
        Transcribe audio using Whisper.
        
        Args:
            audio_path: Path to audio file (.wav, .mp3, etc.)
        
        Returns:
            AudioAnalysisResult with full transcript and segments
        """
        result = AudioAnalysisResult()

        try:
            model = self._load_whisper()
            logger.info(f"Transcribing: {audio_path}")

            transcription = model.transcribe(
                audio_path,
                language=None,  # auto-detect
                task="transcribe",
                verbose=False,
            )

            result.transcript = transcription.get("text", "").strip()
            result.language = transcription.get("language", "en")
            result.transcription_success = True

            # Parse segments
            raw_segments = transcription.get("segments", [])
            for seg in raw_segments:
                result.segments.append(
                    AudioSegment(
                        start=seg.get("start", 0),
                        end=seg.get("end", 0),
                        text=seg.get("text", "").strip(),
                        confidence=abs(seg.get("avg_logprob", -0.5)),
                    )
                )

            # Calculate duration and pace
            if result.segments:
                result.duration = result.segments[-1].end
            elif os.path.exists(audio_path):
                try:
                    import librosa
                    _, sr = librosa.load(audio_path, sr=None, duration=1)
                    import soundfile as sf
                    info = sf.info(audio_path)
                    result.duration = info.duration
                except Exception:
                    pass

            word_count = len(result.transcript.split())
            if result.duration > 0:
                result.words_per_minute = round(word_count / (result.duration / 60), 1)

            # Estimate silence ratio from segments
            if result.duration > 0 and result.segments:
                speech_duration = sum(s.end - s.start for s in result.segments)
                result.silence_ratio = round(
                    1 - (speech_duration / result.duration), 3
                )

            logger.info(
                f"Transcription complete: {word_count} words, "
                f"{result.words_per_minute} WPM, language={result.language}"
            )

        except Exception as e:
            result.error_message = str(e)
            logger.error(f"Transcription failed: {e}")

        return result

    def process_video(self, video_path: str) -> AudioAnalysisResult:
        """
        Full pipeline: extract audio from video and transcribe.
        
        Args:
            video_path: Path to video file
        
        Returns:
            AudioAnalysisResult
        """
        result = AudioAnalysisResult()
        audio_path = None

        try:
            audio_path = self.extract_audio(video_path)
            result.audio_extracted = True
            transcription_result = self.transcribe(audio_path)
            # Merge results
            result.transcript = transcription_result.transcript
            result.segments = transcription_result.segments
            result.language = transcription_result.language
            result.duration = transcription_result.duration
            result.words_per_minute = transcription_result.words_per_minute
            result.silence_ratio = transcription_result.silence_ratio
            result.transcription_success = transcription_result.transcription_success
            result.error_message = transcription_result.error_message

        except Exception as e:
            result.error_message = str(e)
            logger.error(f"Video audio processing failed: {e}")

        finally:
            if audio_path and os.path.exists(audio_path):
                try:
                    os.unlink(audio_path)
                except Exception:
                    pass

        return result
