import os
import tempfile
import time
import logging
from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
from pathlib import Path
from interview_chat_backend import chat_bp
# Import your modules
from modules.pipeline import InterviewAnalysisPipeline

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)
app.register_blueprint(chat_bp)
# Store the last report in memory for demo/dev purposes
last_report = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze_video():
    global last_report
    
    if 'video' not in request.files:
        return jsonify({"error": "No video file provided"}), 400
    
    video_file = request.files['video']
    if video_file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    # Save to temp file
    temp_dir = tempfile.gettempdir()
    temp_path = os.path.join(temp_dir, f"upload_{int(time.time())}_{video_file.filename}")
    video_file.save(temp_path)
    
    logger.info(f"Analyzing uploaded video: {temp_path}")
    
    try:
        # Initialize pipeline
        pipeline = InterviewAnalysisPipeline(
            whisper_model="base",
            cv_sample_rate=5
        )
        
        # Run full analysis
        report = pipeline.analyze(temp_path)
        last_report = report
        
        # Prepare response
        # Convert dataclasses/objects to dictionary for JSON
        response_data = {
            "success": True,
            "processing_time": report.processing_time,
            "overall_score": report.scores.overall_score,
            "performance_tier": report.scores.performance_tier,
            "scores": {
                "confidence": report.scores.confidence_score,
                "communication": report.scores.communication_score,
                "eye_contact": report.cv_result.eye_contact_percentage,
                "composure": max(0, 100 - report.cv_result.nervous_behavior_score),
                "clarity": report.nlp_result.communication_clarity_score
            },
            "breakdown": report.scores.score_breakdown,
            "strengths": report.scores.strengths,
            "weaknesses": report.scores.weaknesses,
            "suggestions": report.scores.suggestions,
            "transcript": report.audio_result.transcript,
            "filler_words_found": report.nlp_result.filler_words_found,
            "filler_word_count": report.nlp_result.filler_word_count,
            "filler_word_rate": report.nlp_result.filler_word_rate,
            "words_per_minute": report.audio_result.words_per_minute,
            "silence_ratio": report.audio_result.silence_ratio,
            "word_count": report.nlp_result.word_count,
            "avg_sentence_length": report.nlp_result.avg_sentence_length,
            "readability_score": report.nlp_result.readability_score,
            "vocabulary_richness": report.nlp_result.vocabulary_richness,
            "top_keywords": report.nlp_result.top_keywords,
            "nervous_timestamps": report.cv_result.timestamps_nervous,
            "nervous_percentage": report.cv_result.nervous_behavior_score,
            "duration": report.audio_result.duration,
            "errors": report.errors
        }
        
        # Add timeline data
        timeline_data = []
        for f in report.cv_result.frame_features:
            if f.face_detected:
                # Map emotion labels to numbers for charts
                emotion_map = {"nervous": 1, "neutral": 2, "happy": 3, "confident": 4, "confused": 0}
                timeline_data.append({
                    "t": round(f.timestamp, 1),
                    "e": emotion_map.get(f.emotion, 2)
                })
        response_data["timeline"] = timeline_data
        
        return jsonify(response_data)

    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500
    
    finally:
        # Clean up temp file
        if os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except:
                pass

@app.route('/demo-data', methods=['GET'])
def get_demo_data():
    from utils.demo_data import create_demo_report
    report = create_demo_report()
    
    # Same mapping as above for consistency
    response_data = {
        "success": True,
        "processing_time": report.processing_time,
        "overall_score": report.scores.overall_score,
        "performance_tier": report.scores.performance_tier,
        "scores": {
            "confidence": report.scores.confidence_score,
            "communication": report.scores.communication_score,
            "eye_contact": report.cv_result.eye_contact_percentage,
            "composure": max(0, 100 - report.cv_result.nervous_behavior_score),
            "clarity": report.nlp_result.communication_clarity_score
        },
        "breakdown": report.scores.score_breakdown,
        "strengths": report.scores.strengths,
        "weaknesses": report.scores.weaknesses,
        "suggestions": report.scores.suggestions,
        "transcript": report.audio_result.transcript,
        "filler_words_found": report.nlp_result.filler_words_found,
        "filler_word_count": report.nlp_result.filler_word_count,
        "filler_word_rate": report.nlp_result.filler_word_rate,
        "words_per_minute": report.audio_result.words_per_minute,
        "silence_ratio": report.audio_result.silence_ratio,
        "word_count": report.nlp_result.word_count,
        "avg_sentence_length": report.nlp_result.avg_sentence_length,
        "readability_score": report.nlp_result.readability_score,
        "vocabulary_richness": report.nlp_result.vocabulary_richness,
        "top_keywords": report.nlp_result.top_keywords,
        "nervous_timestamps": report.cv_result.timestamps_nervous,
        "nervous_percentage": report.cv_result.nervous_behavior_score,
        "duration": report.audio_result.duration,
        "errors": []
    }
    
    # Mock timeline for demo
    response_data["timeline"] = [
        {"t": 5, "e": 1}, {"t": 15, "e": 2}, {"t": 30, "e": 3}, {"t": 45, "e": 2},
        {"t": 60, "e": 4}, {"t": 80, "e": 3}, {"t": 100, "e": 2}, {"t": 120, "e": 3},
        {"t": 140, "e": 4}, {"t": 160, "e": 3}
    ]
    
    return jsonify(response_data)

if __name__ == '__main__':
    # Ensure models are trained
    from modules.ml_module import load_or_train_models
    load_or_train_models()
    
    logger.info("Starting InterviewIQ Server on http://localhost:5000")
    app.run(host='0.0.0.0', port=5000, debug=False)
