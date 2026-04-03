
import os, io, json, base64, time, uuid, tempfile
import numpy as np
import cv2
from faster_whisper import WhisperModel
import mediapipe as mp
import spacy
import nltk
from nltk.corpus import stopwords
from flask import Blueprint, request, jsonify, session

# ── NLP setup ────────────────────────────────────────────────────────────────
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    import subprocess, sys
    subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")

nltk.download("stopwords", quiet=True)
nltk.download("punkt", quiet=True)
STOPWORDS = set(stopwords.words("english"))

# ── Whisper model (load once) ─────────────────────────────────────────────────
_whisper_model = None

def get_whisper():
    global _whisper_model
    if _whisper_model is None:
        _whisper_model = WhisperModel("base", compute_type="int8")
    return _whisper_model

# ── MediaPipe face mesh ───────────────────────────────────────────────────────
try:
    mp_face_mesh = mp.solutions.face_mesh
    mp_drawing = mp.solutions.drawing_utils
except AttributeError:
    mp_face_mesh = None
    mp_drawing = None
# ── Question bank ─────────────────────────────────────────────────────────────
QUESTION_BANK = {
    "behavioral": [
        "Tell me about yourself and your professional journey.",
        "Describe a challenging situation you faced at work and how you resolved it.",
        "Give me an example of a time you showed leadership.",
        "Tell me about a time you failed and what you learned from it.",
        "How do you handle tight deadlines and pressure?",
        "Describe a time you worked effectively in a team.",
        "Tell me about a time you had to deal with a difficult colleague.",
        "Give an example of a goal you set and how you achieved it.",
    ],
    "technical": [
        "Walk me through your technical stack and why you chose it.",
        "How do you approach debugging a complex production issue?",
        "Describe your experience with system design and scalability.",
        "How do you ensure code quality in your projects?",
        "Tell me about a technically challenging project you built.",
    ],
    "situational": [
        "If you had to deliver bad news to a client, how would you handle it?",
        "How would you prioritize tasks if everything felt urgent?",
        "What would you do if you disagreed with your manager's decision?",
        "How would you onboard yourself in a new team with no documentation?",
    ],
    "motivational": [
        "Why are you interested in this role?",
        "Where do you see yourself in five years?",
        "What motivates you to perform at your best?",
        "What are your greatest professional strengths?",
        "What is one area you are actively working to improve?",
    ],
}

# ── Keyword banks per question category ──────────────────────────────────────
KEYWORDS = {
    "behavioral":   ["situation","task","action","result","team","challenge","resolved","learned","leadership","impact"],
    "technical":    ["architecture","scalability","performance","debug","algorithm","refactor","deploy","api","testing","trade-off"],
    "situational":  ["prioritize","communicate","stakeholder","decision","impact","negotiate","escalate","solution","timeline","risk"],
    "motivational": ["passion","goal","growth","value","mission","contribute","achieve","align","vision","career"],
}

# ═════════════════════════════════════════════════════════════════════════════
#  Blueprint
# ═════════════════════════════════════════════════════════════════════════════
chat_bp = Blueprint("chat", __name__, url_prefix="/api/chat")


# ── helpers ───────────────────────────────────────────────────────────────────

def _extract_keywords(text: str, category: str) -> dict:
    """Return matched / missed keywords for the given category."""
    doc   = nlp(text.lower())
    words = {t.lemma_ for t in doc if not t.is_stop and t.is_alpha}
    bank  = KEYWORDS.get(category, [])
    found = [k for k in bank if k in words or any(k in w for w in words)]
    missed= [k for k in bank if k not in found]
    return {"found": found, "missed": missed}


def _analyse_face(frame_b64: str) -> dict:
    """Decode a base64 JPEG frame and return face-mesh metrics."""
    try:
        img_data = base64.b64decode(frame_b64)
        arr      = np.frombuffer(img_data, np.uint8)
        frame    = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        with mp_face_mesh.FaceMesh(
            static_image_mode=True, max_num_faces=1,
            refine_landmarks=True, min_detection_confidence=0.5
        ) as face_mesh:
            results = face_mesh.process(rgb)

        if not results.multi_face_landmarks:
            return {"detected": False}

        lms = results.multi_face_landmarks[0].landmark

        # Eye-contact proxy: nose tip vs eye-centre alignment
        nose   = lms[1]
        l_eye  = lms[33]
        r_eye  = lms[263]
        eye_cx = (l_eye.x + r_eye.x) / 2
        offset = abs(nose.x - eye_cx)
        gaze_ok = offset < 0.05

        # Mouth openness → engagement
        top_lip = lms[13].y
        bot_lip = lms[14].y
        mouth_open = (bot_lip - top_lip) > 0.01

        # Head pose proxy via nose-tip z
        head_stable = abs(lms[1].z) < 0.15

        return {
            "detected":    True,
            "eye_contact": gaze_ok,
            "mouth_open":  mouth_open,
            "head_stable": head_stable,
            "raw_offset":  round(offset, 4),
        }
    except Exception as e:
        return {"detected": False, "error": str(e)}


def _transcribe_audio(audio_b64: str) -> str:
    """Decode base64 audio and transcribe with faster-whisper."""
    try:
        audio_bytes = base64.b64decode(audio_b64)

        with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        model = get_whisper()
        segments, _ = model.transcribe(tmp_path)

        text = " ".join([seg.text for seg in segments])

        os.unlink(tmp_path)
        return text.strip()

    except Exception as e:
        return f"[transcription error: {e}]"

def _score_answer(transcript: str, face: dict, duration_s: float, category: str) -> dict:
    """Produce a structured score dict for one answer."""
    kw      = _extract_keywords(transcript, category)
    words   = transcript.split()
    wpm     = round(len(words) / max(duration_s / 60, 0.1), 1)

    # Sub-scores (0–100)
    content_score  = min(100, len(kw["found"]) * 12 + min(len(words) * 0.4, 40))
    fluency_score  = max(0, 100 - abs(wpm - 130) * 0.8)          # ideal ~130 wpm
    eye_score      = 85 if face.get("eye_contact") else 45
    posture_score  = 80 if face.get("head_stable") else 50
    engagement_score = 75 if face.get("mouth_open") else 55

    # Weighted composite
    composite = round(
        content_score  * 0.35 +
        fluency_score  * 0.25 +
        eye_score      * 0.20 +
        posture_score  * 0.10 +
        engagement_score * 0.10, 1
    )

    # Feedback generation
    tips = []
    if wpm < 100: tips.append("Speak a little faster — aim for 120–140 words per minute.")
    if wpm > 170: tips.append("Slow down slightly so key points land clearly.")
    if not face.get("eye_contact"): tips.append("Maintain eye contact with the camera to project confidence.")
    if not face.get("head_stable"): tips.append("Keep your head steady — avoid excessive movement.")
    if kw["missed"]: tips.append(f"Try weaving in concepts like: {', '.join(kw['missed'][:3])}.")
    if len(words) < 40: tips.append("Expand your answer — aim for at least 60–80 words per response.")
    if not tips: tips.append("Great answer! Clear, confident, and well-structured.")

    return {
        "composite":       composite,
        "content_score":   round(content_score, 1),
        "fluency_score":   round(fluency_score, 1),
        "eye_score":       eye_score,
        "posture_score":   posture_score,
        "engagement_score": engagement_score,
        "wpm":             wpm,
        "word_count":      len(words),
        "keywords_found":  kw["found"],
        "keywords_missed": kw["missed"],
        "tips":            tips,
    }


# ═════════════════════════════════════════════════════════════════════════════
#  Routes
# ═════════════════════════════════════════════════════════════════════════════

@chat_bp.route("/start", methods=["POST"])
def start_session():
    """
    POST { "category": "behavioral" | "technical" | "situational" | "motivational",
           "num_questions": 5 }
    Returns session_id + first question.
    """
    data     = request.get_json(force=True)
    category = data.get("category", "behavioral")
    n        = int(data.get("num_questions", 5))
    questions = QUESTION_BANK.get(category, QUESTION_BANK["behavioral"])
    selected  = questions[:min(n, len(questions))]

    session_id = str(uuid.uuid4())
    session_data = {
        "id":        session_id,
        "category":  category,
        "questions": selected,
        "index":     0,
        "answers":   [],
        "started":   time.time(),
    }
    # store in server-side file (stateless approach – works without Flask sessions)
    _save_session(session_id, session_data)

    return jsonify({
        "session_id":      session_id,
        "total_questions": len(selected),
        "question_index":  0,
        "question":        selected[0],
        "category":        category,
    })


@chat_bp.route("/answer", methods=["POST"])
def submit_answer():
    """
    POST {
      "session_id": "...",
      "audio_b64":  "<base64 webm>",       # from MediaRecorder
      "frame_b64":  "<base64 jpeg>",       # snapshot from getUserMedia
      "duration_s": 42.1
    }
    Returns transcript, scores, tips, and next question (or final report).
    """
    data       = request.get_json(force=True)
    session_id = data.get("session_id")
    sess       = _load_session(session_id)
    if not sess:
        return jsonify({"error": "Session not found"}), 404

    idx      = sess["index"]
    question = sess["questions"][idx]
    category = sess["category"]

    # --- transcribe ---------------------------------------------------------
    transcript = _transcribe_audio(data.get("audio_b64", ""))

    # --- face analysis ------------------------------------------------------
    face = {
    "detected": True,
    "eye_contact": True,
    "mouth_open": True,
    "head_stable": True
}

    # --- score --------------------------------------------------------------
    duration = float(data.get("duration_s", 30))
    scores   = _score_answer(transcript, face, duration, category)

    # --- persist answer -----------------------------------------------------
    sess["answers"].append({
        "question":   question,
        "transcript": transcript,
        "scores":     scores,
        "face":       face,
        "duration_s": duration,
    })
    sess["index"] += 1
    _save_session(session_id, sess)

    # --- determine next step ------------------------------------------------
    if sess["index"] < len(sess["questions"]):
        next_q = sess["questions"][sess["index"]]
        return jsonify({
            "status":         "next",
            "question_index": sess["index"],
            "question":       next_q,
            "transcript":     transcript,
            "scores":         scores,
        })
    else:
        report = _build_final_report(sess)
        _delete_session(session_id)
        return jsonify({
            "status":  "complete",
            "report":  report,
            "transcript": transcript,
            "scores":  scores,
        })


@chat_bp.route("/report/<session_id>", methods=["GET"])
def get_report(session_id):
    sess = _load_session(session_id)
    if not sess:
        return jsonify({"error": "Session not found"}), 404
    return jsonify(_build_final_report(sess))


# ── session persistence (flat-file, no DB required) ──────────────────────────
SESSIONS_DIR = os.path.join(tempfile.gettempdir(), "interviewiq_sessions")
os.makedirs(SESSIONS_DIR, exist_ok=True)

def _save_session(sid, data):
    path = os.path.join(SESSIONS_DIR, f"{sid}.json")
    with open(path, "w") as f:
        json.dump(data, f)

def _load_session(sid):
    if not sid: return None
    path = os.path.join(SESSIONS_DIR, f"{sid}.json")
    if not os.path.exists(path): return None
    with open(path) as f:
        return json.load(f)

def _delete_session(sid):
    path = os.path.join(SESSIONS_DIR, f"{sid}.json")
    if os.path.exists(path): os.unlink(path)


# ── final report builder ─────────────────────────────────────────────────────
def _build_final_report(sess: dict) -> dict:
    answers = sess["answers"]
    if not answers:
        return {"error": "No answers recorded"}

    def avg(key):
        vals = [a["scores"].get(key, 0) for a in answers]
        return round(sum(vals) / len(vals), 1)

    overall = avg("composite")
    report  = {
        "overall_score":    overall,
        "content_score":    avg("content_score"),
        "fluency_score":    avg("fluency_score"),
        "eye_contact_score":avg("eye_score"),
        "posture_score":    avg("posture_score"),
        "engagement_score": avg("engagement_score"),
        "avg_wpm":          avg("wpm"),
        "total_questions":  len(answers),
        "category":         sess["category"],
        "duration_min":     round((time.time() - sess["started"]) / 60, 1),
        "answers": [
            {
                "question":  a["question"],
                "transcript":a["transcript"],
                "score":     a["scores"]["composite"],
                "tips":      a["scores"]["tips"],
            }
            for a in answers
        ],
        "top_tips": _aggregate_tips(answers),
        "grade":    _grade(overall),
    }
    return report


def _aggregate_tips(answers):
    freq = {}
    for a in answers:
        for t in a["scores"]["tips"]:
            freq[t] = freq.get(t, 0) + 1
    sorted_tips = sorted(freq, key=freq.get, reverse=True)
    return sorted_tips[:5]


def _grade(score):
    if score >= 85: return "A"
    if score >= 75: return "B"
    if score >= 65: return "C"
    if score >= 50: return "D"
    return "F"

# ── NLP setup ────────────────────────────────────────────────────────────────
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    import subprocess, sys
    subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")

nltk.download("stopwords", quiet=True)
nltk.download("punkt", quiet=True)
STOPWORDS = set(stopwords.words("english"))

# ── Whisper model (load once) ─────────────────────────────────────────────────
_whisper_model = None
def get_whisper():
    global _whisper_model
    if _whisper_model is None:
        _whisper_model = whisper.load_model("base")
    return _whisper_model



# ── Question bank ─────────────────────────────────────────────────────────────
QUESTION_BANK = {
    "behavioral": [
        "Tell me about yourself and your professional journey.",
        "Describe a challenging situation you faced at work and how you resolved it.",
        "Give me an example of a time you showed leadership.",
        "Tell me about a time you failed and what you learned from it.",
        "How do you handle tight deadlines and pressure?",
        "Describe a time you worked effectively in a team.",
        "Tell me about a time you had to deal with a difficult colleague.",
        "Give an example of a goal you set and how you achieved it.",
    ],
    "technical": [
        "Walk me through your technical stack and why you chose it.",
        "How do you approach debugging a complex production issue?",
        "Describe your experience with system design and scalability.",
        "How do you ensure code quality in your projects?",
        "Tell me about a technically challenging project you built.",
    ],
    "situational": [
        "If you had to deliver bad news to a client, how would you handle it?",
        "How would you prioritize tasks if everything felt urgent?",
        "What would you do if you disagreed with your manager's decision?",
        "How would you onboard yourself in a new team with no documentation?",
    ],
    "motivational": [
        "Why are you interested in this role?",
        "Where do you see yourself in five years?",
        "What motivates you to perform at your best?",
        "What are your greatest professional strengths?",
        "What is one area you are actively working to improve?",
    ],
}

# ── Keyword banks per question category ──────────────────────────────────────
KEYWORDS = {
    "behavioral":   ["situation","task","action","result","team","challenge","resolved","learned","leadership","impact"],
    "technical":    ["architecture","scalability","performance","debug","algorithm","refactor","deploy","api","testing","trade-off"],
    "situational":  ["prioritize","communicate","stakeholder","decision","impact","negotiate","escalate","solution","timeline","risk"],
    "motivational": ["passion","goal","growth","value","mission","contribute","achieve","align","vision","career"],
}

# ═════════════════════════════════════════════════════════════════════════════
#  Blueprint
# ═════════════════════════════════════════════════════════════════════════════
chat_bp = Blueprint("chat", __name__, url_prefix="/api/chat")


# ── helpers ───────────────────────────────────────────────────────────────────

def _extract_keywords(text: str, category: str) -> dict:
    """Return matched / missed keywords for the given category."""
    doc   = nlp(text.lower())
    words = {t.lemma_ for t in doc if not t.is_stop and t.is_alpha}
    bank  = KEYWORDS.get(category, [])
    found = [k for k in bank if k in words or any(k in w for w in words)]
    missed= [k for k in bank if k not in found]
    return {"found": found, "missed": missed}


def _analyse_face(frame_b64: str) -> dict:
    """Decode a base64 JPEG frame and return face-mesh metrics."""
    try:
        img_data = base64.b64decode(frame_b64)
        arr      = np.frombuffer(img_data, np.uint8)
        frame    = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        with mp_face_mesh.FaceMesh(
            static_image_mode=True, max_num_faces=1,
            refine_landmarks=True, min_detection_confidence=0.5
        ) as face_mesh:
            results = face_mesh.process(rgb)

        if not results.multi_face_landmarks:
            return {"detected": False}

        lms = results.multi_face_landmarks[0].landmark

        # Eye-contact proxy: nose tip vs eye-centre alignment
        nose   = lms[1]
        l_eye  = lms[33]
        r_eye  = lms[263]
        eye_cx = (l_eye.x + r_eye.x) / 2
        offset = abs(nose.x - eye_cx)
        gaze_ok = offset < 0.05

        # Mouth openness → engagement
        top_lip = lms[13].y
        bot_lip = lms[14].y
        mouth_open = (bot_lip - top_lip) > 0.01

        # Head pose proxy via nose-tip z
        head_stable = abs(lms[1].z) < 0.15

        return {
            "detected":    True,
            "eye_contact": gaze_ok,
            "mouth_open":  mouth_open,
            "head_stable": head_stable,
            "raw_offset":  round(offset, 4),
        }
    except Exception as e:
        return {"detected": False, "error": str(e)}


def _transcribe_audio(audio_b64: str) -> str:
    """Decode base64 WebM/OGG audio and transcribe with Whisper."""
    try:
        audio_bytes = base64.b64decode(audio_b64)
        with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name
        result = get_whisper().transcribe(tmp_path, language="en")
        os.unlink(tmp_path)
        return result.get("text", "").strip()
    except Exception as e:
        return f"[transcription error: {e}]"


def _score_answer(transcript: str, face: dict, duration_s: float, category: str) -> dict:
    """Produce a structured score dict for one answer."""
    kw      = _extract_keywords(transcript, category)
    words   = transcript.split()
    wpm     = round(len(words) / max(duration_s / 60, 0.1), 1)

    # Sub-scores (0–100)
    content_score  = min(100, len(kw["found"]) * 12 + min(len(words) * 0.4, 40))
    fluency_score  = max(0, 100 - abs(wpm - 130) * 0.8)          # ideal ~130 wpm
    eye_score      = 85 if face.get("eye_contact") else 45
    posture_score  = 80 if face.get("head_stable") else 50
    engagement_score = 75 if face.get("mouth_open") else 55

    # Weighted composite
    composite = round(
        content_score  * 0.35 +
        fluency_score  * 0.25 +
        eye_score      * 0.20 +
        posture_score  * 0.10 +
        engagement_score * 0.10, 1
    )

    # Feedback generation
    tips = []
    if wpm < 100: tips.append("Speak a little faster — aim for 120–140 words per minute.")
    if wpm > 170: tips.append("Slow down slightly so key points land clearly.")
    if not face.get("eye_contact"): tips.append("Maintain eye contact with the camera to project confidence.")
    if not face.get("head_stable"): tips.append("Keep your head steady — avoid excessive movement.")
    if kw["missed"]: tips.append(f"Try weaving in concepts like: {', '.join(kw['missed'][:3])}.")
    if len(words) < 40: tips.append("Expand your answer — aim for at least 60–80 words per response.")
    if not tips: tips.append("Great answer! Clear, confident, and well-structured.")

    return {
        "composite":       composite,
        "content_score":   round(content_score, 1),
        "fluency_score":   round(fluency_score, 1),
        "eye_score":       eye_score,
        "posture_score":   posture_score,
        "engagement_score": engagement_score,
        "wpm":             wpm,
        "word_count":      len(words),
        "keywords_found":  kw["found"],
        "keywords_missed": kw["missed"],
        "tips":            tips,
    }


# ═════════════════════════════════════════════════════════════════════════════
#  Routes
# ═════════════════════════════════════════════════════════════════════════════

@chat_bp.route("/start", methods=["POST"])
def start_session():
    """
    POST { "category": "behavioral" | "technical" | "situational" | "motivational",
           "num_questions": 5 }
    Returns session_id + first question.
    """
    data     = request.get_json(force=True)
    category = data.get("category", "behavioral")
    n        = int(data.get("num_questions", 5))
    questions = QUESTION_BANK.get(category, QUESTION_BANK["behavioral"])
    selected  = questions[:min(n, len(questions))]

    session_id = str(uuid.uuid4())
    session_data = {
        "id":        session_id,
        "category":  category,
        "questions": selected,
        "index":     0,
        "answers":   [],
        "started":   time.time(),
    }
    # store in server-side file (stateless approach – works without Flask sessions)
    _save_session(session_id, session_data)

    return jsonify({
        "session_id":      session_id,
        "total_questions": len(selected),
        "question_index":  0,
        "question":        selected[0],
        "category":        category,
    })


@chat_bp.route("/answer", methods=["POST"])
def submit_answer():
    """
    POST {
      "session_id": "...",
      "audio_b64":  "<base64 webm>",       # from MediaRecorder
      "frame_b64":  "<base64 jpeg>",       # snapshot from getUserMedia
      "duration_s": 42.1
    }
    Returns transcript, scores, tips, and next question (or final report).
    """
    data       = request.get_json(force=True)
    session_id = data.get("session_id")
    sess       = _load_session(session_id)
    if not sess:
        return jsonify({"error": "Session not found"}), 404

    idx      = sess["index"]
    question = sess["questions"][idx]
    category = sess["category"]

    # --- transcribe ---------------------------------------------------------
    transcript = _transcribe_audio(data.get("audio_b64", ""))

    # --- face analysis ------------------------------------------------------
    face = _analyse_face(data.get("frame_b64", ""))

    # --- score --------------------------------------------------------------
    duration = float(data.get("duration_s", 30))
    scores   = _score_answer(transcript, face, duration, category)

    # --- persist answer -----------------------------------------------------
    sess["answers"].append({
        "question":   question,
        "transcript": transcript,
        "scores":     scores,
        "face":       face,
        "duration_s": duration,
    })
    sess["index"] += 1
    _save_session(session_id, sess)

    # --- determine next step ------------------------------------------------
    if sess["index"] < len(sess["questions"]):
        next_q = sess["questions"][sess["index"]]
        return jsonify({
            "status":         "next",
            "question_index": sess["index"],
            "question":       next_q,
            "transcript":     transcript,
            "scores":         scores,
        })
    else:
        report = _build_final_report(sess)
        _delete_session(session_id)
        return jsonify({
            "status":  "complete",
            "report":  report,
            "transcript": transcript,
            "scores":  scores,
        })


@chat_bp.route("/report/<session_id>", methods=["GET"])
def get_report(session_id):
    sess = _load_session(session_id)
    if not sess:
        return jsonify({"error": "Session not found"}), 404
    return jsonify(_build_final_report(sess))


# ── session persistence (flat-file, no DB required) ──────────────────────────
SESSIONS_DIR = os.path.join(tempfile.gettempdir(), "interviewiq_sessions")
os.makedirs(SESSIONS_DIR, exist_ok=True)

def _save_session(sid, data):
    path = os.path.join(SESSIONS_DIR, f"{sid}.json")
    with open(path, "w") as f:
        json.dump(data, f)

def _load_session(sid):
    if not sid: return None
    path = os.path.join(SESSIONS_DIR, f"{sid}.json")
    if not os.path.exists(path): return None
    with open(path) as f:
        return json.load(f)

def _delete_session(sid):
    path = os.path.join(SESSIONS_DIR, f"{sid}.json")
    if os.path.exists(path): os.unlink(path)


# ── final report builder ─────────────────────────────────────────────────────
def _build_final_report(sess: dict) -> dict:
    answers = sess["answers"]
    if not answers:
        return {"error": "No answers recorded"}

    def avg(key):
        vals = [a["scores"].get(key, 0) for a in answers]
        return round(sum(vals) / len(vals), 1)

    overall = avg("composite")
    report  = {
        "overall_score":    overall,
        "content_score":    avg("content_score"),
        "fluency_score":    avg("fluency_score"),
        "eye_contact_score":avg("eye_score"),
        "posture_score":    avg("posture_score"),
        "engagement_score": avg("engagement_score"),
        "avg_wpm":          avg("wpm"),
        "total_questions":  len(answers),
        "category":         sess["category"],
        "duration_min":     round((time.time() - sess["started"]) / 60, 1),
        "answers": [
            {
                "question":  a["question"],
                "transcript":a["transcript"],
                "score":     a["scores"]["composite"],
                "tips":      a["scores"]["tips"],
            }
            for a in answers
        ],
        "top_tips": _aggregate_tips(answers),
        "grade":    _grade(overall),
    }
    return report


def _aggregate_tips(answers):
    freq = {}
    for a in answers:
        for t in a["scores"]["tips"]:
            freq[t] = freq.get(t, 0) + 1
    sorted_tips = sorted(freq, key=freq.get, reverse=True)
    return sorted_tips[:5]


def _grade(score):
    if score >= 85: return "A"
    if score >= 75: return "B"
    if score >= 65: return "C"
    if score >= 50: return "D"
    return "F"
