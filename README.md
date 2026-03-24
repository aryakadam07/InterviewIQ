# InterviewIQ: AI-Powered Interview Analytics 🚀

InterviewIQ is a premium, browser-native application designed to help professionals master their interview skills. By combining Computer Vision, NLP, and Machine Learning, the platform provides deep, actionable insights into your performance, delivery, and body language.

## 🌟 Key Features

### 🏛️ Professional Success Toolkit
A comprehensive suite of career resources directly integrated into the dashboard:
- **The STAR Method Guide**: Master behavioral questions with a structured storytelling framework.
- **Salary Negotiation Masterclass**: Strategy-driven advice for handling compensation discussions.
- **Body Language Pro**: Learn non-verbal cues that project confidence and authority.
- **Follow-up Templates**: High-impact thank-you note templates for various professional scenarios.

### 👨‍💼 Career Coach Perspective
Beyond raw data, InterviewIQ provides a virtual coaching layer. Our **Senior Career Advisor** persona analyzes your final scores to give context-aware advice, helping you bridge the gap between "good" and "exceptional."

### 📊 Comprehensive Analytics
- **Visual Intelligence**: Real-time emotion tracking, eye contact analysis, and nervousness detection.
- **Speech Metrics**: Words per minute (WPM), filler word rate, and silence-to-speech ratios.
- **NLP Insights**: Vocabulary richness, communication clarity, and key topic extraction.
- **Interactive Timeline**: Replay your interview with a synchronized emotion and nervousness timeline.

## 🛠️ Technology Stack
- **Backend**: Flask (Python)
- **Frontend**: Custom HTML5, CSS3 (Vanilla), and JavaScript
- **AI/ML Core**:
    - **Whisper**: High-accuracy speech-to-text transcription.
    - **MediaPipe**: Facial landmarking and movement tracking.
    - **scikit-learn**: Random Forest scoring model.
    - **spaCy & NLTK**: Natural language processing and keyword extraction.

## 🚀 Quick Start

### 1. Prerequisites
Ensure you have Python 3.9+ installed and the required dependencies:
```bash
pip install -r requirements.txt
```

### 2. Run the Application
Start the Flask server:
```bash
python main.py
```
Access the dashboard at `http://localhost:5000`.


 Live App: https://interviewiq-2-hip6.onrender.com

## Features

* AI-based interview analysis
* Speech + NLP + CV insights
* Performance scoring


### 3. Usage
- **Analyze**: Upload a video of your mock interview or recorded talk.
- **Review**: Navigate through the interactive dashboard to see your scores.
- **Learn**: Use the **Professional Toolkit** and **Career Coach** sections to refine your approach.

## 📈 Accuracy & Methodology
The scoring system is powered by a proprietary Random Forest model trained on 18 distinct features including facial stability, communication pace, and linguistic variety. Accuracy is currently optimized for professional executive interviews.

---
*Built for professionals who refuse to settle for "average."*
