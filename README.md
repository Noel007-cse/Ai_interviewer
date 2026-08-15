https://interviewgenius.streamlit.app/
released with errors

<div align="center">

<img src="https://img.shields.io/badge/Python-Core-3776AB?style=for-the-badge&logo=python&logoColor=white" />
<img src="https://img.shields.io/badge/OpenCV-Facial%20Tracking-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white" />
<img src="https://img.shields.io/badge/MediaPipe-Basics-00A98F?style=for-the-badge" />
<img src="https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge" />

<br /><br />

# 🎤 AI Interviewer

### *Practice interviews. Get real-time feedback. Walk in ready.*

**A mock interview system that combines AI-driven questioning with real-time facial tracking to help candidates practice and improve their interview performance.**

[Features](#-features) • [Tech Stack](#-tech-stack) • [Getting Started](#-getting-started) • [Architecture](#-architecture) • [Roadmap](#-roadmap)

</div>

---

## 🌱 About the Project

AI Interviewer simulates a real interview experience — asking questions and tracking the candidate's facial expressions and engagement in real time. It's built as a mini project to explore how computer vision and AI can be combined to give candidates actionable, judgment-free feedback on their interview presence.

---

## ✨ Features

- **Mock interview flow** — AI-driven interview questions delivered in sequence
- **Real-time facial tracking** — monitors expressions, eye contact, and engagement cues during the session via webcam
- **Live feedback signals** — visual indicators of tracked facial metrics as the interview progresses
- **Session-based practice** — run through a full mock interview from start to finish

---

## 🛠 Tech Stack

| Layer | Technology |
|-------|------------|
| **Core Language** | Python |
| **Computer Vision** | OpenCV |
| **Facial Landmark Detection** | MediaPipe |


## 🚀 Getting Started

### Prerequisites

- Python ≥ 3.9
- A webcam-enabled device
- pip

### 1. Clone the repository

```bash
git clone https://github.com/Noel007-cse/ai-interviewer.git
cd ai-interviewer
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Set up environment variables (if using an AI API for questions)

Create a `.env` file in the project root:

```env
AI_API_KEY=your_api_key_here
```

> ⚠️ Never commit your `.env` file — add it to `.gitignore`.

### 4. Run the app

```bash
python main.py
```

---

## 🏗 Architecture

```
ai-interviewer/
├── main.py                  # Entry point — launches interview session
├── facial_tracking/         # OpenCV / MediaPipe facial detection & metrics
├── interview/                # Question flow, AI prompt logic
├── utils/                    # Helper functions
└── requirements.txt
```

> Update this tree to match your actual folder structure.

---

## 🗺 Roadmap

- [ ] Voice/speech analysis (tone, pace, filler words)
- [ ] Post-interview performance report
- [ ] Role-specific question sets (technical, HR, behavioral)
- [ ] Web-based interface for browser access
- [ ] Session history and progress tracking

---

## 🤝 Contributing

Contributions are welcome! Fork the repo, create a feature branch, and open a pull request.

```bash
git checkout -b feature/your-feature-name
git commit -m "feat: describe your change"
git push origin feature/your-feature-name
```

---

## 📄 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

<div align="center">

**Built to make interview practice smarter and a little less nerve-wracking.**

</div>
