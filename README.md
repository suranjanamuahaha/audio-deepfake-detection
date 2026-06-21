# Audio Deepfake Detection System for Voice Fraud Prevention

## Overview

This project presents a real-time AI-powered audio forensics system designed to detect deepfake voice attacks in financial communication systems such as banking IVR and customer support calls.

Developed as part of **PSB Hackathon 2026 in collaboration with IIT Kharagpur**, this system addresses the rapidly growing threat of AI-generated voice fraud by analyzing incoming audio streams and identifying synthetic speech patterns with high precision.

The system integrates deep learning-based speech representation models with a scalable backend pipeline to provide proactive fraud detection before financial damage occurs.

Prototype: [https://audio-deepfake-detection-pink.vercel.app/](https://audio-deepfake-detection-pink.vercel.app/)

---

## Problem Statement

Voice-based fraud has escalated significantly due to advancements in generative AI and voice cloning technologies.

### Key Challenges

* AI-generated voices are nearly indistinguishable from real human speech
* Attackers can bypass traditional authentication (OTP, PIN, KYC)
* Fraud occurs in real-time with minimal traceability
* Existing systems are reactive rather than preventive

As shown in the hackathon documentation, attackers can:

* Clone voices from social media or call recordings
* Impersonate customers or officials
* Spoof caller identity metadata
* Execute fraudulent transactions within minutes 

---

## Proposed Solution

We propose a **real-time deepfake voice detection pipeline** that operates on incoming call audio and flags suspicious activity within seconds.

### Core Capabilities

* Real-time audio stream processing
* Deepfake detection using transformer-based speech embeddings
* Risk scoring and decision engine
* Fraud alert system for immediate action
* Feedback loop for continuous model improvement

### System Workflow

1. Audio Input (live call / recorded input)
2. Preprocessing (noise reduction, segmentation)
3. Feature Extraction (speech embeddings)
4. Deepfake Detection Model
5. Risk Scoring
6. Decision Layer (Safe / Fraud)
7. Alert + Logging System

This pipeline is illustrated in the system architecture diagram on page 3 of the document 

---

## Dataset

### Source

* Fake-or-Real (FoR) Audio Dataset (Kaggle)

### Dataset Details

* Task: Binary Classification (Real vs Fake)
* Labels:

  * 0 → Real Audio
  * 1 → Fake Audio
* Input Type: Short-duration audio clips
* Sampling Rate: 16 kHz

### Dataset Size

* Training Samples: ~5100
* Testing Samples: ~400
* Distribution: Balanced dataset 

---

## Model Architecture

### Feature Extractor

* **Wav2Vec2 (facebook/wav2vec2-base)**
* Extracts 768-dimensional contextual speech embeddings

### Processing Pipeline

* Raw audio converted to float32 waveform
* Processed using Wav2Vec2Processor
* Generates:

  * `input_values`
  * `attention_mask`
* Mean pooling applied across temporal features

### Classifier

* Custom MLP-based Deepfake Detector
* Outputs probability score (Real vs Fake)

### Why Wav2Vec2?

* Learns representations directly from raw audio
* Captures:

  * Temporal dependencies
  * Frequency patterns
  * Phonetic structures
* Highly sensitive to synthetic artifacts
* Robust to noise and variations 

---

## Model Performance

### Metrics

* Accuracy: ~82.11%
* ROC AUC: ~0.985
* Strong precision-recall performance

### Confusion Matrix Insights

* Real correctly classified: 131
* Real misclassified as fake: 73
* Fake correctly classified: 204
* Fake misclassified: 0

This indicates a **zero-miss policy for fraud detection**, prioritizing security over minor false positives.

Evaluation plots (ROC curve, precision-recall curve, probability distributions) are shown in pages 8–10 

---

## System Features

### Real-Time Detection

* Detects deepfake audio within **10 seconds**
* Designed for integration with banking IVR and call systems

### Fraud Alert System

* Instantly flags suspicious audio
* Generates warning for users and systems
* Prevents unauthorized transactions

### Audio Analysis

* Spectrogram visualization
* Confidence scoring
* Behavioral anomaly detection

Example outputs showing authentic vs deepfake audio are illustrated on page 14 

---

## Tech Stack

### Frontend

* React.js
* Tailwind CSS

### Backend

* FastAPI
* REST API architecture

### Machine Learning

* PyTorch
* HuggingFace Transformers (Wav2Vec2)
* Librosa (audio processing)

### Deployment

* Backend: Render
* Frontend: Vercel
* GPU Server support (planned)

### Audio Capture

* MediaRecorder API (browser-based recording) 

---

## System Architecture

### Backend Pipeline

* Audio Input → Preprocessing → Feature Extraction → Model Inference → Risk Scoring → Decision

### Key Components

* Audio preprocessing (resampling, normalization, segmentation)
* Wav2Vec2 encoder
* Binary classifier
* Threshold-based decision system

As shown in the backend model pipeline diagram (page 4) 

---

## Current Limitations

1. **False Positives Trade-off**

   * Tuned for 100% fraud recall
   * May flag some real calls as suspicious

2. **Fixed Window Analysis**

   * Only initial audio segment analyzed
   * Does not yet track entire conversation

These design choices prioritize security over user convenience 

---

## Future Scope

* Continuous sliding window detection across full calls
* Detection of partially synthetic conversations
* API hardening (rate limiting, authentication, validation)
* Optimization for real-time telecom deployment
* Fine-tuning on regional Indian languages and accents
* GPU-based low-latency inference

---

## Comparison with Existing Systems

### Current Banking Systems

* OTP / PIN-based authentication
* Reactive fraud detection
* No real-time voice validation

### Our System

* Proactive fraud prevention
* Real-time deepfake detection
* AI-driven voice analysis
* Seamless integration into IVR and call centers 

---

## Demo

The current implementation uses a **recording interface** to simulate live call audio due to deployment constraints.

In production:

* System runs directly on live call streams
* Provides detection within 10 seconds

Interface preview available in the prototype

---

## Team

* Pushpam Kumari
* Suranjana Paul
* Anant Krishna Tiwari
* Anusha Gupta

---

## Conclusion

This project demonstrates a practical and scalable approach to combating AI-driven voice fraud using deep learning and real-time audio processing. By integrating advanced speech representation models with a robust backend pipeline, the system enables proactive fraud prevention and strengthens trust in digital banking systems.

