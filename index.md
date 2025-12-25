---
layout: default
title: Portfolio
---

## 🚀 AI/ML + GenAI Portfolio (Lead/Architect)

A showcase of my work across **Generative AI, NLP, LLMOps/MLOps, Computer Vision, and production automation**.  
Most projects are **production-minded**, focused on **quality, safety (PHI/PII), monitoring, and measurable outcomes**.

---

## 🧠 What I Build

- **RAG & Knowledge Systems**: hybrid retrieval, grounding strategies, eval-driven improvements  
- **LLMOps / MLOps**: MLflow tracking, model registry/versioning, evaluation suites, monitoring dashboards  
- **Applied NLP**: multi-label classification, NER, document intelligence, explainability + evidence highlighting  
- **Secure AI Pipelines**: PHI/PII masking, compliance-aware artifacts, audit-ready outputs  
- **Computer Vision**: real-time detection pipelines, gesture systems, event triggers  

---

## ⚙️ Tech Stack

<div class="badges" markdown="1">

[![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-FFD21E?logo=huggingface&logoColor=black)](https://huggingface.co/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![SentenceTransformers](https://img.shields.io/badge/SentenceTransformers-111827?logo=semanticweb&logoColor=white)](https://www.sbert.net/)
[![LangChain](https://img.shields.io/badge/LangChain-0F172A?logo=chainlink&logoColor=white)](https://www.langchain.com/)
[![MLflow](https://img.shields.io/badge/MLflow-1D4ED8?logo=mlflow&logoColor=white)](https://mlflow.org/)
[![Databricks](https://img.shields.io/badge/Databricks-FF3621?logo=databricks&logoColor=white)](https://www.databricks.com/)
[![FAISS](https://img.shields.io/badge/FAISS-111827?logo=vectorworks&logoColor=white)](https://github.com/facebookresearch/faiss)
[![Pinecone](https://img.shields.io/badge/Pinecone-0EA5E9?logo=pinecone&logoColor=white)](https://www.pinecone.io/)
[![Elasticsearch](https://img.shields.io/badge/Elasticsearch-005571?logo=elasticsearch&logoColor=white)](https://www.elastic.co/elasticsearch/)
[![spaCy](https://img.shields.io/badge/spaCy-09A3D5?logo=spacy&logoColor=white)](https://spacy.io/)
[![NLTK](https://img.shields.io/badge/NLTK-111827?logo=readthedocs&logoColor=white)](https://www.nltk.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?logo=opencv&logoColor=white)](https://opencv.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-2496ED?logo=docker&logoColor=white)](https://www.docker.com/)
[![Airflow](https://img.shields.io/badge/Airflow-017CEE?logo=apacheairflow&logoColor=white)](https://airflow.apache.org/)
[![Whisper](https://img.shields.io/badge/Whisper-412991?logo=openai&logoColor=white)](https://openai.com/research/whisper)
[![AWS](https://img.shields.io/badge/AWS-232F3E?logo=amazon-aws&logoColor=FF9900)](https://aws.amazon.com/)
[![Azure](https://img.shields.io/badge/Azure-0078D4?logo=microsoft-azure&logoColor=white)](https://azure.microsoft.com/)
[![GitHub Actions](https://img.shields.io/badge/GitHub%20Actions-2088FF?logo=githubactions&logoColor=white)](https://github.com/features/actions)
[![Jekyll](https://img.shields.io/badge/Jekyll-CC0000?logo=jekyll&logoColor=white)](https://jekyllrb.com/)

</div>

---

## ⭐ Featured GenAI / NLP / LLMOps Projects

<div class="projects" markdown="1">

<div class="project-card" markdown="1">

### 🧩 Behavior Scoring Engine (GAP Pipeline) — Multi-label Email Classification (80+ behaviors)
**Goal:** Predict multiple behaviors in emails with strong evaluation and low-risk fallbacks.  
- Built local ML baselines using **SentenceTransformers (multi-qa-mpnet-base-dot-v1)** + per-behavior classifiers.  
- Improved weak behaviors using **ANN + class weights + early stopping + threshold calibration**.  
- Benchmarked LLM prompt strategies (few-shot, decision checklists, CoT+checklist) for **low-confidence hybrid fallback**.  
- Delivered **per-behavior precision/recall/F1** and reproducible artifacts.

**Repo:** Private / internal (available on request)

</div>

<div class="project-card" markdown="1">

### 📚 RAG Knowledge Assistant (Enterprise)
**Goal:** Build grounded Q&A over enterprise documents with evaluation-driven quality improvements.  
- RAG design with chunking strategies, embeddings, retrieval tuning, and grounded responses.  
- Built evaluation + monitoring patterns (quality, latency, drift signals) and dashboards.

**Repo:** Private / internal (available on request)

</div>

<div class="project-card" markdown="1">

### 🩺 Medical Insurance Claims Automation — PHI Masking + NER + Case Classification
**Goal:** Automate medical claim processing from PDFs while protecting PHI/PII.  
- Document extraction + **PHI masking** pipeline before LLM usage.  
- GPT-based entity extraction and fine-tuned classification for case type/category.

**Repo:** Private / client (available on request)

</div>

<div class="project-card" markdown="1">

### 📄 JD–Resume Parser & ATS Matching System
**Goal:** Match resumes to JDs with explainable scoring.  
- Extracted skills, tools, titles, experience; normalized synonyms for better recall.  
- Combined **keyword + embedding similarity** with “matched vs missing skills” reports.

**Repo:** Private / internal (available on request)

</div>

<div class="project-card" markdown="1">

### 🎓 AI Coach for Learning Platform — Personalized Pathways + RAG Q&A
**Goal:** Help learners with study plans and grounded Q&A.  
- Used learning signals + ML to recommend next-best content and identify gaps.  
- Used RAG to reduce hallucinations and keep answers grounded.

**Repo:** Private / client (available on request)

</div>

</div>

---

## 🤖 NLP + Search Projects (Public)

<div class="projects" markdown="1">

<div class="project-card" markdown="1">

### 🍳 Restaurant Chatbot for Chefs (Doc Q&A + Search)
**Goal:** Recipe assistance chatbot for chefs and food enthusiasts.  
- **BERT embeddings, Elasticsearch, AllenNLP, AWS Lex, AWS Comprehend**.  
- Accurate, context-aware culinary Q&A.

[View on GitHub »](https://github.com/Prasun0512/ResturantChatbot){: .btn }

</div>

</div>

---

## 👁️ Computer Vision Projects (Public)

<div class="projects" markdown="1">

<div class="project-card" markdown="1">

### 🎭 Employee Emotion Detection
**Goal:** Track employee emotions and alert HR when dissatisfaction/stress is detected.  
- Built using **Computer Vision & NLP** for emotion classification.  
- Generates actionable reports for proactive intervention.

[View on GitHub »](https://github.com/Prasun0512/Employee_Emotion_Detection){: .btn }

</div>

<div class="project-card" markdown="1">

### 🩺 Melanoma Detection
**Goal:** Early detection of **melanoma** using dermoscopic images.  
- CNNs with **TensorFlow/Keras**.  
- Achieved **75–84% accuracy**, improving early diagnosis potential.

[View on GitHub »](https://github.com/Prasun0512/Melanoma-Detection-Assignment){: .btn }

</div>

<div class="project-card" markdown="1">

### 🫁 Lung Cancer Detection
**Goal:** Detect lung cancer from chest X-ray/CT images.  
- **TensorFlow, Keras, OpenCV** for training and preprocessing.  
- Trained on **LIDC-IDRI** dataset.

[View on GitHub »](https://github.com/Prasun0512/-Lung-Cancer-Detection/){: .btn }

</div>

<div class="project-card" markdown="1">

### 📺 Smart-TV Gesture Recognition
**Goal:** Control Smart TVs via **hand gestures**, no remote required.  
- Deep learning for **gesture classification**.  
- Better **HCI** experience for smart devices.

[View on GitHub »](https://github.com/Prasun0512/Neural-Networks-Project---Gesture-Recognition){: .btn }

</div>

</div>

---

## 📊 Data Science Projects (Public)

<div class="projects" markdown="1">

<div class="project-card" markdown="1">

### 💳 Risky Loan Applicant Predictor
**Goal:** Identify risky loan applicants to reduce credit loss.  
- **EDA + feature engineering** to find key default drivers.  
- ML models for **risk assessment and portfolio optimization**.

[View on GitHub »](https://github.com/Prasun0512/LendingClubCaseStudy){: .btn }

</div>

</div>

---

## 🔒 Private / Client Projects (Summary Only)

<div class="projects" markdown="1">

<div class="project-card" markdown="1">

### ✅ Explainable Behavior + Toxicity Highlighting for Emails
**Goal:** Make predictions auditable with evidence-level explanations.  
- Highlighted evidence lines/spans to justify model decisions.  
- Toxicity detection with sentence-level highlighting to speed compliance review.

**Repo:** Private / client (available on request)

</div>

<div class="project-card" markdown="1">

### 🎥 Smart Consultation Recording Trigger — YOLO Object Detection
**Goal:** Automatically start/stop recording based on stethoscope presence.  
- Stable ON/OFF logic with smoothing and confidence thresholds to reduce noise.

**Repo:** Private / client (available on request)

</div>

<div class="project-card" markdown="1">

### 🧑‍🤝‍🧑 Face Search & Attribute Analytics — DeepFace
**Goal:** Find all photos of a person in a large dataset and export results.  
- Similarity matching + attribute extraction (emotion/age-style).  
- Automated export pipeline for matched outputs.

**Repo:** Private / internal (available on request)

</div>

</div>

---

## 📄 Resume & Links

[Download Resume]({{ "/Prasun_Kumar.pdf" | relative_url }}){: .btn }
[LinkedIn](https://www.linkedin.com/in/prasun-kumar-1708/){: .btn }
[GitHub](https://github.com/Prasun0512){: .btn }
[YouTube](https://www.youtube.com/@AIWizardry277){: .btn }
