---
layout: default
title: Portfolio
---

<section class="hero" id="about">
  <div class="hero-copy">
    <p class="eyebrow">AI/ML Technical Lead - GenAI Solution Architect - Agentic AI & RAG Specialist</p>
    <h1>Designing enterprise <span>AI, Agentic AI, and RAG systems</span> that transform operations.</h1>
    <p class="lead">
      I lead AI and ML delivery across Agentic AI, LLM applications, LangGraph,
      LangChain, Retrieval-Augmented Generation, intelligent document processing,
      OCR + LLM extraction, automation, and cloud-native platforms on Azure and AWS.
    </p>
    <div class="hero-actions">
      <a class="btn primary" href="{{ '/Prasun_Kumar.pdf' | relative_url }}">Download Resume</a>
      <a class="btn" href="{{ '/Prasun_Kumar_AI_ML_Architect_Resume.docx' | relative_url }}">Word Resume</a>
      <a class="btn" href="https://github.com/Prasun0512">GitHub</a>
      <a class="btn" href="https://www.linkedin.com/in/prasun-kumar-1708/">LinkedIn</a>
    </div>
  </div>

  <aside class="hero-card" aria-label="Prasun Kumar profile summary">
    <div class="avatar">
      {% if site.logo %}
        <img src="{{ site.logo | relative_url }}" alt="Prasun Kumar">
      {% else %}
        <span>PK</span>
      {% endif %}
    </div>
    <h2>Prasun Kumar</h2>
    <p>AI/ML Engineer | Technical Lead | GenAI & Agentic AI Solution Architect</p>
    <div class="hero-card-meta">
      <span>Pune, India</span>
      <span>8.5+ years experience</span>
      <span>Azure OpenAI - LangGraph - RAG - Python</span>
    </div>
  </aside>
</section>

<section class="metric-grid" id="impact" aria-label="Career highlights">
  <article>
    <strong>8.5+ years</strong>
    <span>AI, ML, automation, and enterprise engineering experience</span>
  </article>
  <article>
    <strong>60%+</strong>
    <span>improvement in document extraction accuracy on GenAI automation</span>
  </article>
  <article>
    <strong>77 TB</strong>
    <span>storage optimized through model lifecycle cleanup and governance</span>
  </article>
</section>

## Focus Areas

<div class="capability-grid" markdown="1">

<div class="capability-card" markdown="1">
### GenAI and RAG Architecture
Azure OpenAI, GPT-4, Llama/open-source LLMs, Azure AI Search, vector search,
hybrid retrieval, chunking, semantic indexing, metadata filters, answer
grounding, and retrieval evaluation.
</div>

<div class="capability-card" markdown="1">
### Agentic AI and Multi-Agent Workflows
LangGraph, LangChain, tool routing, graph-style orchestration, memory-aware
agents, approval gates, enterprise system actions, and audit-ready automation.
</div>

<div class="capability-card" markdown="1">
### Intelligent Automation
Event-driven workflows with Azure Functions, Service Bus, Blob Storage,
Microsoft Graph API, TrackOps integrations, retry handling, audit trails, and
human-in-the-loop validation.
</div>

<div class="capability-card" markdown="1">
### File and Document Intelligence
OCR, Azure AI Document Intelligence, layout extraction, LLM-based information
extraction, confidence scoring, PII/PHI masking, and validation workflows.
</div>

<div class="capability-card" markdown="1">
### LLMOps and MLOps
QLoRA fine-tuning, adapter merging, model benchmarking, Databricks experiments,
MLflow-style evaluation, monitoring dashboards, and production governance.
</div>

<div class="capability-card" markdown="1">
### Technical Leadership
Architecture reviews, client workshops, cross-functional delivery planning,
engineering standards, mentoring, and delivery ownership across AI teams.
</div>

</div>

<h2 id="featured-enterprise-ai-work">Featured Enterprise AI Work</h2>

<p class="section-note">
  Sanitized architecture notes and starter implementation patterns for the
  private/internal projects are available in the
  <a href="https://github.com/Prasun0512/enterprise-ai-ml-case-studies">Enterprise AI/ML Case Studies repository</a>.
</p>

<div class="projects" markdown="1">

<div class="project-card featured" markdown="1">
### GenAI Email-to-Case Automation Platform
**Role:** Technical Lead / AI Solution Architect

Architected an event-driven Azure platform that converts inbound emails and
unstructured documents into structured case records. The solution used Azure
OpenAI, GPT-4, Azure AI Document Intelligence, Microsoft Graph API, Service Bus,
Blob Storage, MySQL, and TrackOps APIs.

**Impact:** Improved extraction accuracy by more than 60% and reduced manual
case-processing effort.

**Source:** Client/internal implementation. Sanitized architecture summary only.
</div>

<div class="project-card featured" markdown="1">
### Requirements Discovery Agent
**Role:** Technical Lead - Generative AI and Knowledge Discovery

Designed a RAG platform for conversational access to organizational knowledge.
Built ingestion patterns for Azure Blob Storage, Document Intelligence, Azure AI
Search, vector indexing, metadata enrichment, and hybrid retrieval.

**Impact:** Improved answer relevance through retrieval benchmarking, chunk
optimization, and hallucination reduction practices.

**Source:** Enterprise architecture summary. Source code is not public.
</div>

<div class="project-card featured" markdown="1">
### Behavioral Intelligence Platform
**Role:** Technical Lead

Led training, evaluation, and deployment of behavior scoring models using Llama,
Gemma, and Qwen architectures. Implemented QLoRA fine-tuning, adapter merging,
dynamic multi-adapter serving, and Databricks-based evaluation dashboards.

**Impact:** Reduced storage footprint by more than 77 TB through lifecycle
management and cleanup processes.

**Source:** Private/internal implementation. Sanitized summary only.
</div>

<div class="project-card featured" markdown="1">
### AI-Powered Learning Content Generator
**Role:** Technical Lead - Generative AI for Learning

Built a GenAI platform that creates instructional content, quizzes, and
assessments tailored to learning objectives. Used RAG pipelines and
human-in-the-loop quality review to keep generated learning content grounded.

**Impact:** Reduced course-development time and enabled scalable personalized
learning content delivery.

**Source:** Client/internal implementation. Sanitized summary only.
</div>

<div class="project-card featured" markdown="1">
### MLflow MLOps Platform
**Role:** AI Platform / MLOps Architecture

Built a portfolio-safe MLOps reference implementation for experiment tracking,
evaluation gates, model registry decisions, deployment readiness, and monitoring
handoffs. The demo uses dependency-light local tracking while mapping cleanly to
MLflow Tracking and Model Registry patterns.

**Source:** Public runnable demo repository.
</div>

<div class="project-card featured" markdown="1">
### Virtual HR Assistant and Candidate Screening Agent
**Role:** Technical Lead - Conversational AI and NLP

Designed an AI assistant for HR inquiries, routine task automation, and
candidate screening. Integrated LLMs with enterprise HR systems and used NLP
workflows to extract candidate signals from resumes and assessments.

**Impact:** Improved HR response consistency and candidate experience.

**Source:** Client/internal implementation. Sanitized summary only.
</div>

<div class="project-card featured" markdown="1">
### Multi-Agent Workflow Automation Platform
**Role:** Technical Lead - AI Agents and Orchestration

Developed multi-agent orchestration patterns for HR, CRM, and support workflows,
including dynamic tool selection, memory retrieval, reasoning, and enterprise
system updates.

**Impact:** Reduced manual intervention across multi-step operational workflows.

**Source:** Internal architecture summary. Source code is not public.
</div>

</div>

## Additional ML/AI Project Portfolio

<div class="projects" markdown="1">

<div class="project-card lab" markdown="1">
### Behavior Scoring Engine
Multi-label email classification pipeline for 80+ behavioral signals using
SentenceTransformers, per-behavior classifiers, ANN retrieval, class weights,
threshold calibration, and LLM fallback for low-confidence predictions.

<p class="project-meta">
  <span>NLP</span>
  <span>LLM Evaluation</span>
  <span>Classification</span>
</p>
</div>

<div class="project-card lab" markdown="1">
### Medical Insurance Claims Automation
Document AI pipeline for extracting claim information from PDFs, scanned
documents, and images while applying PHI/PII masking before downstream LLM
processing.

<p class="project-meta">
  <span>NER</span>
  <span>PHI Masking</span>
  <span>Document AI</span>
</p>
</div>

<div class="project-card lab" markdown="1">
### JD-Resume Parser and ATS Matcher
Resume-to-job matching system that extracts skills, tools, roles, and
experience, normalizes synonyms, and combines keyword matching with embedding
similarity for explainable scoring.

<p class="project-meta">
  <span>Embeddings</span>
  <span>Search</span>
  <span>ATS</span>
</p>
</div>

<div class="project-card lab" markdown="1">
### Explainable Behavior and Toxicity Highlighting
Auditable NLP workflow that highlights evidence spans and sentence-level signals
behind behavior and toxicity predictions for faster human review and governance.

<p class="project-meta">
  <span>Explainable AI</span>
  <span>NLP</span>
  <span>Governance</span>
</p>
</div>

<div class="project-card lab" markdown="1">
### Smart Consultation Recording Trigger
Computer vision trigger that detects stethoscope presence using object detection
and applies smoothing, confidence thresholds, and stable on/off logic to reduce
false recording events.

<p class="project-meta">
  <span>YOLO</span>
  <span>Computer Vision</span>
  <span>Automation</span>
</p>
</div>

<div class="project-card lab" markdown="1">
### Face Search and Attribute Analytics
DeepFace-based search workflow for finding all images of a person in a large
dataset, extracting attributes, and exporting matched outputs for review.

<p class="project-meta">
  <span>DeepFace</span>
  <span>Similarity Search</span>
  <span>Analytics</span>
</p>
</div>

<div class="project-card lab" markdown="1">
### AI Coach for Learning Pathways
Personalized learning assistant that combines learner signals, content
recommendations, and grounded RAG Q&A to help users identify gaps and choose
next-best learning content.

<p class="project-meta">
  <span>RAG</span>
  <span>Recommendation</span>
  <span>EdTech</span>
</p>
</div>

<div class="project-card lab" markdown="1">
### Prompt and Retrieval Quality Evaluation Suite
Evaluation practice for comparing few-shot prompts, checklist prompts, retrieval
settings, latency, answer relevance, hallucination risk, and confidence-based
fallback strategies.

<p class="project-meta">
  <span>LLMOps</span>
  <span>Evaluation</span>
  <span>RAG Quality</span>
</p>
</div>

</div>

## Public AI Architecture Repositories

<div class="projects public-projects" markdown="1">

<div class="project-card" markdown="1">
### Enterprise AI/ML Case Studies
Sanitized architecture notes and starter implementation patterns for GenAI,
RAG, document intelligence, agentic AI, NLP, and computer-vision work.

[View Repository](https://github.com/Prasun0512/enterprise-ai-ml-case-studies){: .btn }
</div>

<div class="project-card" markdown="1">
### Agentic AI LangGraph Workflows
Multi-agent workflow system with planner, research, retrieval, validation,
execution, approval gates, retry logic, and audit traces.

[View Repository](https://github.com/Prasun0512/agentic-ai-langgraph-workflows){: .btn }
</div>

<div class="project-card" markdown="1">
### Enterprise RAG Quality Lab
RAG quality project demonstrating redaction, chunking, retrieval metrics,
confidence routing, grounded answer checks, demo data, and tests.

[View Repository](https://github.com/Prasun0512/enterprise-rag-quality-lab){: .btn }
</div>

<div class="project-card" markdown="1">
### Document Intelligence LLM Pipeline
OCR + LLM extraction pipeline with PII redaction, schema validation, confidence
scoring, exception routing, and human review.

[View Repository](https://github.com/Prasun0512/document-intelligence-llm-pipeline){: .btn }
</div>

<div class="project-card" markdown="1">
### Email-to-Case GenAI Automation
Event-driven GenAI automation pattern for email ingestion, attachment handling,
case payload creation, retries, dead-letter handling, and validation.

[View Repository](https://github.com/Prasun0512/email-to-case-genai-automation){: .btn }
</div>

<div class="project-card" markdown="1">
### AI Resume Job Matcher
Resume/JD matching assistant with skill extraction, semantic scoring, missing
skill analysis, and explainable suitability summaries.

[View Repository](https://github.com/Prasun0512/ai-resume-job-matcher){: .btn }
</div>

<div class="project-card" markdown="1">
### MLflow MLOps AI Platform
Experiment tracking, evaluation gates, model registry decisions, deployment
readiness, and monitoring handoffs using a local MLflow-style demo.

[View Repository](https://github.com/Prasun0512/mlflow-mlops-ai-platform){: .btn }
</div>

<div class="project-card" markdown="1">
### Enterprise AI Architecture Showcase
Architecture decision records, HLDs, LLDs, sequence diagrams, deployment
patterns, security controls, and multi-tenant AI reference architectures.

[View Repository](https://github.com/Prasun0512/enterprise-ai-architecture-showcase){: .btn }
</div>

<div class="project-card" markdown="1">
### Enterprise LLM Evaluation Framework
Evaluation framework for faithfulness, groundedness, recall, relevance,
regression gates, and release-readiness checks for LLM and RAG systems.

[View Repository](https://github.com/Prasun0512/enterprise-llm-evaluation-framework){: .btn }
</div>

<div class="project-card" markdown="1">
### Enterprise AI Platform Engineering
Model gateway, prompt registry, guardrails, caching, routing, evaluation hooks,
and cost-control patterns for platform-style AI delivery.

[View Repository](https://github.com/Prasun0512/enterprise-ai-platform-engineering){: .btn }
</div>

<div class="project-card" markdown="1">
### AI Reliability Engineering
Retry, circuit breaker, fallback, poison queue, runbook, and postmortem
patterns for operating AI systems reliably.

[View Repository](https://github.com/Prasun0512/ai-reliability-engineering){: .btn }
</div>

<div class="project-card" markdown="1">
### Production AI Operations
Deployment, rollback, canary, embedding refresh, index rebuild, monitoring,
alerting, and cost-management playbooks.

[View Repository](https://github.com/Prasun0512/production-ai-operations){: .btn }
</div>

<div class="project-card" markdown="1">
### Multi-Tenant AI Platform
Tenant isolation, RBAC, feature flags, model/vector routing, usage metering,
and cost tracking for SaaS-style AI platforms.

[View Repository](https://github.com/Prasun0512/multi-tenant-ai-platform){: .btn }
</div>

<div class="project-card" markdown="1">
### Enterprise Integration Patterns
Webhooks, REST integrations, async messaging, retries, idempotency, DLQs, and
audit logging for enterprise workflow automation.

[View Repository](https://github.com/Prasun0512/enterprise-integration-patterns){: .btn }
</div>

<div class="project-card" markdown="1">
### AI Technical Lead Playbook
AI delivery lifecycle, design reviews, risk management, code review standards,
and architecture review practices for AI teams.

[View Repository](https://github.com/Prasun0512/ai-technical-lead-playbook){: .btn }
</div>

<div class="project-card" markdown="1">
### Engineering Leadership
Architecture reviews, technical decisions, mentoring, delivery planning,
stakeholder communication, and team-scaling practices.

[View Repository](https://github.com/Prasun0512/engineering-leadership){: .btn }
</div>

<div class="project-card" markdown="1">
### Restaurant AI Chatbot
Legacy NLP assistant for recipe and food-domain Q&A. This repo is kept as a
foundational AI project; rename recommended to `restaurant-ai-chatbot`.

[View Repository](https://github.com/Prasun0512/ResturantChatbot){: .btn }
</div>

</div>

## Foundational ML Projects

Earlier machine-learning and computer-vision work includes emotion detection,
medical image classification experiments, gesture recognition, credit-risk
analysis, fraud detection, and regression modeling. These are now treated as
foundational ML projects rather than the headline portfolio because the current
GitHub profile is focused on enterprise GenAI, Agentic AI, RAG, LLMOps, and
production-ready AI architecture.

## Architecture Showcase

<figure class="architecture-diagram">
  <img
    src="https://raw.githubusercontent.com/Prasun0512/Resume/master/assets/img/enterprise-ai-architecture-flow.png"
    alt="Enterprise AI architecture flow from workflow intake to queue processing, AI workflow, validation gates, business update or human review, and audit monitoring"
  >
</figure>

My architecture work emphasizes event-driven design, idempotency, DLQs, human
approval gates, retrieval quality, AI safety controls, observability, and
business-system integration.

## Leadership and Mentorship

I operate as a technical lead across AI delivery: architecture reviews, client
workshops, solution design, delivery planning, mentoring, review standards,
stakeholder communication, and production-readiness gates.

## Awards and Recognition

Recognized consistently between 2017 and 2026 for technical excellence,
innovation, leadership, ownership, collaboration, and delivery impact.

<p class="skill-cloud">
  <span>3x Quarterly Team Award Winner</span>
  <span>2x Superstar Award Winner</span>
  <span>4x Technical Star Recognition</span>
  <span>We-NOT-Me</span>
  <span>Going-the-extra-mile</span>
  <span>Thank You Recognition</span>
  <span>FunStar</span>
  <span>PEP Silver</span>
  <span>Hackathon Recognition - 2017</span>
</p>

## Certifications

**Post Graduate Program in Machine Learning & Artificial Intelligence**  
Credential: [Credential.net verification](https://www.credential.net/20453200-981b-4985-9ddf-2a44c5ee66cf)

Covered ML, deep learning, NLP, reinforcement learning, production AI concepts,
and MLOps foundations.

## Thought Leadership

I write and share practical AI engineering perspectives around Agentic AI,
production AI agents, GenAI architecture, RAG systems, LLM reliability, AI
governance, and emerging AI trends.

[Follow my LinkedIn posts](https://www.linkedin.com/in/prasun-kumar-1708/){: .btn }

## Open Source Contributions

My public GitHub focuses on sanitized, interview-safe AI architecture examples:
agentic workflows, RAG quality, document intelligence, email-to-case automation,
MLOps patterns, reliability, platform engineering, and leadership playbooks.

<h2 id="experience">Experience</h2>

<div class="timeline" markdown="1">

<div markdown="1">
**Harbinger Group** - Technical Lead, AI and ML<br>
Aug 2024 - Present
</div>

<div markdown="1">
**Harbinger Group** - Senior Software Engineer<br>
Apr 2021 - Jul 2024
</div>

<div markdown="1">
**Extentia Information Technology** - Software Engineer<br>
Nov 2019 - Apr 2021
</div>

<div markdown="1">
**Symantec Software India Pvt Ltd** - Associate IT Applications Specialist<br>
Jul 2017 - Oct 2019
</div>

</div>

## Core Stack

<p class="skill-cloud">
  <span>Azure OpenAI</span>
  <span>GPT-4</span>
  <span>LangGraph</span>
  <span>LangChain</span>
  <span>Agentic AI</span>
  <span>RAG</span>
  <span>Llama</span>
  <span>Azure AI Search</span>
  <span>Azure AI Document Intelligence</span>
  <span>OCR + LLM Pipelines</span>
  <span>Vector Search</span>
  <span>AI Agents</span>
  <span>Python</span>
  <span>Databricks</span>
  <span>QLoRA</span>
  <span>Gemma</span>
  <span>Qwen</span>
  <span>MLflow</span>
  <span>TensorFlow</span>
  <span>PyTorch</span>
  <span>Docker</span>
  <span>Kubernetes</span>
  <span>AWS</span>
  <span>Azure Functions</span>
  <span>Service Bus</span>
  <span>Microsoft Graph API</span>
</p>

<h2 id="contact">Contact</h2>

I am based in Pune, India and open to Technical Lead, AI/ML Architect, and
Generative AI Solution Architect roles.

[Download Resume]({{ '/Prasun_Kumar.pdf' | relative_url }}){: .btn .primary }
[Word Resume]({{ '/Prasun_Kumar_AI_ML_Architect_Resume.docx' | relative_url }}){: .btn }
[LinkedIn](https://www.linkedin.com/in/prasun-kumar-1708/){: .btn }
[GitHub](https://github.com/Prasun0512){: .btn }
[YouTube](https://www.youtube.com/@AIWizardry277){: .btn }
