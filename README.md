# Prasun Kumar - AI/ML Architect Portfolio

This repository powers the GitHub Pages portfolio for Prasun Kumar:

<https://prasun0512.github.io/Resume/>

The site presents a recruiter-friendly portfolio for an AI/ML Engineer,
Technical Lead, and GenAI / Agentic AI Solution Architect with 8.5+ years of
software engineering experience. It focuses on sanitized enterprise AI patterns,
not private client code.

## About

The portfolio highlights practical work across Agentic AI, LangGraph-style
workflows, LLM applications, RAG systems, document intelligence, OCR + LLM
extraction, AI automation, Azure OpenAI, and production-ready AI engineering.

## Featured Projects

- [Enterprise AI/ML Case Studies](https://github.com/Prasun0512/enterprise-ai-ml-case-studies)
- [Agentic AI LangGraph Workflows](https://github.com/Prasun0512/agentic-ai-langgraph-workflows)
- [Document Intelligence LLM Pipeline](https://github.com/Prasun0512/document-intelligence-llm-pipeline)
- [Email-to-Case GenAI Automation](https://github.com/Prasun0512/email-to-case-genai-automation)
- [Enterprise RAG Quality Lab](https://github.com/Prasun0512/enterprise-rag-quality-lab)
- [AI Resume Job Matcher](https://github.com/Prasun0512/ai-resume-job-matcher)

## Architecture Strengths

- Event-driven AI automation with queues, retries, idempotency, and DLQ handling.
- RAG quality evaluation with chunking, retrieval metrics, citations, and review thresholds.
- OCR + LLM extraction with PII masking, schema validation, confidence scoring, and human review.
- Agentic AI workflow design with planner, retriever, validator, executor, tool routing, and audit logs.
- Production readiness: Docker, CI, tests, configuration hygiene, monitoring, and cost controls.

## Technical Leadership

The site is written to support AI architecture and technical-lead interviews:
business problem framing, design tradeoffs, governance, failure modes,
production-readiness notes, and sanitized case-study summaries.

## Resume Download

The live site links to the resume PDF:

<https://prasun0512.github.io/Resume/Prasun_Kumar.pdf>

The updated AI/ML Architect Word resume is available at:

<https://prasun0512.github.io/Resume/Prasun_Kumar_AI_ML_Architect_Resume.docx>

## Local Preview

This is a Jekyll/GitHub Pages site. With Ruby and Bundler installed:

```bash
bundle install
bundle exec jekyll serve
```

Then open:

```text
http://localhost:4000/Resume/
```

## Repository Structure

- `index.md` - main portfolio page
- `_config.yml` - GitHub Pages and SEO metadata
- `_layouts/default.html` - site shell and header
- `assets/css/style.scss` - custom portfolio styling
- `assets/js/ai-cursor.js` - interactive cursor trail effect
- `Prasun_Kumar.pdf` - downloadable resume

## Notes

Some enterprise projects are represented as sanitized case studies because the
source code belongs to client or internal systems and should not be published.
No private data, credentials, client names, or proprietary implementation details
should be committed to this repository.
