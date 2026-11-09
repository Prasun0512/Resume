# Public resume maintenance

`profile.json` contains only reviewed, public resume facts. The private candidate
profile is intentionally excluded from GitHub because it includes job-search
preferences, compensation notes and unresolved source details.

The resume prioritizes prior-role professional AI/ML projects. Independent GitHub
work is a separate section, not employer delivery. Formal IBM employment details
come from the supplied November 2026 PDF; no IBM project outcomes are invented.

## Build and validate

From the repository root, using Python 3.12:

```bash
python -m pip install -r script/requirements-resume.txt
python script/build_resume.py
python script/test_resume.py
```

The stable output is `Prasun_Kumar_AI_ML_Architect_Resume.pdf`. The generator uses
standard PDF fonts and a single-column layout, without tables or image-only text.
After content edits, render both pages and inspect them before publishing:

```bash
pdftoppm -r 120 -png Prasun_Kumar_AI_ML_Architect_Resume.pdf /tmp/resume-page
```

Do not infer client outcomes from public code, add an aspirational role as an
employment title, or replace uncertain education details with guessed values.
Reconfirm exact dates against HR records when needed for formal applications.
