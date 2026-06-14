(function () {
  "use strict";

  var profile = {
    name: "Prasun Kumar",
    title: "AI/ML Engineer | Technical Lead | GenAI & Agentic AI Solution Architect",
    location: "Pune, India",
    totalExperience: "8.5+ years",
    contact: {
      email: "cehprasunsinha@gmail.com",
      linkedin: "https://www.linkedin.com/in/prasun-kumar-1708/",
      github: "https://github.com/Prasun0512",
      portfolio: "https://prasun0512.github.io/Resume/"
    },
    summary: [
      "Prasun Kumar is an AI/ML Engineer, Technical Lead, and GenAI & Agentic AI Solution Architect with 8.5+ years of experience.",
      "He focuses on Agentic AI, LLM applications, LangGraph, LangChain, RAG systems, Azure OpenAI, AI automation, document intelligence, OCR plus LLM pipelines, and production-ready AI engineering.",
      "His work spans healthcare, insurance, HRTech, EdTech, background verification, learning platforms, and enterprise workflow automation."
    ],
    experience: [
      {
        company: "Harbinger Group",
        role: "Technical Lead, AI and ML",
        dates: "Aug 2024 - Present",
        focus: "Enterprise GenAI, AI automation, RAG, document intelligence, LLM workflows, and solution architecture."
      },
      {
        company: "Harbinger Group",
        role: "Senior Software Engineer",
        dates: "Apr 2021 - Jul 2024",
        focus: "AI/ML applications, cloud-native engineering, NLP, computer vision, automation, and enterprise integrations."
      },
      {
        company: "Extentia Information Technology",
        role: "Software Engineer",
        dates: "Nov 2019 - Apr 2021",
        focus: "Enterprise application engineering, automation workflows, APIs, and integration components."
      },
      {
        company: "Symantec Software India Pvt Ltd",
        role: "Associate IT Applications Specialist",
        dates: "Jul 2017 - Oct 2019",
        focus: "Enterprise IT applications, automation, support workflows, and reliability improvements."
      }
    ],
    projects: [
      {
        name: "GenAI Email-to-Case Automation Platform",
        keywords: ["email", "case", "azure functions", "service bus", "blob", "graph api", "ocr", "document intelligence", "azure openai", "gpt-4", "trackops"],
        answer: "Prasun architected a GenAI-powered email-to-case platform that converts inbound emails and unstructured documents into structured case records using Azure Functions, Service Bus, Blob Storage, Azure OpenAI, GPT-4, Azure AI Document Intelligence, Microsoft Graph API, and TrackOps integrations. The work included retries, dead-letter handling, auditability, confidence scoring, and human-in-the-loop validation, with 60%+ extraction accuracy improvement."
      },
      {
        name: "Enterprise RAG / Requirements Discovery Agent",
        keywords: ["rag", "retrieval", "azure ai search", "vector", "embedding", "requirements", "knowledge discovery", "semantic"],
        answer: "Prasun designed enterprise RAG and knowledge discovery patterns using Azure Blob Storage, Azure AI Document Intelligence, Azure AI Search, vector retrieval, semantic indexing, metadata enrichment, hybrid retrieval, chunk optimization, and retrieval benchmarking."
      },
      {
        name: "Agentic AI LangGraph Workflows",
        keywords: ["agent", "agentic", "langgraph", "langchain", "multi-agent", "workflow", "tool routing", "memory", "approval"],
        answer: "The Agentic AI LangGraph Workflows project demonstrates graph-style orchestration, tool routing, state handling, approval checkpoints, memory-aware execution, and audit-friendly traces for enterprise workflow automation."
      },
      {
        name: "Document Intelligence LLM Pipeline",
        keywords: ["document", "ocr", "extraction", "redaction", "pii", "phi", "schema", "validation"],
        answer: "The Document Intelligence LLM Pipeline project shows OCR plus LLM extraction with redaction, schema validation, confidence scoring, PII handling, and review routing for enterprise document processing."
      },
      {
        name: "Behavioral Intelligence Platform",
        keywords: ["behavior", "classification", "llama", "gemma", "qwen", "qlora", "databricks", "fine-tuning", "adapter"],
        answer: "Prasun led behavior scoring and LLM experimentation work using Llama, Gemma, Qwen, QLoRA, adapter evaluation, multilingual datasets, and Databricks dashboards. His portfolio highlights 77 TB+ storage lifecycle savings through cleanup and governance."
      },
      {
        name: "AI Resume Job Matcher",
        keywords: ["resume", "job", "ats", "matcher", "scoring", "skills", "recruiter"],
        answer: "The AI Resume Job Matcher project extracts skills, analyzes role fit, explains matching strengths, and highlights gaps between a resume and job description."
      }
    ],
    skills: {
      "Agentic AI": ["agentic ai", "ai agents", "multi-agent", "langgraph", "langchain", "tool routing", "memory", "approval", "orchestration"],
      "LLM Applications": ["llm", "gpt", "gpt-4", "azure openai", "prompt", "llama", "gemma", "qwen", "open-source llm"],
      "RAG": ["rag", "retrieval augmented generation", "azure ai search", "vector", "embedding", "hybrid retrieval", "semantic search", "chunking"],
      "Document Intelligence": ["ocr", "document intelligence", "document ai", "extraction", "pdf", "scanned", "pii", "phi", "redaction"],
      "Cloud AI Architecture": ["azure", "azure functions", "service bus", "blob storage", "graph api", "azure monitor", "aws", "serverless"],
      "ML and MLOps": ["machine learning", "nlp", "computer vision", "classification", "qlora", "lora", "databricks", "mlflow", "tensorflow", "pytorch"],
      "Leadership": ["technical lead", "architect", "solution architect", "stakeholder", "client", "mentor", "design review", "delivery"]
    }
  };

  var matcherWeights = {
    "Agentic AI": 16,
    "LLM Applications": 16,
    "RAG": 15,
    "Document Intelligence": 14,
    "Cloud AI Architecture": 14,
    "ML and MLOps": 13,
    "Leadership": 12
  };

  var llmState = {
    modelId: "Qwen2-0.5B-Instruct-q4f16_1-MLC",
    engine: null,
    loadingPromise: null,
    enabled: false,
    failed: false
  };

  function normalize(text) {
    return (text || "").toLowerCase().replace(/[^a-z0-9+#.\s/-]/g, " ");
  }

  function unique(items) {
    return items.filter(function (item, index) {
      return items.indexOf(item) === index;
    });
  }

  function includesAny(text, terms) {
    return terms.some(function (term) {
      return text.indexOf(term) !== -1;
    });
  }

  function createMessage(role, html) {
    var message = document.createElement("div");
    message.className = "assistant-message " + role;
    message.innerHTML = html;
    return message;
  }

  function updateMessage(message, html, role) {
    message.className = "assistant-message " + role;
    message.innerHTML = html;
  }

  function escapeHtml(text) {
    var element = document.createElement("div");
    element.textContent = text;
    return element.innerHTML;
  }

  function formatList(items) {
    return "<ul>" + items.map(function (item) {
      return "<li>" + item + "</li>";
    }).join("") + "</ul>";
  }

  function htmlToText(html) {
    var element = document.createElement("div");
    element.innerHTML = html;
    return element.textContent.replace(/\s+/g, " ").trim();
  }

  function buildProfileContext() {
    return [
      "Candidate: " + profile.name,
      "Title: " + profile.title,
      "Location: " + profile.location,
      "Total experience: " + profile.totalExperience,
      "Summary: " + profile.summary.join(" "),
      "Experience:",
      profile.experience.map(function (item) {
        return "- " + item.company + " | " + item.role + " | " + item.dates + " | " + item.focus;
      }).join("\n"),
      "Major projects:",
      profile.projects.map(function (item) {
        return "- " + item.name + ": " + item.answer;
      }).join("\n"),
      "Skill categories:",
      Object.keys(profile.skills).map(function (category) {
        return "- " + category + ": " + profile.skills[category].join(", ");
      }).join("\n"),
      "Contact: " + profile.contact.email + " | LinkedIn: " + profile.contact.linkedin + " | GitHub: " + profile.contact.github + " | Portfolio: " + profile.contact.portfolio
    ].join("\n");
  }

  function answerSummary() {
    return "<p><strong>Recruiter summary:</strong> " + profile.summary.join(" ") + "</p>" +
      formatList([
        "Best fit: Technical Lead AI/ML, GenAI Solution Architect, Agentic AI Engineer, RAG Architect, AI Automation Lead.",
        "Core strengths: Azure OpenAI, LangGraph, LangChain, RAG, OCR + LLM extraction, Azure Functions, Service Bus, AI governance, and production delivery.",
        "Public proof: GitHub portfolio includes Agentic AI LangGraph Workflows, Document Intelligence LLM Pipeline, Email-to-Case GenAI Automation, Enterprise RAG Quality Lab, and AI Resume Job Matcher."
      ]);
  }

  function answerSkills() {
    return "<p><strong>Strongest AI/ML areas:</strong></p>" + formatList([
      "Agentic AI and multi-agent workflow orchestration with LangGraph and LangChain patterns.",
      "Enterprise RAG with Azure AI Search, embeddings, vector search, hybrid retrieval, chunking, grounding, and evaluation.",
      "LLM applications using Azure OpenAI, GPT-4, Llama, Gemma, Qwen, prompt governance, and structured extraction.",
      "File and document intelligence with OCR, Azure AI Document Intelligence, PII/PHI masking, confidence scoring, and human validation.",
      "Production AI architecture using Azure Functions, Service Bus, Blob Storage, Microsoft Graph API, monitoring, retries, and dead-letter handling.",
      "Technical leadership across architecture reviews, delivery planning, mentoring, and client-facing solution discussions."
    ]);
  }

  function answerExperience() {
    return "<p><strong>Experience timeline:</strong></p>" + formatList(profile.experience.map(function (item) {
      return "<strong>" + item.company + " - " + item.role + "</strong> (" + item.dates + "): " + item.focus;
    }));
  }

  function answerProjects(query) {
    var text = normalize(query);
    var project = profile.projects.find(function (item) {
      return includesAny(text, item.keywords);
    });

    if (project) {
      return "<p><strong>" + project.name + ":</strong> " + project.answer + "</p>";
    }

    return "<p><strong>Major AI portfolio projects:</strong></p>" + formatList(profile.projects.map(function (item) {
      return "<strong>" + item.name + "</strong>";
    })) + "<p>Ask about a specific project like email-to-case, RAG, LangGraph, document intelligence, behavior scoring, or resume matcher.</p>";
  }

  function answerContact() {
    return "<p><strong>Contact and links:</strong></p>" + formatList([
      "Email: <a href=\"mailto:" + profile.contact.email + "\">" + profile.contact.email + "</a>",
      "LinkedIn: <a href=\"" + profile.contact.linkedin + "\" target=\"_blank\" rel=\"noopener\">Prasun Kumar</a>",
      "GitHub: <a href=\"" + profile.contact.github + "\" target=\"_blank\" rel=\"noopener\">github.com/Prasun0512</a>",
      "Portfolio: <a href=\"" + profile.contact.portfolio + "\" target=\"_blank\" rel=\"noopener\">prasun0512.github.io/Resume</a>"
    ]);
  }

  function analyzeJobDescription(input) {
    var text = normalize(input);
    var categories = Object.keys(profile.skills);
    var matches = [];
    var gaps = [];
    var score = 0;
    var maxScore = 0;

    categories.forEach(function (category) {
      var terms = profile.skills[category];
      var matchedTerms = unique(terms.filter(function (term) {
        return text.indexOf(term) !== -1;
      }));
      var weight = matcherWeights[category] || 10;
      maxScore += weight;

      if (matchedTerms.length) {
        var categoryScore = Math.min(weight, Math.ceil((matchedTerms.length / Math.min(terms.length, 5)) * weight));
        score += categoryScore;
        matches.push({
          category: category,
          terms: matchedTerms.slice(0, 6),
          score: categoryScore,
          weight: weight
        });
      } else {
        gaps.push(category);
      }
    });

    var percent = Math.min(96, Math.max(32, Math.round((score / maxScore) * 100)));
    var rating = percent >= 82 ? "Strong match" : percent >= 68 ? "Good match" : percent >= 52 ? "Partial match" : "Stretch match";
    var matchedStrengths = matches.map(function (item) {
      return "<strong>" + item.category + ":</strong> " + item.terms.join(", ");
    });
    var gapNotes = gaps.slice(0, 4).map(function (gap) {
      return gap + " is not strongly visible in the pasted JD or needs clearer evidence during screening.";
    });
    var talkTracks = [
      "Lead with production AI architecture: Azure OpenAI, RAG, document intelligence, event-driven automation, and governance.",
      "Use the email-to-case automation and document intelligence pipeline as proof of real enterprise workflow impact.",
      "Highlight LangGraph/LangChain and multi-agent workflow projects when the JD mentions agents, orchestration, or automation.",
      "Mention 8.5+ years, Technical Lead ownership, architecture reviews, client workshops, and cross-functional delivery."
    ];

    return "<p><strong>JD suitability:</strong> " + rating + " - estimated " + percent + "% fit based on public resume and portfolio signals.</p>" +
      "<p><strong>Matched strengths:</strong></p>" + formatList(matchedStrengths.length ? matchedStrengths : ["The JD text was too short or generic to find strong keyword overlap."]) +
      "<p><strong>Potential gaps to clarify:</strong></p>" + formatList(gapNotes.length ? gapNotes : ["No major gaps detected from the pasted JD keywords."]) +
      "<p><strong>Best interview positioning:</strong></p>" + formatList(talkTracks);
  }

  function looksLikeJobDescription(input) {
    var text = normalize(input);
    var jdSignals = [
      "job description",
      "responsibilities",
      "requirements",
      "qualifications",
      "role",
      "we are looking",
      "candidate",
      "experience with",
      "years of experience",
      "must have",
      "preferred"
    ];
    return input.length > 550 || includesAny(text, jdSignals);
  }

  function routeQuestion(input) {
    var text = normalize(input);

    if (!input.trim()) {
      return "<p>Ask about Prasun's experience, projects, skills, contact details, or paste a job description for a suitability match.</p>";
    }

    if (looksLikeJobDescription(input) || text.indexOf("jd match") !== -1 || text.indexOf("suitable") !== -1 || text.indexOf("suitability") !== -1) {
      if (input.length < 160 && (text.indexOf("jd") !== -1 || text.indexOf("suitable") !== -1)) {
        return "<p>Paste the full job description here and I will estimate fit, matched strengths, gaps, and interview positioning based on Prasun's resume and portfolio.</p>";
      }
      return analyzeJobDescription(input);
    }

    if (includesAny(text, ["summary", "summarize", "recruiter", "about", "who is", "intro"])) {
      return answerSummary();
    }

    if (includesAny(text, ["skill", "stack", "technology", "tools", "strength"])) {
      return answerSkills();
    }

    if (includesAny(text, ["experience", "timeline", "company", "harbinger", "extentia", "symantec"])) {
      return answerExperience();
    }

    if (includesAny(text, ["project", "email", "case", "rag", "agent", "langgraph", "document", "ocr", "behavior", "resume matcher", "portfolio"])) {
      return answerProjects(input);
    }

    if (includesAny(text, ["contact", "email", "linkedin", "github", "location"])) {
      return answerContact();
    }

    return "<p>I can answer from Prasun's public resume and portfolio. Useful questions include:</p>" + formatList([
      "Summarize Prasun for a recruiter.",
      "What AI/ML projects prove production experience?",
      "Tell me about LangGraph, RAG, or document intelligence work.",
      "Paste a job description and ask for suitability."
    ]);
  }

  function isWebLLMSupported() {
    return Boolean(window.isSecureContext && navigator.gpu && window.WebAssembly);
  }

  function setStatus(statusElement, text) {
    if (statusElement) {
      statusElement.textContent = text;
    }
  }

  function setLlmButton(button, text, disabled) {
    if (button) {
      button.textContent = text;
      button.disabled = Boolean(disabled);
    }
  }

  function loadLocalLLM(statusElement, button) {
    if (llmState.engine) {
      return Promise.resolve(llmState.engine);
    }

    if (llmState.loadingPromise) {
      return llmState.loadingPromise;
    }

    if (!isWebLLMSupported()) {
      llmState.failed = true;
      setStatus(statusElement, "LLM mode needs HTTPS + WebGPU. Using instant mode.");
      setLlmButton(button, "LLM unavailable", true);
      return Promise.reject(new Error("WebGPU is not available in this browser."));
    }

    setStatus(statusElement, "Loading browser LLM...");
    setLlmButton(button, "Loading model...", true);

    llmState.loadingPromise = import("https://esm.run/@mlc-ai/web-llm")
      .then(function (webllm) {
        return webllm.CreateMLCEngine(llmState.modelId, {
          initProgressCallback: function (progress) {
            var percent = progress && typeof progress.progress === "number"
              ? " " + Math.round(progress.progress * 100) + "%"
              : "";
            var text = progress && progress.text ? progress.text : "Loading browser LLM";
            setStatus(statusElement, text + percent);
          }
        });
      })
      .then(function (engine) {
        llmState.engine = engine;
        llmState.enabled = true;
        llmState.failed = false;
        setStatus(statusElement, "LLM mode active - Qwen2 0.5B in browser");
        setLlmButton(button, "LLM Active", true);
        return engine;
      })
      .catch(function (error) {
        llmState.failed = true;
        llmState.enabled = false;
        setStatus(statusElement, "LLM load failed. Using instant resume mode.");
        setLlmButton(button, "Retry LLM", false);
        llmState.loadingPromise = null;
        throw error;
      });

    return llmState.loadingPromise;
  }

  function buildLlmPrompt(question) {
    var deterministicAnswer = htmlToText(routeQuestion(question));
    var jdContext = looksLikeJobDescription(question)
      ? "\nDeterministic JD analysis to preserve scoring logic:\n" + htmlToText(analyzeJobDescription(question))
      : "";

    return [
      {
        role: "system",
        content: [
          "You are Prasun Kumar's portfolio assistant.",
          "Answer only from the provided resume and portfolio context.",
          "Do not invent employers, metrics, dates, certifications, degrees, or production claims.",
          "If the user pasted a job description, explain fit using the deterministic JD analysis and resume context.",
          "Be concise, recruiter-friendly, and specific. Use bullets when helpful.",
          "If information is not in context, say it is not available in the public resume/portfolio."
        ].join(" ")
      },
      {
        role: "user",
        content: [
          "Resume and portfolio context:\n" + buildProfileContext(),
          jdContext,
          "\nRule-based fallback answer:\n" + deterministicAnswer,
          "\nUser question or JD:\n" + question
        ].join("\n")
      }
    ];
  }

  function generateLlmAnswer(question, statusElement, button) {
    return loadLocalLLM(statusElement, button)
      .then(function (engine) {
        setStatus(statusElement, "Generating with free local LLM...");
        return engine.chat.completions.create({
          messages: buildLlmPrompt(question),
          temperature: 0.35,
          top_p: 0.9,
          max_tokens: 420
        });
      })
      .then(function (reply) {
        var content = reply &&
          reply.choices &&
          reply.choices[0] &&
          reply.choices[0].message &&
          reply.choices[0].message.content;

        setStatus(statusElement, "LLM mode active - Qwen2 0.5B in browser");
        if (!content) {
          return routeQuestion(question);
        }

        return "<p><strong>LLM answer:</strong></p><p>" +
          escapeHtml(content).replace(/\n{2,}/g, "</p><p>").replace(/\n/g, "<br>") +
          "</p>";
      })
      .catch(function () {
        return routeQuestion(question) +
          "<p><strong>Note:</strong> LLM mode could not run on this browser/device, so this answer used the instant resume matcher.</p>";
      });
  }

  function initAssistant() {
    var assistant = document.querySelector(".profile-assistant");
    if (!assistant) {
      return;
    }

    var toggle = assistant.querySelector(".assistant-toggle");
    var panel = assistant.querySelector(".assistant-panel");
    var closeButton = assistant.querySelector(".assistant-close");
    var status = assistant.querySelector(".assistant-status");
    var llmButton = assistant.querySelector(".assistant-llm-toggle");
    var messages = assistant.querySelector(".assistant-messages");
    var form = assistant.querySelector(".assistant-form");
    var input = assistant.querySelector("#assistant-input");
    var prompts = assistant.querySelectorAll(".assistant-prompts button");

    function openAssistant() {
      panel.hidden = false;
      assistant.classList.add("is-open");
      toggle.setAttribute("aria-expanded", "true");
      if (!messages.dataset.started) {
        messages.appendChild(createMessage("bot", "<p><strong>Hi, I am Prasun's portfolio assistant demo.</strong> Ask about AI/ML experience, projects, skills, or paste a JD for a fit check. Instant mode is always available; optional LLM mode runs only when the browser supports it.</p>"));
        messages.dataset.started = "true";
      }
      input.focus();
    }

    function closeAssistant(shouldFocusToggle) {
      assistant.classList.remove("is-open");
      panel.hidden = true;
      toggle.setAttribute("aria-expanded", "false");
      if (shouldFocusToggle !== false) {
        toggle.focus();
      }
    }

    function submitQuestion(question) {
      var cleanQuestion = question.trim();
      if (!cleanQuestion) {
        return;
      }

      messages.appendChild(createMessage("user", "<p>" + escapeHtml(cleanQuestion) + "</p>"));
      var botMessage = createMessage("bot loading", "<p>Thinking...</p>");
      messages.appendChild(botMessage);
      messages.scrollTop = messages.scrollHeight;
      input.value = "";

      if (llmState.enabled && llmState.engine) {
        generateLlmAnswer(cleanQuestion, status, llmButton).then(function (answer) {
          updateMessage(botMessage, answer, "bot");
          messages.scrollTop = messages.scrollHeight;
        });
      } else {
        updateMessage(botMessage, routeQuestion(cleanQuestion), "bot");
        messages.scrollTop = messages.scrollHeight;
      }
    }

    toggle.addEventListener("click", function () {
      if (!assistant.classList.contains("is-open")) {
        openAssistant();
      } else {
        closeAssistant(true);
      }
    });

    closeButton.addEventListener("click", function () {
      closeAssistant(true);
    });

    document.addEventListener("keydown", function (event) {
      if (event.key === "Escape" && assistant.classList.contains("is-open")) {
        closeAssistant(true);
      }
    });

    document.addEventListener("click", function (event) {
      if (assistant.classList.contains("is-open") && !assistant.contains(event.target)) {
        closeAssistant(false);
      }
    });

    if (llmButton) {
      llmButton.addEventListener("click", function () {
        openAssistant();
        messages.appendChild(createMessage("bot loading", "<p>Loading a free open-source LLM in your browser. First load may take a little while and depends on WebGPU support.</p>"));
        messages.scrollTop = messages.scrollHeight;
        loadLocalLLM(status, llmButton)
          .then(function () {
            messages.appendChild(createMessage("bot", "<p><strong>LLM mode is active.</strong> Ask a profile question or paste a JD and I will generate a grounded response using the local Qwen2 0.5B model plus Prasun's resume context.</p>"));
            messages.scrollTop = messages.scrollHeight;
          })
          .catch(function () {
            messages.appendChild(createMessage("bot", "<p><strong>LLM mode is unavailable here.</strong> This browser may not support WebGPU, or the model download failed. The instant resume assistant and JD matcher still work.</p>"));
            messages.scrollTop = messages.scrollHeight;
          });
      });
    }

    prompts.forEach(function (prompt) {
      prompt.addEventListener("click", function () {
        openAssistant();
        input.value = prompt.dataset.question || "";
        if (input.value.toLowerCase().indexOf("job description") === -1) {
          submitQuestion(input.value);
        } else {
          input.focus();
        }
      });
    });

    form.addEventListener("submit", function (event) {
      event.preventDefault();
      submitQuestion(input.value);
    });
  }

  document.addEventListener("DOMContentLoaded", initAssistant);
})();
