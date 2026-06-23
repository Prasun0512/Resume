(function () {
  const buttons = Array.from(document.querySelectorAll("[data-project-filter]"));
  const cards = Array.from(document.querySelectorAll(".project-card"));
  const countTarget = document.querySelector("[data-filter-count]");
  const emptyState = document.querySelector(".filter-empty");

  if (!buttons.length || !cards.length) {
    return;
  }

  const normalize = (value) => value.toLowerCase().replace(/\s+/g, " ").trim();

  const repoTypes = [
    { match: "langgraph", label: "Runnable Demo" },
    { match: "rag quality", label: "Evaluation Framework" },
    { match: "document intelligence", label: "Runnable Demo" },
    { match: "email-to-case", label: "Runnable Demo" },
    { match: "resume job matcher", label: "Runnable Demo" },
    { match: "jailguard", label: "Guardrails Service" },
    { match: "mlflow", label: "Runnable Demo" },
    { match: "llm evaluation", label: "Evaluation Framework" },
    { match: "technical lead", label: "Leadership Playbook" },
    { match: "engineering leadership", label: "Leadership Playbook" },
    { match: "case studies", label: "Architecture Playbook" },
    { match: "architecture showcase", label: "Architecture Playbook" },
    { match: "platform engineering", label: "Architecture Playbook" },
    { match: "reliability", label: "Architecture Playbook" },
    { match: "operations", label: "Architecture Playbook" },
    { match: "multi-tenant", label: "Architecture Playbook" },
    { match: "integration", label: "Architecture Playbook" }
  ];

  const addRepoTypeBadges = () => {
    document.querySelectorAll(".public-projects .project-card").forEach((card) => {
      if (card.querySelector(".type-badge")) {
        return;
      }

      const text = normalize(card.textContent || "");
      const match = repoTypes.find((item) => text.includes(item.match));
      if (!match) {
        return;
      }

      const badge = document.createElement("span");
      badge.className = "type-badge";
      badge.textContent = match.label;
      const heading = card.querySelector("h3");
      if (heading) {
        heading.insertAdjacentElement("afterend", badge);
      } else {
        card.prepend(badge);
      }
    });
  };

  const getHaystack = (card) => {
    const explicitSkills = card.getAttribute("data-skills") || "";
    return normalize(`${explicitSkills} ${card.textContent || ""}`);
  };

  const matchesFilter = (card, filter) => {
    if (filter === "all") {
      return true;
    }

    const terms = filter.split("|").map(normalize).filter(Boolean);
    const haystack = getHaystack(card);
    return terms.some((term) => haystack.includes(term));
  };

  const setActiveButton = (activeButton) => {
    buttons.forEach((button) => {
      const isActive = button === activeButton;
      button.classList.toggle("active", isActive);
      button.setAttribute("aria-pressed", String(isActive));
    });
  };

  const applyFilter = (button) => {
    const filter = button.getAttribute("data-project-filter") || "all";
    let visibleCount = 0;

    cards.forEach((card) => {
      const isVisible = matchesFilter(card, filter);
      card.classList.toggle("is-hidden", !isVisible);
      if (isVisible) {
        visibleCount += 1;
      }
    });

    setActiveButton(button);

    if (countTarget) {
      countTarget.textContent = filter === "all" ? "all" : String(visibleCount);
    }

    if (emptyState) {
      emptyState.hidden = visibleCount > 0;
    }
  };

  buttons.forEach((button) => {
    button.setAttribute("aria-pressed", button.classList.contains("active") ? "true" : "false");
    button.addEventListener("click", () => applyFilter(button));
  });

  const activeButton = buttons.find((button) => button.classList.contains("active")) || buttons[0];
  addRepoTypeBadges();
  applyFilter(activeButton);
})();
