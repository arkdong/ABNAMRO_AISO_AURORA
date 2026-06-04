WRITING_GUIDE = {
    "version": "2026 V1.1",
    "hard_rules": [
        "No unsubstantiated forward-looking claims",
        "No competitor mentions without legal sign-off",
        "No greenwashing language (ESG policy)",
        "Must cite internal or approved external data sources",
        "Max 1200 words (standard) / 800 words (headlines)",
        "No superlative comparatives without evidence ('best', 'most')",
        "Avoid first-person plural assertions ('we believe', 'we think')",
    ],
    "soft_guidance": [
        "Tone: forward-thinking, engaged, smart — not corporate or salesy",
        "Plain language B1 level even for complex topics",
        "Structure: hook → context → insight → implication",
        "Prefer active voice and concrete examples",
        "Dutch articles: formal 'u'-form",
        "English articles: British English spelling",
    ],
}

ROLES = {
    "Insights Editorial": {"color": "#1a56db", "bg": "#e8f0fe", "emoji": "📰"},
    "Chatbot (Anna)": {"color": "#7c3aed", "bg": "#ede9fe", "emoji": "💬"},
    "Mobile App (UX)": {"color": "#059669", "bg": "#ecfdf5", "emoji": "📱"},
    "Web / IB": {"color": "#d97706", "bg": "#fef3c7", "emoji": "🌐"},
}

CHANNEL_PROFILES = {
    "Insights Editorial": {
        "description": "Expert-authored ABN AMRO Insights articles (sectors, trends, research).",
        "hard_rules": WRITING_GUIDE["hard_rules"],
        "soft_guidance": WRITING_GUIDE["soft_guidance"],
        "skills": [
            "Sector analysis & macro context",
            "Citing internal & external data",
            "Bilingual NL/EN editorial style",
            "Headline & hook writing",
        ],
    },
    "Chatbot (Anna)": {
        "description": 'Customer-facing chatbot dialogs, based on Ruby "Chatbot Anna" guide.',
        "hard_rules": [
            "Use 'u'-form; never 'je/jij'.",
            "No emojis, also not as punctuation.",
            "No abbreviations in answers.",
            "Max 3 bubbles per dialog step.",
            "Max 160 characters per bubble (incl. spaces).",
            "1 bubble = 1 message; avoid multiple topics in one bubble.",
            "Buttons: 2–4 per set, max 30 characters, no questions in buttons.",
            "Yes/positive choice always comes before no.",
            "Use only one link per set of bubbles; no inline links in text.",
            "Use deeplinks to Internet Bankieren where appropriate; never to old IB.",
        ],
        "soft_guidance": [
            "Tone: friendly, calm, helpful; follow brand voice but slightly more conversational.",
            "Always acknowledge the question before answering (e.g. 'Oké', 'Duidelijk').",
            "Use discourse markers for multi-step flows ('In 3 stappen geregeld', 'U bent er bijna').",
            "Apply 'voice first': end turns with a question or clear next action.",
            "Never create a dead end; always offer a way forward or handover.",
            "Apply 'Jenga': aggressively trim unnecessary words (often >50%).",
            "Avoid 'zien/kijken/hieronder' and positional references except when strictly necessary.",
            "Use tapering to shorten dialogs once the topic is clear.",
        ],
        "skills": [
            "Conversation design & flows",
            "Grey intents & disambiguation",
            "Logged-in vs not-logged-in logic",
            "Deep link usage in IB & app contexts",
        ],
    },
    "Mobile App (UX)": {
        "description": "UX microcopy and future proof content for ABN AMRO App + IB.",
        "hard_rules": [
            "Write in plain, understandable language at B1 level.",
            "Explain difficult banking terminology; exceptions only for legal wording.",
            "NL: follow Woordenlijst der Nederlandse Taal & Taalunie.",
            "EN: follow Oxford English Dictionary / UK-EN spelling.",
            "Follow WCAG accessibility guidelines and inclusive, gender-neutral phrasing.",
            "Reuse generic keys/UX-copy; only diverge when well-founded.",
            "Figma copy is leading in preparation; Lokalise copy is leading in execution & dev.",
        ],
        "soft_guidance": [
            "Short, action-oriented microcopy tied to the user’s goal.",
            "Positive, reassuring tone; avoid pressure sales language.",
            "Design copy to be channel and segment agnostic unless explicitly app-specific.",
            "Consider all required languages when defining base keys.",
        ],
        "skills": [
            "Microcopy & push notifications",
            "Figma/Lokalise key management",
            "Accessibility labels & alt-text",
        ],
    },
    "Web / IB": {
        "description": "Content for public web and Internet Banking flows.",
        "hard_rules": [
            "Use B1 reading level; explain complex terms.",
            "Respect WCAG guidelines for headings, links, and structure.",
            "Include all required regulatory disclosures for product pages.",
            "Deep links only into authenticated Internet Banking; not to legacy IB.",
        ],
        "soft_guidance": [
            "Make pages scannable: short paragraphs, headings, and bullets.",
            "Guide users with stepwise instructions without over-specific UI coordinates.",
            "Use deep links to reduce steps for logged-in customers.",
        ],
        "skills": [
            "Accessibility (WCAG)",
            "Deep linking & login state awareness",
            "IA & headings design",
        ],
    },
}

DOCUMENT_REGISTRY = {
    "writing_guide": {
        "id": "Writing-Guide-2026-V1.1",
        "path": "Writing-Guide-2026-V1.1.pdf",
        "type": "guideline",
        "authority": 100,
        "description": "Normative editorial Writing Guide 2026 for ABN AMRO Insights.",
    },
    "channel_guidelines": {
        "id": "Contentrichtlijnen-per-kanaal-v7",
        "path": "Contentrichtlijnen-per-kanaal-v7-20260324_100002.pdf",
        "type": "channel_guidelines",
        "authority": 95,
        "description": "Ruby channel-specific content guidelines, including Chatbot Anna.",
    },
}

TASK_HINTS = {
    "translate": ("T1_TRANSLATE", "Translation request detected"),
    "vertaal": ("T1_TRANSLATE", "Translation request detected (NL keyword)"),
    "older than": ("T4_RENEWAL", "Article renewal / aging scan request"),
    "ouder dan": ("T4_RENEWAL", "Article renewal / aging scan request (NL)"),
    "renew": ("T4_RENEWAL", "Article renewal request"),
    "check": ("T2_COMPLIANCE", "Compliance check request"),
    "checklist": ("T2_COMPLIANCE", "Compliance checklist request"),
    "pass": ("T2_COMPLIANCE", "Pass/fail compliance check"),
    "draft": ("T1_DRAFT", "Draft generation request"),
    "write": ("T1_DRAFT", "Draft generation request"),
    "schrijf": ("T1_DRAFT", "Draft generation request (NL)"),
    "article about": ("T1_DRAFT", "New article request"),
    "exist": ("T1_SEARCH", "Corpus search / prior art request"),
    "any article": ("T1_SEARCH", "Corpus search request"),
    "related": ("T1_SEARCH", "Related content search"),
}

TASK_LABELS = {
    "T1_DRAFT": "T1 – Draft new content",
    "T1_TRANSLATE": "T1 – Translate existing content",
    "T1_SEARCH": "T1 – Search corpus for related articles",
    "T2_COMPLIANCE": "T2 – Quality & compliance check",
    "T4_RENEWAL": "T4 – Detect & renew aging articles",
}
