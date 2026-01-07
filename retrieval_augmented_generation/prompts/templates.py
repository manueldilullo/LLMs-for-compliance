"""
Prompt templates for RAG patterns and agentic workflows.
"""

# =============================================================================
# JSON Output Instructions
# =============================================================================

INSTRUCTIONS_ANSWER_JSON = """
CRITICAL: You MUST respond with ONLY valid JSON. Do not include any text before or after the JSON object.

Instructions:
- Your entire response must be a single JSON object starting with "{" and ending with "}"
- Do not include explanations, reasoning, or extra text
- The JSON must have exactly one key: "answer"
- The "answer" value must be at most 2 sentences (max 60 words)

Output format (JSON only):
{
"answer": "<your concise factual answer>"
}
END_JSON
"""

# =============================================================================
# Main RAG Prompts
# =============================================================================

RAG_PROMPT = """
You are an expert assistant specialized in the EU AI Act, GDPR and regulatory compliance.
Your task is to answer the user's question as accurately and concisely as possible, strictly based on the resources provided below.

Question: {query}

Resources:
----------------------------------------
{context}
----------------------------------------

""" + INSTRUCTIONS_ANSWER_JSON

BASELINE_PROMPT = """
You are an expert assistant specialized in the EU AI Act, GDPR and regulatory compliance.
Your role is to provide accurate and concise answers.

CRITICAL: You MUST respond with ONLY valid JSON. Do not include any text before or after the JSON object.

Question: {query}

""" + INSTRUCTIONS_ANSWER_JSON

# =============================================================================
# Self-Refinement Prompts
# =============================================================================

SELF_REFINEMENT_PROMPT = """
You are an expert assistant specialized in the EU AI Act, GDPR and regulatory compliance.
Your role is to Review and improve the given answers for accuracy. You will receive the starting question, the draft answer and the context text used by another expert.
Provide a more accurate and concise answer reviewing the input.

CRITICAL: You MUST respond with ONLY valid JSON. Do not include any text before or after the JSON object.

Question: {query}
Draft: {answer_text}

Resources:
----------------------------------------
{context}
----------------------------------------

""" + INSTRUCTIONS_ANSWER_JSON

REFINE_FROM_ISSUES_PROMPT = """
You are an expert assistant specialized in the EU AI Act, GDPR and regulatory compliance.
Revise the Draft to address the listed Issues using ONLY the Resources.

Rules:
- Only change what is needed to resolve Issues; otherwise keep the Draft wording stable.
- Do NOT add information not present in Resources.

Question: {query}

Draft:
{draft}

Issues (from auditor):
{issues_json}

Resources:
----------------------------------------
{context}
----------------------------------------
""" + INSTRUCTIONS_ANSWER_JSON

# =============================================================================
# Critic/Collaboration Prompts
# =============================================================================

INSTRUCTIONS_CRITIQUE_JSON = """
CRITICAL: Respond with ONLY valid JSON, no surrounding text.

Rules:
- Output must be exactly one JSON object.
- Keys MUST be exactly: "verdict", "issues"
- "verdict" is either "APPROVE" or "REVISE"
- "issues" is a list. If verdict is "APPROVE", issues MUST be [].
- Every issue MUST include an evidence quote copied from Resources; if you cannot quote, do not add the issue.

Return ONLY this JSON shape:
{
  "verdict": "<APPROVE | REVISE>",
  "issues": [
    {
      "type": "<unsupported | missing | contradiction>",
      "claim": "<what is wrong/missing>",
      "evidence_quote": "<verbatim quote from Resources that supports the critique>"
    }
  ]
}
END_JSON
"""

CRITIC_PROMPT = """
You are a strict compliance auditor.
Task: Evaluate whether the Draft answer is fully supported by the Resources and answers the Question.

Important:
- Do NOT use outside knowledge.
- If the Draft is correct and sufficiently supported, output verdict=APPROVE.
- If you request changes, each issue MUST include an evidence_quote copied from Resources.

Question: {query}

Draft:
{draft}

Resources:
----------------------------------------
{context}
----------------------------------------
""" + INSTRUCTIONS_CRITIQUE_JSON
