"""
Prompt templates for synthetic Q&A generation.

These templates are used to generate questions and answers about
GDPR and AI Act regulations.
"""

# =============================================================================
# Article Unity Prompts
# =============================================================================

PROMPT_ARTICLE_UNITY_Q = """
You are a legal-question generator. Write one precise question about the GDPR
that, when answered correctly, must be the excerpt provided below from
Article {number} (or its faithful paraphrase).

Requirements:
- Base the question only on the inputs below; do not use external context.
- The question must uniquely target the "Required Answer" excerpt, not any
other part of the article.
- Avoid yes/no questions. Ask for a short factual answer (who/what/which/
under what conditions/according to which criteria).
- Keep the question under 50 words.
- Output only valid JSON as specified.

Article metadata (optional):
\"\"\"{metadata}\"\"\"
Article {number} (full text):
\"\"\"{full_text}\"\"\"
Contextual reference for framing the question (required excerpt):
\"\"\"{excerpt}\"\"\"

Output format (JSON):
{{
"article_number": {number},
"question": "<your question>"
}}
END_JSON

!CRITICAL! Return ONLY the JSON—do NOT include any reasoning or extra text.
Any explanation is strictly forbidden

Validation checklist (do not include in output):
- Would the required excerpt alone fully answer the question?
- Is the question specific enough that a different annex passage would not fit?
- Is the question non-yes/no and <50 words?
"""

PROMPT_ARTICLE_UNITY_A = """
You are a legal assistant. Provide a precise and concise answer based
solely on the following excerpt from Article {number} of the GDPR.

Requirements:
- Base your answer only on the inputs below; do not use external context.
- The answer must exactly or faithfully paraphrase the "Required Answer"
excerpt.
- Keep the answer factual and concise.
- You must elaborate the answer as your own, do not copy it.

Article {number} (full text):
\"\"\"{full_text}\"\"\"
Required answer excerpt:
\"\"\"{excerpt}\"\"\"

Output format (JSON):
{{
"article_number": {number},
"answer": "<your concise factual answer>"
}}
END_JSON

!CRITICAL! Return ONLY the JSON—do NOT include any reasoning or extra text.
Any explanation is strictly forbidden
"""

# =============================================================================
# Article-Recital Binding Prompts
# =============================================================================

PROMPT_ARTICLE_RECITAL_BINDING_Q = """
You are a legal-question generator. Write one precise, legally relevant
question about the GDPR
that explicitly captures the relationship between the specified Recital and
the specified Article.

Requirements:
- The question must be answerable solely from Recital {recital_num} and Article {article_num} (below);
do not rely on external context.
- Make the linkage explicit (e.g., "According to Recital {recital_num}, how does it inform/clarify/apply
to Article {article_num} (text below) regarding X?").
- The question must require using both texts together (not just one of them).
- Avoid yes/no questions. Ask for a short factual answer (who/what/which/under what conditions/according to criteria).
- Keep the question under 50 words.
- Make it specific enough that a different recital or article passage would not fit.
- Output only valid JSON as specified.

Context:
Article metadata (optional):
\"\"\"{metadata}\"\"\"
Article {article_num} (full text):
\"\"\"{article_text}\"\"\"
Recital {recital_num} (full text):
\"\"\"{recital_text}\"\"\"

Output format (JSON):
{{
"recital_number": {recital_num},
"article_number": {article_num},
"question": "<your question>"
}}
END_JSON

!CRITICAL! Return ONLY the JSON—do NOT include any reasoning or extra text.
Any explanation is strictly forbidden

Validation checklist (do not include in output):
- Would the required excerpt alone fully answer the question?
- Is the question specific enough that a different annex passage would not fit?
- Is the question non-yes/no and <50 words?
"""

PROMPT_ARTICLE_RECITAL_BINDING_A = """
You are a legal assistant. Provide a precise and concise answer based
solely on the following Recital {recital_number} and Article {article_number} of the GDPR.

Requirements:
- Base your answer only on the inputs below; do not use external context.
- The answer must explain faithfully how the recital informs, clarifies, or
applies to the article's provision.
- Keep the answer factual and concise.
- You must elaborate the answer as your own, do not copy it.

Article {article_number} (full text):
\"\"\"{article_text}\"\"\"

Recital {recital_number} (full text):
\"\"\"{recital_text}\"\"\"

Required answer excerpt:
\"\"\"{answer_excerpt}\"\"\"

Output format (JSON):
{{
  "recital_number": {recital_number},
  "article_number": {article_number},
  "answer": "<your concise factual answer>"
}}
END_JSON

!CRITICAL! Return ONLY the JSON—do NOT include any reasoning or extra text.
Any explanation is strictly forbidden!
"""

# =============================================================================
# Recital Unity Prompts
# =============================================================================

PROMPT_RECITAL_UNITY_Q = """
You are a legal-question generator. Task: write one precise question about
the GDPR whose correct answer is the provided recital (or its faithful paraphrase).

Requirements:
- Base the question only on the recital text given below; do not invent
facts outside it.
- The question must be answerable uniquely by that recital's content
and not by other recitals.
- Prefer specific "According to Recital {recital_number}..." phrasing.
- Avoid yes/no questions; ask for a short factual answer (who/what/when/where/which/under what conditions).
- Keep the question under 50 words.
- Output only valid JSON as specified.

Recital number: {recital_number}
Recital text:
\"\"\"{recital_text}\"\"\"

Output format (JSON):
{{
  "recital_number": {recital_number},
  "question": "<your question>"
}}
END_JSON

!CRITICAL! Return ONLY the JSON—do NOT include any reasoning or extra text.
Any explanation is strictly forbidden!

Validation checklist (do not include in output):
- Would the required excerpt alone fully answer the question?
- Is the question specific enough that a different annex passage would not fit?
- Is the question non-yes/no and <50 words?
"""

PROMPT_RECITAL_UNITY_A = """
You are a legal assistant. Provide a precise and concise answer based
solely on the following excerpt from Recital {recital_number} of the GDPR.

Requirements:
- Base your answer only on the excerpt; do not use external context.
- The answer must faithfully paraphrase or restate the recital's content.
- Keep the answer factual and concise.
- You must elaborate the answer as your own; do not copy it.

Recital {recital_number} (full text):
\"\"\"{recital_text}\"\"\"

Required answer excerpt:
\"\"\"{answer_excerpt}\"\"\"

Output format (JSON):
{{
  "recital_number": {recital_number},
  "answer": "<your concise factual answer>"
}}
END_JSON

!CRITICAL! Return ONLY the JSON—do NOT include any reasoning or extra text.
Any explanation is strictly forbidden!
"""

# =============================================================================
# Augmentation Prompt
# =============================================================================

PROMPT_AUGMENTATION_Q = """
You are an assistant that augments questions about the GDPR.
Create variations of the given question that ask for the same information in different ways.

Rules:
- Do NOT directly reference any paragraph, article, annex, or recital.
- The answer's core meaning must remain the same.

Original question:
```
{original_question}
```
Full context:
```
{full_context}
```
Reference excerpt (original answer):
```
{original_answer}
```
Number of questions to generate: {n}

Generate alternative phrasings of the original question that ask for the same information but with different words.

Output format (JSON List):
{{
  "questions": [
    "<your augmented question 1>",
    "<your augmented question 2>",
    ...
    "<your augmented question n>"
  ]
}}
END_JSON

Return ONLY the questions in a JSON List format—do NOT include any reasoning or extra text.
Any explanation is strictly forbidden!
"""

# =============================================================================
# Annex-Recital Binding Prompts
# =============================================================================

PROMPT_ANNEX_RECITAL_BINDING_Q = """
You are a legal-question generator. Write one precise question about the EU AI Act that explicitly captures the relationship between the specified Recital and the specified Annex.

Requirements:
- The question must be answerable solely from the provided Recital {recital_num} and Annex {annex_num} texts; do not rely on external context.
- Make the linkage explicit (e.g., "According to Recital {recital_num}, how does it inform/clarify/apply to Annex {annex_num} (text below) regarding X?").
- The question must require using both texts together (not just one of them).
- Avoid yes/no questions. Ask for a short factual answer (who/what/which/under what conditions/according to which criteria).
- Keep the question under 50 words.
- Make it specific enough that a different recital or annex passage would not fit.
- Output only valid JSON as specified.

Annex {annex_num} (full text):
\"\"\"{annex_text}\"\"\"
Recital {recital_num} (full text):
\"\"\"{recital_text}\"\"\"

Output format (JSON):
{{
"recital_number": {recital_num},
"annex_number": {annex_num},
"question": "<your question>"
}}
END_JSON

!CRITICAL! Return ONLY the JSON—do NOT include any reasoning or extra text.
Any explanation is strictly forbidden!

Validation checklist (do not include in output):
- Does the question necessitate consulting both the recital and annex texts?
- Is the connection unique to these two inputs (not generic across the AI Act)?
- Is the question specific, non-yes/no, and <50 words?
"""

PROMPT_ANNEX_RECITAL_BINDING_A = """
You are a legal assistant. Provide a precise and concise answer based solely on the following Recital {recital_number} and Annex {annex_number} from the EU AI Act.

Requirements:
- Base your answer only on the provided texts; do not use external context.
- The answer must explicitly reflect the relationship between Recital {recital_number} and Annex {annex_number}.
- Keep the answer factual, short, and specific to these two inputs.
- Avoid generic statements or interpretations not grounded in the texts.
- You must elaborate the answer as your own, do not copy it.

Annex {annex_number} (full text):
\"\"\"{annex_text}\"\"\"
Recital {recital_number} (full text):
\"\"\"{recital_text}\"\"\"

Output format (JSON):
{{
"recital_number": {recital_number},
"annex_number": {annex_number},
"answer": "<your concise factual answer>"
}}
END_JSON

!CRITICAL! Return ONLY the JSON—do NOT include any reasoning or extra text.
Any explanation is strictly forbidden!
"""

# =============================================================================
# Annex-Article Binding Prompts
# =============================================================================

PROMPT_ANNEX_ARTICLE_BINDING_Q = """
You are a legal-question generator. Write one precise question about the EU AI Act that explicitly captures the relationship between the specified Article and the specified Annex.

Requirements:
- The question must be answerable solely from the provided Article {article_num} and Annex {annex_num} texts; do not rely on external context.
- Make the linkage explicit (e.g., "According to Article {article_num}, how does it inform/clarify/apply to Annex {annex_num} (text below) regarding X?").
- The question must require using both texts together (not just one of them).
- Avoid yes/no questions. Ask for a short factual answer (who/what/which/under what conditions/according to which criteria).
- Keep the question under 50 words.
- Make it specific enough that a different article or annex passage would not fit.
- Output only valid JSON as specified.

Annex {annex_num} (full text):
\"\"\"{annex_text}\"\"\"
Article {article_num} (full text):
\"\"\"{article_text}\"\"\"

Output format (JSON):
{{
"article_number": {article_num},
"annex_number": {annex_num},
"question": "<your question>"
}}
END_JSON

!CRITICAL! Return ONLY the JSON—do NOT include any reasoning or extra text.
Any explanation is strictly forbidden!

Validation checklist (do not include in output):
- Does the question necessitate consulting both the article and annex texts?
- Is the connection unique to these two inputs (not generic across the AI Act)?
- Is the question specific, non-yes/no, and <50 words?
"""

PROMPT_ANNEX_ARTICLE_BINDING_A = """
You are a legal assistant. Provide a precise and concise answer based solely on the following Article {article_number} and Annex {annex_number} from the EU AI Act.

Requirements:
- Base your answer only on the provided texts; do not use external context.
- The answer must explicitly reflect the relationship between Article {article_number} and Annex {annex_number}.
- Keep the answer factual, short, and specific to these two inputs.
- Avoid generic statements or interpretations not grounded in the texts.
- You must elaborate the answer as your own, do not copy it.

Annex {annex_number} (full text):
\"\"\"{annex_text}\"\"\"
Article {article_number} (full text):
\"\"\"{article_text}\"\"\"

Output format (JSON):
{{
"article_number": {article_number},
"annex_number": {annex_number},
"answer": "<your concise factual answer>"
}}
END_JSON

!CRITICAL! Return ONLY the JSON—do NOT include any reasoning or extra text.
Any explanation is strictly forbidden!
"""

# =============================================================================
# Annex Unity Prompts
# =============================================================================

PROMPT_ANNEX_UNITY_Q = """
You are a legal-question generator. Write one precise question about the EU AI Act that, when answered correctly, must be the excerpt provided below from
Annex {number} (or its faithful paraphrase).

Requirements:
- Base the question only on the inputs below; do not use external context.
- The question must uniquely target the "Required Answer" excerpt, not any other part of the annex.
- Avoid yes/no questions. Ask for a short factual answer (who/what/which/under what conditions/according to which criteria).
- Keep the question under 50 words.
- Output only valid JSON as specified.

Annex {number} (full text):
\"\"\"{full_text}\"\"\"
Contextual reference for framing the question (required excerpt):
\"\"\"{excerpt}\"\"\"

Output format (JSON):
{{
"annex_number": {number},
"question": "<your question>"
}}
END_JSON

!CRITICAL! Return ONLY the JSON—do NOT include any reasoning or extra text.
Any explanation is strictly forbidden!

Validation checklist (do not include in output):
- Would the required excerpt alone fully answer the question?
- Is the question specific enough that a different annex passage would not fit?
- Is the question non-yes/no and <50 words?
"""

PROMPT_ANNEX_UNITY_A = """
You are a legal assistant. Provide a precise and concise answer based solely on the following excerpt from Annex {number} of the EU AI Act.

Requirements:
- Base your answer only on the provided text; do not use external context.
- The answer must faithfully paraphrase or restate the "Required Answer" excerpt.
- Keep the answer factual and concise.
- You must elaborate the answer as your own, do not copy it.

Annex {number} (full text):
\"\"\"{full_text}\"\"\"
Required answer excerpt:
\"\"\"{excerpt}\"\"\"

Output format (JSON):
{{
"annex_number": {number},
"answer": "<your concise factual answer>"
}}
END_JSON

!CRITICAL! Return ONLY the JSON—do NOT include any reasoning or extra text.
Any explanation is strictly forbidden!
"""

# =============================================================================
# All Prompts Dictionary
# =============================================================================

ALL_PROMPTS = {
    'PROMPT_ARTICLE_UNITY_Q': PROMPT_ARTICLE_UNITY_Q,
    'PROMPT_ARTICLE_UNITY_A': PROMPT_ARTICLE_UNITY_A,
    'PROMPT_RECITAL_UNITY_Q': PROMPT_RECITAL_UNITY_Q,
    'PROMPT_RECITAL_UNITY_A': PROMPT_RECITAL_UNITY_A,
    'PROMPT_ANNEX_UNITY_Q': PROMPT_ANNEX_UNITY_Q,
    'PROMPT_ANNEX_UNITY_A': PROMPT_ANNEX_UNITY_A,
    'PROMPT_ARTICLE_RECITAL_BINDING_Q': PROMPT_ARTICLE_RECITAL_BINDING_Q,
    'PROMPT_ARTICLE_RECITAL_BINDING_A': PROMPT_ARTICLE_RECITAL_BINDING_A,
    'PROMPT_ANNEX_ARTICLE_BINDING_Q': PROMPT_ANNEX_ARTICLE_BINDING_Q,
    'PROMPT_ANNEX_ARTICLE_BINDING_A': PROMPT_ANNEX_ARTICLE_BINDING_A,
    'PROMPT_ANNEX_RECITAL_BINDING_Q': PROMPT_ANNEX_RECITAL_BINDING_Q,
    'PROMPT_ANNEX_RECITAL_BINDING_A': PROMPT_ANNEX_RECITAL_BINDING_A,
    'PROMPT_AUGMENTATION_Q': PROMPT_AUGMENTATION_Q,
}
