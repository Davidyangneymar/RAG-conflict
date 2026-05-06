# services/p6_runner.py
import os
import logging
import re
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
logger = logging.getLogger(__name__)

_llm_client = None

def get_llm_client():
    """Return an OpenAI-compatible client (supports Zhipu, OpenAI, vLLM, Ollama, etc.)"""
    global _llm_client
    if _llm_client is None:
        api_key = os.getenv("LLM_API_KEY")
        if not api_key:
            raise ValueError("LLM_API_KEY not set (e.g., ZHIPUAI_API_KEY or OPENAI_API_KEY)")
        base_url = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")
        _llm_client = OpenAI(api_key=api_key, base_url=base_url)
        logger.info(f"LLM client initialized with base_url={base_url}")
    return _llm_client

def _extract_keywords(text: str) -> set:
    stopwords = {"what","how","why","where","when","which","who","is","are","was","were",
                 "be","the","a","an","and","or","of","to","for","in","on","at","with",
                 "by","from","that","this","these","those","it","they","we","you"}
    words = re.findall(r'\b[a-z]{3,}\b', text.lower())
    return {w for w in words if w not in stopwords}

def _evidence_relevant_to_query(evidence_texts: list, query: str) -> bool:
    query_keywords = _extract_keywords(query)
    if not query_keywords:
        return True
    for text in evidence_texts:
        text_keywords = _extract_keywords(text)
        if query_keywords & text_keywords:
            return True
    return False

def _generate_plain_answer(query: str, evidence_texts: list) -> str:
    if not evidence_texts or not _evidence_relevant_to_query(evidence_texts, query):
        return "I don't know based on the provided evidence."

    client = get_llm_client()
    model = os.getenv("LLM_MODEL", "gpt-3.5-turbo")
    evidence = "\n".join([f"- {t[:500]}" for t in evidence_texts])
    custom_prompt = f"""
Answer the user's question based ONLY on the evidence below.

User question: {query}

Evidence:
{evidence}

Instructions:
- Use only information from the evidence. Do not add external knowledge.
- If the evidence does not directly answer the question, say "The evidence does not provide an answer."
- Keep answer concise, plain English, max 3 sentences.
- Do not output JSON, markdown, or any code blocks.
"""
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": custom_prompt}],
            temperature=0.3,
            max_tokens=200,
        )
        answer = response.choices[0].message.content
        return answer.strip() if answer else "No answer generated."
    except Exception as e:
        logger.error(f"LLM call failed: {e}")
        return "Error: Could not generate answer."

def run_p6(typed_output, input_record, p6_func) -> dict:
    if p6_func is None:
        raise RuntimeError("P6 function not initialized")
    plans = p6_func(typed_output.samples, [input_record])
    if not plans:
        return {
            "final_answer": "No answer generated",
            "abstention": {"should_abstain": True},
            "answer_context": {"citations": []},
            "confidence": 0.0
        }
    plan = plans[0]
    plan_dict = plan.to_dict() if hasattr(plan, 'to_dict') else plan

    answer_ctx = plan_dict.get("answer_context", {})
    query = answer_ctx.get("query", "")
    evidence_clusters = answer_ctx.get("evidence_clusters", {})
    evidence_texts = []
    for cluster in evidence_clusters.values():
        for item in cluster:
            text = item.get("text", "") if isinstance(item, dict) else getattr(item, "text", "")
            if text:
                evidence_texts.append(text)

    final_answer = _generate_plain_answer(query, evidence_texts)
    plan_dict["final_answer"] = final_answer
    plan_dict["confidence"] = 0.7
    return plan_dict

def run_p6_benchmark(typed_output, input_record, p6_func, p1_output: dict) -> dict:
    """
    Benchmark version: returns predicted label and best scores based on P1 NLI results.
    This avoids LLM call and uses the strongest NLI signal.
    """
    # Extract NLI results from P1 output
    nli_results = p1_output.get("nli_results", [])
    if not nli_results:
        # No cross-source pairs, fallback to neutral
        return {
            "predicted_label": "neutral",
            "best_entailment_score": 0.0,
            "best_contradiction_score": 0.0,
            "best_neutral_score": 1.0,
            "answer_context": {}
        }

    # Find highest score among all NLI results for each label
    best_ent = max((r.get("entailment_score", 0.0) for r in nli_results), default=0.0)
    best_con = max((r.get("contradiction_score", 0.0) for r in nli_results), default=0.0)
    best_neu = max((r.get("neutral_score", 0.0) for r in nli_results), default=0.0)

    # Determine predicted label
    scores = {"entailment": best_ent, "contradiction": best_con, "neutral": best_neu}
    predicted_label = max(scores, key=scores.get)

    return {
        "predicted_label": predicted_label,
        "best_entailment_score": best_ent,
        "best_contradiction_score": best_con,
        "best_neutral_score": best_neu,
        "answer_context": {}
    }