import os
import json
import logging
from typing import List, Dict, Any
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

DEEPINFRA_API_KEY = os.getenv("DEEPINFRA_API_KEY")
MODEL_NAME = "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo"

try:
    client = OpenAI(
        api_key=DEEPINFRA_API_KEY,
        base_url="https://api.deepinfra.com/v1/openai"
    )
except Exception as e:
    logger.warning(f"Could not initialize OpenAI client: {e}")
    client = None

def _format_diagnostics(diagnostics: List[Dict[str, Any]]) -> str:
    """Format the diagnostics table into a readable string for the LLM."""
    lines = []
    for row in diagnostics:
        param = row.get("parameter", "")
        val = row.get("value", "")
        ref = row.get("reference", "")
        diff = row.get("diff", "")
        status = "ABNORMAL" if row.get("is_abnormal") else "Normal"
        lines.append(f"- {param}: {val} (Ref: {ref}, Diff: {diff}) [{status}]")
    return "\n".join(lines)


def generate_summary_stream(diagnostics: List[Dict[str, Any]]):
    """Auto-generate a 3-sentence clinical summary, yielding chunks as they stream in."""
    if not client or not DEEPINFRA_API_KEY:
        yield "AI Summary unavailable (API key missing)."
        return
        
    system_prompt = (
        "You are an expert orthodontist. Review these cephalometric measurements. "
        "Write a purely objective 3-sentence clinical summary of the skeletal class, "
        "growth pattern, and soft-tissue profile. Do not recommend treatments.\n"
        "CRITICAL RULES:\n"
        "- Flag any measurement with |diff| > 4° or |diff| > 4mm as 'significant' and ensure it is mentioned in the summary.\n"
        "- Never describe a finding as 'normal' if its |diff| > 2 units.\n"
        "- Explicitly mention BOTH maxillary AND mandibular positions (e.g., both maxillary and mandibular retrusion) if they deviate from the norm.\n"
        "OUTPUT ONLY THE MEDICAL SUMMARY TEXT. Do not include any introductory phrases like 'Here is the summary' or conversational filler."
    )
    
    user_content = f"Patient Diagnostics:\n{_format_diagnostics(diagnostics)}"
    
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ],
            temperature=0.3,
            max_tokens=200,
            stream=True,  # Enable streaming
        )
        for chunk in response:
            if chunk.choices[0].delta.content is not None:
                yield chunk.choices[0].delta.content
    except Exception as e:
        logger.error(f"Error generating AI summary stream: {e}")
        yield "AI Summary generation failed."

def generate_summary(diagnostics: List[Dict[str, Any]]) -> str:
    """Non-streaming fallback (retained for backward compatibility if needed)."""
    return "".join([chunk for chunk in generate_summary_stream(diagnostics)])


def ask_question(diagnostics: List[Dict[str, Any]], question: str) -> str:
    """Answer a specific question about the patient's X-Ray diagnostics."""
    if not client or not DEEPINFRA_API_KEY:
        return "AI Chat unavailable (API key missing)."
        
    system_prompt = (
        "You are an orthodontic AI assistant. Use the provided patient cephalometric "
        "data to answer the doctor's question concisely. Be clinical, objective, and direct."
    )
    
    user_content = f"Patient Diagnostics:\n{_format_diagnostics(diagnostics)}\n\nDoctor's Question: {question}"
    
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ],
            temperature=0.3,
            max_tokens=300,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Error answering AI question: {e}")
        return "I encountered an error analyzing the diagnostics for your question."
