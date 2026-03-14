import os
import json
from openai import OpenAI


def _extrair_json(texto: str) -> dict:
    texto = (texto or "").strip()

    if texto.startswith("```json"):
        texto = texto[len("```json"):].strip()
    elif texto.startswith("```"):
        texto = texto[len("```"):].strip()

    if texto.endswith("```"):
        texto = texto[:-3].strip()

    return json.loads(texto)


def revisar_codigo_xai(codigo: str) -> dict:
    api_key = os.getenv("XAI_API_KEY")
    if not api_key:
        raise ValueError("XAI_API_KEY não encontrada.")

    client = OpenAI(
        api_key=api_key,
        base_url="https://api.x.ai/v1",
    )

    prompt = f"""
Você é um revisor sênior de código Python e Streamlit.

Analise o código abaixo e responda SOMENTE em JSON válido com esta estrutura:
{{
  "ia": "xai",
  "nome_melhoria": "...",
  "objetivo": "...",
  "erros_encontrados": ["...", "..."],
  "onde_colocar": "...",
  "o_que_melhora": "...",
  "riscos": "...",
  "prioridade": "baixa|média|alta",
  "patch_recomendado": "código corrigido ou trecho sugerido"
}}

Código:
{codigo}
"""

    resp = client.chat.completions.create(
        model=os.getenv("XAI_REVIEW_MODEL", "grok-3"),
        messages=[
            {"role": "system", "content": "Responda apenas JSON válido."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
    )

    return _extrair_json(resp.choices[0].message.content)
