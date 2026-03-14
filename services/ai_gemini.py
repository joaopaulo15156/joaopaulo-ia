import json
from google import genai


def _extrair_json(texto: str) -> dict:
    texto = (texto or "").strip()

    if texto.startswith("```json"):
        texto = texto[len("```json"):].strip()
    elif texto.startswith("```"):
        texto = texto[len("```"):].strip()

    if texto.endswith("```"):
        texto = texto[:-3].strip()

    return json.loads(texto)


def revisar_codigo_gemini(codigo: str, api_key: str) -> dict:
    if not api_key:
        raise ValueError("GEMINI_API_KEY não encontrada.")

    client = genai.Client(api_key=api_key)

    prompt = f"""
Você é um revisor sênior de código Python e Streamlit.

Analise o código abaixo e responda SOMENTE em JSON válido com esta estrutura:
{{
  "ia": "gemini",
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

    response = client.models.generate_content(
        model="gemini-3-flash-preview",
        contents=prompt,
    )

    return _extrair_json(response.text)
