import os
import json
from openai import OpenAI


def revisar_codigo_openai(codigo: str) -> dict:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY não encontrada.")

    client = OpenAI(api_key=api_key)

    prompt = f"""
Você é um revisor sênior de código Python e Streamlit.

Analise o código abaixo e responda SOMENTE em JSON válido com esta estrutura:
{{
  "ia": "openai",
  "nome_melhoria": "...",
  "objetivo": "...",
  "erros_encontrados": ["...", "..."],
  "onde_colocar": "...",
  "o_que_melhora": "...",
  "riscos": "...",
  "prioridade": "baixa|média|alta",
  "patch_recomendado": "código corrigido ou trecho sugerido"
}}

Se não encontrar erro grave, ainda assim proponha a melhoria mais útil.

Código:
{codigo}
"""

    resp = client.chat.completions.create(
        model=os.getenv("OPENAI_REVIEW_MODEL", "gpt-5"),
        messages=[
            {"role": "system", "content": "Responda apenas JSON válido."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
        response_format={"type": "json_object"},
    )

    return json.loads(resp.choices[0].message.content)
