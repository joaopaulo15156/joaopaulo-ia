import os
import json
from openai import OpenAI
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


def gerar_sugestao_openai(codigo: str, api_key: str, tentativa: int) -> dict:
    client = OpenAI(api_key=api_key)

    prompt = f"""
Você é um arquiteto sênior de software.

Analise o código abaixo e gere UMA melhoria objetiva e realista.
A sugestão deve ser pequena ou média, útil e implementável.
Não proponha refatoração gigante.

Tentativa atual: {tentativa}

Responda SOMENTE em JSON:
{{
  "tentativa": {tentativa},
  "titulo": "...",
  "resumo": "...",
  "motivo": "...",
  "tipo": "correcao|seguranca|refatoracao|performance|usabilidade|organizacao",
  "prioridade": "baixa|média|alta",
  "onde_colocar": "...",
  "codigo_sugerido": "...",
  "riscos": "...",
  "pontos_fortes": ["...", "..."],
  "pontos_fracos": ["...", "..."]
}}

Código:
{codigo}
"""

    resp = client.chat.completions.create(
        model=os.getenv("OPENAI_AUTO_MODEL", "gpt-5"),
        messages=[
            {"role": "system", "content": "Responda somente JSON válido."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.35,
        response_format={"type": "json_object"},
    )

    return json.loads(resp.choices[0].message.content)


def avaliar_sugestao_openai(sugestao: dict, codigo: str, api_key: str) -> dict:
    client = OpenAI(api_key=api_key)

    prompt = f"""
Você é um revisor técnico.
Avalie se a sugestão abaixo vale a pena ser implementada no código.

Responda SOMENTE em JSON:
{{
  "ia": "openai",
  "aprovado": true,
  "nota": 0,
  "motivo": "...",
  "riscos": "...",
  "onde_colocar": "...",
  "patch_recomendado": "...",
  "pontos_fortes": ["...", "..."],
  "pontos_fracos": ["...", "..."]
}}

Código atual:
{codigo}

Sugestão:
{json.dumps(sugestao, ensure_ascii=False)}
"""

    resp = client.chat.completions.create(
        model=os.getenv("OPENAI_AUTO_MODEL", "gpt-5"),
        messages=[
            {"role": "system", "content": "Responda somente JSON válido."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
        response_format={"type": "json_object"},
    )

    return json.loads(resp.choices[0].message.content)


def avaliar_sugestao_xai(sugestao: dict, codigo: str, api_key: str) -> dict:
    client = OpenAI(api_key=api_key, base_url="https://api.x.ai/v1")

    prompt = f"""
Você é um revisor técnico.
Avalie se a sugestão abaixo vale a pena ser implementada no código.

Responda SOMENTE em JSON:
{{
  "ia": "xai",
  "aprovado": true,
  "nota": 0,
  "motivo": "...",
  "riscos": "...",
  "onde_colocar": "...",
  "patch_recomendado": "...",
  "pontos_fortes": ["...", "..."],
  "pontos_fracos": ["...", "..."]
}}

Código atual:
{codigo}

Sugestão:
{json.dumps(sugestao, ensure_ascii=False)}
"""

    resp = client.chat.completions.create(
        model=os.getenv("XAI_AUTO_MODEL", "grok-3"),
        messages=[
            {"role": "system", "content": "Responda somente JSON válido."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
    )

    return _extrair_json(resp.choices[0].message.content)


def avaliar_sugestao_gemini(sugestao: dict, codigo: str, api_key: str) -> dict:
    client = genai.Client(api_key=api_key)

    prompt = f"""
Você é um revisor técnico.
Avalie se a sugestão abaixo vale a pena ser implementada no código.

Responda SOMENTE em JSON:
{{
  "ia": "gemini",
  "aprovado": true,
  "nota": 0,
  "motivo": "...",
  "riscos": "...",
  "onde_colocar": "...",
  "patch_recomendado": "...",
  "pontos_fortes": ["...", "..."],
  "pontos_fracos": ["...", "..."]
}}

Código atual:
{codigo}

Sugestão:
{json.dumps(sugestao, ensure_ascii=False)}
"""

    response = client.models.generate_content(
        model=os.getenv("GEMINI_AUTO_MODEL", "gemini-3-flash-preview"),
        contents=prompt,
    )

    return _extrair_json(response.text)
