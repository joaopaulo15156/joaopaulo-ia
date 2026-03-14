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


def _erro_de_cota_ou_chave(msg: str) -> bool:
    msg = (msg or "").lower()
    sinais = [
        "insufficient_quota",
        "exceeded your current quota",
        "invalid_api_key",
        "incorrect api key provided",
        "401",
        "429",
    ]
    return any(s in msg for s in sinais)


def gerar_sugestao_openai(codigo: str, api_key: str, tentativa: int) -> dict:
    if not api_key:
        raise ValueError("OPENAI_API_KEY não encontrada.")

    client = OpenAI(api_key=api_key)

    prompt = f"""
Você é um arquiteto sênior de software.

Analise o código abaixo e gere UMA melhoria objetiva e realista.
A sugestão deve ser pequena ou média, útil e implementável.
Não proponha refatoração gigante.

Tentativa atual: {tentativa}

Responda SOMENTE em JSON:
{{
  "fonte_geradora": "openai",
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
        model="gpt-5",
        messages=[
            {"role": "system", "content": "Responda somente JSON válido."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.35,
        response_format={"type": "json_object"},
    )

    return json.loads(resp.choices[0].message.content)


def gerar_sugestao_xai(codigo: str, api_key: str, tentativa: int) -> dict:
    if not api_key:
        raise ValueError("XAI_API_KEY não encontrada.")

    client = OpenAI(api_key=api_key, base_url="https://api.x.ai/v1")

    prompt = f"""
Você é um arquiteto sênior de software.

Analise o código abaixo e gere UMA melhoria objetiva e realista.
A sugestão deve ser pequena ou média, útil e implementável.
Não proponha refatoração gigante.

Tentativa atual: {tentativa}

Responda SOMENTE em JSON:
{{
  "fonte_geradora": "xai",
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
        model="grok-3",
        messages=[
            {"role": "system", "content": "Responda somente JSON válido."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.35,
    )

    return _extrair_json(resp.choices[0].message.content)


def gerar_sugestao_gemini(codigo: str, api_key: str, tentativa: int) -> dict:
    if not api_key:
        raise ValueError("GEMINI_API_KEY não encontrada.")

    client = genai.Client(api_key=api_key)

    prompt = f"""
Você é um arquiteto sênior de software.

Analise o código abaixo e gere UMA melhoria objetiva e realista.
A sugestão deve ser pequena ou média, útil e implementável.
Não proponha refatoração gigante.

Tentativa atual: {tentativa}

Responda SOMENTE em JSON:
{{
  "fonte_geradora": "gemini",
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

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
    )

    return _extrair_json(response.text)


def avaliar_sugestao_openai(sugestao: dict, codigo: str, api_key: str) -> dict:
    if not api_key:
        raise ValueError("OPENAI_API_KEY não encontrada.")

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
        model="gpt-5",
        messages=[
            {"role": "system", "content": "Responda somente JSON válido."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
        response_format={"type": "json_object"},
    )

    return json.loads(resp.choices[0].message.content)


def avaliar_sugestao_xai(sugestao: dict, codigo: str, api_key: str) -> dict:
    if not api_key:
        raise ValueError("XAI_API_KEY não encontrada.")

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
        model="grok-3",
        messages=[
            {"role": "system", "content": "Responda somente JSON válido."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
    )

    return _extrair_json(resp.choices[0].message.content)


def avaliar_sugestao_gemini(sugestao: dict, codigo: str, api_key: str) -> dict:
    if not api_key:
        raise ValueError("GEMINI_API_KEY não encontrada.")

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
        model="gemini-2.5-flash",
        contents=prompt,
    )

    return _extrair_json(response.text)


def gerar_sugestao_com_fallback(
    codigo: str,
    tentativa: int,
    openai_api_key: str = "",
    xai_api_key: str = "",
    gemini_api_key: str = "",
) -> tuple[dict, list[dict]]:
    erros_geracao = []

    if openai_api_key:
        try:
            return gerar_sugestao_openai(codigo, openai_api_key, tentativa), erros_geracao
        except Exception as e:
            erros_geracao.append({"fonte": "openai", "erro": str(e)})
            if not _erro_de_cota_ou_chave(str(e)):
                raise

    if xai_api_key:
        try:
            return gerar_sugestao_xai(codigo, xai_api_key, tentativa), erros_geracao
        except Exception as e:
            erros_geracao.append({"fonte": "xai", "erro": str(e)})

    if gemini_api_key:
        try:
            return gerar_sugestao_gemini(codigo, gemini_api_key, tentativa), erros_geracao
        except Exception as e:
            erros_geracao.append({"fonte": "gemini", "erro": str(e)})

    raise RuntimeError(
        "Nenhum provedor conseguiu gerar a sugestão inicial. "
        f"Erros: {json.dumps(erros_geracao, ensure_ascii=False)}"
    )
