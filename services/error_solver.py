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


def resolver_erro_openai(
    api_key: str,
    origem: str,
    ia_que_falhou: str,
    tipo_erro: str,
    mensagem_erro: str,
    codigo_analisado: str,
) -> dict:
    client = OpenAI(api_key=api_key)

    prompt = f"""
Você é um engenheiro de software especialista em depuração.

Analise este erro de sistema e proponha uma correção.

Responda SOMENTE em JSON:
{{
  "ia": "openai",
  "explicacao_erro": "...",
  "motivo_provavel": "...",
  "vale_a_pena_corrigir_agora": true,
  "prioridade": "baixa|média|alta",
  "onde_colocar": "...",
  "codigo_sugerido": "...",
  "confianca": "baixa|média|alta"
}}

Origem: {origem}
IA que falhou: {ia_que_falhou}
Tipo do erro: {tipo_erro}
Mensagem do erro: {mensagem_erro}

Código analisado:
{codigo_analisado}
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


def resolver_erro_xai(
    api_key: str,
    origem: str,
    ia_que_falhou: str,
    tipo_erro: str,
    mensagem_erro: str,
    codigo_analisado: str,
) -> dict:
    client = OpenAI(api_key=api_key, base_url="https://api.x.ai/v1")

    prompt = f"""
Você é um engenheiro de software especialista em depuração.

Analise este erro de sistema e proponha uma correção.

Responda SOMENTE em JSON:
{{
  "ia": "xai",
  "explicacao_erro": "...",
  "motivo_provavel": "...",
  "vale_a_pena_corrigir_agora": true,
  "prioridade": "baixa|média|alta",
  "onde_colocar": "...",
  "codigo_sugerido": "...",
  "confianca": "baixa|média|alta"
}}

Origem: {origem}
IA que falhou: {ia_que_falhou}
Tipo do erro: {tipo_erro}
Mensagem do erro: {mensagem_erro}

Código analisado:
{codigo_analisado}
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


def resolver_erro_gemini(
    api_key: str,
    origem: str,
    ia_que_falhou: str,
    tipo_erro: str,
    mensagem_erro: str,
    codigo_analisado: str,
) -> dict:
    client = genai.Client(api_key=api_key)

    prompt = f"""
Você é um engenheiro de software especialista em depuração.

Analise este erro de sistema e proponha uma correção.

Responda SOMENTE em JSON:
{{
  "ia": "gemini",
  "explicacao_erro": "...",
  "motivo_provavel": "...",
  "vale_a_pena_corrigir_agora": true,
  "prioridade": "baixa|média|alta",
  "onde_colocar": "...",
  "codigo_sugerido": "...",
  "confianca": "baixa|média|alta"
}}

Origem: {origem}
IA que falhou: {ia_que_falhou}
Tipo do erro: {tipo_erro}
Mensagem do erro: {mensagem_erro}

Código analisado:
{codigo_analisado}
"""

    response = client.models.generate_content(
        model="gemini-3-flash-preview",
        contents=prompt,
    )

    return _extrair_json(response.text)


def resolver_erro_multi_ia(
    origem: str,
    ia_que_falhou: str,
    tipo_erro: str,
    mensagem_erro: str,
    codigo_analisado: str,
    openai_api_key: str = "",
    xai_api_key: str = "",
    gemini_api_key: str = "",
) -> dict:
    resultados = {}

    if openai_api_key:
        try:
            resultados["openai"] = resolver_erro_openai(
                openai_api_key,
                origem,
                ia_que_falhou,
                tipo_erro,
                mensagem_erro,
                codigo_analisado,
            )
        except Exception as e:
            resultados["openai"] = {"erro": str(e)}

    if xai_api_key:
        try:
            resultados["xai"] = resolver_erro_xai(
                xai_api_key,
                origem,
                ia_que_falhou,
                tipo_erro,
                mensagem_erro,
                codigo_analisado,
            )
        except Exception as e:
            resultados["xai"] = {"erro": str(e)}

    if gemini_api_key:
        try:
            resultados["gemini"] = resolver_erro_gemini(
                gemini_api_key,
                origem,
                ia_que_falhou,
                tipo_erro,
                mensagem_erro,
                codigo_analisado,
            )
        except Exception as e:
            resultados["gemini"] = {"erro": str(e)}

    return resultados
