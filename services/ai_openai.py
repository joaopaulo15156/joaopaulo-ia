import ast
from typing import Any

from services.ai_openai import revisar_codigo_openai
from services.ai_xai import revisar_codigo_xai
from services.ai_gemini import revisar_codigo_gemini
from services.auto_programmer import (
    gerar_sugestao_openai,
    avaliar_sugestao_openai,
    avaliar_sugestao_xai,
    avaliar_sugestao_gemini,
)


def validar_codigo_python(codigo: str) -> dict[str, Any]:
    try:
        ast.parse(codigo)
        return {
            "ok": True,
            "mensagem": "Sintaxe Python válida.",
            "linha": None,
            "offset": None,
        }
    except SyntaxError as e:
        return {
            "ok": False,
            "mensagem": str(e),
            "linha": getattr(e, "lineno", None),
            "offset": getattr(e, "offset", None),
        }


def revisar_codigo_multi_ia(
    codigo: str,
    openai_api_key: str = "",
    xai_api_key: str = "",
    gemini_api_key: str = "",
    usar_openai: bool = True,
    usar_xai: bool = True,
    usar_gemini: bool = True,
    validar_sintaxe: bool = True,
) -> dict:
    resultados = {}

    if validar_sintaxe:
        resultados["validacao_local"] = validar_codigo_python(codigo)
    else:
        resultados["validacao_local"] = {
            "ok": None,
            "mensagem": "Validação local desativada.",
            "linha": None,
            "offset": None,
        }

    if usar_openai:
        try:
            resultados["openai"] = revisar_codigo_openai(codigo, openai_api_key)
        except Exception as e:
            resultados["openai"] = {"erro": str(e)}

    if usar_xai:
        try:
            resultados["xai"] = revisar_codigo_xai(codigo, xai_api_key)
        except Exception as e:
            resultados["xai"] = {"erro": str(e)}

    if usar_gemini:
        try:
            resultados["gemini"] = revisar_codigo_gemini(codigo, gemini_api_key)
        except Exception as e:
            resultados["gemini"] = {"erro": str(e)}

    return resultados


def executar_modo_programador_automatico(
    codigo: str,
    openai_api_key: str,
    xai_api_key: str = "",
    gemini_api_key: str = "",
    max_tentativas: int = 3,
) -> dict:
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY não encontrada. Ela é obrigatória para gerar a sugestão inicial.")

    validacao_local = validar_codigo_python(codigo)

    rodadas = []
    sugestao_escolhida = None
    avaliacoes_escolhidas = None

    for tentativa in range(1, max_tentativas + 1):
        sugestao = gerar_sugestao_openai(codigo, openai_api_key, tentativa)

        avaliacoes = []

        try:
            avaliacoes.append(avaliar_sugestao_openai(sugestao, codigo, openai_api_key))
        except Exception as e:
            avaliacoes.append({"ia": "openai", "erro": str(e), "aprovado": False, "nota": 0})

        if xai_api_key:
            try:
                avaliacoes.append(avaliar_sugestao_xai(sugestao, codigo, xai_api_key))
            except Exception as e:
                avaliacoes.append({"ia": "xai", "erro": str(e), "aprovado": False, "nota": 0})

        if gemini_api_key:
            try:
                avaliacoes.append(avaliar_sugestao_gemini(sugestao, codigo, gemini_api_key))
            except Exception as e:
                avaliacoes.append({"ia": "gemini", "erro": str(e), "aprovado": False, "nota": 0})

        aprovadas = [a for a in avaliacoes if a.get("aprovado") is True]
        reprovadas = [a for a in avaliacoes if a.get("aprovado") is not True]

        rodada = {
            "tentativa": tentativa,
            "sugestao": sugestao,
            "avaliacoes": avaliacoes,
            "aprovadas": len(aprovadas),
            "reprovadas": len(reprovadas),
        }
        rodadas.append(rodada)

        if len(aprovadas) >= 2:
            sugestao_escolhida = sugestao
            avaliacoes_escolhidas = avaliacoes
            break

    sucesso = sugestao_escolhida is not None

    return {
        "modo": "automatico",
        "validacao_local": validacao_local,
        "sucesso": sucesso,
        "max_tentativas": max_tentativas,
        "rodadas": rodadas,
        "sugestao_final": sugestao_escolhida,
        "avaliacoes_finais": avaliacoes_escolhidas or [],
    }
