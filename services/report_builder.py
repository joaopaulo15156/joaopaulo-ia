from datetime import datetime


def _escolher_prioridade(prioridades: list[str]) -> str:
    prioridades_norm = [str(p).lower().strip() for p in prioridades if p]

    if "alta" in prioridades_norm:
        return "alta"
    if "média" in prioridades_norm or "media" in prioridades_norm:
        return "média"
    if "baixa" in prioridades_norm:
        return "baixa"
    return "não definida"


def _primeiro_valido(lista):
    for item in lista:
        if item and str(item).strip():
            return item
    return ""


def montar_relatorio_final(resultados: dict) -> dict:
    agora = datetime.now().isoformat()

    if resultados.get("modo") == "automatico":
        validacao_local = resultados.get("validacao_local", {})
        sugestao_final = resultados.get("sugestao_final") or {}
        avaliacoes_finais = resultados.get("avaliacoes_finais", [])
        rodadas = resultados.get("rodadas", [])
        sucesso = resultados.get("sucesso", False)

        pontos_fortes = []
        pontos_fracos = []
        aprovacoes = []
        reprovacoes = []
        prioridades = []
        onde_colocar = []
        patchs = []
        riscos = []

        for av in avaliacoes_finais:
            if av.get("aprovado") is True:
                aprovacoes.append(av.get("ia", "desconhecida"))
            else:
                reprovacoes.append(av.get("ia", "desconhecida"))

            for item in av.get("pontos_fortes", []) or []:
                pontos_fortes.append({"fonte": av.get("ia", "-"), "item": item})

            for item in av.get("pontos_fracos", []) or []:
                pontos_fracos.append({"fonte": av.get("ia", "-"), "item": item})

            if av.get("onde_colocar"):
                onde_colocar.append(av["onde_colocar"])

            if av.get("patch_recomendado"):
                patchs.append(av["patch_recomendado"])

            if av.get("riscos"):
                riscos.append({"fonte": av.get("ia", "-"), "risco": av["riscos"]})

        if sugestao_final.get("prioridade"):
            prioridades.append(sugestao_final["prioridade"])

        prioridade_final = _escolher_prioridade(prioridades)
        onde_mexer_primeiro = (
            _primeiro_valido(onde_colocar)
            or sugestao_final.get("onde_colocar", "")
            or "Não definido."
        )
        patch_recomendado = (
            _primeiro_valido(patchs)
            or sugestao_final.get("codigo_sugerido", "")
            or "# Nenhum patch recomendado retornado."
        )

        if sucesso:
            resumo_executivo = (
                f"A sugestão automática foi aprovada após {len(rodadas)} tentativa(s). "
                f"O sistema encontrou uma melhoria considerada útil por pelo menos duas IAs."
            )
            melhor_acao_agora = "Aplicar o patch sugerido em ambiente de teste e validar o comportamento."
            vale_a_pena = "Sim"
        else:
            resumo_executivo = (
                f"Nenhuma sugestão automática atingiu o critério mínimo de aprovação em {len(rodadas)} tentativa(s)."
            )
            melhor_acao_agora = "Não aplicar mudanças automáticas agora. Revisar manualmente os pontos fracos."
            vale_a_pena = "Não"

        return {
            "data": agora,
            "modo": "automatico",
            "validacao_local": validacao_local,
            "resumo_executivo": resumo_executivo,
            "vale_a_pena_implementar": vale_a_pena,
            "melhor_acao_agora": melhor_acao_agora,
            "prioridade": prioridade_final,
            "onde_mexer_primeiro": onde_mexer_primeiro,
            "titulo_sugestao": sugestao_final.get("titulo", "Nenhuma sugestão aprovada"),
            "resumo_sugestao": sugestao_final.get("resumo", ""),
            "motivo_sugestao": sugestao_final.get("motivo", ""),
            "tipo_sugestao": sugestao_final.get("tipo", ""),
            "fonte_geradora": sugestao_final.get("fonte_geradora", ""),
            "pontos_fortes": pontos_fortes,
            "pontos_fracos": pontos_fracos,
            "riscos": riscos,
            "ias_que_aprovaram": aprovacoes,
            "ias_que_reprovaram": reprovacoes,
            "patch_recomendado": patch_recomendado,
            "rodadas": rodadas,
        }

    validacao_local = resultados.get("validacao_local", {})
    analises_por_ia = {}
    problemas_detectados = []
    prioridades = []
    onde_colocar_lista = []
    patches = []
    resumos = []

    for fonte, conteudo in resultados.items():
        if fonte == "validacao_local":
            continue

        if "erro" in conteudo:
            analises_por_ia[fonte] = {"erro": conteudo["erro"]}
            continue

        analises_por_ia[fonte] = conteudo

        prioridade = conteudo.get("prioridade", "")
        if prioridade:
            prioridades.append(prioridade)

        onde_colocar = conteudo.get("onde_colocar", "")
        if onde_colocar:
            onde_colocar_lista.append(onde_colocar)

        patch = conteudo.get("patch_recomendado", "") or conteudo.get("codigo_sugerido", "")
        if patch:
            patches.append(patch)

        nome_melhoria = conteudo.get("nome_melhoria", "")
        if nome_melhoria:
            resumos.append(f"{fonte}: {nome_melhoria}")

        erros = conteudo.get("erros_encontrados", [])
        if isinstance(erros, list):
            for erro in erros:
                problemas_detectados.append({
                    "fonte": fonte,
                    "problema": erro
                })
        elif erros:
            problemas_detectados.append({
                "fonte": fonte,
                "problema": str(erros)
            })

    prioridade_final = _escolher_prioridade(prioridades)
    onde_mexer_primeiro = _primeiro_valido(onde_colocar_lista)
    patch_recomendado = _primeiro_valido(patches)

    if validacao_local.get("ok") is False:
        resumo_executivo = (
            f"Foi detectado erro de sintaxe local antes mesmo da revisão por IA. "
            f"Linha: {validacao_local.get('linha')} | Detalhe: {validacao_local.get('mensagem')}"
        )
        melhor_acao_agora = "Corrigir o erro de sintaxe antes de confiar em qualquer refatoração maior."
    elif problemas_detectados:
        resumo_executivo = "As IAs encontraram problemas reais e sugeriram correções pontuais no código."
        melhor_acao_agora = "Aplicar primeiro o patch mais consistente e depois testar localmente."
    else:
        resumo_executivo = "Nenhum erro crítico foi confirmado, mas há melhorias estruturais recomendadas."
        melhor_acao_agora = "Aplicar a melhoria estrutural sugerida e validar o comportamento do app."

    return {
        "data": agora,
        "modo": "manual",
        "validacao_local": validacao_local,
        "resumo_executivo": resumo_executivo,
        "melhor_acao_agora": melhor_acao_agora,
        "prioridade": prioridade_final,
        "onde_mexer_primeiro": onde_mexer_primeiro or "Não definido pelas IAs.",
        "problemas_detectados": problemas_detectados,
        "analises_por_ia": analises_por_ia,
        "patch_recomendado": patch_recomendado or "# Nenhum patch recomendado retornado.",
        "resumo_fontes": resumos,
    }
