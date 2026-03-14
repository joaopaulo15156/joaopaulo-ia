import json


def montar_contexto_chat(
    pergunta: str,
    data_longa_ptbr: str,
    memory: dict,
    contexto_busca: str,
    usar_pesquisa: bool,
    resultados: list[dict],
    mensagens_anteriores: list[dict],
    system_prompt: str,
) -> list[dict]:
    memory_context = json.dumps(memory, ensure_ascii=False) if memory else "{}"

    messages = [{"role": "system", "content": system_prompt}]
    for m in mensagens_anteriores[-8:-1]:
        messages.append(m)

    if usar_pesquisa and resultados:
        conteudo_user = (
            f"Pergunta do usuário:\n{pergunta}\n\n"
            f"Data atual: {data_longa_ptbr}\n\n"
            f"Memória global:\n{memory_context}\n\n"
            f"Resultados atuais da pesquisa:\n{contexto_busca}\n\n"
            f"Responda usando primeiro os resultados atuais."
        )
    else:
        conteudo_user = (
            f"Pergunta do usuário:\n{pergunta}\n\n"
            f"Data atual: {data_longa_ptbr}\n\n"
            f"Memória global:\n{memory_context}\n\n"
            f"Se a pergunta exigir fato atual e não houver pesquisa, deixe isso claro."
        )

    messages.append({"role": "user", "content": conteudo_user})
    return messages
