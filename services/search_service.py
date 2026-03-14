def build_search_query(pergunta: str, hoje: str, ano: int, deep_search: bool) -> str:
    return f'{pergunta} "{hoje}" {ano}' if deep_search else f"{pergunta} {ano}"


def formatar_resultados(resultados: list[dict]) -> str:
    if not resultados:
        return "Nenhum resultado atual encontrado."

    blocos = []
    for i, r in enumerate(resultados, start=1):
        blocos.append(
            f"{i}. Título: {r.get('title', 'Sem título')}\n"
            f"Link: {r.get('link', '')}\n"
            f"Resumo: {r.get('snippet', 'Sem descrição')}"
        )
    return "\n\n".join(blocos)


def exibir_fontes_streamlit(st, resultados: list[dict]):
    if not resultados:
        st.info("Nenhuma fonte encontrada.")
        return

    with st.expander("Fontes pesquisadas"):
        for i, r in enumerate(resultados, start=1):
            st.markdown(f"**{i}. {r.get('title', 'Sem título')}**")
            st.write(r.get("snippet", "Sem descrição"))
            link = r.get("link", "")
            if link:
                st.write(link)
            st.markdown("---")
