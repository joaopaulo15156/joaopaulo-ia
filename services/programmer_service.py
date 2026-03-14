import json
import streamlit as st

from services.code_review import (
    revisar_codigo_multi_ia,
    executar_modo_programador_automatico,
)
from services.report_builder import montar_relatorio_final, extrair_patch_consolidado
from services.error_solver import resolver_erro_multi_ia


def render_programmer_manual(
    get_secret_func,
    data_curta_func,
    db_save_daily_report_func,
    db_save_ia_error_func,
):
    st.markdown("## Revisão manual multi-IA")
    st.caption("Cole um código abaixo para analisar com OpenAI, xAI e Gemini.")

    codigo_dev = st.text_area(
        "Cole o código para análise",
        height=320,
        placeholder="Cole aqui o código que você quer revisar",
        key="manual_codigo_dev",
    )

    col_a, col_b = st.columns(2)

    with col_a:
        usar_openai = st.checkbox("Usar OpenAI", value=True, key="manual_usar_openai")
        usar_xai = st.checkbox("Usar xAI / Grok", value=True, key="manual_usar_xai")
        usar_gemini = st.checkbox("Usar Gemini", value=True, key="manual_usar_gemini")

    with col_b:
        validar_sintaxe = st.checkbox("Validar sintaxe Python antes", value=True, key="manual_validar")
        salvar_relatorio = st.checkbox("Salvar relatório no banco", value=True, key="manual_salvar")

    if st.button("Analisar com múltiplas IAs", key="btn_manual_multi"):
        if not codigo_dev.strip():
            st.warning("Cole um código para análise.")
            return

        try:
            with st.spinner("Consultando múltiplas IAs..."):
                resultados = revisar_codigo_multi_ia(
                    codigo=codigo_dev,
                    openai_api_key=get_secret_func("OPENAI_API_KEY"),
                    xai_api_key=get_secret_func("XAI_API_KEY"),
                    gemini_api_key=get_secret_func("GEMINI_API_KEY"),
                    usar_openai=usar_openai,
                    usar_xai=usar_xai,
                    usar_gemini=usar_gemini,
                    validar_sintaxe=validar_sintaxe,
                )
                relatorio_final = montar_relatorio_final(resultados)

            st.markdown("### Relatório consolidado")
            st.json(relatorio_final)

            patch_consolidado = extrair_patch_consolidado(relatorio_final)
            if patch_consolidado:
                st.markdown("### Patch consolidado")
                st.code(patch_consolidado, language="python")

            if salvar_relatorio:
                try:
                    db_save_daily_report_func(data_curta_func(), relatorio_final)
                except Exception as e:
                    st.warning(f"Não foi possível salvar no banco: {e}")

            st.download_button(
                "Baixar relatório técnico",
                data=json.dumps(relatorio_final, ensure_ascii=False, indent=2),
                file_name=f"relatorio_multi_ia_{data_curta_func()}.json",
                mime="application/json",
                key="download_relatorio_manual",
            )

        except Exception as e:
            erro_txt = str(e)

            try:
                db_save_ia_error_func(
                    origem="modo_programador_manual",
                    ia_que_falhou="desconhecida",
                    tipo_erro="analise_manual",
                    mensagem_erro=erro_txt,
                    codigo_analisado=codigo_dev,
                )
            except Exception:
                pass

            st.error(f"Erro na análise multi-IA: {e}")


def render_programmer_automatic(
    get_secret_func,
    data_curta_func,
    db_save_daily_report_func,
    db_save_ia_error_func,
    codigo_padrao: str,
):
    st.markdown("## Geração automática de melhoria")
    st.caption("Você aperta um botão e a IA tenta encontrar sozinha uma melhoria que realmente compense.")

    usar_codigo_do_app = st.checkbox(
        "Usar automaticamente o código atual do app",
        value=True,
        key="auto_usar_codigo_app",
    )

    if usar_codigo_do_app:
        codigo_auto = codigo_padrao
        st.text_area(
            "Código detectado automaticamente",
            value=codigo_auto,
            height=220,
            disabled=True,
            key="auto_codigo_detectado",
        )
    else:
        codigo_auto = st.text_area(
            "Cole o código que será usado no modo automático",
            height=320,
            placeholder="Cole aqui o código que a IA vai melhorar automaticamente",
            key="auto_codigo_colado",
        )

    col1, col2, col3 = st.columns(3)

    with col1:
        max_tentativas = st.slider("Máximo de tentativas", 1, 5, 3, key="auto_max_tentativas")

    with col2:
        salvar_automatico = st.checkbox("Salvar relatório no banco", value=True, key="auto_salvar")

    with col3:
        mostrar_rodadas = st.checkbox("Mostrar rodadas da análise", value=True, key="auto_rodadas")

    if st.button("🚀 Gerar melhoria automática", key="btn_auto_melhoria"):
        if not codigo_auto.strip():
            st.warning("Não há código para analisar.")
            return

        try:
            with st.spinner("Gerando e validando melhoria automática..."):
                resultado_auto = executar_modo_programador_automatico(
                    codigo=codigo_auto,
                    openai_api_key=get_secret_func("OPENAI_API_KEY"),
                    xai_api_key=get_secret_func("XAI_API_KEY"),
                    gemini_api_key=get_secret_func("GEMINI_API_KEY"),
                    max_tentativas=max_tentativas,
                )
                relatorio_final = montar_relatorio_final(resultado_auto)

            st.markdown("### Relatório automático final")

            colm1, colm2, colm3 = st.columns(3)
            with colm1:
                st.metric("Vale a pena implementar?", relatorio_final.get("vale_a_pena_implementar", "-"))
            with colm2:
                st.metric("Prioridade", relatorio_final.get("prioridade", "-"))
            with colm3:
                st.metric("Rodadas", len(relatorio_final.get("rodadas", [])))

            st.write(f"**Resumo executivo:** {relatorio_final.get('resumo_executivo', '-')}")
            st.write(f"**Melhor ação agora:** {relatorio_final.get('melhor_acao_agora', '-')}")
            st.write(f"**Sugestão escolhida:** {relatorio_final.get('titulo_sugestao', '-')}")
            st.write(f"**Tipo da sugestão:** {relatorio_final.get('tipo_sugestao', '-')}")
            st.write(f"**Fonte geradora:** {relatorio_final.get('fonte_geradora', '-')}")
            st.write(f"**Onde mexer primeiro:** {relatorio_final.get('onde_mexer_primeiro', '-')}")

            st.markdown("#### Pontos fortes")
            st.json(relatorio_final.get("pontos_fortes", []))

            st.markdown("#### Pontos fracos")
            st.json(relatorio_final.get("pontos_fracos", []))

            st.markdown("#### Riscos")
            st.json(relatorio_final.get("riscos", []))

            st.markdown("#### IAs que aprovaram")
            st.write(relatorio_final.get("ias_que_aprovaram", []))

            st.markdown("#### IAs que reprovaram")
            st.write(relatorio_final.get("ias_que_reprovaram", []))

            st.markdown("#### Patch recomendado")
            st.code(relatorio_final.get("patch_recomendado", ""), language="python")

            patch_consolidado = extrair_patch_consolidado(relatorio_final)
            if patch_consolidado:
                st.markdown("#### Patch consolidado")
                st.code(patch_consolidado, language="python")

            if mostrar_rodadas:
                st.markdown("#### Rodadas da análise")
                for rodada in relatorio_final.get("rodadas", []):
                    with st.expander(f"Tentativa {rodada.get('tentativa')}"):
                        st.write("**Sugestão gerada:**")
                        st.json(rodada.get("sugestao", {}))
                        if rodada.get("erros_geracao"):
                            st.write("**Erros ao gerar a sugestão:**")
                            st.json(rodada.get("erros_geracao", []))
                        st.write("**Avaliações:**")
                        st.json(rodada.get("avaliacoes", []))

            if salvar_automatico:
                try:
                    db_save_daily_report_func(data_curta_func(), relatorio_final)
                except Exception as e:
                    st.warning(f"Não foi possível salvar no banco: {e}")

            st.download_button(
                "Baixar relatório automático",
                data=json.dumps(relatorio_final, ensure_ascii=False, indent=2),
                file_name=f"relatorio_auto_{data_curta_func()}.json",
                mime="application/json",
                key="download_relatorio_auto",
            )

        except Exception as e:
            erro_txt = str(e)

            try:
                db_save_ia_error_func(
                    origem="modo_programador_automatico",
                    ia_que_falhou="desconhecida",
                    tipo_erro="execucao_automatica",
                    mensagem_erro=erro_txt,
                    codigo_analisado=codigo_auto,
                )
            except Exception:
                pass

            st.error(f"Erro no modo programador automático: {e}")


def render_programmer_errors(
    get_secret_func,
    db_list_ia_errors_func,
    db_update_ia_error_report_func,
    db_mark_error_resolved_func,
):
    st.markdown("## Erros da IA")

    erros_ia = db_list_ia_errors_func()

    status_opcoes = ["todos", "pendente", "analisado", "resolvido"]
    origem_opcoes = ["todos"] + sorted(list({str(x.get("origem", "desconhecida")) for x in erros_ia})) if erros_ia else ["todos"]
    ia_opcoes = ["todos"] + sorted(list({str(x.get("ia_que_falhou", "desconhecida")) for x in erros_ia})) if erros_ia else ["todos"]

    c1, c2, c3 = st.columns(3)
    with c1:
        filtro_status = st.selectbox("Filtrar por status", status_opcoes, key="erro_filtro_status")
    with c2:
        filtro_origem = st.selectbox("Filtrar por origem", origem_opcoes, key="erro_filtro_origem")
    with c3:
        filtro_ia = st.selectbox("Filtrar por IA", ia_opcoes, key="erro_filtro_ia")

    if st.button("Atualizar lista de erros", key="btn_atualizar_erros"):
        st.rerun()

    if not erros_ia:
        st.info("Nenhum erro da IA salvo ainda.")
        return

    erros_filtrados = []
    for item in erros_ia:
        ok_status = filtro_status == "todos" or item.get("status_resolucao", "") == filtro_status
        ok_origem = filtro_origem == "todos" or item.get("origem", "") == filtro_origem
        ok_ia = filtro_ia == "todos" or item.get("ia_que_falhou", "") == filtro_ia

        if ok_status and ok_origem and ok_ia:
            erros_filtrados.append(item)

    if not erros_filtrados:
        st.info("Nenhum erro encontrado com esses filtros.")
        return

    st.caption(f"Total filtrado: {len(erros_filtrados)}")

    for item in erros_filtrados:
        error_id = item["id"]
        origem = item.get("origem", "-")
        ia_que_falhou = item.get("ia_que_falhou", "-")
        tipo_erro = item.get("tipo_erro", "-")
        mensagem_erro = item.get("mensagem_erro", "-")
        codigo_analisado = item.get("codigo_analisado", "")
        status_resolucao = item.get("status_resolucao", "pendente")
        relatorio_json = item.get("relatorio_json") or {}

        with st.expander(f"Erro #{error_id} • {origem} • {status_resolucao}"):
            st.write(f"**IA que falhou:** {ia_que_falhou}")
            st.write(f"**Tipo do erro:** {tipo_erro}")
            st.write(f"**Mensagem:** {mensagem_erro}")

            if codigo_analisado:
                st.markdown("**Código analisado:**")
                st.code(codigo_analisado, language="python")

            col1, col2 = st.columns(2)

            with col1:
                if st.button(
                    "Mandar erro para outras IAs resolverem",
                    key=f"resolver_erro_{error_id}"
                ):
                    try:
                        with st.spinner("Enviando erro para análise multi-IA..."):
                            relatorio_erro = resolver_erro_multi_ia(
                                origem=origem,
                                ia_que_falhou=ia_que_falhou,
                                tipo_erro=tipo_erro,
                                mensagem_erro=mensagem_erro,
                                codigo_analisado=codigo_analisado,
                                openai_api_key=get_secret_func("OPENAI_API_KEY"),
                                xai_api_key=get_secret_func("XAI_API_KEY"),
                                gemini_api_key=get_secret_func("GEMINI_API_KEY"),
                            )

                            db_update_ia_error_report_func(
                                error_id=error_id,
                                relatorio_json=relatorio_erro,
                                status_resolucao="analisado"
                            )

                        st.success("Erro enviado para as IAs e relatório salvo.")
                        st.rerun()

                    except Exception as e:
                        st.error(f"Erro ao resolver com múltiplas IAs: {e}")

            with col2:
                if st.button(
                    "Marcar como resolvido",
                    key=f"resolver_status_{error_id}"
                ):
                    try:
                        db_mark_error_resolved_func(error_id)
                        st.success("Erro marcado como resolvido.")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Erro ao marcar como resolvido: {e}")

            if relatorio_json:
                st.markdown("### Relatório do erro")
                st.json(relatorio_json)

                patchs = []
                for nome_ia, dados in relatorio_json.items():
                    if isinstance(dados, dict) and "codigo_sugerido" in dados:
                        st.markdown(f"#### Código sugerido por {nome_ia}")
                        st.code(dados.get("codigo_sugerido", ""), language="python")
                        st.write(f"**Por que deu esse erro?** {dados.get('motivo_provavel', '-')}")
                        st.write(f"**Onde colocar:** {dados.get('onde_colocar', '-')}")
                        patch = dados.get("codigo_sugerido", "").strip()
                        if patch:
                            patchs.append(patch)

                patch_consolidado = patchs[0] if patchs else ""
                if patch_consolidado:
                    st.markdown("### Patch consolidado do erro")
                    st.code(patch_consolidado, language="python")
