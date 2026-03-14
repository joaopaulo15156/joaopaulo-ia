def db_save_ia_error(
    get_supabase_func,
    get_secret_func,
    origem: str,
    ia_que_falhou: str,
    tipo_erro: str,
    mensagem_erro: str,
    codigo_analisado: str,
    relatorio_json: dict | None = None,
    status_resolucao: str = "pendente",
):
    if not (get_secret_func("SUPABASE_URL") and get_secret_func("SUPABASE_ANON_KEY")):
        return

    supabase = get_supabase_func()
    supabase.table("ia_error_logs").insert({
        "origem": origem,
        "ia_que_falhou": ia_que_falhou,
        "tipo_erro": tipo_erro,
        "mensagem_erro": mensagem_erro,
        "codigo_analisado": codigo_analisado,
        "relatorio_json": relatorio_json or {},
        "status_resolucao": status_resolucao,
    }).execute()


def db_list_ia_errors(get_supabase_func, get_secret_func, limit: int = 50):
    if not (get_secret_func("SUPABASE_URL") and get_secret_func("SUPABASE_ANON_KEY")):
        return []

    supabase = get_supabase_func()
    resp = (
        supabase.table("ia_error_logs")
        .select("*")
        .order("created_at", desc=True)
        .limit(limit)
        .execute()
    )
    return resp.data or []


def db_update_ia_error_report(
    get_supabase_func,
    get_secret_func,
    error_id: int,
    relatorio_json: dict,
    status_resolucao: str = "analisado",
):
    if not (get_secret_func("SUPABASE_URL") and get_secret_func("SUPABASE_ANON_KEY")):
        return

    supabase = get_supabase_func()
    supabase.table("ia_error_logs").update({
        "relatorio_json": relatorio_json,
        "status_resolucao": status_resolucao,
    }).eq("id", error_id).execute()


def db_mark_error_resolved(
    get_supabase_func,
    get_secret_func,
    error_id: int,
):
    if not (get_secret_func("SUPABASE_URL") and get_secret_func("SUPABASE_ANON_KEY")):
        return

    supabase = get_supabase_func()
    supabase.table("ia_error_logs").update({
        "status_resolucao": "resolvido",
    }).eq("id", error_id).execute()
