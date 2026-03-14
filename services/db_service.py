from datetime import datetime


def supabase_disponivel(get_secret_func) -> bool:
    return bool(get_secret_func("SUPABASE_URL")) and bool(get_secret_func("SUPABASE_ANON_KEY"))


def db_insert_message(get_supabase_func, get_secret_func, session_id: str, role: str, content: str):
    if not supabase_disponivel(get_secret_func):
        return

    supabase = get_supabase_func()
    supabase.table("chat_messages").insert({
        "session_id": session_id,
        "role": role,
        "content": content,
    }).execute()


def db_load_messages(get_supabase_func, get_secret_func, session_id: str, limit: int = 50) -> list[dict]:
    if not supabase_disponivel(get_secret_func):
        return []

    supabase = get_supabase_func()
    resp = (
        supabase.table("chat_messages")
        .select("role,content,created_at")
        .eq("session_id", session_id)
        .order("created_at", desc=False)
        .limit(limit)
        .execute()
    )
    return [{"role": x["role"], "content": x["content"]} for x in (resp.data or [])]


def db_upsert_memory(get_supabase_func, get_secret_func, memory_key: str, memory_value: dict):
    if not supabase_disponivel(get_secret_func):
        return

    supabase = get_supabase_func()
    supabase.table("user_memory").upsert({
        "memory_key": memory_key,
        "memory_value": memory_value,
        "updated_at": datetime.utcnow().isoformat()
    }, on_conflict="memory_key").execute()


def db_get_memory(get_supabase_func, get_secret_func, memory_key: str) -> dict:
    if not supabase_disponivel(get_secret_func):
        return {}

    supabase = get_supabase_func()
    resp = (
        supabase.table("user_memory")
        .select("memory_value")
        .eq("memory_key", memory_key)
        .limit(1)
        .execute()
    )
    if resp.data:
        return resp.data[0]["memory_value"]
    return {}


def db_save_daily_report(get_supabase_func, get_secret_func, report_date: str, report_json: dict):
    if not supabase_disponivel(get_secret_func):
        return

    supabase = get_supabase_func()
    supabase.table("daily_reports").upsert({
        "report_date": report_date,
        "report_json": report_json,
    }, on_conflict="report_date").execute()


def db_list_reports(get_supabase_func, get_secret_func, limit: int = 15):
    if not supabase_disponivel(get_secret_func):
        return []

    supabase = get_supabase_func()
    resp = (
        supabase.table("daily_reports")
        .select("report_date,report_json,created_at")
        .order("report_date", desc=True)
        .limit(limit)
        .execute()
    )
    return resp.data or []
