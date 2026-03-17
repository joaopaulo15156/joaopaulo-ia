import os
import io
import json
import base64
import sqlite3
import difflib
import shutil
import subprocess
import tempfile
import ast
import shlex
from datetime import datetime
from pathlib import Path

import requests
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from openai import OpenAI

try:
    from groq import Groq
except Exception:
    Groq = None


BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
REPORTS_DIR = DATA_DIR / "reports"
GENERATED_DIR = DATA_DIR / "generated"
BACKUPS_DIR = DATA_DIR / "backups"
MANAGED_DIR = BASE_DIR / "managed_project"
DB_PATH = DATA_DIR / "app.db"
TESTS_CONFIG_PATH = BASE_DIR / ".jpia_tests.json"

DATA_DIR.mkdir(exist_ok=True)
REPORTS_DIR.mkdir(exist_ok=True)
GENERATED_DIR.mkdir(exist_ok=True)
BACKUPS_DIR.mkdir(exist_ok=True)
MANAGED_DIR.mkdir(exist_ok=True)

APP_PORT = int(os.getenv("PORT", os.getenv("APP_PORT", "3000")))
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
SERPAPI_KEY = os.getenv("SERPAPI_KEY", "").strip()
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "123456").strip()

OPENAI_TEXT_MODEL = os.getenv("OPENAI_TEXT_MODEL", "gpt-4.1-mini")
OPENAI_VISION_MODEL = os.getenv("OPENAI_VISION_MODEL", "gpt-4.1-mini")
OPENAI_IMAGE_MODEL = os.getenv("OPENAI_IMAGE_MODEL", "gpt-image-1")
GROQ_TEXT_MODEL = os.getenv("GROQ_TEXT_MODEL", "llama-3.3-70b-versatile")

app = FastAPI(title="João Paulo-IA")

openai_client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None
groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY and Groq else None


def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def init_db():
    conn = get_conn()
    cur = conn.cursor()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS messages (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT NOT NULL,
        role TEXT NOT NULL,
        content TEXT NOT NULL,
        created_at TEXT NOT NULL
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        password TEXT NOT NULL,
        is_admin INTEGER NOT NULL DEFAULT 0,
        created_at TEXT NOT NULL
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS programmer_proposals (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT NOT NULL,
        username TEXT,
        target_files_json TEXT NOT NULL,
        original_files_json TEXT NOT NULL,
        suggested_files_json TEXT NOT NULL,
        report_json TEXT NOT NULL,
        source_mode TEXT NOT NULL DEFAULT 'manual',
        test_json TEXT,
        score_json TEXT,
        status TEXT NOT NULL DEFAULT 'pending',
        backup_json TEXT,
        created_at TEXT NOT NULL,
        reviewed_at TEXT
    )
    """)

    conn.commit()

    admin = cur.execute("SELECT id FROM users WHERE username = ?", ("admin",)).fetchone()
    if not admin:
        cur.execute(
            "INSERT INTO users (username, password, is_admin, created_at) VALUES (?, ?, ?, ?)",
            ("admin", ADMIN_PASSWORD, 1, now_iso()),
        )
        conn.commit()

    conn.close()


init_db()


class ChatIn(BaseModel):
    session_id: str
    message: str


class ProgrammerLoginIn(BaseModel):
    password: str


class UserLoginIn(BaseModel):
    username: str
    password: str


class UserCreateIn(BaseModel):
    username: str
    password: str
    admin_password: str


class AutoImproveIn(BaseModel):
    session_id: str
    username: str
    target_files: list[str]
    notes: str = ""


def log_error(source: str, message: str):
    print(f"[ERRO] {source}: {message}")


def save_message(session_id: str, role: str, content: str):
    conn = get_conn()
    conn.execute(
        "INSERT INTO messages (session_id, role, content, created_at) VALUES (?, ?, ?, ?)",
        (session_id, role, content, now_iso()),
    )
    conn.commit()
    conn.close()


def get_all_messages(session_id: str):
    conn = get_conn()
    rows = conn.execute(
        "SELECT role, content, created_at FROM messages WHERE session_id = ? ORDER BY id ASC",
        (session_id,),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def list_users():
    conn = get_conn()
    rows = conn.execute(
        "SELECT id, username, is_admin, created_at FROM users ORDER BY id DESC"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def authenticate_user(username: str, password: str) -> bool:
    conn = get_conn()
    row = conn.execute(
        "SELECT * FROM users WHERE username = ? AND password = ?",
        (username.strip(), password.strip()),
    ).fetchone()
    conn.close()
    return row is not None


def create_user(username: str, password: str, admin_password: str):
    if admin_password != ADMIN_PASSWORD:
        raise HTTPException(status_code=403, detail="Senha admin inválida.")

    conn = get_conn()
    exists = conn.execute(
        "SELECT id FROM users WHERE username = ?",
        (username,),
    ).fetchone()

    if exists:
        conn.close()
        raise HTTPException(status_code=400, detail="Usuário já existe.")

    conn.execute(
        "INSERT INTO users (username, password, is_admin, created_at) VALUES (?, ?, ?, ?)",
        (username, password, 0, now_iso()),
    )
    conn.commit()
    conn.close()


def sanitize_target_file(target_file: str) -> Path:
    target_file = (target_file or "").strip().replace("\\", "/")
    if not target_file:
        raise HTTPException(status_code=400, detail="Arquivo alvo vazio.")
    if ".." in target_file or target_file.startswith("/"):
        raise HTTPException(status_code=400, detail="Arquivo alvo inválido.")

    final_path = (MANAGED_DIR / target_file).resolve()
    if MANAGED_DIR.resolve() not in final_path.parents and final_path != MANAGED_DIR.resolve():
        raise HTTPException(status_code=400, detail="Arquivo fora da pasta permitida.")
    final_path.parent.mkdir(parents=True, exist_ok=True)
    return final_path


def sanitize_target_files(target_files: list[str]) -> list[Path]:
    return [sanitize_target_file(f) for f in target_files]


def text_llm(messages):
    if openai_client:
        resp = openai_client.chat.completions.create(
            model=OPENAI_TEXT_MODEL,
            messages=messages,
            temperature=0.3,
        )
        return resp.choices[0].message.content

    if groq_client:
        resp = groq_client.chat.completions.create(
            model=GROQ_TEXT_MODEL,
            messages=messages,
            temperature=0.3,
        )
        return resp.choices[0].message.content

    return "Nenhuma IA de texto configurada."


def image_analysis_llm(question: str, image_bytes: bytes, mime_type: str):
    if not openai_client:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY não configurada.")

    b64 = base64.b64encode(image_bytes).decode("utf-8")
    data_url = f"data:{mime_type};base64,{b64}"

    resp = openai_client.chat.completions.create(
        model=OPENAI_VISION_MODEL,
        messages=[
            {"role": "system", "content": "Responda em português do Brasil."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": question},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            },
        ],
        temperature=0.2,
    )
    return resp.choices[0].message.content


def generate_image(prompt: str) -> str:
    if not openai_client:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY não configurada.")

    result = openai_client.images.generate(
        model=OPENAI_IMAGE_MODEL,
        prompt=prompt,
        size="1024x1024",
    )

    image_base64 = result.data[0].b64_json
    image_bytes = base64.b64decode(image_base64)

    filename = f"img_{int(datetime.now().timestamp())}.png"
    path = GENERATED_DIR / filename
    path.write_bytes(image_bytes)
    return filename


def create_pdf_for_session(session_id: str) -> str:
    messages = get_all_messages(session_id)
    if not messages:
        raise HTTPException(status_code=404, detail="Conversa vazia.")

    filename = f"conversa_{session_id}_{int(datetime.now().timestamp())}.pdf"
    path = REPORTS_DIR / filename

    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    _, height = A4

    c.setTitle("Última conversa")
    c.setFont("Helvetica-Bold", 14)
    c.drawString(40, height - 40, "Última conversa")
    c.setFont("Helvetica", 11)

    text_obj = c.beginText(40, height - 70)
    text_obj.setLeading(15)

    def wrap_line(line, max_chars=95):
        words = line.split()
        current = ""
        lines = []
        for word in words:
            test = (current + " " + word).strip()
            if len(test) <= max_chars:
                current = test
            else:
                if current:
                    lines.append(current)
                current = word
        if current:
            lines.append(current)
        return lines or [""]

    for m in messages:
        block = f"{m['role'].upper()}: {m['content']}"
        for paragraph in block.split("\n"):
            for line in wrap_line(paragraph):
                if text_obj.getY() < 50:
                    c.drawText(text_obj)
                    c.showPage()
                    c.setFont("Helvetica", 11)
                    text_obj = c.beginText(40, height - 40)
                    text_obj.setLeading(15)
                text_obj.textLine(line)
            text_obj.textLine("")

    c.drawText(text_obj)
    c.save()
    buffer.seek(0)
    path.write_bytes(buffer.getvalue())
    return filename


def search_technical_references(query: str) -> list[dict]:
    if not SERPAPI_KEY:
        return []

    try:
        params = {
            "engine": "google",
            "q": query,
            "hl": "pt-br",
            "gl": "br",
            "num": 5,
            "api_key": SERPAPI_KEY,
        }
        resp = requests.get("https://serpapi.com/search", params=params, timeout=25)
        resp.raise_for_status()
        data = resp.json()

        out = []
        for item in data.get("organic_results", [])[:5]:
            out.append({
                "title": item.get("title", ""),
                "link": item.get("link", ""),
                "snippet": item.get("snippet", ""),
            })
        return out
    except Exception as e:
        log_error("tech_search", str(e))
        return []


def validate_python_syntax_for_map(files_map: dict) -> dict:
    results = {}
    all_ok = True
    for filename, code in files_map.items():
        try:
            ast.parse(code)
            results[filename] = {"ok": True, "message": "Sintaxe Python válida."}
        except SyntaxError as e:
            all_ok = False
            results[filename] = {"ok": False, "message": str(e)}
    return {"all_ok": all_ok, "files": results}


def load_test_commands() -> list[str]:
    commands = []
    if TESTS_CONFIG_PATH.exists():
        try:
            data = json.loads(TESTS_CONFIG_PATH.read_text(encoding="utf-8"))
            commands.extend(data.get("commands", []))
        except Exception as e:
            log_error("load_test_commands", str(e))
    return [c for c in commands if c]


def run_smoke_test(files_map: dict) -> dict:
    temp_dir = Path(tempfile.mkdtemp(prefix="jpia_"))
    try:
        managed_temp = temp_dir / "managed_project"
        managed_temp.mkdir(parents=True, exist_ok=True)

        for rel_name, code in files_map.items():
            full = managed_temp / rel_name
            full.parent.mkdir(parents=True, exist_ok=True)
            full.write_text(code, encoding="utf-8")

        syntax_results = []
        for pyf in managed_temp.rglob("*.py"):
            proc = subprocess.run(
                ["python", "-m", "py_compile", str(pyf)],
                capture_output=True,
                text=True,
                cwd=temp_dir,
            )
            syntax_results.append({
                "file": str(pyf.relative_to(temp_dir)),
                "ok": proc.returncode == 0,
                "stderr": proc.stderr,
            })

        custom_results = []
        for cmd in load_test_commands():
            proc = subprocess.run(
                shlex.split(cmd),
                capture_output=True,
                text=True,
                cwd=temp_dir,
            )
            custom_results.append({
                "command": cmd,
                "ok": proc.returncode == 0,
                "stderr": proc.stderr,
            })

        syntax_ok = all(x["ok"] for x in syntax_results) if syntax_results else True
        custom_ok = all(x["ok"] for x in custom_results) if custom_results else True

        return {
            "all_ok": syntax_ok and custom_ok,
            "syntax_results": syntax_results,
            "custom_results": custom_results,
        }
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def score_proposal(report: dict, syntax_check: dict, smoke_test: dict) -> dict:
    score = 100
    reasons = []

    if not syntax_check.get("all_ok"):
        score -= 50
        reasons.append("Sintaxe inválida.")

    if not smoke_test.get("all_ok"):
        score -= 35
        reasons.append("Smoke test falhou.")

    if len(report.get("riscos", []) or []) >= 3:
        score -= 10
        reasons.append("Muitos riscos.")

    score = max(0, min(100, score))
    return {
        "score": score,
        "can_approve": score >= 70 and syntax_check.get("all_ok") and smoke_test.get("all_ok"),
        "reasons": reasons,
    }


def build_programmer_report_multi(original_map: dict, target_files: list[str], notes: str) -> dict:
    if not openai_client:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY necessária.")

    search_results = search_technical_references(
        "Python FastAPI code improvement best practices multi file refactor"
    )

    prompt = f"""
Você é uma IA de apoio ao programador.

Analise os arquivos abaixo e gere um relatório técnico em JSON.
Você deve sugerir código melhorado para cada arquivo.

Responda SOMENTE em JSON com este formato:
{{
  "titulo": "...",
  "resumo": "...",
  "pontos_bons": ["...", "..."],
  "pontos_ruins": ["...", "..."],
  "riscos": ["...", "..."],
  "para_que_ajuda": "...",
  "arquivos": {{
    "arquivo1.py": "codigo completo melhorado",
    "arquivo2.py": "codigo completo melhorado"
  }}
}}

Arquivos alvo:
{json.dumps(target_files, ensure_ascii=False)}

Observações extras:
{notes}

Resultados de pesquisa técnica:
{json.dumps(search_results, ensure_ascii=False)}

Código atual:
{json.dumps(original_map, ensure_ascii=False)}
"""

    resp = openai_client.chat.completions.create(
        model=OPENAI_TEXT_MODEL,
        messages=[
            {"role": "system", "content": "Responda apenas JSON válido."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.25,
        response_format={"type": "json_object"},
    )
    return json.loads(resp.choices[0].message.content)


def auto_generate_improvement_multi(target_files: list[str], notes: str = "") -> dict:
    paths = sanitize_target_files(target_files)
    original_map = {}
    for p in paths:
        rel = str(p.relative_to(MANAGED_DIR))
        if not p.exists():
            raise HTTPException(status_code=404, detail=f"Arquivo não encontrado: {rel}")
        original_map[rel] = p.read_text(encoding="utf-8")

    report = build_programmer_report_multi(original_map, list(original_map.keys()), notes)
    suggested_map = report.get("arquivos", {})
    if not suggested_map:
        raise HTTPException(status_code=500, detail="A IA não gerou arquivos sugeridos.")

    syntax_check = validate_python_syntax_for_map(suggested_map)
    smoke_test = run_smoke_test(suggested_map)
    score_json = score_proposal(report, syntax_check, smoke_test)

    diffs = {}
    for name, original_code in original_map.items():
        diffs[name] = make_diff(original_code, suggested_map.get(name, ""))

    report["syntax_check"] = syntax_check
    report["smoke_test"] = smoke_test
    report["score"] = score_json
    report["diffs"] = diffs

    return {
        "original_map": original_map,
        "suggested_map": suggested_map,
        "report": report,
        "test_json": {"syntax_check": syntax_check, "smoke_test": smoke_test},
        "score_json": score_json,
    }


def save_programmer_proposal(session_id, username, target_files, original_map, suggested_map, report, source_mode, test_json, score_json):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO programmer_proposals
        (session_id, username, target_files_json, original_files_json, suggested_files_json,
         report_json, source_mode, test_json, score_json, status, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            session_id,
            username,
            json.dumps(target_files, ensure_ascii=False),
            json.dumps(original_map, ensure_ascii=False),
            json.dumps(suggested_map, ensure_ascii=False),
            json.dumps(report, ensure_ascii=False),
            source_mode,
            json.dumps(test_json, ensure_ascii=False),
            json.dumps(score_json, ensure_ascii=False),
            "pending",
            now_iso(),
        ),
    )
    proposal_id = cur.lastrowid
    conn.commit()
    conn.close()
    return proposal_id


def list_proposals():
    conn = get_conn()
    rows = conn.execute(
        "SELECT * FROM programmer_proposals ORDER BY id DESC"
    ).fetchall()
    conn.close()

    result = []
    for r in rows:
        result.append({
            "id": r["id"],
            "session_id": r["session_id"],
            "username": r["username"],
            "target_files": json.loads(r["target_files_json"]),
            "report": json.loads(r["report_json"]),
            "source_mode": r["source_mode"],
            "test_json": json.loads(r["test_json"]) if r["test_json"] else {},
            "score_json": json.loads(r["score_json"]) if r["score_json"] else {},
            "status": r["status"],
            "backup_json": json.loads(r["backup_json"]) if r["backup_json"] else {},
            "created_at": r["created_at"],
            "reviewed_at": r["reviewed_at"],
        })
    return result


def make_backups_for_files(target_files: list[str]) -> dict:
    backup_map = {}
    for rel_file in target_files:
        path = sanitize_target_file(rel_file)
        if path.exists():
            stamp = int(datetime.now().timestamp())
            backup_name = f"{path.name}.{stamp}.bak"
            backup_path = BACKUPS_DIR / backup_name
            shutil.copy2(path, backup_path)
            backup_map[rel_file] = backup_name
    return backup_map


def approve_proposal(proposal_id: int):
    conn = get_conn()
    row = conn.execute(
        "SELECT * FROM programmer_proposals WHERE id = ?",
        (proposal_id,),
    ).fetchone()

    if not row:
        conn.close()
        raise HTTPException(status_code=404, detail="Proposta não encontrada.")

    if row["status"] != "pending":
        conn.close()
        raise HTTPException(status_code=400, detail="Proposta já revisada.")

    target_files = json.loads(row["target_files_json"])
    suggested_map = json.loads(row["suggested_files_json"])
    score_json = json.loads(row["score_json"]) if row["score_json"] else {}

    if not score_json.get("can_approve", False):
        conn.close()
        raise HTTPException(status_code=400, detail="Não pode aprovar: score insuficiente ou testes falharam.")

    backup_map = make_backups_for_files(target_files)

    for rel_file, code in suggested_map.items():
        path = sanitize_target_file(rel_file)
        path.write_text(code, encoding="utf-8")

    conn.execute(
        "UPDATE programmer_proposals SET status = ?, backup_json = ?, reviewed_at = ? WHERE id = ?",
        ("approved", json.dumps(backup_map, ensure_ascii=False), now_iso(), proposal_id),
    )
    conn.commit()
    conn.close()

    return {
        "applied_files": target_files,
        "backup_map": backup_map,
    }


def reject_proposal(proposal_id: int):
    conn = get_conn()
    row = conn.execute(
        "SELECT * FROM programmer_proposals WHERE id = ?",
        (proposal_id,),
    ).fetchone()

    if not row:
        conn.close()
        raise HTTPException(status_code=404, detail="Proposta não encontrada.")

    if row["status"] != "pending":
        conn.close()
        raise HTTPException(status_code=400, detail="Proposta já revisada.")

    conn.execute(
        "UPDATE programmer_proposals SET status = ?, reviewed_at = ? WHERE id = ?",
        ("rejected", now_iso(), proposal_id),
    )
    conn.commit()
    conn.close()


def rollback_proposal(proposal_id: int):
    conn = get_conn()
    row = conn.execute(
        "SELECT * FROM programmer_proposals WHERE id = ?",
        (proposal_id,),
    ).fetchone()

    if not row:
        conn.close()
        raise HTTPException(status_code=404, detail="Proposta não encontrada.")

    if row["status"] != "approved":
        conn.close()
        raise HTTPException(status_code=400, detail="Só dá para rollback em proposta aprovada.")

    backup_map = json.loads(row["backup_json"]) if row["backup_json"] else {}
    restored = []
    for rel_file, backup_name in backup_map.items():
        backup_path = BACKUPS_DIR / backup_name
        if backup_path.exists():
            target_path = sanitize_target_file(rel_file)
            shutil.copy2(backup_path, target_path)
            restored.append(rel_file)

    conn.execute(
        "UPDATE programmer_proposals SET status = ?, reviewed_at = ? WHERE id = ?",
        ("rolled_back", now_iso(), proposal_id),
    )
    conn.commit()
    conn.close()

    return restored


@app.get("/", response_class=HTMLResponse)
def home():
    return APP_HTML


@app.get("/api/health")
def health():
    return {
        "ok": True,
        "openai_configured": bool(openai_client),
        "groq_configured": bool(groq_client),
        "serpapi_configured": bool(SERPAPI_KEY),
        "test_commands": load_test_commands(),
    }


@app.post("/api/chat")
def api_chat(data: ChatIn):
    message = (data.message or "").strip()
    if not message:
        raise HTTPException(status_code=400, detail="Mensagem vazia.")

    save_message(data.session_id, "user", message)
    lower = message.lower()

    if lower.startswith("gerar imagem de "):
        prompt = message[len("gerar imagem de "):].strip()
        filename = generate_image(prompt)
        answer = "Imagem gerada com sucesso."
        save_message(data.session_id, "assistant", answer)
        return {"type": "image", "message": answer, "file_url": f"/files/generated/{filename}"}

    if "pdf da última conversa" in lower or "pdf da ultima conversa" in lower:
        filename = create_pdf_for_session(data.session_id)
        answer = "PDF gerado com sucesso."
        save_message(data.session_id, "assistant", answer)
        return {"type": "pdf", "message": answer, "file_url": f"/files/reports/{filename}"}

    if message.startswith("\\programador"):
        answer = "Modo programador solicitado. Use o painel Programador."
        save_message(data.session_id, "assistant", answer)
        return {"type": "text", "message": answer}

    answer = text_llm([
        {"role": "system", "content": "Responda em português do Brasil. Seja clara e útil."},
        {"role": "user", "content": message},
    ])
    save_message(data.session_id, "assistant", answer)
    return {"type": "text", "message": answer}


@app.post("/api/analyze-image")
async def api_analyze_image(
    session_id: str = Form(...),
    question: str = Form(...),
    file: UploadFile = File(...),
):
    content = await file.read()
    answer = image_analysis_llm(question, content, file.content_type or "image/png")
    save_message(session_id, "user", f"[Imagem enviada] {question}")
    save_message(session_id, "assistant", answer)
    return {"message": answer}


@app.post("/api/programmer/login")
def api_programmer_login(data: ProgrammerLoginIn):
    return {"ok": data.password == ADMIN_PASSWORD}


@app.post("/api/user/login")
def api_user_login(data: UserLoginIn):
    return {"ok": authenticate_user(data.username, data.password)}


@app.post("/api/user/create")
def api_user_create(data: UserCreateIn):
    create_user(data.username, data.password, data.admin_password)
    return {"ok": True, "message": "Usuário criado com sucesso."}


@app.get("/api/users")
def api_users():
    return {"items": list_users()}


@app.post("/api/programmer/auto-improve")
def api_auto_improve(data: AutoImproveIn):
    if not data.username.strip():
        raise HTTPException(status_code=400, detail="Informe um usuário.")

    auto_data = auto_generate_improvement_multi(data.target_files, data.notes)

    proposal_id = save_programmer_proposal(
        session_id=data.session_id,
        username=data.username,
        target_files=data.target_files,
        original_map=auto_data["original_map"],
        suggested_map=auto_data["suggested_map"],
        report=auto_data["report"],
        source_mode="automatico",
        test_json=auto_data["test_json"],
        score_json=auto_data["score_json"],
    )

    return {
        "proposal_id": proposal_id,
        "report": auto_data["report"],
        "message": "Melhoria automática gerada com sucesso.",
    }


@app.get("/api/programmer/proposals")
def api_programmer_proposals():
    return {"items": list_proposals()}


@app.post("/api/programmer/proposals/{proposal_id}/approve")
def api_programmer_approve(proposal_id: int):
    result = approve_proposal(proposal_id)
    return {
        "ok": True,
        "applied_files": result["applied_files"],
        "backup_map": result["backup_map"],
        "message": "Código aprovado e aplicado com sucesso.",
    }


@app.post("/api/programmer/proposals/{proposal_id}/reject")
def api_programmer_reject(proposal_id: int):
    reject_proposal(proposal_id)
    return {"ok": True, "message": "Código reprovado. A proposta foi descartada."}


@app.post("/api/programmer/proposals/{proposal_id}/rollback")
def api_programmer_rollback(proposal_id: int):
    restored = rollback_proposal(proposal_id)
    return {"ok": True, "restored_files": restored, "message": "Rollback aplicado com sucesso."}


@app.get("/files/reports/{filename}")
def download_report(filename: str):
    path = REPORTS_DIR / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail="Arquivo não encontrado.")
    return FileResponse(path)


@app.get("/files/generated/{filename}")
def download_generated(filename: str):
    path = GENERATED_DIR / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail="Arquivo não encontrado.")
    return FileResponse(path)


APP_HTML = """
<!DOCTYPE html>
<html lang="pt-BR">
<head>
<meta charset="UTF-8" />
<title>João Paulo-IA</title>
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<style>
:root{--bg:#0b1220;--card:#121a2b;--muted:#8da2c0;--text:#e9eef8;--line:#22304a}
*{box-sizing:border-box}
body{margin:0;font-family:Arial,Helvetica,sans-serif;background:linear-gradient(135deg,#0b1220,#101828);color:var(--text)}
.wrap{max-width:1200px;margin:0 auto;padding:20px}
.card{background:var(--card);border:1px solid var(--line);border-radius:18px;padding:16px;margin-bottom:20px}
.title{font-size:20px;margin-bottom:12px}
.chat-box{height:350px;overflow:auto;border:1px solid var(--line);border-radius:14px;padding:12px;background:#0c1525;margin-bottom:12px}
.msg{padding:10px 12px;border-radius:12px;margin-bottom:10px;line-height:1.45;white-space:pre-wrap}
.user{background:#17304d}
.assistant{background:#192235}
input, textarea, button{width:100%;padding:12px;border-radius:12px;border:1px solid var(--line);background:#0f1828;color:var(--text);margin-bottom:10px}
button{cursor:pointer;background:linear-gradient(135deg,#2f7de1,#4da3ff);border:none;font-weight:bold}
button.secondary{background:#17233a;border:1px solid var(--line)}
.row{display:grid;grid-template-columns:1fr 1fr;gap:10px}
.grid{display:grid;grid-template-columns:1.2fr 1fr;gap:20px}
.proposal{border:1px solid var(--line);border-radius:14px;padding:12px;margin-bottom:12px;background:#0e1625}
pre{white-space:pre-wrap;background:#0b1220;padding:10px;border-radius:12px;border:1px solid #22304a;overflow:auto}
@media(max-width:900px){.grid,.row{grid-template-columns:1fr}}
</style>
</head>
<body>
<div class="wrap">
  <div class="card">
    <div class="title">🧠 João Paulo-IA</div>
    <div>Chat, imagem, PDF e modo programador automático.</div>
  </div>

  <div class="grid">
    <div>
      <div class="card">
        <div class="title">💬 Conversa</div>
        <div id="chatBox" class="chat-box"></div>
        <input id="chatInput" placeholder="Digite sua mensagem..." />
        <div class="row">
          <button onclick="sendChat()">Enviar</button>
          <button class="secondary" onclick="quickPdf()">PDF da última conversa</button>
        </div>
      </div>

      <div class="card">
        <div class="title">🖼 Analisar imagem</div>
        <input type="file" id="imageFile" accept="image/*" />
        <input id="imageQuestion" placeholder="Ex.: descreva essa imagem..." />
        <button onclick="analyzeImage()">Analisar imagem</button>
        <div id="imageResult"></div>
      </div>
    </div>

    <div>
      <div class="card">
        <div class="title">👤 Usuário</div>
        <input id="loginUser" placeholder="Usuário" />
        <input id="loginPass" type="password" placeholder="Senha" />
        <button onclick="loginUser()">Entrar</button>
        <div id="userStatus"></div>
      </div>

      <div class="card">
        <div class="title">🛠 Programador</div>
        <input id="progPassword" type="password" placeholder="Senha do programador" />
        <button onclick="programmerLogin()">Entrar no modo programador</button>
        <div id="progStatus"></div>

        <input id="autoFiles" value="app_logic.py,utils.py" placeholder="Arquivos separados por vírgula" />
        <textarea id="autoNotes" placeholder="Observações extras"></textarea>
        <button onclick="autoImprove()">Gerar melhoria automática</button>
        <div id="autoImproveResult"></div>
      </div>

      <div class="card">
        <div class="title">📋 Propostas</div>
        <button class="secondary" onclick="loadProposals()">Atualizar propostas</button>
        <div id="proposalList"></div>
      </div>
    </div>
  </div>
</div>

<script>
const sessionId = localStorage.getItem("jp_session_id") || crypto.randomUUID();
localStorage.setItem("jp_session_id", sessionId);

let programmerLogged = false;
let loggedUser = localStorage.getItem("jp_user") || "";

function addMessage(role, text, extraHtml="") {
  const box = document.getElementById("chatBox");
  const div = document.createElement("div");
  div.className = "msg " + role;
  div.innerHTML = "<b>" + (role === "user" ? "Você" : "IA") + ":</b><br>" + escapeHtml(text).replace(/\\n/g, "<br>") + extraHtml;
  box.appendChild(div);
  box.scrollTop = box.scrollHeight;
}

function escapeHtml(text) {
  return String(text).replaceAll("&","&amp;").replaceAll("<","&lt;").replaceAll(">","&gt;");
}

function parseFiles() {
  return document.getElementById("autoFiles").value.split(",").map(x => x.trim()).filter(Boolean);
}

async function sendChat() {
  const input = document.getElementById("chatInput");
  const message = input.value.trim();
  if (!message) return;
  addMessage("user", message);
  input.value = "";

  const res = await fetch("/api/chat", {
    method: "POST",
    headers: {"Content-Type":"application/json"},
    body: JSON.stringify({session_id: sessionId, message})
  });
  const data = await res.json();

  if (data.type === "image") {
    addMessage("assistant", data.message, `<br><a target="_blank" href="${data.file_url}">Abrir imagem</a><br><img style="max-width:100%;margin-top:10px;border-radius:10px;" src="${data.file_url}" />`);
  } else if (data.type === "pdf") {
    addMessage("assistant", data.message, `<br><a target="_blank" href="${data.file_url}">Baixar PDF</a>`);
  } else {
    addMessage("assistant", data.message || "Sem resposta.");
  }
}

async function quickPdf() {
  document.getElementById("chatInput").value = "pdf da última conversa";
  await sendChat();
}

async function analyzeImage() {
  const fileInput = document.getElementById("imageFile");
  const question = document.getElementById("imageQuestion").value.trim();
  const result = document.getElementById("imageResult");

  if (!fileInput.files.length) return result.innerText = "Selecione uma imagem.";
  if (!question) return result.innerText = "Digite a pergunta.";

  const form = new FormData();
  form.append("session_id", sessionId);
  form.append("question", question);
  form.append("file", fileInput.files[0]);

  result.innerText = "Analisando...";
  const res = await fetch("/api/analyze-image", {method:"POST", body: form});
  const data = await res.json();
  result.innerText = data.message || "Sem resposta.";
}

async function loginUser() {
  const username = document.getElementById("loginUser").value.trim();
  const password = document.getElementById("loginPass").value.trim();
  const status = document.getElementById("userStatus");

  const res = await fetch("/api/user/login", {
    method: "POST",
    headers: {"Content-Type":"application/json"},
    body: JSON.stringify({username, password})
  });
  const data = await res.json();

  if (data.ok) {
    loggedUser = username;
    localStorage.setItem("jp_user", loggedUser);
    status.innerText = "Login feito como: " + loggedUser;
  } else {
    status.innerText = "Usuário ou senha inválidos.";
  }
}

async function programmerLogin() {
  const password = document.getElementById("progPassword").value;
  const status = document.getElementById("progStatus");
  const res = await fetch("/api/programmer/login", {
    method: "POST",
    headers: {"Content-Type":"application/json"},
    body: JSON.stringify({password})
  });
  const data = await res.json();
  programmerLogged = !!data.ok;
  status.innerText = programmerLogged ? "Modo programador liberado." : "Senha inválida.";
}

async function autoImprove() {
  if (!programmerLogged) return alert("Entre no modo programador primeiro.");
  if (!loggedUser) return alert("Faça login como usuário primeiro.");

  const files = parseFiles();
  const notes = document.getElementById("autoNotes").value.trim();
  const box = document.getElementById("autoImproveResult");
  box.innerHTML = "Gerando melhoria automática...";

  const res = await fetch("/api/programmer/auto-improve", {
    method: "POST",
    headers: {"Content-Type":"application/json"},
    body: JSON.stringify({
      session_id: sessionId,
      username: loggedUser,
      target_files: files,
      notes
    })
  });

  const data = await res.json();
  if (!res.ok) return box.innerHTML = "Erro: " + (data.detail || "falha");

  box.innerHTML = `<pre>${escapeHtml(JSON.stringify(data.report, null, 2))}</pre>`;
  loadProposals();
}

async function loadProposals() {
  const box = document.getElementById("proposalList");
  box.innerHTML = "Carregando...";

  const res = await fetch("/api/programmer/proposals");
  const data = await res.json();
  const items = data.items || [];

  if (!items.length) {
    box.innerHTML = "Nenhuma proposta ainda.";
    return;
  }

  box.innerHTML = items.map(item => {
    const score = item.score_json || {};
    return `
      <div class="proposal">
        <div><b>#${item.id}</b> - ${item.status}</div>
        <div><b>Arquivos:</b> ${(item.target_files || []).join(", ")}</div>
        <div><b>Score:</b> ${score.score ?? "-"}</div>
        <div><b>Pode aprovar:</b> ${score.can_approve ? "Sim" : "Não"}</div>
        <div class="row">
          <button onclick="approveProposal(${item.id})">Aprovar</button>
          <button class="secondary" onclick="rejectProposal(${item.id})">Reprovar</button>
        </div>
        <button class="secondary" onclick="rollbackProposal(${item.id})">Rollback</button>
      </div>
    `;
  }).join("");
}

async function approveProposal(id) {
  const res = await fetch(`/api/programmer/proposals/${id}/approve`, {method:"POST"});
  const data = await res.json();
  if (!res.ok) return alert(data.detail || "Erro ao aprovar.");
  alert(data.message);
  loadProposals();
}

async function rejectProposal(id) {
  const res = await fetch(`/api/programmer/proposals/${id}/reject`, {method:"POST"});
  const data = await res.json();
  if (!res.ok) return alert(data.detail || "Erro ao reprovar.");
  alert(data.message);
  loadProposals();
}

async function rollbackProposal(id) {
  const res = await fetch(`/api/programmer/proposals/${id}/rollback`, {method:"POST"});
  const data = await res.json();
  if (!res.ok) return alert(data.detail || "Erro no rollback.");
  alert(data.message);
  loadProposals();
}

loadProposals();
</script>
</body>
</html>
"""

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=APP_PORT, reload=False)
