import os
import io
import json
import base64
import tempfile
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo

import requests
import streamlit as st
import streamlit.components.v1 as components
import replicate

from groq import Groq
from openai import OpenAI
from supabase import create_client, Client
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from services.code_review import (
    revisar_codigo_multi_ia,
    executar_modo_programador_automatico,
)
from services.report_builder import montar_relatorio_final


# =========================
# CONFIG
# =========================
TIMEZONE = "America/Sao_Paulo"

MODEL_TEXT = os.getenv("MODEL_TEXT", "llama-3.3-70b-versatile")
MODEL_SECOND = os.getenv("MODEL_SECOND", "llama-3.1-8b-instant")
MODEL_VISION = os.getenv("MODEL_VISION", "meta-llama/llama-4-scout-17b-16e-instruct")
MODEL_TRANSCRIBE = os.getenv("MODEL_TRANSCRIBE", "whisper-large-v3-turbo")

IMAGE_PROVIDERS = ["OpenAI", "Replicate", "HuggingFace"]

SYSTEM_PROMPT_BASE = """
Você é a João Paulo-IA.
Foi desenvolvida por João Paulo e está em desenvolvimento contínuo.

Regras obrigatórias:
- Responda em português do Brasil.
- Seja clara, direta e útil.
- Quando houver resultados de pesquisa atuais, use-os como prioridade.
- Nunca invente fatos.
- Se faltarem dados, diga isso claramente.
"""

SEARCH_TRIGGER_WORDS = [
    "hoje", "agora", "atual", "atualmente", "notícia", "noticias",
    "última", "ultimo", "último", "preço", "precos", "cotação", "cotacao",
    "presidente", "resultado", "placar", "2026", "2027", "ontem", "amanhã",
    "versão", "versao", "lançamento", "lancamento", "google", "site"
]

CUSTOM_CSS = """
<style>
:root {
    --border: rgba(255,255,255,0.10);
    --card: rgba(255,255,255,0.06);
    --card-strong: rgba(255,255,255,0.09);
    --text-soft: #d1d5db;
    --text-muted: #9ca3af;
}
.stApp {
    background:
        radial-gradient(circle at top left, rgba(56,189,248,0.16) 0%, transparent 28%),
        radial-gradient(circle at top right, rgba(139,92,246,0.12) 0%, transparent 25%),
        linear-gradient(135deg, #0f172a 0%, #111827 45%, #020617 100%);
}
.block-container {
    padding-top: 1.2rem;
    padding-bottom: 2rem;
    max-width: 1280px;
}
.jp-hero {
    padding: 24px 24px;
    border: 1px solid var(--border);
    border-radius: 24px;
    background: linear-gradient(135deg, rgba(56,189,248,0.14), rgba(139,92,246,0.14));
    box-shadow: 0 12px 40px rgba(0,0,0,0.30);
    margin-bottom: 16px;
}
.jp-title {
    font-size: 2.1rem;
    font-weight: 800;
    color: white;
    margin-bottom: 6px;
}
.jp-sub {
    color: var(--text-soft);
    font-size: 1rem;
}
.jp-pill {
    display: inline-block;
    padding: 7px 12px;
    border: 1px solid var(--border);
    background: rgba(255,255,255,0.05);
    border-radius: 999px;
    color: #e5e7eb;
    font-size: 0.84rem;
    margin-top: 10px;
}
.jp-card {
    border: 1px solid var(--border);
    background: var(--card);
    border-radius: 20px;
    padding: 16px 18px;
    margin-bottom: 12px;
    box-shadow: 0 8px 22px rgba(0,0,0,0.18);
}
.jp-card-strong {
    border: 1px solid rgba(255,255,255,0.12);
    background: var(--card-strong);
    border-radius: 22px;
    padding: 18px 20px;
    margin-bottom: 14px;
    box-shadow: 0 10px 28px rgba(0,0,0,0.22);
}
.jp-section-title {
    font-size: 1.1rem;
    font-weight: 700;
    color: white;
    margin-bottom: 8px;
}
.small-muted {
    color: var(--text-muted);
    font-size: 0.90rem;
}
.jp-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
    gap: 12px;
    margin: 10px 0 4px 0;
}
.jp-mini {
    border: 1px solid var(--border);
    background: rgba(255,255,255,0.04);
    border-radius: 16px;
    padding: 12px 14px;
}
.jp-mini strong {
    display: block;
    color: white;
    margin-bottom: 4px;
}
div[data-testid="stMetric"] {
    background: rgba(255,255,255,0.04);
    border: 1px solid var(--border);
    padding: 10px 12px;
    border-radius: 16px;
}
.stButton > button {
    border-radius: 14px;
    border: 1px solid rgba(255,255,255,0.10);
    background: linear-gradient(135deg, rgba(56,189,248,0.18), rgba(139,92,246,0.18));
    color: white;
    font-weight: 600;
}
.stTextArea textarea, .stTextInput input {
    border-radius: 14px !important;
}
</style>
"""


# =========================
# STREAMLIT CONFIG
# =========================
st.set_page_config(page_title="João Paulo-IA", page_icon="🧠", layout="wide")
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# =========================
# SECRETS
# =========================
def get_secret(name: str, default: str = "") -> str:
    try:
        if name in st.secrets:
            return str(st.secrets[name])
    except Exception:
        pass
    return os.getenv(name, default)


def chave_ok(name: str) -> bool:
    return bool(get_secret(name, "").strip())


# =========================
# CLIENTES
# =========================
@st.cache_resource
def get_http() -> requests.Session:
    retry = Retry(
        total=3,
        read=3,
        connect=3,
        backoff_factor=1.2,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET", "POST"],
    )
    adapter = HTTPAdapter(max_retries=retry)
    session = requests.Session()
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    session.headers.update({"User-Agent": "JoaoPaulo-IA/3.5"})
    return session


@st.cache_resource
def get_groq_client() -> Groq:
    return Groq(api_key=get_secret("GROQ_API_KEY"))


@st.cache_resource
def get_supabase() -> Client:
    url = get_secret("SUPABASE_URL")
    key = get_secret("SUPABASE_ANON_KEY")
    if not url or not key:
        raise ValueError("SUPABASE_URL ou SUPABASE_ANON_KEY ausentes.")
    return create_client(url, key)


HTTP = get_http()


# =========================
# HELPERS
# =========================
def agora_sp() -> datetime:
    return datetime.now(ZoneInfo(TIMEZONE))


def data_longa_ptbr() -> str:
    meses = [
        "janeiro", "fevereiro", "março", "abril", "maio", "junho",
        "julho", "agosto", "setembro", "outubro", "novembro", "dezembro"
    ]
    agora = agora_sp()
    return f"{agora.day} de {meses[agora.month - 1]} de {agora.year}"


def data_curta() -> str:
    return agora_sp().strftime("%Y-%m-%d")


def precisa_pesquisar(pergunta: str) -> bool:
    texto = pergunta.lower()
    return any(p in texto for p in SEARCH_TRIGGER_WORDS)


def montar_system_prompt() -> str:
    return SYSTEM_PROMPT_BASE + f"\n\nData atual em São Paulo, Brasil: {data_longa_ptbr()}."


def build_search_query(pergunta: str, deep_search: bool) -> str:
    hoje = data_longa_ptbr()
    ano = agora_sp().year
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


def exibir_fontes(resultados: list[dict]):
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


def salvar_historico_json(hist: list[dict]) -> bytes:
    return json.dumps(hist, ensure_ascii=False, indent=2).encode("utf-8")


def gerar_pdf_bytes(texto: str, titulo: str = "Documento") -> bytes:
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    _, altura = A4

    c.setTitle(titulo)
    c.setFont("Helvetica-Bold", 14)
    c.drawString(40, altura - 40, titulo)

    c.setFont("Helvetica", 11)
    text_obj = c.beginText(40, altura - 70)
    text_obj.setLeading(15)

    largura_max = 95

    def quebrar_linha(linha: str):
        palavras = linha.split()
        atual = ""
        linhas = []

        for palavra in palavras:
            teste = (atual + " " + palavra).strip()
            if len(teste) <= largura_max:
                atual = teste
            else:
                if atual:
                    linhas.append(atual)
                atual = palavra

        if atual:
            linhas.append(atual)

        return linhas or [""]

    for paragrafo in texto.split("\n"):
        for linha in quebrar_linha(paragrafo):
            if text_obj.getY() < 50:
                c.drawText(text_obj)
                c.showPage()
                c.setFont("Helvetica", 11)
                text_obj = c.beginText(40, altura - 40)
                text_obj.setLeading(15)

            text_obj.textLine(linha)

        text_obj.textLine("")

    c.drawText(text_obj)
    c.save()
    buffer.seek(0)
    return buffer.getvalue()


def falar_texto(texto: str):
    components.html(
        f"""
        <script>
        const texto = {texto!r};
        const fala = new SpeechSynthesisUtterance(texto);
        fala.lang = "pt-BR";
        fala.rate = 1.0;
        speechSynthesis.cancel();
        speechSynthesis.speak(fala);
        </script>
        """,
        height=0,
    )


def imagem_para_data_url(uploaded_file) -> str:
    mime = uploaded_file.type or "image/png"
    b64 = base64.b64encode(uploaded_file.getvalue()).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def provider_status(provider: str):
    if provider == "OpenAI":
        return chave_ok("OPENAI_API_KEY"), "OPENAI_API_KEY"
    if provider == "Replicate":
        return chave_ok("REPLICATE_API_TOKEN"), "REPLICATE_API_TOKEN"
    if provider == "HuggingFace":
        return chave_ok("HF_TOKEN"), "HF_TOKEN"
    return False, "CHAVE_DESCONHECIDA"


# =========================
# SUPABASE
# =========================
def supabase_disponivel() -> bool:
    return chave_ok("SUPABASE_URL") and chave_ok("SUPABASE_ANON_KEY")


def get_session_id() -> str:
    if "persistent_session_id" not in st.session_state:
        raw = os.urandom(12)
        st.session_state.persistent_session_id = base64.urlsafe_b64encode(raw).decode().rstrip("=")
    return st.session_state.persistent_session_id


def db_insert_message(session_id: str, role: str, content: str):
    if not supabase_disponivel():
        return
    supabase = get_supabase()
    supabase.table("chat_messages").insert({
        "session_id": session_id,
        "role": role,
        "content": content,
    }).execute()


def db_load_messages(session_id: str, limit: int = 50) -> list[dict]:
    if not supabase_disponivel():
        return []
    supabase = get_supabase()
    resp = (
        supabase.table("chat_messages")
        .select("role,content,created_at")
        .eq("session_id", session_id)
        .order("created_at", desc=False)
        .limit(limit)
        .execute()
    )
    return [{"role": x["role"], "content": x["content"]} for x in (resp.data or [])]


def db_upsert_memory(memory_key: str, memory_value: dict):
    if not supabase_disponivel():
        return
    supabase = get_supabase()
    supabase.table("user_memory").upsert({
        "memory_key": memory_key,
        "memory_value": memory_value,
        "updated_at": datetime.utcnow().isoformat()
    }, on_conflict="memory_key").execute()


def db_get_memory(memory_key: str) -> dict:
    if not supabase_disponivel():
        return {}
    supabase = get_supabase()
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


def db_save_daily_report(report_date: str, report_json: dict):
    if not supabase_disponivel():
        return
    supabase = get_supabase()
    supabase.table("daily_reports").upsert({
        "report_date": report_date,
        "report_json": report_json,
    }, on_conflict="report_date").execute()


def db_list_reports(limit: int = 15):
    if not supabase_disponivel():
        return []
    supabase = get_supabase()
    resp = (
        supabase.table("daily_reports")
        .select("report_date,report_json,created_at")
        .order("report_date", desc=True)
        .limit(limit)
        .execute()
    )
    return resp.data or []


# =========================
# PESQUISA
# =========================
@st.cache_data(ttl=600, show_spinner=False)
def pesquisar_google(pergunta: str, api_key: str, deep_search: bool = False, max_results: int = 6):
    q = build_search_query(pergunta, deep_search)
    params = {
        "engine": "google",
        "q": q,
        "hl": "pt-br",
        "gl": "br",
        "num": max_results,
        "api_key": api_key,
    }
    resp = HTTP.get("https://serpapi.com/search", params=params, timeout=25)
    resp.raise_for_status()
    data = resp.json()

    resultados = []
    for item in data.get("organic_results", [])[:max_results]:
        resultados.append({
            "title": item.get("title", "Sem título"),
            "link": item.get("link", ""),
            "snippet": item.get("snippet", "Sem descrição"),
        })
    return resultados


@st.cache_data(ttl=600, show_spinner=False)
def pesquisar_google_news(pergunta: str, api_key: str, max_results: int = 6):
    params = {
        "engine": "google_news",
        "q": pergunta,
        "hl": "pt-br",
        "gl": "br",
        "sort_by": "1",
        "api_key": api_key,
    }
    resp = HTTP.get("https://serpapi.com/search", params=params, timeout=25)
    resp.raise_for_status()
    data = resp.json()

    resultados = []
    for item in data.get("news_results", [])[:max_results]:
        resultados.append({
            "title": item.get("title", "Sem título"),
            "link": item.get("link", ""),
            "snippet": item.get("snippet", "Sem descrição"),
        })
    return resultados


# =========================
# GROQ
# =========================
def gerar_resposta_texto(client: Groq, messages: list[dict]) -> str:
    resp = client.chat.completions.create(
        model=MODEL_TEXT,
        messages=messages,
        temperature=0.25,
        max_tokens=1400,
    )
    return resp.choices[0].message.content


def gerar_segunda_opiniao(client: Groq, pergunta: str) -> str:
    resp = client.chat.completions.create(
        model=MODEL_SECOND,
        messages=[
            {"role": "system", "content": "Responda em português do Brasil. Seja objetivo e útil."},
            {"role": "user", "content": pergunta},
        ],
        temperature=0.2,
        max_tokens=800,
    )
    return resp.choices[0].message.content


def transcrever_audio(client: Groq, audio_file) -> str:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(audio_file.getvalue())
        tmp_path = tmp.name

    try:
        with open(tmp_path, "rb") as f:
            transcription = client.audio.transcriptions.create(
                file=f,
                model=MODEL_TRANSCRIBE,
                language="pt",
                temperature=0.0,
            )
        return getattr(transcription, "text", "").strip()
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass


def analisar_imagem(client: Groq, uploaded_file, pergunta_usuario: str) -> str:
    data_url = imagem_para_data_url(uploaded_file)
    messages = [
        {"role": "system", "content": montar_system_prompt()},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": pergunta_usuario},
                {"type": "image_url", "image_url": {"url": data_url}},
            ],
        },
    ]
    resp = client.chat.completions.create(
        model=MODEL_VISION,
        messages=messages,
        temperature=0.2,
        max_tokens=1200,
    )
    return resp.choices[0].message.content


# =========================
# IMAGEM
# =========================
def gerar_imagem_openai(prompt: str):
    client = OpenAI(api_key=get_secret("OPENAI_API_KEY"))
    result = client.images.generate(
        model="gpt-image-1",
        prompt=prompt,
        size="1024x1024"
    )
    image_base64 = result.data[0].b64_json
    image_bytes = base64.b64decode(image_base64)
    caminho = "imagem_openai.png"
    with open(caminho, "wb") as f:
        f.write(image_bytes)
    return caminho


def gerar_imagem_replicate(prompt: str):
    os.environ["REPLICATE_API_TOKEN"] = get_secret("REPLICATE_API_TOKEN")
    output = replicate.run(
        "black-forest-labs/flux-schnell",
        input={"prompt": prompt}
    )
    if isinstance(output, list) and len(output) > 0:
        return str(output[0])
    return str(output)


def gerar_imagem_huggingface(prompt: str):
    api_url = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-xl-base-1.0"
    headers = {"Authorization": f"Bearer {get_secret('HF_TOKEN')}"}
    response = HTTP.post(
        api_url,
        headers=headers,
        json={"inputs": prompt},
        timeout=120
    )
    response.raise_for_status()
    caminho = "imagem_hf.png"
    with open(caminho, "wb") as f:
        f.write(response.content)
    return caminho


def gerar_imagem(prompt: str, provedor: str):
    ok, chave = provider_status(provedor)
    if not ok:
        raise RuntimeError(f"Chave ausente para {provedor}: {chave}")

    if provedor == "OpenAI":
        return gerar_imagem_openai(prompt)
    if provedor == "Replicate":
        return gerar_imagem_replicate(prompt)
    if provedor == "HuggingFace":
        return gerar_imagem_huggingface(prompt)

    raise ValueError("Provedor de imagem inválido.")


# =========================
# INIT
# =========================
if not chave_ok("GROQ_API_KEY"):
    st.error("GROQ_API_KEY não encontrada.")
    st.stop()

client = get_groq_client()
session_id = get_session_id()

if "db_bootstrapped" not in st.session_state:
    try:
        st.session_state.messages = db_load_messages(session_id)
    except Exception:
        st.session_state.messages = []
    st.session_state.db_bootstrapped = True

if "ultima_resposta" not in st.session_state:
    st.session_state.ultima_resposta = ""

if "pending_prompt" not in st.session_state:
    st.session_state.pending_prompt = ""

if "audio_prompt" not in st.session_state:
    st.session_state.audio_prompt = ""

if "segunda_opiniao" not in st.session_state:
    st.session_state.segunda_opiniao = ""


# =========================
# HEADER
# =========================
st.markdown(
    f"""
    <div class="jp-hero">
        <div class="jp-title">🧠 João Paulo-IA 3.5</div>
        <div class="jp-sub">Chat, imagem, áudio, PDF e modo programador automático com validação multi-IA.</div>
        <div class="jp-pill">Desenvolvida por João Paulo • Data atual: {data_longa_ptbr()}</div>
    </div>
    """,
    unsafe_allow_html=True,
)


# =========================
# SIDEBAR
# =========================
with st.sidebar:
    st.header("Configurações")

    force_google = st.toggle("Forçar pesquisa no Google", value=False)
    deep_search = st.toggle("Pesquisa profunda com data exata", value=True)
    mostrar_fontes = st.toggle("Mostrar fontes da pesquisa", value=True)
    habilitar_voz = st.toggle("Mostrar botão para ouvir resposta", value=True)
    image_provider = st.selectbox("Provedor de imagem", IMAGE_PROVIDERS)

    st.markdown("---")
    st.subheader("Status das chaves")

    st.success("GROQ_API_KEY ok") if chave_ok("GROQ_API_KEY") else st.error("GROQ_API_KEY ausente")
    st.success("SERPAPI_KEY ok") if chave_ok("SERPAPI_KEY") else st.warning("SERPAPI_KEY ausente")
    st.success("SUPABASE ok") if supabase_disponivel() else st.warning("SUPABASE não configurado")
    st.success("OPENAI_API_KEY ok") if chave_ok("OPENAI_API_KEY") else st.info("OPENAI_API_KEY ausente")
    st.success("XAI_API_KEY ok") if chave_ok("XAI_API_KEY") else st.info("XAI_API_KEY ausente")
    st.success("GEMINI_API_KEY ok") if chave_ok("GEMINI_API_KEY") else st.info("GEMINI_API_KEY ausente")
    st.success("REPLICATE_API_TOKEN ok") if chave_ok("REPLICATE_API_TOKEN") else st.info("REPLICATE_API_TOKEN ausente")
    st.success("HF_TOKEN ok") if chave_ok("HF_TOKEN") else st.info("HF_TOKEN ausente")

    st.markdown("---")
    st.caption(f"Session ID: {session_id}")

    if st.button("Limpar conversa da sessão"):
        st.session_state.messages = []
        st.session_state.ultima_resposta = ""
        st.session_state.pending_prompt = ""
        st.session_state.audio_prompt = ""
        st.session_state.segunda_opiniao = ""
        st.rerun()


# =========================
# TABS
# =========================
aba_chat, aba_imagem, aba_gerar_img, aba_arquivos, aba_dev = st.tabs(
    ["💬 Conversar", "🖼 Analisar imagem", "🎨 Gerar imagem", "📄 PDF e histórico", "🛠 Programador"]
)


with aba_chat:
    st.markdown(
        '<div class="jp-card"><div class="small-muted">Se o Supabase estiver configurado, a conversa desta sessão fica salva no banco.</div></div>',
        unsafe_allow_html=True
    )

    audio = st.audio_input("Gravar pergunta por voz")

    if st.button("Transcrever áudio"):
        if audio is None:
            st.warning("Nenhum áudio gravado.")
        else:
            try:
                st.session_state.audio_prompt = transcrever_audio(client, audio)
                st.success("Áudio transcrito com sucesso.")
            except Exception as e:
                st.error(f"Erro ao transcrever áudio: {e}")

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    prompt_digitado = st.chat_input("Digite sua pergunta")
    if prompt_digitado:
        st.session_state.pending_prompt = prompt_digitado

    pergunta = st.session_state.pending_prompt or st.session_state.audio_prompt

    if pergunta:
        st.session_state.messages.append({"role": "user", "content": pergunta})

        try:
            db_insert_message(session_id, "user", pergunta)
        except Exception:
            pass

        with st.chat_message("user"):
            st.markdown(pergunta)

        with st.chat_message("assistant"):
            try:
                resultados = []
                usar_pesquisa = force_google or precisa_pesquisar(pergunta)

                if usar_pesquisa:
                    if not chave_ok("SERPAPI_KEY"):
                        st.warning("Pesquisa solicitada, mas SERPAPI_KEY não está configurada.")
                    else:
                        with st.spinner("Buscando informações atuais..."):
                            if deep_search:
                                resultados = pesquisar_google(pergunta, get_secret("SERPAPI_KEY"), deep_search=True)
                            else:
                                resultados = pesquisar_google_news(pergunta, get_secret("SERPAPI_KEY"))

                        if mostrar_fontes and resultados:
                            exibir_fontes(resultados)

                contexto_busca = formatar_resultados(resultados)
                memory = db_get_memory("global_memory")
                memory_context = json.dumps(memory, ensure_ascii=False) if memory else "{}"

                messages = [{"role": "system", "content": montar_system_prompt()}]
                for m in st.session_state.messages[-8:-1]:
                    messages.append(m)

                if usar_pesquisa and resultados:
                    conteudo_user = (
                        f"Pergunta do usuário:\n{pergunta}\n\n"
                        f"Data atual: {data_longa_ptbr()}\n\n"
                        f"Memória global:\n{memory_context}\n\n"
                        f"Resultados atuais da pesquisa:\n{contexto_busca}\n\n"
                        f"Responda usando primeiro os resultados atuais."
                    )
                else:
                    conteudo_user = (
                        f"Pergunta do usuário:\n{pergunta}\n\n"
                        f"Data atual: {data_longa_ptbr()}\n\n"
                        f"Memória global:\n{memory_context}\n\n"
                        f"Se a pergunta exigir fato atual e não houver pesquisa, deixe isso claro."
                    )

                messages.append({"role": "user", "content": conteudo_user})

                resposta = gerar_resposta_texto(client, messages)
                if not resposta or not resposta.strip():
                    resposta = "A IA não retornou texto. Tente reformular a pergunta."

                st.markdown(resposta)

                st.session_state.messages.append({"role": "assistant", "content": resposta})
                st.session_state.ultima_resposta = resposta
                st.session_state.pending_prompt = ""
                st.session_state.audio_prompt = ""
                st.session_state.segunda_opiniao = ""

                try:
                    db_insert_message(session_id, "assistant", resposta)
                except Exception:
                    pass

            except Exception as e:
                erro = f"Erro ao responder: {e}"
                st.error(erro)
                st.session_state.messages.append({"role": "assistant", "content": erro})
                st.session_state.ultima_resposta = erro
                st.session_state.pending_prompt = ""
                st.session_state.audio_prompt = ""

    if st.session_state.ultima_resposta:
        col1, col2 = st.columns(2)

        with col1:
            if habilitar_voz and st.button("🔊 Ouvir última resposta"):
                falar_texto(st.session_state.ultima_resposta)

        with col2:
            if st.button("🧠 Pedir segunda opinião"):
                try:
                    ultima_pergunta = None
                    for m in reversed(st.session_state.messages):
                        if m["role"] == "user":
                            ultima_pergunta = m["content"]
                            break

                    if ultima_pergunta:
                        st.session_state.segunda_opiniao = gerar_segunda_opiniao(client, ultima_pergunta)
                except Exception as e:
                    st.session_state.segunda_opiniao = f"Erro na segunda opinião: {e}"

    if st.session_state.segunda_opiniao:
        st.markdown("### Segunda opinião")
        st.markdown(st.session_state.segunda_opiniao)


with aba_imagem:
    uploaded_image = st.file_uploader(
        "Envie uma imagem",
        type=["png", "jpg", "jpeg", "webp"],
        key="vision_upload"
    )

    pergunta_imagem = st.text_area(
        "O que você quer que a IA analise na imagem?",
        placeholder="Ex.: descreva essa imagem, leia o texto visível, identifique objetos..."
    )

    if uploaded_image is not None:
        st.image(uploaded_image, caption="Imagem enviada", use_container_width=True)

    if st.button("Analisar imagem"):
        if uploaded_image is None:
            st.warning("Envie uma imagem primeiro.")
        elif not pergunta_imagem.strip():
            st.warning("Escreva o que você quer analisar.")
        else:
            try:
                with st.spinner("Analisando imagem..."):
                    analise = analisar_imagem(client, uploaded_image, pergunta_imagem)
                st.markdown("### Resultado da análise")
                st.markdown(analise)
                st.session_state.ultima_resposta = analise
            except Exception as e:
                st.error(f"Erro ao analisar imagem: {e}")


with aba_gerar_img:
    prompt_imagem = st.text_area(
        "Descreva a imagem que você quer criar",
        placeholder="Ex.: um robô estudando em uma sala futurista azul",
        height=120,
        key="gen_img_prompt"
    )

    if st.button("Gerar imagem"):
        if not prompt_imagem.strip():
            st.warning("Descreva a imagem primeiro.")
        else:
            try:
                with st.spinner(f"Gerando imagem com {image_provider}..."):
                    img = gerar_imagem(prompt_imagem, image_provider)

                if isinstance(img, str):
                    st.image(img, caption=prompt_imagem, use_container_width=True)
                    st.session_state.ultima_resposta = f"Imagem gerada com {image_provider}: {prompt_imagem}"
                else:
                    st.error("O provedor retornou um formato inválido.")
            except Exception as e:
                st.error(f"Erro ao gerar imagem: {e}")


with aba_arquivos:
    ultima = st.session_state.ultima_resposta or "Ainda não há resposta para exportar."
    conversa_texto = "\n\n".join(
        [f"{m['role'].upper()}: {m['content']}" for m in st.session_state.messages]
    ) or "Ainda não há conversa para exportar."

    pdf_ultima = gerar_pdf_bytes(ultima, titulo="Última resposta - João Paulo-IA")
    pdf_conversa = gerar_pdf_bytes(conversa_texto, titulo="Conversa - João Paulo-IA")
    historico_json = salvar_historico_json(st.session_state.messages)

    st.download_button(
        "Baixar PDF da última resposta",
        data=pdf_ultima,
        file_name="ultima_resposta_joao_paulo_ia.pdf",
        mime="application/pdf",
    )

    st.download_button(
        "Baixar PDF da conversa",
        data=pdf_conversa,
        file_name="conversa_joao_paulo_ia.pdf",
        mime="application/pdf",
    )

    st.download_button(
        "Baixar histórico em JSON",
        data=historico_json,
        file_name="historico_sessao_joao_paulo_ia.json",
        mime="application/json",
    )


with aba_dev:
    admin_password = get_secret("ADMIN_PASSWORD", "")
    senha = st.text_input("Senha do programador", type="password")

    if not admin_password:
        st.info("ADMIN_PASSWORD não configurada nos segredos.")
    elif senha != admin_password:
        st.info("Área bloqueada.")
    else:
        st.success("Acesso liberado.")

        st.markdown(
            """
            <div class="jp-card-strong">
                <div class="jp-section-title">🛠 Modo Programador Inteligente</div>
                <div class="small-muted">
                    Você pode revisar código manualmente ou apertar um botão para a IA gerar uma melhoria,
                    mandar para várias IAs avaliarem e só aprovar quando realmente compensar.
                </div>
                <div class="jp-grid">
                    <div class="jp-mini">
                        <strong>Modo 1</strong>
                        Revisão manual multi-IA
                    </div>
                    <div class="jp-mini">
                        <strong>Modo 2</strong>
                        Geração automática de melhoria
                    </div>
                    <div class="jp-mini">
                        <strong>Resultado</strong>
                        Relatório técnico pronto
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        modo_programador = st.radio(
            "Escolha o modo",
            ["Manual", "Automático"],
            horizontal=True,
        )

        codigo_padrao = Path(__file__).read_text(encoding="utf-8")

        if modo_programador == "Manual":
            st.markdown("## Revisão manual multi-IA")
            st.caption("Cole um código abaixo para analisar com OpenAI, xAI e Gemini.")

            codigo_dev = st.text_area(
                "Cole o código para análise",
                height=320,
                placeholder="Cole aqui o código que você quer revisar"
            )

            col_a, col_b = st.columns(2)

            with col_a:
                usar_openai = st.checkbox("Usar OpenAI", value=True)
                usar_xai = st.checkbox("Usar xAI / Grok", value=True)
                usar_gemini = st.checkbox("Usar Gemini", value=True)

            with col_b:
                validar_sintaxe = st.checkbox("Validar sintaxe Python antes", value=True)
                salvar_relatorio = st.checkbox("Salvar relatório no banco", value=True)

            if st.button("Analisar com múltiplas IAs"):
                if not codigo_dev.strip():
                    st.warning("Cole um código para análise.")
                else:
                    try:
                        with st.spinner("Consultando múltiplas IAs..."):
                           resultados = revisar_codigo_multi_ia(
    codigo=codigo_dev,
    openai_api_key=get_secret("OPENAI_API_KEY"),
    xai_api_key=get_secret("XAI_API_KEY"),
    gemini_api_key=get_secret("GEMINI_API_KEY"),
    usar_openai=usar_openai,
    usar_xai=usar_xai,
    usar_gemini=usar_gemini,
    validar_sintaxe=validar_sintaxe,
)

                            )
                            relatorio_final = montar_relatorio_final(resultados)

                        st.markdown("### Relatório consolidado")
                        st.json(relatorio_final)

                        if salvar_relatorio:
                            try:
                                db_save_daily_report(data_curta(), relatorio_final)
                            except Exception as e:
                                st.warning(f"Não foi possível salvar no banco: {e}")

                        st.download_button(
                            "Baixar relatório técnico",
                            data=json.dumps(relatorio_final, ensure_ascii=False, indent=2),
                            file_name=f"relatorio_multi_ia_{data_curta()}.json",
                            mime="application/json",
                        )
                    except Exception as e:
                        st.error(f"Erro na análise multi-IA: {e}")

        else:
            st.markdown("## Geração automática de melhoria")
            st.caption("Você aperta um botão e a IA tenta encontrar sozinha uma melhoria que realmente compense.")

            usar_codigo_do_app = st.checkbox(
                "Usar automaticamente o código atual do app",
                value=True
            )

            if usar_codigo_do_app:
                codigo_auto = codigo_padrao
                st.text_area(
                    "Código detectado automaticamente",
                    value=codigo_auto,
                    height=220,
                    disabled=True
                )
            else:
                codigo_auto = st.text_area(
                    "Cole o código que será usado no modo automático",
                    height=320,
                    placeholder="Cole aqui o código que a IA vai melhorar automaticamente"
                )

            col1, col2, col3 = st.columns(3)
            with col1:
                max_tentativas = st.slider("Máximo de tentativas", 1, 5, 3)
            with col2:
                salvar_automatico = st.checkbox("Salvar relatório no banco", value=True)
            with col3:
                mostrar_rodadas = st.checkbox("Mostrar rodadas da análise", value=True)

            if st.button("🚀 Gerar melhoria automática"):
                if not codigo_auto.strip():
                    st.warning("Não há código para analisar.")
                else:
                    try:
                        with st.spinner("Gerando e validando melhoria automática..."):
                            resultado_auto = executar_modo_programador_automatico(
                                codigo=codigo_auto,
                                openai_api_key=get_secret("OPENAI_API_KEY"),
                                xai_api_key=get_secret("XAI_API_KEY"),
                                gemini_api_key=get_secret("GEMINI_API_KEY"),
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

                        if mostrar_rodadas:
                            st.markdown("#### Rodadas da análise")
                            for rodada in relatorio_final.get("rodadas", []):
                                with st.expander(f"Tentativa {rodada.get('tentativa')}"):
                                    st.write("**Sugestão gerada:**")
                                    st.json(rodada.get("sugestao", {}))
                                    st.write("**Avaliações:**")
                                    st.json(rodada.get("avaliacoes", []))

                        if salvar_automatico:
                            try:
                                db_save_daily_report(data_curta(), relatorio_final)
                            except Exception as e:
                                st.warning(f"Não foi possível salvar no banco: {e}")

                        st.download_button(
                            "Baixar relatório automático",
                            data=json.dumps(relatorio_final, ensure_ascii=False, indent=2),
                            file_name=f"relatorio_auto_{data_curta()}.json",
                            mime="application/json",
                        )
                    except Exception as e:
                        st.error(f"Erro no modo programador automático: {e}")

        st.markdown("---")
        st.markdown("### Relatórios salvos")
        try:
            relatorios = db_list_reports()
            if not relatorios:
                st.info("Nenhum relatório salvo ainda.")
            else:
                for item in relatorios:
                    report_date = item["report_date"]
                    report_json = item["report_json"]

                    with st.expander(f"Relatório {report_date}"):
                        st.write(f"**Modo:** {report_json.get('modo', '-')}")
                        st.write(f"**Resumo executivo:** {report_json.get('resumo_executivo', '-')}")
                        st.write(f"**Melhor ação agora:** {report_json.get('melhor_acao_agora', '-')}")
                        st.write(f"**Prioridade:** {report_json.get('prioridade', '-')}")
                        st.write(f"**Onde mexer primeiro:** {report_json.get('onde_mexer_primeiro', '-')}")
                        if report_json.get("vale_a_pena_implementar"):
                            st.write(f"**Vale a pena implementar?:** {report_json.get('vale_a_pena_implementar')}")
                        st.write("**Patch recomendado:**")
                        st.code(report_json.get("patch_recomendado", ""), language="python")
                        st.download_button(
                            f"Baixar relatório {report_date}",
                            data=json.dumps(report_json, ensure_ascii=False, indent=2),
                            file_name=f"relatorio_{report_date}.json",
                            mime="application/json",
                            key=f"dl_{report_date}",
                        )
        except Exception as e:
            st.error(f"Erro ao listar relatórios: {e}")


st.markdown("---")
st.caption("João Paulo-IA")
st.caption("Desenvolvida por João Paulo")
st.caption("Versão 3.5 com modo programador automático.")
