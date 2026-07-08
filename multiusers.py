"""멀티유저/멀티세션/저장 RAG 챗봇.

Supabase를 사용자 저장소·세션 저장소·벡터 데이터베이스로 사용하는 Streamlit RAG 앱.

핵심 특징:
- Supabase Authentication(auth.users)을 사용하지 않고, "user" 테이블에서
  login_id / password_hash(PBKDF2-SHA256) 기반으로 직접 회원가입/로그인 처리.
- 로그인 성공/회원가입 성공 시에만 메인 대시보드(헤더+사이드바+채팅) 표시.
- 모든 세션/메시지/벡터 데이터는 user_id로 필터링하여 사용자별로 분리.
- PDF 처리 및 대화 후 자동 세션 저장, LLM(gpt-4o-mini) 기반 세션 제목 자동 생성.
- 답변은 스트리밍으로 표시, 마지막에 후속 질문 3개 추가.
- 로컬 실행(.env)과 Streamlit Cloud 배포(st.secrets)를 모두 지원.

DB 셋팅: 같은 폴더의 multiusers-ref.sql 을 Supabase SQL Editor에서 먼저 실행하세요.
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import os
import re
import tempfile
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

try:
    from supabase import Client, create_client
except ModuleNotFoundError:  # pragma: no cover - 의존성 안내용
    Client = Any  # type: ignore[assignment,misc]
    create_client = None  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# 경로 & 환경변수
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[2]
ENV_PATH = REPO_ROOT / ".env"
LOGO_PATH = SCRIPT_DIR / "logo.png"

# .env는 실행 위치와 무관하게 절대경로 기준으로 로드
load_dotenv(dotenv_path=ENV_PATH)

# 고정 설정값 (프롬프트 지정) --------------------------------------------------
LLM_MODEL = "gpt-4o-mini"
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1536
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
EMBED_BATCH_SIZE = 10
CHATBOT_NAME = "숭실대학교 RAG 챗봇"

# 비밀번호 해시 파라미터
_PBKDF2_ALGO = "sha256"
_PBKDF2_ITERATIONS = 200_000


# ---------------------------------------------------------------------------
# 로깅 (Streamlit Cloud 대응: 쓰기 가능한 경로 fallback → 콘솔)
# ---------------------------------------------------------------------------
def _resolve_log_dir() -> Path | None:
    """쓰기 가능한 로그 디렉터리를 순서대로 탐색해 반환한다.

    Streamlit Cloud(/mount/src/...)는 소스 디렉터리에 쓰기 권한이 없으므로,
    fintech/logs → OS 임시 폴더 순으로 시도하고, 모두 실패하면 None을 반환한다.

    Returns:
        Path | None: 사용 가능한 로그 디렉터리. 없으면 None(콘솔 전용).
    """
    candidates = [
        REPO_ROOT / "logs",
        Path(tempfile.gettempdir()) / "multiusers_logs",
    ]
    for candidate in candidates:
        try:
            candidate.mkdir(parents=True, exist_ok=True)
            probe = candidate / ".write_test"
            probe.write_text("ok", encoding="utf-8")
            probe.unlink(missing_ok=True)
            return candidate
        except (PermissionError, OSError):
            continue
    return None


def _setup_logging() -> logging.Logger:
    """파일/콘솔 로거를 설정하고 애플리케이션 로거를 반환한다.

    파일 로깅이 불가능한 환경(Cloud 등)에서는 콘솔(StreamHandler)만 사용하며,
    어떤 경우에도 예외로 앱 시작을 막지 않는다.
    """
    file_handler: logging.Handler | None = None
    log_dir = _resolve_log_dir()
    if log_dir is not None:
        try:
            log_path = log_dir / f"multiusers_{datetime.now().strftime('%Y%m%d')}.log"
            file_handler = logging.FileHandler(log_path, encoding="utf-8")
        except (PermissionError, OSError):
            file_handler = None

    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(logging.WARNING)

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    ch = logging.StreamHandler()
    ch.setLevel(logging.WARNING)
    ch.setFormatter(fmt)
    root.addHandler(ch)
    if file_handler is not None:
        file_handler.setLevel(logging.WARNING)
        file_handler.setFormatter(fmt)
        root.addHandler(file_handler)

    for name in ("httpx", "httpcore", "urllib3", "openai", "langchain", "hpack"):
        logging.getLogger(name).setLevel(logging.WARNING)

    return logging.getLogger("multiusers")


logger = _setup_logging()


# ---------------------------------------------------------------------------
# 설정값 로딩 (st.secrets 우선 → .env/os.getenv)
# ---------------------------------------------------------------------------
def get_config_value(key: str) -> str:
    """secrets 또는 환경변수에서 설정값을 읽어 반환한다.

    우선순위: st.secrets 값이 있으면 사용, 없으면 .env/os.getenv.

    Args:
        key: 조회할 설정 키 이름.

    Returns:
        str: 값 (없으면 빈 문자열).
    """
    try:
        if key in st.secrets:  # type: ignore[operator]
            return str(st.secrets[key]).strip()
    except Exception:  # noqa: BLE001 - secrets 미설정 환경
        pass
    return os.getenv(key, "").strip()


def missing_keys() -> list[str]:
    """필수 키 중 누락된 항목 목록을 반환한다."""
    required = ("OPENAI_API_KEY", "SUPABASE_URL", "SUPABASE_ANON_KEY")
    return [k for k in required if not get_config_value(k)]


# ---------------------------------------------------------------------------
# 프롬프트 & 텍스트 유틸
# ---------------------------------------------------------------------------
ANSWER_STYLE_SYSTEM = """당신은 친절하고 공손한 AI 어시스턴트입니다.

답변 규칙:
- 반드시 마크다운 헤딩(# ## ###)으로 구조화하세요. 주요 주제는 #, 세부는 ##, 구체 설명은 ###.
- 서술형으로 완전한 문장을 사용하고 존댓말로 작성하세요.
- 구분선(---, ===, ___)은 사용하지 마세요.
- 취소선(~~텍스트~~)은 사용하지 마세요.
- 참조 표시, 각주, 출처 문구, URL 인용 문장은 넣지 마세요.
"""


def remove_separators(text: str) -> str:
    """마크다운 취소선/구분선/과도한 빈 줄을 제거한다."""
    out = re.sub(r"~~([^~]*)~~", r"\1", text)
    out = re.sub(r"(?m)^\s*-{3,}\s*$", "", out)
    out = re.sub(r"(?m)^\s*={3,}\s*$", "", out)
    out = re.sub(r"(?m)^\s*_{3,}\s*$", "", out)
    out = re.sub(r"\n{3,}", "\n\n", out)
    return out.strip()


# ---------------------------------------------------------------------------
# 비밀번호 해시 (PBKDF2-SHA256, 평문 저장 금지)
# ---------------------------------------------------------------------------
def hash_password(password: str) -> str:
    """비밀번호를 PBKDF2-SHA256으로 해시해 저장용 문자열을 만든다.

    반환 형식: ``pbkdf2_sha256$<iterations>$<salt_hex>$<hash_hex>``

    Args:
        password: 평문 비밀번호.

    Returns:
        str: 저장 가능한 해시 문자열.
    """
    salt = os.urandom(16)
    dk = hashlib.pbkdf2_hmac(
        _PBKDF2_ALGO, password.encode("utf-8"), salt, _PBKDF2_ITERATIONS
    )
    return f"pbkdf2_{_PBKDF2_ALGO}${_PBKDF2_ITERATIONS}${salt.hex()}${dk.hex()}"


def verify_password(password: str, stored: str) -> bool:
    """평문 비밀번호가 저장된 해시와 일치하는지 상수시간 비교로 검증한다.

    Args:
        password: 입력된 평문 비밀번호.
        stored: DB에 저장된 해시 문자열.

    Returns:
        bool: 일치 여부.
    """
    try:
        _, iterations_s, salt_hex, hash_hex = stored.split("$")
        iterations = int(iterations_s)
        salt = bytes.fromhex(salt_hex)
        expected = bytes.fromhex(hash_hex)
    except (ValueError, AttributeError):
        return False
    dk = hashlib.pbkdf2_hmac(
        _PBKDF2_ALGO, password.encode("utf-8"), salt, iterations
    )
    return hmac.compare_digest(dk, expected)


# ---------------------------------------------------------------------------
# LLM / 임베딩
# ---------------------------------------------------------------------------
def get_llm(temperature: float = 0.7) -> ChatOpenAI:
    """gpt-4o-mini 채팅 모델을 반환한다."""
    key = get_config_value("OPENAI_API_KEY")
    if not key:
        raise ValueError("OPENAI_API_KEY가 설정되어 있지 않습니다.")
    return ChatOpenAI(model=LLM_MODEL, temperature=temperature, api_key=key)


def get_embeddings() -> OpenAIEmbeddings:
    """OpenAI 임베딩 모델을 반환한다."""
    key = get_config_value("OPENAI_API_KEY")
    if not key:
        raise ValueError("OPENAI_API_KEY가 설정되어 있지 않습니다.")
    return OpenAIEmbeddings(model=EMBEDDING_MODEL, api_key=key)


def generate_session_title(first_q: str, first_a: str) -> str:
    """첫 질문/답변을 요약해 세션 제목을 생성한다.

    Args:
        first_q: 첫 사용자 질문.
        first_a: 첫 어시스턴트 답변.

    Returns:
        str: 20자 내외의 세션 제목. 실패 시 질문 앞부분으로 대체.
    """
    fallback = (first_q or "새 세션").strip().replace("\n", " ")[:30] or "새 세션"
    try:
        llm = get_llm(temperature=0.3)
        prompt = (
            "다음 대화의 핵심 주제를 한국어로 20자 이내의 짧은 제목으로 요약하세요.\n"
            "따옴표나 마침표, 설명 없이 제목 텍스트만 출력하세요.\n\n"
            f"[질문]\n{first_q}\n\n[답변]\n{first_a[:1500]}"
        )
        out = llm.invoke([HumanMessage(content=prompt)])
        title = str(getattr(out, "content", "") or "").strip()
        title = title.strip("\"'` \n")
        return title[:40] if title else fallback
    except Exception as exc:  # noqa: BLE001
        logger.warning("세션 제목 생성 실패: %s", exc)
        return fallback


def generate_followup_section(user_q: str, answer: str) -> str:
    """이어서 물어볼 만한 후속 질문 3개 마크다운 블록을 생성한다."""
    try:
        llm = get_llm(temperature=0.3)
        prompt = (
            "다음 사용자 질문과 답변을 바탕으로, 이어서 물어볼 만한 후속 질문을 "
            "한국어로 정확히 3개만 작성하세요.\n"
            "형식:\n1. ...\n2. ...\n3. ...\n"
            "설명 문장이나 다른 텍스트는 출력하지 마세요.\n\n"
            f"[사용자 질문]\n{user_q}\n\n[답변]\n{answer[:8000]}"
        )
        out = llm.invoke([HumanMessage(content=prompt)])
        raw = remove_separators(str(getattr(out, "content", "") or ""))
        if not raw.strip():
            return ""
        return f"\n\n### 💡 다음에 물어볼 수 있는 질문들\n\n{raw.strip()}\n"
    except Exception as exc:  # noqa: BLE001
        logger.warning("후속 질문 생성 실패: %s", exc)
        return ""


# ---------------------------------------------------------------------------
# Supabase 클라이언트
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_supabase() -> Any:
    """Supabase 클라이언트를 생성해 캐시한다."""
    if create_client is None:
        raise RuntimeError(
            "supabase 패키지가 설치되어 있지 않습니다. "
            "'pip install -r requirements.txt' 를 실행해 주세요."
        )
    url = get_config_value("SUPABASE_URL")
    key = get_config_value("SUPABASE_ANON_KEY")
    if not url or not key:
        raise RuntimeError("SUPABASE_URL 또는 SUPABASE_ANON_KEY가 설정되어 있지 않습니다.")
    return create_client(url, key)


# ---------------------------------------------------------------------------
# 사용자 인증 (회원가입/로그인) — "user" 테이블 기반
# ---------------------------------------------------------------------------
def db_get_user_by_login_id(login_id: str) -> dict[str, Any] | None:
    """login_id로 사용자 행을 조회한다(없으면 None)."""
    supabase = get_supabase()
    resp = (
        supabase.table("user")
        .select("id, login_id, password_hash")
        .eq("login_id", login_id)
        .limit(1)
        .execute()
    )
    rows = resp.data or []
    return rows[0] if rows else None


def db_create_user(login_id: str, password: str) -> dict[str, Any]:
    """새 사용자를 생성하고 생성된 행(id, login_id)을 반환한다.

    Raises:
        ValueError: 이미 존재하는 login_id인 경우.
    """
    if db_get_user_by_login_id(login_id) is not None:
        raise ValueError("이미 사용 중인 아이디입니다.")
    supabase = get_supabase()
    resp = (
        supabase.table("user")
        .insert(
            {
                "login_id": login_id,
                "password_hash": hash_password(password),
            }
        )
        .execute()
    )
    rows = resp.data or []
    if not rows:
        raise RuntimeError("회원가입에 실패했습니다. 잠시 후 다시 시도해 주세요.")
    return rows[0]


def authenticate_user(login_id: str, password: str) -> dict[str, Any] | None:
    """login_id/password를 검증하고 성공 시 사용자 정보를 반환한다."""
    user = db_get_user_by_login_id(login_id)
    if user is None:
        return None
    if not verify_password(password, str(user.get("password_hash", ""))):
        return None
    return {"id": user["id"], "login_id": user["login_id"]}


# ---------------------------------------------------------------------------
# DB 헬퍼 (모든 접근은 user_id로 필터링)
# ---------------------------------------------------------------------------
def db_save_session(
    user_id: int,
    session_id: str,
    title: str,
    messages: list[dict[str, str]],
    file_names: list[str],
) -> None:
    """세션과 메시지를 Supabase에 저장(upsert)한다.

    chat_sessions 는 session_id 기준으로 upsert하고,
    chat_messages 는 해당 세션 기존 메시지를 지운 뒤 현재 대화를 다시 기록한다.
    """
    supabase = get_supabase()
    now = datetime.utcnow().isoformat()
    supabase.table("chat_sessions").upsert(
        {
            "user_id": user_id,
            "session_id": session_id,
            "title": title,
            "file_names": file_names,
            "updated_at": now,
        },
        on_conflict="session_id",
    ).execute()

    supabase.table("chat_messages").delete().eq("session_id", session_id).eq(
        "user_id", user_id
    ).execute()
    rows = [
        {
            "user_id": user_id,
            "session_id": session_id,
            "role": m["role"],
            "content": m["content"],
            "seq": i,
        }
        for i, m in enumerate(messages)
    ]
    if rows:
        supabase.table("chat_messages").insert(rows).execute()


def db_list_sessions(user_id: int) -> list[dict[str, Any]]:
    """해당 사용자의 세션 목록을 최신순으로 반환한다."""
    supabase = get_supabase()
    resp = (
        supabase.table("chat_sessions")
        .select("session_id, title, updated_at, file_names")
        .eq("user_id", user_id)
        .order("updated_at", desc=True)
        .execute()
    )
    return list(resp.data or [])


def db_load_session(user_id: int, session_id: str) -> dict[str, Any]:
    """해당 사용자의 세션 제목/메시지/파일명을 로드한다."""
    supabase = get_supabase()
    sess = (
        supabase.table("chat_sessions")
        .select("title, file_names")
        .eq("user_id", user_id)
        .eq("session_id", session_id)
        .limit(1)
        .execute()
    )
    title = ""
    file_names: list[str] = []
    if sess.data:
        title = sess.data[0].get("title") or ""
        file_names = sess.data[0].get("file_names") or []

    msgs = (
        supabase.table("chat_messages")
        .select("role, content, seq")
        .eq("user_id", user_id)
        .eq("session_id", session_id)
        .order("seq")
        .execute()
    )
    messages = [
        {"role": r["role"], "content": r["content"]} for r in (msgs.data or [])
    ]
    return {"title": title, "messages": messages, "file_names": file_names}


def db_delete_session(user_id: int, session_id: str) -> None:
    """해당 사용자의 세션과 관련 메시지/벡터 문서를 모두 삭제한다."""
    supabase = get_supabase()
    supabase.table("vector_documents").delete().eq("user_id", user_id).eq(
        "session_id", session_id
    ).execute()
    supabase.table("chat_messages").delete().eq("user_id", user_id).eq(
        "session_id", session_id
    ).execute()
    supabase.table("chat_sessions").delete().eq("user_id", user_id).eq(
        "session_id", session_id
    ).execute()


def db_ensure_session_row(user_id: int, session_id: str) -> None:
    """벡터 문서 저장 전, FK 제약을 위해 세션 행이 존재하도록 보장한다."""
    supabase = get_supabase()
    now = datetime.utcnow().isoformat()
    supabase.table("chat_sessions").upsert(
        {"user_id": user_id, "session_id": session_id, "updated_at": now},
        on_conflict="session_id",
    ).execute()


def db_insert_vector_documents(
    user_id: int,
    session_id: str,
    splits: list[Document],
) -> int:
    """분할된 문서를 임베딩하여 vector_documents에 배치 저장한다.

    file_name과 user_id는 각 청크에 명시적으로 포함하여 NOT NULL 제약 위반을 방지한다.

    Returns:
        int: 저장된 청크 수.
    """
    if not splits:
        return 0
    db_ensure_session_row(user_id, session_id)
    supabase = get_supabase()
    embeddings = get_embeddings()

    saved = 0
    for start in range(0, len(splits), EMBED_BATCH_SIZE):
        batch = splits[start : start + EMBED_BATCH_SIZE]
        texts = [d.page_content for d in batch]
        vectors = embeddings.embed_documents(texts)
        rows = []
        for doc, vec in zip(batch, vectors):
            file_name = str(doc.metadata.get("file_name") or "unknown.pdf")
            rows.append(
                {
                    "user_id": user_id,
                    "session_id": session_id,
                    "file_name": file_name,
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "embedding": vec,
                }
            )
        supabase.table("vector_documents").insert(rows).execute()
        saved += len(rows)
    return saved


def db_retrieve_documents(
    user_id: int,
    session_id: str,
    query: str,
    k: int = 10,
) -> list[Document]:
    """RPC(match_vector_documents)로 user_id·session_id 필터링된 문서를 검색한다."""
    supabase = get_supabase()
    embeddings = get_embeddings()
    query_vec = embeddings.embed_query(query)
    try:
        resp = supabase.rpc(
            "match_vector_documents",
            {
                "query_embedding": query_vec,
                "match_count": k,
                "filter_session_id": session_id,
                "filter_user_id": user_id,
            },
        ).execute()
        rows = resp.data or []
    except Exception as exc:  # noqa: BLE001 - RPC 실패 시 폴백
        logger.warning("RPC 검색 실패, 폴백 사용: %s", exc)
        rows = _fallback_retrieve(user_id, session_id, k)

    docs: list[Document] = []
    for r in rows:
        docs.append(
            Document(
                page_content=r.get("content", ""),
                metadata={
                    "file_name": r.get("file_name"),
                    **(r.get("metadata") or {}),
                },
            )
        )
    return docs


def _fallback_retrieve(user_id: int, session_id: str, k: int) -> list[dict[str, Any]]:
    """RPC가 없을 때 사용자·세션의 문서를 단순 조회하는 폴백."""
    try:
        supabase = get_supabase()
        resp = (
            supabase.table("vector_documents")
            .select("content, file_name, metadata")
            .eq("user_id", user_id)
            .eq("session_id", session_id)
            .limit(k)
            .execute()
        )
        return list(resp.data or [])
    except Exception as exc:  # noqa: BLE001
        logger.warning("폴백 검색 실패: %s", exc)
        return []


def db_list_vector_files(user_id: int, session_id: str) -> list[str]:
    """현재 사용자·세션 벡터 DB에 저장된 고유 파일명 목록을 반환한다."""
    supabase = get_supabase()
    resp = (
        supabase.table("vector_documents")
        .select("file_name")
        .eq("user_id", user_id)
        .eq("session_id", session_id)
        .execute()
    )
    names = {r.get("file_name") for r in (resp.data or []) if r.get("file_name")}
    return sorted(names)


# ---------------------------------------------------------------------------
# PDF 처리
# ---------------------------------------------------------------------------
def process_pdf_uploads(uploaded_files: list[Any]) -> list[Document]:
    """PDF들을 로드/분할하고 각 청크에 파일명을 명시적으로 부여한다."""
    all_splits: list[Document] = []
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    for uf in uploaded_files:
        suffix = Path(uf.name).suffix.lower() or ".pdf"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uf.getvalue())
            tmp_path = tmp.name
        try:
            docs = PyPDFLoader(tmp_path).load()
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

        splits = splitter.split_documents(docs)
        for s in splits:
            s.metadata = {**(s.metadata or {}), "file_name": uf.name}
        all_splits.extend(splits)
    return all_splits


def build_rag_messages(question: str, context: str, memory_text: str) -> list[Any]:
    """RAG 답변용 LangChain 메시지를 구성한다."""
    sys = f"""{ANSWER_STYLE_SYSTEM}

아래 [대화 맥락]과 [참고 문서]를 활용해 답하세요. 참고 문서에 없는 내용은 추측하지 말고 한계를 밝히세요.
[대화 맥락]
{memory_text or "(없음)"}

[참고 문서]
{context or "(없음)"}
"""
    return [SystemMessage(content=sys), HumanMessage(content=question)]


def format_memory_block(messages: list[dict[str, str]], max_items: int = 20) -> str:
    """최근 대화를 RAG 맥락 문자열로 변환한다."""
    tail = messages[-max_items:] if len(messages) > max_items else messages
    lines: list[str] = []
    for m in tail:
        content = (m.get("content") or "").strip()
        if not content:
            continue
        prefix = "사용자" if m.get("role") == "user" else "어시스턴트"
        lines.append(f"{prefix}: {content}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 세션 상태
# ---------------------------------------------------------------------------
def init_session() -> None:
    """st.session_state 기본값을 초기화한다."""
    defaults: dict[str, Any] = {
        "auth_user": None,  # {"id": int, "login_id": str} 또는 None
        "session_id": uuid.uuid4().hex,
        "chat_history": [],
        "processed_names": [],
        "current_title": "",
        "has_vectors": False,
        "loaded_selection": "__new__",
        "vectordb_files": [],
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def current_user_id() -> int:
    """로그인한 사용자의 id를 반환한다."""
    user = st.session_state.get("auth_user") or {}
    return int(user["id"])


def reset_to_new_session() -> None:
    """새 세션으로 화면을 초기화한다(저장된 데이터는 보존)."""
    st.session_state.session_id = uuid.uuid4().hex
    st.session_state.chat_history = []
    st.session_state.processed_names = []
    st.session_state.current_title = ""
    st.session_state.has_vectors = False
    st.session_state.loaded_selection = "__new__"
    st.session_state.vectordb_files = []


def autosave_current_session() -> None:
    """현재 세션을 자동 저장한다(제목이 없으면 첫 Q&A로 생성)."""
    history = st.session_state.chat_history
    if not history:
        return
    title = st.session_state.current_title
    if not title:
        first_q = next((m["content"] for m in history if m["role"] == "user"), "")
        first_a = next(
            (m["content"] for m in history if m["role"] == "assistant"), ""
        )
        title = generate_session_title(first_q, first_a)
        st.session_state.current_title = title
    try:
        db_save_session(
            current_user_id(),
            st.session_state.session_id,
            title,
            history,
            st.session_state.processed_names,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("자동 저장 실패: %s", exc)
        st.toast(f"자동 저장 실패: {exc}", icon="⚠️")


def load_session_into_state(session_id: str) -> None:
    """지정한 세션을 로드해 화면 상태에 반영한다."""
    uid = current_user_id()
    data = db_load_session(uid, session_id)
    st.session_state.session_id = session_id
    st.session_state.chat_history = data["messages"]
    st.session_state.current_title = data["title"]
    st.session_state.processed_names = list(data["file_names"] or [])
    st.session_state.vectordb_files = db_list_vector_files(uid, session_id)
    st.session_state.has_vectors = len(st.session_state.vectordb_files) > 0


# ---------------------------------------------------------------------------
# 공통 스타일 / 헤더
# ---------------------------------------------------------------------------
def inject_style() -> None:
    """ref.py 스타일(핑크/골드/블루 헤딩, 핑크 버튼)을 주입한다."""
    st.markdown(
        """
<style>
h1 { color: #ff69b4 !important; font-size: 1.4rem !important; }
h2 { color: #ffd700 !important; font-size: 1.2rem !important; }
h3 { color: #1f77b4 !important; font-size: 1.1rem !important; }
div.stButton > button:first-child {
  background-color: #ff69b4;
  color: #ffffff;
  border: none;
}
</style>
""",
        unsafe_allow_html=True,
    )


def render_header() -> None:
    """로고 + 제목 헤더를 렌더링한다."""
    c1, c2, c3 = st.columns([1, 4, 1])
    with c1:
        if LOGO_PATH.is_file():
            st.image(str(LOGO_PATH), width=180)
        else:
            st.markdown("### 📚")
    with c2:
        st.markdown(
            """
<h1 style="text-align:center; margin:0;">
  <span style="color:#1f77b4;">숭실대학교</span>
  <span style="color:#ff8c00;">RAG 챗봇</span>
</h1>
<p style="text-align:center; color:#888; margin-top:4px;">
  Supabase 기반 멀티유저·멀티세션 RAG
</p>
""",
            unsafe_allow_html=True,
        )
    with c3:
        st.empty()


# ---------------------------------------------------------------------------
# 로그인 / 회원가입 화면 (화면 중앙)
# ---------------------------------------------------------------------------
def render_auth_screen() -> None:
    """미로그인 상태에서 화면 중앙에 로그인/회원가입 UI를 표시한다."""
    render_header()

    _, center, _ = st.columns([1, 2, 1])
    with center:
        st.markdown(
            """
<h2 style="text-align:center;">🔐 로그인 / 회원가입</h2>
<p style="text-align:center; color:#888;">
  서비스를 이용하려면 로그인하거나 새 계정을 만들어 주세요.
</p>
""",
            unsafe_allow_html=True,
        )

        login_tab, signup_tab = st.tabs(["로그인", "회원가입"])

        with login_tab:
            with st.form("login_form", clear_on_submit=False):
                login_id = st.text_input("아이디", key="login_id_input")
                password = st.text_input(
                    "비밀번호", type="password", key="login_pw_input"
                )
                submitted = st.form_submit_button("로그인", use_container_width=True)
            if submitted:
                _handle_login(login_id.strip(), password)

        with signup_tab:
            with st.form("signup_form", clear_on_submit=False):
                new_id = st.text_input("아이디", key="signup_id_input")
                new_pw = st.text_input(
                    "비밀번호", type="password", key="signup_pw_input"
                )
                new_pw2 = st.text_input(
                    "비밀번호 확인", type="password", key="signup_pw2_input"
                )
                submitted2 = st.form_submit_button(
                    "회원가입", use_container_width=True
                )
            if submitted2:
                _handle_signup(new_id.strip(), new_pw, new_pw2)


def _login_success(user: dict[str, Any]) -> None:
    """로그인/회원가입 성공 처리: 상태를 초기화하고 대시보드로 이동한다."""
    st.session_state.auth_user = user
    reset_to_new_session()
    st.rerun()


def _handle_login(login_id: str, password: str) -> None:
    """로그인 폼 제출 처리."""
    if not login_id or not password:
        st.warning("아이디와 비밀번호를 모두 입력해 주세요.")
        return
    try:
        user = authenticate_user(login_id, password)
    except Exception as exc:  # noqa: BLE001
        logger.warning("로그인 실패: %s", exc)
        st.error(f"로그인 중 오류가 발생했습니다: {exc}")
        return
    if user is None:
        st.error("아이디 또는 비밀번호가 올바르지 않습니다.")
        return
    st.success("로그인 성공! 대시보드로 이동합니다.")
    _login_success(user)


def _handle_signup(login_id: str, pw: str, pw2: str) -> None:
    """회원가입 폼 제출 처리(성공 시 자동 로그인)."""
    if not login_id or not pw:
        st.warning("아이디와 비밀번호를 입력해 주세요.")
        return
    if len(pw) < 4:
        st.warning("비밀번호는 4자 이상으로 설정해 주세요.")
        return
    if pw != pw2:
        st.error("비밀번호와 비밀번호 확인이 일치하지 않습니다.")
        return
    try:
        created = db_create_user(login_id, pw)
    except ValueError as exc:
        st.error(str(exc))
        return
    except Exception as exc:  # noqa: BLE001
        logger.warning("회원가입 실패: %s", exc)
        st.error(f"회원가입 중 오류가 발생했습니다: {exc}")
        return
    st.success("회원가입 성공! 자동 로그인 후 대시보드로 이동합니다.")
    _login_success({"id": created["id"], "login_id": created["login_id"]})


# ---------------------------------------------------------------------------
# 사이드바 (로그인 후)
# ---------------------------------------------------------------------------
def render_sidebar() -> None:
    """세션 관리 사이드바를 렌더링한다."""
    uid = current_user_id()
    with st.sidebar:
        user_login = (st.session_state.auth_user or {}).get("login_id", "")
        st.markdown(f"### 🤖 {CHATBOT_NAME}")
        st.caption(f"LLM: {LLM_MODEL}")
        st.markdown(f"**👤 {user_login} 님**")
        if st.button("🚪 로그아웃", use_container_width=True):
            _handle_logout()

        st.divider()

        # 세션 선택 (선택 시 자동 로드) --------------------------------------
        try:
            sessions = db_list_sessions(uid)
        except Exception as exc:  # noqa: BLE001
            sessions = []
            st.error(f"세션 목록을 불러오지 못했습니다: {exc}")

        options = ["__new__"] + [s["session_id"] for s in sessions]
        labels = {"__new__": "➕ 새 세션"}
        for s in sessions:
            title = s.get("title") or "(제목 없음)"
            labels[s["session_id"]] = f"📄 {title}"

        current = st.session_state.session_id
        if current not in options:
            options.insert(1, current)
            labels[current] = f"📄 {st.session_state.current_title or '(현재 세션)'}"

        try:
            default_idx = options.index(current)
        except ValueError:
            default_idx = 0

        selected = st.selectbox(
            "세션 선택",
            options,
            index=default_idx,
            format_func=lambda x: labels.get(x, x),
            key="session_selectbox",
        )

        # 풀다운에서 다른 세션을 고르면 자동 로드
        if selected != st.session_state.loaded_selection:
            st.session_state.loaded_selection = selected
            if selected == "__new__":
                reset_to_new_session()
            elif selected != current:
                try:
                    load_session_into_state(selected)
                    st.toast("세션을 불러왔습니다.", icon="✅")
                except Exception as exc:  # noqa: BLE001
                    st.error(f"세션 로드 실패: {exc}")
            st.rerun()

        st.divider()

        # 관리 버튼 ---------------------------------------------------------
        col1, col2 = st.columns(2)
        with col1:
            if st.button("💾 세션저장", use_container_width=True):
                _handle_save_button()
        with col2:
            if st.button("📂 세션로드", use_container_width=True):
                _handle_load_button(selected)

        col3, col4 = st.columns(2)
        with col3:
            if st.button("🗑️ 세션삭제", use_container_width=True):
                _handle_delete_button()
        with col4:
            if st.button("🧹 화면초기화", use_container_width=True):
                reset_to_new_session()
                st.rerun()

        if st.button("🔎 vectordb", use_container_width=True):
            _handle_vectordb_button()

        st.divider()

        # PDF 업로드 --------------------------------------------------------
        uploads = st.file_uploader(
            "PDF 파일 업로드",
            type=["pdf"],
            accept_multiple_files=True,
        )
        if st.button("📥 파일 처리하기", use_container_width=True):
            _handle_process_button(uploads)

        if st.session_state.processed_names:
            st.markdown("**처리된 파일**")
            for name in st.session_state.processed_names:
                st.text(f"- {name}")

        # 상태 표시 ---------------------------------------------------------
        st.divider()
        st.text(
            f"세션 제목: {st.session_state.current_title or '(미저장)'}\n"
            f"메시지 수: {len(st.session_state.chat_history)}\n"
            f"벡터 문서: {'있음' if st.session_state.has_vectors else '없음'}"
        )


def _handle_logout() -> None:
    """로그아웃 버튼: 인증 상태와 화면을 초기화하고 로그인 화면으로 돌아간다."""
    st.session_state.auth_user = None
    reset_to_new_session()
    st.toast("로그아웃되었습니다.", icon="👋")
    st.rerun()


def _handle_save_button() -> None:
    """세션저장 버튼: 제목 생성 후 Supabase에 저장(upsert)."""
    if not st.session_state.chat_history:
        st.warning("저장할 대화가 없습니다.")
        return
    try:
        autosave_current_session()
        st.success("세션을 저장했습니다.")
        st.rerun()
    except Exception as exc:  # noqa: BLE001
        st.error(f"세션 저장 실패: {exc}")


def _handle_load_button(selected: str) -> None:
    """세션로드 버튼: 선택된 세션을 로드."""
    if selected == "__new__":
        st.warning("불러올 세션을 먼저 선택하세요.")
        return
    try:
        load_session_into_state(selected)
        st.session_state.loaded_selection = selected
        st.success("세션을 불러왔습니다.")
        st.rerun()
    except Exception as exc:  # noqa: BLE001
        st.error(f"세션 로드 실패: {exc}")


def _handle_delete_button() -> None:
    """세션삭제 버튼: 현재 선택된 세션과 관련 데이터를 삭제."""
    try:
        db_delete_session(current_user_id(), st.session_state.session_id)
        reset_to_new_session()
        st.success("세션을 삭제했습니다.")
        st.rerun()
    except Exception as exc:  # noqa: BLE001
        st.error(f"세션 삭제 실패: {exc}")


def _handle_vectordb_button() -> None:
    """vectordb 버튼: 현재 세션 벡터 DB의 파일명 목록 표시."""
    try:
        files = db_list_vector_files(current_user_id(), st.session_state.session_id)
        st.session_state.vectordb_files = files
        if files:
            st.info("벡터 DB 파일 목록:\n" + "\n".join(f"- {f}" for f in files))
        else:
            st.info("현재 세션의 벡터 DB에 저장된 파일이 없습니다.")
    except Exception as exc:  # noqa: BLE001
        st.error(f"벡터 DB 조회 실패: {exc}")


def _handle_process_button(uploads: list[Any] | None) -> None:
    """파일 처리 버튼: PDF 임베딩을 Supabase에 저장하고 자동 저장."""
    if not uploads:
        st.warning("업로드된 PDF가 없습니다.")
        return
    try:
        uid = current_user_id()
        with st.spinner("PDF를 처리하고 임베딩을 저장하는 중..."):
            splits = process_pdf_uploads(list(uploads))
            count = db_insert_vector_documents(uid, st.session_state.session_id, splits)
            names = sorted({u.name for u in uploads})
            merged = sorted(set(st.session_state.processed_names) | set(names))
            st.session_state.processed_names = merged
            st.session_state.has_vectors = True
            st.session_state.vectordb_files = db_list_vector_files(
                uid, st.session_state.session_id
            )
            autosave_current_session()
        st.success(f"PDF 처리 완료: {count}개 청크를 저장했습니다.")
    except Exception as exc:  # noqa: BLE001
        logger.warning("PDF 처리 실패: %s", exc)
        st.error(f"PDF 처리 중 오류가 발생했습니다: {exc}")


# ---------------------------------------------------------------------------
# 본문 (채팅)
# ---------------------------------------------------------------------------
def render_welcome() -> None:
    """로그인 후 환영 메시지를 표시한다."""
    if st.session_state.chat_history:
        return
    user_login = (st.session_state.auth_user or {}).get("login_id", "")
    st.markdown(
        f"""
### 👋 환영합니다, {user_login} 님!

무엇이든 물어보세요. PDF를 업로드하면 문서 기반(RAG)으로 답변해 드립니다.
왼쪽 사이드바에서 세션을 저장·불러오기·삭제할 수 있습니다.
"""
    )


def render_chat_and_input() -> None:
    """대화 내역 표시 및 입력 처리(스트리밍 답변)."""
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(remove_separators(msg["content"]))

    user_input = st.chat_input("질문을 입력하세요")
    if not user_input:
        return

    uid = current_user_id()
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(remove_separators(user_input))

    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_answer = ""
        try:
            llm = get_llm()
            memory = format_memory_block(st.session_state.chat_history[:-1])

            context = ""
            if st.session_state.has_vectors:
                docs = db_retrieve_documents(
                    uid, st.session_state.session_id, user_input, k=10
                )
                context = "\n\n".join(d.page_content for d in docs)

            messages = build_rag_messages(user_input, context, memory)

            acc = ""
            for chunk in llm.stream(messages):
                piece = getattr(chunk, "content", "") or ""
                if piece:
                    acc += piece
                    placeholder.markdown(remove_separators(acc) + "▌")
            full_answer = remove_separators(acc)
            placeholder.markdown(full_answer)

            follow = generate_followup_section(user_input, full_answer)
            if follow:
                full_answer += follow
                placeholder.markdown(remove_separators(full_answer))
        except Exception as exc:  # noqa: BLE001
            logger.warning("답변 생성 실패: %s", exc)
            full_answer = f"# 오류\n\n요청을 처리하는 중 문제가 발생했습니다.\n\n`{exc}`"
            placeholder.markdown(remove_separators(full_answer))

    st.session_state.chat_history.append(
        {"role": "assistant", "content": full_answer}
    )

    # 대화 후 자동 저장
    if not full_answer.lstrip().startswith("# 오류"):
        autosave_current_session()
    st.rerun()


# ---------------------------------------------------------------------------
# 진입점
# ---------------------------------------------------------------------------
def main() -> None:
    """Streamlit 앱 진입점."""
    st.set_page_config(page_title=CHATBOT_NAME, page_icon="📚", layout="wide")
    inject_style()
    init_session()

    missing = missing_keys()
    if missing:
        render_header()
        st.warning(
            "다음 키가 설정되어 있지 않습니다: "
            + ", ".join(missing)
            + "\n\n프로젝트 루트의 `.env` 파일 또는 Streamlit secrets에 값을 추가해 주세요."
        )
        st.stop()

    # 미로그인 → 로그인/회원가입 전용 화면만 표시
    if not st.session_state.get("auth_user"):
        render_auth_screen()
        return

    # 로그인 후 → 메인 대시보드
    render_header()
    render_sidebar()
    render_welcome()
    render_chat_and_input()


if __name__ == "__main__":
    main()
