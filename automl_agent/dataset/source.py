"""Where the data comes from: a CSV, an sqlite file, or a database URL.

Roles:

* Source kinds — tell the kind from the source string alone.
* SQL input checks — build the SELECT; refuse passwords in URLs.
* Loading — read all rows; the fixed scripts' only entry point.
"""

from __future__ import annotations

import os
import re
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - type checking only
    import pandas as pd

# Needs ``://`` so a Windows drive letter does not match.
URL_SCHEME = re.compile(r"^[A-Za-z][A-Za-z0-9+.\-]*://")
SQLITE_SUFFIXES = frozenset({".db", ".sqlite", ".sqlite3"})
# Put into SQL text, so plain names only
TABLE_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
DB_PASSWORD_ENV = "AUTOML_DB_PASSWORD"


# --- Role: source kinds -----------------------------------------------------------


def is_database(source: Any) -> bool:
    """Tell whether ``source`` is a URL or sqlite file; opens no file."""
    text = str(source)
    return bool(URL_SCHEME.match(text)) or Path(text).suffix.lower() in SQLITE_SUFFIXES


def as_source(text: Any) -> Path | str:
    """Turn a CLI string into a ``Path``, a URL string, or ``""``.

    Never wrap a URL in ``Path``: Windows silently breaks its slashes."""
    if text is None or text == "":
        return ""
    value = str(text)
    return value if URL_SCHEME.match(value) else Path(value)


def source_kind(source: Any) -> str:
    """Name the source kind in one word for the public card.

    Never the source string itself: the description reaches prompts."""
    text = str(source)
    scheme = URL_SCHEME.match(text)
    if scheme:
        return "sql"
    if Path(text).suffix.lower() in SQLITE_SUFFIXES:
        return "sqlite"
    return Path(text).suffix.lstrip(".").lower() or "csv"


# --- Role: SQL input checks -------------------------------------------------------


def select_statement(table: str | None = None, query: str | None = None) -> str:
    """Build the SQL from exactly one of ``--table`` or ``--query``.

    Raises ValueError for both, neither, or a non-plain table name."""
    if bool(table) == bool(query):
        raise ValueError(
            "DB 출처에는 --table 또는 --query 중 정확히 하나가 필요합니다 "
            "(--table 은 SELECT * FROM <이름> 의 줄임입니다)."
        )
    if query:
        return str(query)
    name = str(table)
    if not TABLE_NAME.match(name):
        raise ValueError(
            f"--table {name!r}은 식별자 문법이 아닙니다 (영문자나 _로 시작하고 영숫자와 _만). "
            "따옴표나 스키마 접두사가 필요하면 --query 로 질의를 직접 주십시오."
        )
    return f'SELECT * FROM "{name}"'


def assert_no_inline_password(source: Any) -> None:
    """Raise ValueError for a database URL with a password in it.

    The message leaves out the URL, since it holds the secret."""
    text = str(source)
    if not URL_SCHEME.match(text):
        return
    authority = text.split("://", 1)[1].split("/", 1)[0]
    if "@" not in authority:
        return
    userinfo = authority.rsplit("@", 1)[0]
    if ":" in userinfo:
        raise ValueError(
            "접속 URL에 비밀번호를 담지 마십시오. 출처 문자열은 run_config.json에 그대로 적히고 "
            "--resume 이 그것을 다시 읽으므로, URL에 담은 비밀번호는 디스크에 적힌 비밀번호입니다. "
            f"URL에는 사용자 이름까지만 주고 비밀번호는 {DB_PASSWORD_ENV} 환경변수로 주십시오."
        )


# --- Role: loading ----------------------------------------------------------------


def _engine(url: str) -> Any:
    """_engine | Loading: make a SQLAlchemy engine with the env password."""
    try:
        from sqlalchemy import create_engine
        from sqlalchemy.engine import make_url
    except ImportError as exc:  # pragma: no cover - depends on what is installed
        # Keep ImportError: the scripts' ``main`` print it in one line.
        raise ImportError(
            "접속 URL을 읽으려면 SQLAlchemy가 필요합니다: pip install -e \".[db]\". "
            "드라이버는 따로입니다 — postgres는 psycopg, mysql은 pymysql. "
            "sqlite라면 URL 대신 파일 경로(local/x.db)를 주면 의존성 없이 읽습니다."
        ) from exc
    parsed = make_url(url)
    password = os.environ.get(DB_PASSWORD_ENV)
    if password:
        # Never written to logs, errors, or artifacts.
        parsed = parsed.set(password=password)
    return create_engine(parsed)


def load_frame(
    source: Any, *, table: str | None = None, query: str | None = None
) -> pd.DataFrame:
    """Read all rows of ``source`` into a DataFrame; the scripts' only reader.

    ``table``/``query`` only for databases; DB read errors become RuntimeError."""
    import pandas as pd

    text = str(source)
    if not is_database(text):
        if table or query:
            raise ValueError(
                f"--table/--query 는 DB 출처에만 씁니다 (받은 출처: {source_kind(text)}). "
                "sqlite 파일이라면 확장자를 .db/.sqlite/.sqlite3 로 두거나 접속 URL을 주십시오."
            )
        return pd.read_csv(text)

    statement = select_statement(table, query)
    assert_no_inline_password(text)

    if URL_SCHEME.match(text):
        engine = _engine(text)
        # Broad: driver error types are unknown here.
        try:
            with engine.connect() as connection:
                return pd.read_sql_query(statement, connection)
        except Exception as exc:
            raise RuntimeError(f"DB에서 행을 읽지 못했습니다: {type(exc).__name__}: {exc}") from exc

    # ``sqlite3.connect`` would silently create a missing file.
    path = Path(text)
    if not path.is_file():
        raise FileNotFoundError(f"sqlite 파일이 없습니다: {path}")
    with closing(sqlite3.connect(path)) as connection:
        return pd.read_sql_query(statement, connection)
