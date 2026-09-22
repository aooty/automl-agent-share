"""데이터가 어디서 들어오는가 — 파일 하나, 또는 SQL 하나.

행을 읽는 자리는 셋이고 전부 고정 스크립트다: ``scripts/profile.py``(카드를 만든다),
``scripts/train.py``(적합한다), ``scripts/predict.py``(배치를 채점한다). 셋 다 ``pd.read_csv``를
직접 불렀고, 그래서 출처를 하나 더 받으려면 세 군데를 고쳐야 했다. 여기가 그 하나의 자리다 —
세 스크립트는 이제 :func:`load_frame`만 부른다.

출처를 무엇으로 읽을지는 **문자열이 정한다**. 플래그를 더 두지 않는 이유는 사용자가 이미 아는
것을 다시 묻는 셈이기 때문이다.

* ``local/x.csv`` — CSV. 지금까지의 유일한 경로이고 기본이다
* ``local/x.db`` (``.sqlite``, ``.sqlite3``도) — sqlite 파일. ``sqlite3``가 stdlib라서 **새
  의존성이 없다**
* ``postgresql+psycopg://user@host:5432/db`` — SQLAlchemy가 읽는다. ``pip install -e ".[db]"``,
  드라이버는 따로

DB 출처에는 ``--table`` 또는 ``--query``가 **정확히 하나** 필요하다. 테이블 이름은 질의의
줄임이고(``SELECT * FROM "이름"``), 그 이상은 ``--query``가 한다.

비밀번호는 URL에 담지 못하게 **거부한다**. 출처 문자열은 ``run_config.json``에 그대로 적히고
``--resume``이 그것을 다시 읽으므로, URL에 담긴 비밀번호는 곧 디스크에 적힌 비밀번호다. 환경변수
``AUTOML_DB_PASSWORD``로 받으면 기록에는 사용자 이름까지만 남고 재개도 그대로 된다.

프롬프트 경계는 이 모듈이 바꾸지 않는다. 출처 문자열은 CSV 경로와 똑같이 비공개 ``data_ref``
채널로 가고 :func:`automl_agent.privacy.register_private`에 등록된다 — 접속 URL이 프롬프트에
나타나면 실행이 중단된다([docs/privacy.md](../../docs/privacy.md)).
"""

from __future__ import annotations

import os
import re
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - 타입 검사에서만
    import pandas as pd

# SQLAlchemy 꼴의 URL. 윈도 드라이브 문자(``C:\\x.csv``)가 걸리지 않는 것은 ``://``가 없기
# 때문이고, 그것이 이 정규식이 ``:``이 아니라 ``://``를 보는 이유다.
URL_SCHEME = re.compile(r"^[A-Za-z][A-Za-z0-9+.\-]*://")
SQLITE_SUFFIXES = frozenset({".db", ".sqlite", ".sqlite3"})
# ``--table``의 값은 SQL로 서식되므로 식별자 문법을 벗어나는 것은 받지 않는다. 자기 데이터베이스를
# 가리키는 사람이 주는 값이지만, 그렇다고 해서 따옴표를 닫고 나오는 이름이 안전해지지는 않는다 —
# 그런 이름이 진짜로 필요하면 ``--query``가 있다.
TABLE_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
DB_PASSWORD_ENV = "AUTOML_DB_PASSWORD"


def is_database(source: Any) -> bool:
    """``source``를 SQL로 읽어야 하는가. 문자열만 보고, 파일을 열지 않는다."""
    text = str(source)
    return bool(URL_SCHEME.match(text)) or Path(text).suffix.lower() in SQLITE_SUFFIXES


def as_source(text: Any) -> Path | str:
    """CLI 문자열을 출처로. 파일은 ``Path``, 접속 URL은 문자열 그대로.

    URL을 ``Path``에 넣으면 안 되는 이유는 윈도에서 조용히 망가지기 때문이다:
    ``Path("postgresql://u@h/db")``는 ``postgresql:\\u@h\\db``가 되고 — ``//``가 하나로 줄고
    구분자가 뒤집힌다 — 그 문자열로는 아무 데도 접속할 수 없다. 오류도 나지 않는다.
    """
    if text is None or text == "":
        return ""
    value = str(text)
    return value if URL_SCHEME.match(value) else Path(value)


def source_kind(source: Any) -> str:
    """카드의 ``description``에 적을 출처 종류 한 단어.

    카드의 설명은 공개 필드이고 모든 추론 프롬프트에 실린다. 그래서 출처 *문자열*이 아니라 종류만
    돌려준다 — 접속 URL은 여기로 나갈 수 없다.
    """
    text = str(source)
    scheme = URL_SCHEME.match(text)
    if scheme:
        return "sql"
    if Path(text).suffix.lower() in SQLITE_SUFFIXES:
        return "sqlite"
    return Path(text).suffix.lstrip(".").lower() or "csv"


def select_statement(table: str | None = None, query: str | None = None) -> str:
    """``--table``/``--query``를 읽을 SQL 하나로. 둘 다이거나 둘 다 아니면 거부한다."""
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
    """URL에 비밀번호가 담겨 있으면 거부한다.

    메시지에 URL을 싣지 않는다 — 그게 지금 막으려는 것이기 때문이다.
    """
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


def _engine(url: str) -> Any:
    """SQLAlchemy 엔진. 비밀번호는 환경변수에서 붙인다."""
    try:
        from sqlalchemy import create_engine
        from sqlalchemy.engine import make_url
    except ImportError as exc:  # pragma: no cover - 설치 여부에 달린 분기
        # ``ImportError``로 다시 내는 이유: 세 스크립트의 ``main``이 이미 그것을 잡아 한 줄로
        # 보고한다. 새 예외 종류를 만들면 그 세 자리에 traceback이 뜬다.
        raise ImportError(
            "접속 URL을 읽으려면 SQLAlchemy가 필요합니다: pip install -e \".[db]\". "
            "드라이버는 따로입니다 — postgres는 psycopg, mysql은 pymysql. "
            "sqlite라면 URL 대신 파일 경로(local/x.db)를 주면 의존성 없이 읽습니다."
        ) from exc
    parsed = make_url(url)
    password = os.environ.get(DB_PASSWORD_ENV)
    if password:
        # 값을 읽기만 하고 로그·예외·아티팩트 어디에도 적지 않는다.
        parsed = parsed.set(password=password)
    return create_engine(parsed)


def load_frame(
    source: Any, *, table: str | None = None, query: str | None = None
) -> pd.DataFrame:
    """``source``의 행 전부를 DataFrame으로. 행을 읽는 세 스크립트의 하나뿐인 입구다."""
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
        # ``except Exception``인 이유: 드라이버가 내는 예외 종류는 여기서 이름 부를 수 없다
        # (그러려고 sqlalchemy를 미리 import하면 의존성이 optional이 아니게 된다). 종류 이름은
        # 메시지에 남기므로 진단은 잃지 않는다.
        try:
            with engine.connect() as connection:
                return pd.read_sql_query(statement, connection)
        except Exception as exc:
            raise RuntimeError(f"DB에서 행을 읽지 못했습니다: {type(exc).__name__}: {exc}") from exc

    # ``sqlite3.connect``는 없는 파일을 **만든다**. 그대로 두면 오타 하나가 빈 데이터베이스와
    # "no such table"이 되고, 진단 가치가 전부 사라진다.
    path = Path(text)
    if not path.is_file():
        raise FileNotFoundError(f"sqlite 파일이 없습니다: {path}")
    with closing(sqlite3.connect(path)) as connection:
        return pd.read_sql_query(statement, connection)
