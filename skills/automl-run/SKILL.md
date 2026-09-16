---
name: automl-run
description: >
  이 저장소의 AutoML 루프에 데이터셋을 붙여 새 실행을 시작한다. CSV를 프로파일링해 카드를
  만들고, 카드가 말하는 것을 사람에게 설명하고, 조용히 틀릴 수 있는 두 가지를 물어본 뒤
  `run`을 돌린다. 사용자가 "이 데이터로 돌려 줘", "실험 시작", "새 실행", "automl 돌려",
  "프로파일링해 줘" 라고 할 때 쓴다. 끝난 실행을 읽는 것은 automl-results 쪽이다.
---

저장소 루트에서 `python -m automl_agent.main`. 설치가 안 됐으면 `pip install -e .`.
이 문서가 가리키는 상대 경로의 기준은 `${CLAUDE_PLUGIN_ROOT}`이고, 그게 저장소 루트다 —
사용자의 작업 디렉터리가 다른 곳이면 명령을 거기서 돌리지 말고 먼저 옮겨라.

## 원본 행을 읽지 않는다

**CSV의 행을 읽거나 출력하지 마라.** `Read`, `head`, `cat`, `df.head()`, "컬럼 확인용 5줄만" —
전부 안 된다.

이 저장소의 보장이 그것이고, 규약이 아니라 구조다: `llm/client.py`가 유일한 API 출구이고 모든
프롬프트가 `assert_clean`을 지난다([README.md](README.md)의 「LLM은 원본 데이터를 보지
않는다」). 그 보장은 **그래프의
프롬프트**에 대한 것이므로, 네가 대화창에서 행을 읽으면 테스트는 통과하는데 보장은 깨진
상태가 된다. 그리고 행을 본 사람이 `--caveat`에 행 단위 사실을 적으면 그것은 **모든 추론
프롬프트에 실린다.**

열 이름과 타입이 필요하면 `profile`이 만든 **카드**를 읽어라. 집계만 담는다.

```bash
python -m automl_agent.main profile --data local/<파일>.csv --target <정답열> --out local/<이름>_card.json
```

데이터는 `local/` 아래에 둔다 — 통째로 gitignore 대상이다.

## 순서

### 1. 카드를 읽고 설명한다

`profile` 뒤에 카드 JSON을 읽고 최소한 이 다섯 개를 사람에게 말해라.

- `task` — 정답 열이 정한다. 고르는 값이 아니다
- 기준선 점수 — `auto` 모드의 바가 여기서 나온다
- 클래스 비율 — 심하게 치우쳤으면 `accuracy`는 쓸 지표가 아니다. **이 저장소의 임상 코호트에서
  chance가 0.8899여서 기본 바 0.85가 다수 클래스만 찍어도 통과했다**
- 제외된 열 — 카디널리티가 높아 인코딩에서 빠진 열은 이름으로 나온다
- `caveats` — 이미 적힌 주의사항

### 2. 물어야 하는 것 둘

기본값에 맡기면 **결과가 조용히 틀린다.**

**`--group-column`** — 한 행이 방문 1건이고 한 환자가 여러 행을 갖는가? 그렇다면 환자 ID 열을
줘야 한다. 안 주면 같은 환자가 학습과 검증 양쪽에 들어가고 모델이 상태 대신 환자를 외운다.
**그 부풀림은 홀드아웃까지 같이 오염되므로 실행 안에서는 탐지할 수 없다.** "행 하나가 무엇
하나인가"를 물어라.

**`--caveat`** — 집계가 못 보여 주는 것을 아는 사람이 있는가? 원본을 본 사람의 지식이 들어오는
유일한 통로다. 여러 번 줄 수 있다.

정답 열이 빈 행이 있으면 `--on-missing-target`도 정해야 한다 — `reject`(기본) 또는 `drop`.

### 3. 모델 조합은 고정이다

**`ollama:gemma4:12b`가 제안자, Claude가 심판.** 이 플러그인은 이 조합만 쓴다. 다른 제안자
모델이나 `--no-llm`으로 바꾸지 마라 — 사용자가 명시적으로 요구할 때만이다.

```
--model claude-opus-5 --proposer-model ollama:gemma4:12b
```

**왜 이 절반인가**: planning·model_selection의 출력만 코드 검증(`validate_plan`·레지스트리·
클램프)을 통과하므로, 약한 모델을 놓을 수 있는 유일한 절반이다. critic·report는 그 검증이
없어서 Claude가 맡는다. `gemma4:12b`는 이 저장소에서 실측됐다 — 12,540 토큰 프롬프트까지
fallback 0회, 판정은 Claude와 구분되지 않는다 — 그 probe는 개발 저장소에 있다.

**돌리기 전에 둘 다 살아 있는지 확인해라.** 제안자는 `check_credentials`가 검사하지 않으므로
ollama가 죽어 있으면 planning 노드에서 fallback으로 조용히 새는 실행이 된다.

```bash
ollama list                     # gemma4:12b 가 있나. 없으면 ollama pull gemma4:12b
```

`ANTHROPIC_API_KEY`(또는 `AUTOML_USE_BEDROCK`+`AWS_REGION`)는 **존재만** 확인하고 값을 출력하지
마라. 없으면 거기서 멈추고 사용자에게 말해라 — critic 없이 도는 실행은 이 플러그인이 약속한
것이 아니다.

`--dry-run`은 루프 구조 확인 전용이고 **점수가 가짜다.** 결과로 인용하면 안 된다.

### 4. 돌린다

```bash
python -m automl_agent.main run --dataset-card local/<이름>_card.json --thread-id <아이디> \
  --metric <지표> --max-iterations 5 \
  --model claude-opus-5 --proposer-model ollama:gemma4:12b
```

- `--thread-id`는 필수. 안 정했으면 `<데이터셋>-<날짜>-<번호>`를 제안해라
- 목표는 `auto`가 기본(카드 기준선에서 유도, `--margin`으로 조절). 밖에서 온 숫자를 못박을 때만
  `--threshold` — 그러면 자동으로 fixed 모드다. 둘을 함께 주면 거부된다
- **`run_in_background: true`로 띄우고** thread_id를 알려 준 뒤 진행은 로그로 봐라. 실행은
  분에서 시간 단위다. 붙잡고 기다리지 마라
- 불균형 타깃이면 `--search-past-goal` 없이도 계획이 `tune_threshold`를 쓸 수 있다. 그게 이
  프로젝트에서 측정된 가장 큰 레버다(test `balanced_accuracy` +0.0762 — 측정 문서는 개발
  저장소에 있고, URL은 [README.md](README.md)의 「더 자세한 것」)

### 5. 끝나면

automl-results 쪽으로 넘어가라. **보고할 숫자는 val 최고가 아니라 holdout이다.**

## 거부되면

잘못된 조합은 `main.py`가 한글로 왜 안 되는지 말한다. **그 메시지를 그대로 보여 주고 추측으로
플래그를 바꿔 다시 시도하지 마라** — 거부는 대부분 "이 실행은 측정으로서 성립하지 않는다"는 뜻이다.

전체 명령은 [RUNBOOK.md](RUNBOOK.md), 플래그의 근거는 [README.md](README.md)의 「이 루프를
믿을 수 있는 근거 네 가지」.
