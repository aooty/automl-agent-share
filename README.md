# AutoML Agent — LangGraph 자율 실험 루프

원본 CSV에서 **집계 요약(데이터셋 카드)** 을 만들고, 그 카드만 보고 **계획 → 모델
선택 → 학습 → 평가**를 돌리고, 목표 지표에 미달하면 **실패를 진단해 다시 계획**하는
순환 그래프입니다. 목표를 달성하거나 예산이 끝나면 보고서를 씁니다.

**LLM은 원본 데이터를 한 행도 보지 않습니다.** 프롬프트 문구가 아니라 그래프
구조로 그렇게 됩니다.

```
      raw data (CSV)
            │
            ▼
      [profiling] ──────────────► data_ref  (비공개: 경로 + target 컬럼)
       subprocess                     │        실행 노드만 읽음
            │                         │
            ├──► goal  (목표 임계값 — auto 모드는 기준선에서 도출)
            │                         │
            └──► dataset_card         │       ← 데이터에서 추론으로 넘어가는
                    │ (집계만)         │         유일한 통로
                    ▼                 ▼      
            [planning] ──► [model_selection] ──► [training] ──► [evaluate]
                 ▲             LLM                subprocess         │
                 │                                                   │
                 │                                            route()│  ← 종료 판단은 코드만
                 │                                                   │
              [critic] ◄──────────── critic ──────────────────────────┤
                LLM                                                   │ report
                                                                      ▼
                                                                 [holdout]  ← 반복이 한 번도
                                                                subprocess     보지 않은 test
                                                                      │        20%를 1회 채점
                                                                      ▼
                                                                  [report] ──► END
```

복사해서 쓰는 명령 모음은 [RUNBOOK.md](RUNBOOK.md)에 있습니다.

## 설치

Python 3.11+.

```bash
python -m pip install -e .
python -m pip install -e ".[xgboost]"   # 선택: xgboost 모델까지
python -m pip install -e ".[bedrock]"   # 선택: Bedrock 경유 LLM 호출
```

### Claude Code 플러그인으로 쓰기

이 저장소 자체가 Claude Code 플러그인입니다([.claude-plugin/plugin.json](.claude-plugin/plugin.json)).
설치하면 스킬 두 개가 붙습니다 — [skills/automl-run](skills/automl-run/SKILL.md)이 카드를 만들고
실행을 띄우고, [skills/automl-results](skills/automl-results/SKILL.md)가 끝난 실행을 읽습니다.

```
/plugin marketplace add <이 저장소 경로 또는 URL>
/plugin install automl-agent@automl-agent
```

**모델 조합은 플러그인이 고정합니다** — planning·model_selection은 로컬
`ollama:gemma4:12b`, critic·report는 `claude-opus-5`입니다. 임의로 정한 조합이 아니라
어느 절반에 코드 검증이 걸려 있는지가 정합니다: 제안자의 출력만 `validate_plan`·레지스트리
화이트리스트·클램프를 통과하므로 **약한 모델을 놓을 수 있는 자리가 거기뿐**이고, critic과
report에는 그 관문이 없습니다.

그래서 이 조합은 `ANTHROPIC_API_KEY`(또는 Bedrock)와 **로컬 ollama 둘 다** 필요합니다.
아래 [자격 증명](#자격-증명)의 사전 점검이 검사하는 것은 앞쪽뿐이므로 — ollama가 죽어 있으면
실행은 멈추지 않고 계획이 규칙 폴백으로 조용히 넘어갑니다. 스킬이 돌리기 전에 `ollama list`를
확인하고 끝난 실행에서 `plan_source`를 가장 먼저 보게 하는 이유입니다.

스킬은 **원본 CSV 행을 읽지 않습니다.** 그 보장은 그래프의 프롬프트에 대한 것이라 대화창에서
행을 읽으면 아래 [1번 근거](#1-llm은-원본-데이터를-보지-않는다--채널과-프로세스로)만 조용히
깨집니다.

## 데이터

**데이터는 포함되어 있지 않고, 자기 CSV가 필요합니다.** 정답 열이 하나 있는
표(tabular) 하나면 됩니다 — 라벨 열이면 분류, 연속 열이면 회귀로 자동 판정합니다.

원본 CSV와 거기서 만든 카드는 **`local/`에 두세요.** `.gitignore`가 이 디렉터리를
통째로 막으므로 "이 파일 커밋해도 되나"가 파일마다의 질문이 아니라 디렉터리 하나에
대한 질문이 됩니다.

데이터 없이 루프 구조만 보려면 [`examples/`](examples/)의 손으로 쓴 카드 네 장으로
`--dry-run`을 돌리면 됩니다(아래 [예시 데이터셋 카드](#예시-데이터셋-카드)).

## 자격 증명

API 키는 **환경변수로만** 읽고 코드·로그·아티팩트에 남기지 않습니다.

| 경로 | 설정 | 추가 의존성 |
| --- | --- | --- |
| Anthropic API 직접 | `ANTHROPIC_API_KEY` | 없음 |
| Bedrock 경유 | `AUTOML_USE_BEDROCK=1`, `AWS_REGION` (+ 표준 AWS 자격 증명 체인) | `.[bedrock]` |
| 제안자만 로컬로 | 위 둘 중 하나 + `--proposer-model ollama:<모델>` (서버 주소는 `OLLAMA_HOST`, 기본 `http://localhost:11434`) | 없음 — HTTP로 나갑니다 |
| 자격 증명 없이 | `--dry-run` 또는 `--no-llm` | 없음 |

자격 증명이나 botocore가 없는 상태로 LLM 모드를 실행하면 **시작 전에** 안내와 함께
종료합니다. Bedrock에서는 구조화 출력이 강제 tool use로 자동 강등되고, 어느 쪽이
쓰였는지는 아티팩트의 `structured_mode`에 남습니다.

**사전 점검이 검사하는 것은 Anthropic 쪽뿐입니다** — 로컬 서버가 살아 있는지는 확인하지
않습니다. 그래서 제안자를 로컬로 돌리는데 그 서버가 죽어 있으면 실행은 멈추지 않고 계획만
규칙 폴백으로 넘어갑니다(호출이 실패하면 `fallback_plan`이 받습니다). 학습·채점·홀드아웃은
그대로 유효하므로 실행이 망가진 것은 아니지만, **그 실행이 잰 것은 모델이 아니라 규칙입니다.**
매 반복이 `history.json`에 `plan_source`를 남기므로 끝난 뒤 그것으로 확인하세요 —
`llm`/`fallback`/`rules` 중 하나이고, `fallback`이 호출은 나갔는데 답이 안 쓰인 반복입니다.

## 명령

```bash
# 카드 만들기 — 사람이 읽고 고칠 수 있는 집계 요약. 두 모드의 목표값도 미리 보여줍니다
python -m automl_agent.main profile --data local/data.csv --target died \
  --metric roc_auc --out local/my_card.json

# 실행 (카드로 시작)
python -m automl_agent.main run --dataset-card local/my_card.json \
  --metric roc_auc --max-iterations 5 --thread-id my-run-001

# 실행 (원본 CSV로 시작 — profiling 노드가 그래프 안에서 카드를 만듭니다)
python -m automl_agent.main run --data local/data.csv --target died \
  --metric roc_auc --thread-id my-run-002

# 중단된 실행 재개 (다른 플래그 불필요 — run_config.json에서 복원)
python -m automl_agent.main resume --thread-id my-run-001

# 저장된 상태 확인 (--report 로 보고서 전문까지)
python -m automl_agent.main show --thread-id my-run-001 --report

# 고른 모델을 새 CSV에 적용 (--label-column 을 주면 채점까지)
python -m automl_agent.main predict --thread-id my-run-001 \
  --data local/new_rows.csv --id-column patient_id --out local/pred.csv

# 그래프 구조 저장
python -m automl_agent.main graph --out graph.png
```

반복마다 한 줄 요약을 출력합니다:

```
[iter 2] model=hist_gbdt f1=0.8100 (goal 0.85) → critic: hyperparam
```

### 주요 플래그

| 플래그 | 의미 |
| --- | --- |
| `--dataset-card` / `--data` | 둘 중 하나 필수. 카드로 시작하거나 원본 CSV로 시작합니다 |
| `--target` | 정답 컬럼. `--data`를 쓸 때 필수 |
| `--metric` | 분류 `f1` `accuracy` `balanced_accuracy` `precision` `recall` `roc_auc` `pr_auc` / 회귀 `r2` `mae` `rmse`. 목록은 [metrics.py](automl_agent/scoring/metrics.py) 한 곳에서 나옵니다 |
| `--goal-mode {auto,fixed}` | 기본 `auto`(기준선에서 도출). `--threshold`를 주면 `fixed`로 간주 |
| `--threshold` / `--margin` | 각각 `fixed`의 목표값 / `auto`가 요구할 남은 여유의 비율(기본 `0.25`). 모드와 어긋나면 거부 |
| `--max-iterations` / `--seed` | 반복 상한 / 시드 |
| `--time-budget-sec` | **실행 하나가 일하는 초의 상한**(기본 `3600`). 학습 하나의 timeout이 아니라 모든 노드의 소요를 누적한 값입니다 — 10%는 holdout 몫으로 예약되고, 적합 하나의 몫은 `남은 시간 ÷ 남은 반복 수`입니다. 몫을 넘긴 적합은 `too_slow`, 예산을 다 쓰고 끊긴 루프는 `out_of_time` |
| `--search-past-goal` | 목표를 넘어도 멈추지 않고 예산을 다 씁니다. `auto`의 바는 기준선에서 도출되므로 첫 시도가 넘는 일이 흔하고, 그러면 Critic이 한 번도 안 돕니다 — 진단·재계획 경로를 실제로 돌려 보려면 이 플래그입니다. 바를 올리지도, 우승자 규칙(val 최고)을 바꾸지도 않습니다 |
| `--no-llm` | 학습은 **실제로** 하고 계획하는 쪽만 규칙 기반. 자격 증명 불필요. **자격 증명이 없을 때의 차선이 아니라 대조군입니다** — 학습·분할·지표·홀드아웃 규칙이 전부 같고 계획만 결정론적이므로 같은 카드·같은 시드면 같은 계획이 나옵니다. LLM이 무엇을 더 벌었는지 말하려면 이 팔의 점수가 바입니다 |
| `--dry-run` / `--scenario` | LLM과 학습을 **모두** 모킹. `--scenario {success,fail,oom,stall,slow,crash}` |
| `--force` | 같은 `--thread-id`의 기존 체크포인트를 지우고 처음부터 |
| `--on-missing-target {reject,drop}` | 정답이 빈 행의 처리. 기본은 개수를 알리고 중단 |
| `--keep-models {best,all}` | 실행이 끝날 때 진 시도의 `model.joblib`을 지울지 |
| `--group-column` | **한 값에 속한 행들이 train/val/test로 흩어지면 안 되는 컬럼**(예: 환자 ID). 한 행이 방문 1건이고 한 사람이 여러 행을 가지면 이걸 줘야 합니다 — 안 주면 같은 사람이 학습과 검증 양쪽에 들어가 모델이 상태 대신 그 사람을 외웁니다. **그 부풀림은 홀드아웃까지 같이 오염되므로 실행 안에서 탐지할 수 없습니다.** 주면 그 컬럼은 특성에서 빠지고 모든 점수가 처음 보는 그룹에 대한 성능이 됩니다. 카드에 기록돼 있으면 `run`에서 생략해도 따릅니다 |
| `--caveat` | 집계가 보여주지 못하는 것을 아는 사람의 지식이 들어오는 **유일한 통로**. 여러 번 줄 수 있고 모든 추론 프롬프트에 실립니다 — 그래서 **행 단위 사실을 적으면 안 됩니다** |
| `--model` | 심판(critic·report)이 쓸 Claude 모델 ID. 기본 `claude-opus-5` |
| `--proposer-model` | planning·model_selection에만 쓸 모델. 생략하면 `--model`과 같습니다. `ollama:<모델>`을 주면 로컬 Ollama로 나갑니다(예: `ollama:gemma4:12b`). **이 둘만 코드 검증(`validate_plan`·레지스트리 화이트리스트·클램프)을 통과하므로 약한 모델을 놓을 수 있는 절반입니다** — critic·report에는 그 관문이 없습니다. 제안자는 `check_credentials`가 검사하지 않으므로 로컬 서버가 죽어 있으면 실행은 계속 돌고 계획만 규칙 폴백이 됩니다. 끝난 뒤 `history.json`의 `plan_source`로 확인하세요 |
| `--direction {maximize,minimize}` | **지표가 이미 정하므로 생략이 기본입니다** — `mae`는 `minimize`, `f1`은 `maximize`. 함의와 다르게 주면 선호가 아니라 실수이므로 거부합니다 |
| `--artifacts-root` | 아티팩트 루트 재지정. 두 실행을 나란히 돌릴 때 |

`--force` 없이 이미 쓴 `thread_id`로 `run`을 다시 호출하면 **거부합니다** — 두 실행의
`history`가 `operator.add`로 조용히 이어붙기 때문입니다. 모순되는 플래그 조합
(`auto` + `--threshold` 등)과 실행을 무의미하게 만드는 값(`--max-iterations -1`,
음수 margin, NaN threshold)도 호출 시점에 끊습니다. 조용히 무시된 플래그가
최악이라서입니다.

## 이 루프를 믿을 수 있는 근거 네 가지

### 1. LLM은 원본 데이터를 보지 않는다 — 채널과 프로세스로

원본 행을 여는 코드는 고정 스크립트 세 개뿐이고, 그래프 안의 둘은 **subprocess**로
돕니다. 따라서 프롬프트를 렌더링하는 프로세스는 pandas를 import조차 하지 않습니다.

| | 원본 행을 읽는가 | 무엇을 내보내는가 |
| --- | --- | --- |
| [scripts/profile.py](automl_agent/scripts/profile.py) | 읽음 (subprocess) | 카드 = 열 단위 집계 |
| [scripts/train.py](automl_agent/scripts/train.py) | 읽음 (subprocess) | metrics + status (`model.joblib`은 디스크에만) |
| [scripts/predict.py](automl_agent/scripts/predict.py) | 읽음 (그래프 밖) | 행 단위 예측 CSV — 사용자에게만 |
| 그 외 **모든** 노드 / `llm/client.py` | 읽지 않음 | — |

State도 둘로 쪼갰습니다. `data_ref`는 **비공개**(`{path, target_column}`,
`profiling`·`training`만 읽음), `dataset_card`는 **공개**(집계뿐, 추론 노드 전부가
읽음). 카드가 데이터에서 추론으로 넘어가는 유일한 통로이고, 그 통로는 노드
경계입니다.

카드가 내는 것은 개수·비율·**버킷**뿐입니다 — min·max·평균·분위수·셀 값·예시 행·
범주 level 이름·클래스 라벨 이름은 내지 않습니다. min/max 한 쌍은 실재하는 두
사람의 값이고, 플래너가 알아야 하는 것은 "이 열은 heavy-tailed이고 수백 단위"라는
사실이지 "3번 환자의 크레아티닌이 4.1"이 아닙니다. 이 정책은
[privacy.py](automl_agent/privacy.py)의 **allowlist로 강제**합니다 — 모르는 최상위 키가
하나라도 있으면 실행을 중단합니다.

**적합된 모델 파일도 데이터와 같은 급입니다.** SVC는 support vector를, 트리는 실제
값에서 읽은 분기점을 담습니다. 그래서 `model.joblib`·`feature_schema.json`·예측
CSV는 `.gitignore`가 막는 디렉터리 안에만 있고, 그 경로들은 state로 올라가는
`PUBLIC_RESULT_FIELDS` 밖에 **일부러** 뒀습니다.

백스톱은 유일한 API 호출 지점에 있습니다. 모든 프롬프트가
[render_prompt()](automl_agent/llm/client.py) → `privacy.assert_clean()` 관문을 통과하고,
**등록된** 데이터 경로가 프롬프트에 있으면 `RawDataLeak`으로 실행을 중단합니다
(유출은 폴백으로 감출 버그가 아닙니다). 경로처럼 *보이는* 문자열은 조용히
마스킹합니다 — false positive로 실행을 죽이는 편이 더 나쁩니다. 대가는 산문
쪽입니다: `median/mean`은 슬래시 때문에 마스킹되므로 **프롬프트 산문에서 슬래시를
피합니다.**

자기 데이터로 확인하려면:

```powershell
$files = Get-ChildItem artifacts\<id>\llm\* -File
"검사 대상 $($files.Count)개"                # 0개면 검증이 아니라 경로가 틀린 것
$files | Select-String -Pattern "<파일명>"    # 출력 없음이 정상
```

확장자를 걸지 마세요. 실제 실행은 `.json`, `--dry-run`은 `.dryrun.md`로 남으므로
`*.md`로 좁히면 실제 실행에서는 **0개를 검사하고** "결과 없음"이 나옵니다.

### 2. 목표 임계값은 코드가 정한다 — LLM에게 묻지 않는다

모델이 자기 합격선을 정할 수 있으면 기준을 낮춰 성공을 선언하는 것이 가장 쉬운
전략이 됩니다. 두 모드 모두 결정론적 코드입니다([goal.py](automl_agent/scoring/goal.py)).

| 모드 | 임계값 | 언제 |
| --- | --- | --- |
| **`auto`** (기본) | `기준선 + (1 - 기준선) × margin` | "평범한 기본값을 유의미하게 이겼나"를 재고 싶을 때 |
| **`fixed`** | `--threshold`, 없으면 지표별 기본값 | 숫자가 데이터 밖에서 올 때 — 배포 요구사항, 규제 하한, 이겨야 할 논문 점수 |

`auto`가 두 개 있는 이유: 고정된 `f1 >= 0.85`는 옮겨 쓸 수 없는 숫자입니다. 양성
11% 코호트에서는 도달 불가이고 잘 분리된 카드에서는 공짜라서, 종료 조건이 "이 모델이
좋다"가 아니라 **"이 데이터셋이 쉽다"** 를 재게 됩니다. margin을 점수가 아니라 **남은
여유**에 걸면 임계값의 *의미*가 데이터셋을 건너 같아집니다.

- 기준선은 `LogisticRegression` + median 대치 + 표준화를 `profile.py`가 **같은
  분할·같은 시드·같은 인코딩**으로 한 번 측정한 값이라 모든 시도 점수와 직접
  비교됩니다. 화려하지 않은 모델을 일부러 골랐습니다.
- `chance`(다수 클래스 예측기)보다 낮은 바는 `chance + 0.02`까지 **끌어올립니다.**
  상한 `0.99`도 있는데, **상한을 먼저 걸고 chance를 나중에 봅니다** — 순서가 반대일 때
  다수 클래스 99.5% 코호트에서 목표가 상수 예측기가 이미 넘는 값으로 되돌아갔습니다.
- **`balanced_accuracy`에서는 기준선 랭킹 자신의 최적 컷까지도 끌어올립니다.** 기준선은
  0.5 컷에서 측정되므로(실행기는 항상 `predict()`를 부릅니다) 그 컷이 보는 손해만큼 바가
  덜 나오고, 그러면 **같은 logreg를 다시 자른 것이 이미 넘는 점수**가 목표가 됩니다. 그건
  목표가 아니라서 KS로 계산한 그 값까지 올립니다. 이 지표만 그렇습니다 — 다른 지표는
  margin이 붙는 폭이 컷 손해보다 크고, `precision`·`recall`은 아무것도/전부 예측해서
  최적 컷이 1.0이라 바닥이 무의미합니다. **바가 올라가면 그 밑의 `--margin`은 아무것도
  바꾸지 않으므로**, 목표 줄이 margin이 낸 바와 같은 바를 내는 `--margin` 상한을 함께
  출력합니다. 조용히 무시된 플래그를 남기지 않기 위해서입니다.
- 보정은 목표를 **더 어렵게만** 만듭니다. 바가 도달 불가가 되면 조용히 낮추지 않고
  `(도달 불가)`로 표시하며 `--metric balanced_accuracy` 같은 대안을 말합니다.
- `mae`·`rmse`에는 `fixed` **기본값이 없습니다.** 정답 열의 단위로 나오는 지표에
  이식 가능한 바는 존재하지 않고, `0.85`를 넣으면 모델이 아니라 그 열의 단위에 대한
  바가 됩니다. `--threshold`나 측정된 기준선을 요구합니다.
- 기준선 없는 카드에서 `auto`는 지표별 기본값으로 폴백하고 **경고합니다** — 재지
  않은 숫자를 잰 것처럼 보이게 하지 않습니다.

같은 카드에서 모드가 내는 값(8,000행, 양성 12.5% 예시 코호트):

| 지표 | chance | 기준선(logreg) | `auto` 목표 | `fixed` 기본값 |
| --- | --- | --- | --- | --- |
| `roc_auc` | 0.5 | 0.7444 | **0.8083** | 0.90 (이 데이터에서 도달 불가) |
| `f1` | 0.0 | 0.3108 | **0.4831** | 0.85 (12.5% 유병률에서 불가능) |
| `accuracy` | 0.8744 | 0.8919 | **0.9189** | 0.85 (다수 클래스가 이미 넘음) |
| `pr_auc` | 0.1256 | 0.4851 | **0.6138** | 0.70 |

`profile` 명령이 두 모드의 값을 나란히 출력하므로 실행을 쓰지 않고 고를 수
있습니다. `--margin`도 받습니다 — margin이 남은 여유에 걸리기 때문에 같은 증분이
카드마다 다른 폭으로 움직이고, 머릿속 환산이 안 됩니다.

### 3. 보고하는 점수는 고르지 않은 점수다 — 3분할과 반복 밖 1회 채점

```
전체 행 ─┬─ test  20%   ← 가장 먼저 떼어 둡니다. 루프가 끝날 때까지 아무도 읽지 않습니다
         └─ 나머지 80% ─┬─ train 60%   기준선과 모든 시도가 적합하는 행
                        └─ val   20%   기준선 점수·매 시도 점수·`best` 선택이 나오는 행
```

5회 반복이면 `best`는 **노이즈 섞인 검증 점수 5개의 최댓값**입니다. 계획도
하이퍼파라미터도 `best` 선택도 전부 그 숫자를 보고 정했는데, 같은 숫자를 실행의
결과로 인용하면 재지 않은 폭만큼 과장이고 반복이 늘수록 커집니다.

그래서 [holdout.py](automl_agent/nodes/holdout.py)가 `route()` **다음**에서 그 폭을
잽니다 — 일부러 종료 판단 뒤입니다. test 점수가 루프의 행동을 바꿀 수 있으면 그건
더 이상 떼어 둔 숫자가 아닙니다. 재적합하지 않고 **최고 iteration의 `model.joblib`과
그 iteration의 config를** 그대로 써서 `train.py --score-model`을 부릅니다.

| 실행 | 반복 | 검증 `best` | test | `selection_gap` |
| --- | --- | --- | --- | --- |
| `auto`(→ 0.8083) | 1회에 달성 | 0.8545 | 0.8427 | **+0.0119** |
| `fixed` 0.90 | 3회 소진 | 0.8648 | 0.8412 | **+0.0236** |

같은 데이터·같은 모델 계열인데 시도를 3개 놓고 최댓값을 고르자 격차가 두 배가
됐습니다. **최종 test가 검증 `best`보다 낮은 것이 정상이고, 그게 이 숫자의
용도입니다.** 보고할 숫자는 test 쪽입니다.

분할은 [splits.py](automl_agent/scoring/splits.py) 한 곳에서 나오고 `profile.py`와
`train.py`가 같은 함수를 부릅니다. 세 분할 모두 stratified이고 `--seed`만 받으므로
**test 행은 (파일, 시드)의 함수**입니다 — 인덱스를 저장하지 않고 매번 다시 계산합니다.
같은 이유로 카드의 `baseline.protocol`이 실행 설정과 다르면 `ProfilingFailed`로
멈춥니다: 기준선을 잰 데이터와 학습이 점수를 내는 데이터가 같아야 합니다.

채점할 모델이 없으면(`--dry-run`, 모든 시도 실패, 채점 자체 실패)
`{"status": "skipped", "reason": ...}`를 남기고 보고서에 사유를 적습니다. 최종 측정이
없다고 보고서를 안 쓰면 실행이 실제로 모은 근거를 버리게 됩니다.

### 4. 차이를 개선이라 부르기 전에, 노이즈보다 큰지 함께 낸다

모든 점수는 한 슬라이스에 대한 점 추정이고 모든 판단은 그런 점수 두 개의
비교입니다. 그래서 채점되는 분할마다
[intervals.py](automl_agent/scoring/intervals.py)가 95% 부트스트랩 구간을 같이 내고
(`<metric>_ci_low` / `_ci_high`), 그 구간 안에 들어오는 차이는 Critic·보고서·`holdout`이
**"이 행들로는 구분되지 않는다"** 고 이름 붙입니다. 판정을 뒤집지는 않습니다 —
밝히기만 합니다.

- 두 시도를 비교할 때는 **짝지은** 구간입니다. 같은 val 행에 채점되므로 구간 폭의
  대부분인 행 표집 잡음이 차이에서 상쇄됩니다.
- 재표집 단위가 20개 미만이면 구간을 지어내지 않고 없다고 적습니다
  (`no confidence interval (split too small, or metric degenerate)`).
- 확률 품질은 [calibration.py](automl_agent/scoring/calibration.py)가 따로 잽니다
  (`brier`, 구간별 오차). **진단이고 목표로 삼을 수 없습니다** — 모델을 바꾸지 않습니다.
- 랭킹 상한은 [ranking.py](automl_agent/scoring/ranking.py)가 KS로 잽니다.
  `balanced_accuracy_cut_headroom`은 **더 나은 컷이 이 행들에서 아직 값하는 정확한
  양**이라, 남은 격차가 운영점에 있는지 랭킹에 있는지 구분해 줍니다. **도달한 점수로
  인용하면 안 됩니다** — 그 시도가 쓰지 않은 컷에서 잰 값입니다.
  - **이름의 `balanced_accuracy`는 실수가 아닙니다.** 목표 지표가 `f1`이든 `roc_auc`든
    이 값은 항상 `balanced_accuracy`로 잽니다. 그래서 그 목표까지 남은 거리와 나란히
    놓고 비율로 읽으면 단위가 어긋나고, 이름이 그 사실을 들고 있게 하려고 붙인
    접두사입니다 — `f1` 실행 두 번이 이 숫자를 `f1`의 여유로 읽고 컷을 요청한 뒤에
    개명했습니다. 그 이전에 나온 결과와 로그에는 `cut_headroom`으로 남아 있습니다.
  - 컷을 **실제로 옮긴** 시도에서도 이 값은 남습니다. 컷은 따로 뗀 행에서 골랐고 이
    값은 val에서 재므로, 남은 것은 그 선택이 옮겨오지 못한 나머지입니다.
    적용된 컷의 위치는 `applied_threshold`이고, **없으면 고정 0.5**라는 뜻입니다.
    계획이 컷을 요청했는지와 거부됐다면 그 이유는 `internal_validation`의
    `cut_requested`·`cut_declined`에 있습니다.

## 루프가 끝나는 네 가지 경우 — `route()`

1. 목표 지표 달성 (`--search-past-goal`이면 이 조건만 해제됩니다)
2. `iteration >= max_iterations`
3. 정체: `stall_count >= 2` (개선 없는 반복 연속 2회)
4. 시간 예산 소진: `--time-budget-sec`에서 holdout 몫을 뺀 나머지를 다 씀

2·3번이 함께 무한 루프를 불가능하게 만듭니다 — `--search-past-goal`도 예산을 **더 쓸
수는 있어도 넘길 수는 없습니다.** 목표를 못 채워도 `best` 스냅샷을 근거로 보고서는
**반드시** 작성됩니다.

**4번은 맨 마지막에 검사합니다.** 순서가 종료 사유의 *이름*을 정하기 때문입니다. 앞의
셋은 시계가 멈춰 있었어도 이 실행을 끝냈을 조건들이고, `out_of_time`은 **그 밖에도 계속
갈 수 있었던 루프를 시계가 끊은 경우**만 가리킵니다. 쓴 예산은 콘솔과 `history.json`의
`budget` 블록에 남으므로, 반복 2/5에서 멈춘 실행이 시계 때문인지 정체 때문인지 사후에
구분할 수 있습니다.

1번이 먼저 검사되므로 **첫 시도가 바를 넘으면 `critic`은 한 번도 실행되지
않습니다.** 진단·재계획 경로가 돌지 않은 실행이고, 보고서와 `history.json`의
`critic_runs`가 그 사실을 명시합니다 — 루프가 실제로 루프했는지를 점수와 따로 읽을 수
있어야 하기 때문입니다.

Critic의 출력은 자유 텍스트가 아니라 `{failure_type, evidence, direction,
concrete_changes}` 스키마이고, `failure_type`은 8종으로 제한됩니다 —
`underfitting` · `overfitting` · `data_issue` · `hyperparam` · `oom` · `too_slow` ·
`wrong_model_family` · `unknown`. 목록 밖의 값은 `unknown`으로 강등되어 존재하지 않는
원인이 Planner 프롬프트를 오염시키지 않습니다.

**모델의 실패와 환경의 실패는 다르게 다룹니다.** 디스크가 차거나 체크포인트
데이터베이스가 잠긴 것은 제안을 바꿔서 나아지지 않습니다. `train_config.json` 쓰기
실패는 그 시도만 `write_failed`로 기록하고 루프는 계속되며(예전에는 이 `OSError`가
이미 끝난 iteration 전부를 잃게 했습니다), sqlite는 busy timeout 60초 + WAL로 열고
그래도 잠기면 `database is locked` 대신 원인과 대응이 담긴 한글 메시지를 냅니다.

## 파일 구조

```
automl_agent/          최상위 6개는 오케스트레이션 척추 — 그래프와 그 경계
  main.py              CLI: profile / run / resume / show / predict / graph
  graph.py             route() + 그래프 배선 + SqliteSaver
  state.py             AutoMLState / Attempt / 결정론적 헬퍼
  config.py            RunConfig(불변) + 상수 + subprocess 공용 헬퍼
  capabilities.py      실행기가 하는 일/안 하는 일 — 프롬프트와 계획 점검이 함께 읽는 목록
  privacy.py           원본 데이터 경계: validate_card / public_card / public_result / 프롬프트 관문
  scoring/             무엇을 어떻게 재는가 (결정론, LLM 미사용, state를 모름)
    metrics.py         지표 레지스트리 — 목표·기준선·학습기가 읽는 유일한 목록 (sklearn 없음)
    goal.py            목표 임계값 auto/fixed
    splits.py          채점 프로토콜 — train 60 / val 20 / test 20
    intervals.py       부트스트랩 신뢰구간
    calibration.py     확률이 확률로서 쓸 만한지 — 재기만 하고 모델은 안 바꿈
    ranking.py         랭킹 상한(KS) — 운영점이 아직 살 수 있는 것이 무엇인지
  dataset/             표의 열을 어떻게 읽는가 (데이터 행은 여기 없습니다 — 규칙과 어휘만)
    features.py        특성 열 정책 — 수치 통과, 저카디널리티 one-hot, 나머지는 이름과 함께 제외
    pipeline.py        전처리 step을 정렬된 spec으로 — 이름이 붙은 순서만 실행됩니다
    targets.py         정답 열 인코딩 + task 판정(detect_task) + 결측 정책(reject/drop)
    sentinels.py       측정값처럼 생긴 결측(`-9999`) 후보 — 경고는 하고 변환은 안 함
    caveats.py         집계가 보여주지 못하는 것을 카드가 싣는 채널
  nodes/
    profiling.py       결정론. 진입 노드. data_ref → dataset_card (subprocess)
    planning.py        LLM. iteration 카운터 소유, 이력 기반 재계획
    model_selection.py LLM 판단 + 코드 검증(레지스트리 화이트리스트·클램프)
    training.py        subprocess wrapper (실패를 예외로 올리지 않음)
    evaluate.py        결정론. best / stall_count 계산
    critic.py          LLM. 구조화 진단 + history append
    holdout.py         결정론. route() 다음, 최고 모델을 test 20%에서 1회 채점 (subprocess)
    report.py          LLM. 최종 보고서 + 진 시도 모델 정리
  scripts/             고정 스크립트 — LLM이 생성하지 않고, 파일로 실행됩니다
    profile.py         집계만 방출
    train.py           지표를 실제로 계산하는 scorers()가 여기 하나만 있습니다
                       (profile.py가 같은 것을 부릅니다 — 바와 점수는 같은 측정이어야 합니다)
    predict.py         저장된 인코딩을 재생, 못 맞추면 거부 (노드 아님)
  llm/
    client.py          유일한 API 호출 지점 (백오프, 타임아웃, 구조화 출력, leak 관문)
    prompts/*.md       프롬프트는 코드가 아닌 .md 파일
examples/
  dataset_card*.json   손으로 쓴 카드 — 원본 데이터 없이 돌아감
.claude-plugin/        Claude Code 플러그인 manifest (plugin.json + 로컬 marketplace.json)
skills/                그 플러그인이 붙이는 스킬 둘 — automl-run / automl-results
artifacts/<thread_id>/ 실행 산출물 (git 무시)
local/                 배포하면 안 되는 것 전부 (git 무시)
```

### 산출물

```
artifacts/
  checkpoints.sqlite               모든 스레드의 체크포인트
  <thread_id>/
    run_config.json                resume이 복원하는 실행 설정 (임시 파일 + rename으로 씀)
    dataset_card.json              profiling 노드가 만든 카드 (private `data` 블록 포함)
    report.md                      최종 보고서 (한글)
    history.json                   시도 이력 + 종료 사유 + best + holdout + models
    holdout.json                   최종 test 채점 — 실행당 1회
    llm/NNN_<prompt>_iterN.json    프롬프트·응답·usage 전량 (--dry-run은 .dryrun.md)
    train/iter_NN/
      train_config.json            학습 스크립트 입력
      result.json                  학습 스크립트 출력
      model.joblib                 적합된 파이프라인 — 데이터 등가
      feature_schema.json          그 모델을 만든 인코딩 — predict가 이 짝을 요구함
      train.log                    학습 로그
    predict/                       predict 서브커맨드의 기본 출력 위치
```

**실행이 끝나면 진 시도의 모델을 지웁니다**(`--keep-models best`, 기본값). 근거는
측정값입니다 — 개발 중 `artifacts/`가 1.9 GB였고 그중 1.79 GB가 `model.joblib`
56개, 가장 큰 하나가 **486 MB**였습니다. 이기지도 못한 시도의 깊은 forest였고, 루프에는
이것을 막는 것이 없습니다(`guard_memory`는 **입력 행렬**에 값을 매기므로 추정기가
무엇을 하든 같습니다). 지운 iteration 번호와 회수한 바이트는 콘솔과 `history.json`의
`models`에 남습니다 — 설명할 수 없는 삭제는 아낀 디스크보다 나쁩니다. 성공한 시도가
하나도 없으면 **아무것도 지우지 않습니다**: 실패한 실행이 남긴 파일은 그 실행의 증거
전부입니다. `feature_schema.json`도 지우지 않습니다(킬로바이트짜리 인코딩 기록).

`--dry-run`에서도 프롬프트는 **실제로 렌더링해서** 저장합니다. 미치환 placeholder가
없다는 것과, iteration 2의 계획 프롬프트가 iteration 1의 `failure_type`을 실제로 싣고
있다는 것을 토큰 한 개 없이 감사할 수 있습니다.

## 예시 데이터셋 카드

| 파일 | 목적 |
| --- | --- |
| [dataset_card.json](examples/dataset_card.json) | 기본 tabular 이진 분류 |
| [dataset_card_hard.json](examples/dataset_card_hard.json) | 목표 미달로 예산을 소진하는 어려운 카드 |
| [dataset_card_oom.json](examples/dataset_card_oom.json) | `constraints.memory_limit_mb: 6` — 실제 OOM 경로 |
| [dataset_card_crash.json](examples/dataset_card_crash.json) | `simulate.fail: "crash"` — 서브프로세스 하드 종료 |

네 장 모두 `path` 없이 선언된 형태만 담고 있어 실행기가 그 형태대로 데이터를
합성합니다 — **카드 한 장으로 루프 전체가 돕니다.** 대신 `baseline` 블록이 없으므로
`auto`는 지표별 기본값으로 폴백합니다(그렇다고 경고합니다).

`--dry-run` 4종이 서로 다른 분기로 정상 종료하는 것을 확인했습니다: `success`(iter 3
달성) · `fail`(iter 5 소진) · `oom`(OOM → Critic이 linear + `train_subsample`로 축소 →
iter 3 달성) · `stall`(동일 점수 3회 → 조기 종료). `--no-llm` 실제 실행에서도 OOM
카드는 `logreg`로 회복하고, crash 카드는 `result.json` 없이 자식이 죽어도
오케스트레이터가 살아 보고서까지 씁니다.

## 명세와 다른 점

의도적으로 다르게 구현한 부분입니다.

1. **프레임워크가 pytorch가 아니라 scikit-learn**(+ 선택적 xgboost). 대상이 tabular
   분류이고 이 환경에 torch가 없습니다. `scripts/train.py`의 입출력 계약은 명세
   그대로입니다.
2. **`oom`은 CUDA OOM이 아니라 명시적 메모리 예산 초과**로 재현합니다
   (`constraints.memory_limit_mb`). `status="error", error_type="oom"`, exit 0 계약은
   동일합니다.
3. **`profiling` 노드와 `data_ref` 채널을 추가**했습니다. 명세는 카드를 입력으로 받는
   것까지였는데, 그러면 "LLM은 원본 데이터를 보지 않는다"가 카드를 쓰는 사람에 대한
   규약으로만 남습니다. 카드 생성을 결정론적 진입 노드로 만들고 원본 경로를 별도
   채널로 빼면 그것이 구조가 됩니다.
4. **목표 임계값에 모드를 두었습니다.** 명세의 `goal`은 고정 숫자 하나였지만 그러면
   종료 조건이 모델 품질이 아니라 데이터셋 난이도를 잽니다. 숫자가 데이터 밖에서 올
   때는 고정값이 맞는 답이므로 둘 중 하나를 버리지 않고 모드로 나눴습니다.
5. **분할을 2개가 아니라 3개로 두고 `holdout` 노드를 추가**했습니다. 명세의 루프는
   검증 `best`를 보고하는 것까지였는데, 그러면 보고하는 숫자가 선택에 쓴 숫자와 같은
   숫자가 됩니다.
6. **범주형 열을 버리지 않고 인코딩합니다.** 문자열 열을 통째로 버리는 것은 일반적인
   표에서 신호의 대부분을 버리는 동작입니다.
7. **State에 `evaluation` 채널**, **CLI에 `--no-llm`과 `--force`** 를 추가했습니다.
   `--no-llm`은 처음에 자격 증명 없이 학습을 확인하려고 넣었지만, 남은 이유는 다릅니다 —
   계획하는 쪽만 규칙 기반인 **결정론적 대조군**이고, LLM이 계획하는 것이 무엇을 벌었는지는
   그 팔과 비교하지 않으면 말할 수 없습니다.

## 더 자세한 것

- 실행 명령: [RUNBOOK.md](RUNBOOK.md)
- 각 규칙의 근거: 해당 모듈의 docstring. 위 트리의 파일 이름이 곧 목차입니다
- 설계 논증과 사전 등록 측정 기록, 테스트 스위트(1,880개), 임상 데이터 측정 기록은 비공개
  개발 저장소에 있고 여기 싣지 않습니다. 그래서 이 저장소의 주석은 규칙을 말하되 그 규칙이
  왜 그 모양인지까지는 말하지 않는 곳이 있습니다

## 라이선스

MIT — [LICENSE](LICENSE). 코드와 문서에 붙습니다. `local/`에 두는 원본 데이터는 git이
무시하므로 애초에 여기 없습니다.
