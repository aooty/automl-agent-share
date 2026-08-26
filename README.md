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
                    │ (집계만)         │         유일한 통로, 그리고 그것은
                    ▼                 ▼         노드 경계다
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
python -m pip install -e ".[dev]"
python -m pip install -e ".[xgboost]"   # 선택: xgboost 모델까지
python -m pip install -e ".[bedrock]"   # 선택: Bedrock 경유 LLM 호출
```

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
| 자격 증명 없이 | `--dry-run` 또는 `--no-llm` | 없음 |

자격 증명이나 botocore가 없는 상태로 LLM 모드를 실행하면 **시작 전에** 안내와 함께
종료합니다. Bedrock에서는 구조화 출력이 강제 tool use로 자동 강등되고, 어느 쪽이
쓰였는지는 아티팩트의 `structured_mode`에 남습니다.

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
| `--max-iterations` / `--time-budget-sec` / `--seed` | 반복 예산 |
| `--no-llm` | 학습은 **실제로** 하고 추론 노드만 규칙 기반 폴백. 자격 증명 불필요 |
| `--dry-run` / `--scenario` | LLM과 학습을 **모두** 모킹. `--scenario {success,fail,oom,stall,slow,crash}` |
| `--force` | 같은 `--thread-id`의 기존 체크포인트를 지우고 처음부터 |
| `--on-missing-target {reject,drop}` | 정답이 빈 행의 처리. 기본은 개수를 알리고 중단 |
| `--keep-models {best,all}` | 실행이 끝날 때 진 시도의 `model.joblib`을 지울지 |

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
  `cut_headroom`은 **임계값을 골랐다면 얻었을 정확한 양**이라, 남은 격차가 운영점에
  있는지 랭킹에 있는지 구분해 줍니다. 실행기는 항상 `predict()`를 부르므로 도달한
  점수로 인용하면 안 됩니다.

## 루프가 끝나는 세 가지 경우 — `route()`

1. 목표 지표 달성
2. `iteration >= max_iterations`
3. 정체: `stall_count >= 2` (개선 없는 반복 연속 2회)

2·3번이 함께 무한 루프를 불가능하게 만듭니다. 목표를 못 채워도 `best` 스냅샷을
근거로 보고서는 **반드시** 작성됩니다.

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

## 더 자세한 것

- 실행 명령: [RUNBOOK.md](RUNBOOK.md)
- 각 규칙의 근거: 해당 모듈의 docstring. 위 트리의 파일 이름이 곧 목차입니다
- 테스트 스위트(1227개)와 임상 데이터 측정 기록은 배포본에 없습니다 — 개발 저장소의
  `tests/`와 `FINDINGS-mimic.md`입니다. 이 문서가 하는 주장을 실제로 검증하는 것이 그
  스위트이므로, 코드를 고칠 생각이라면 그쪽에서 작업하세요
