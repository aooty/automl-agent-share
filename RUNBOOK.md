# 실행 예시 모음 (PowerShell)

복사해서 붙여 넣는 용도입니다. 설계 설명은 [README.md](README.md)에 있습니다.

- 셸: PowerShell. 줄바꿈은 백틱(`` ` ``), 경로 구분은 `\`
- **저장소 루트에서** 실행하세요. `pip install -e .` 를 하지 않았으면
  `python -m automl_agent.main`이 패키지를 찾지 못합니다
- **데이터는 포함되어 있지 않습니다.** 정답 열이 있는 자기 CSV가 필요하고, 아래
  예시의 `local\data.csv`·`died` 자리에 넣으면 됩니다
- **배포하면 안 되는 것은 전부 `local\` 안에 둡니다** — 원본 CSV, 거기서 만든 카드,
  예측 결과. 이 디렉터리는 통째로 [.gitignore](.gitignore)에 있습니다

```powershell
python -m pip install -e ".[dev]"
```

## 차례

**처음이면 1 → 2 → 3 순서로 읽으세요. 데이터가 아직 없으면 4번부터.**

| | |
| --- | --- |
| [1. 카드 만들기](#1-카드-만들기--목표-미리보기) | 실행을 쓰지 않고 바만 먼저 확인 |
| [2. 실행](#2-실행) | 기본 경로 |
| [3. 조각 골라 붙이기](#3-조각-골라-붙이기) | 플래그를 조각으로 놓고 고르는 방법 |
| [4. 데이터 없이 루프만](#4-데이터-없이-루프만----dry-run) | `--dry-run` + 포함된 카드 |
| [5. LLM으로 실행](#5-llm으로-실행) | Anthropic API 직접 / Bedrock 경유 |
| [6. 재개·조회](#6-재개조회) | `resume` · `show` |
| [7. 새 데이터에 적용](#7-새-데이터에-적용) | `predict`, 라벨 있으면 채점까지 |
| [8. 거부되는 조합](#8-거부되는-조합) | 일부러 실패시켜 메시지 확인 |
| [9. 자주 걸리는 것](#9-자주-걸리는-것) | 막히면 여기 |

---

## 1. 카드 만들기 + 목표 미리보기

```powershell
python -m automl_agent.main profile `
  --data local\data.csv --target died `
  --metric roc_auc `
  --out local\my_card.json
```

`--metric`은 카드에 기록되지 않습니다 — 그 설정으로 실행했을 때 **두 모드가 낼 바를
미리 보여주기만** 합니다. `--margin`도 같이 받으므로 요구 수준을 올렸을 때의 바를
실행을 쓰지 않고 볼 수 있습니다. margin은 점수가 아니라 남은 여유에 걸리기 때문에
같은 증분이 데이터셋마다 다른 폭으로 움직이고, 미리보기 없이는 맞출 수 없습니다.

```
--margin 0.25 → roc_auc 0.8083   (기준선 0.7444)
--margin 0.40 → roc_auc 0.8466
--margin 0.50 → roc_auc 0.8722
```

카드는 사람이 읽고 고칠 수 있는 집계 요약입니다. `--no-baseline`을 주면 기준선
측정을 건너뛰지만, 그러면 `auto` 모드가 지표별 기본값으로 폴백합니다.

---

## 2. 실행

```powershell
# 카드로 시작
python -m automl_agent.main run `
  --dataset-card local\my_card.json `
  --metric roc_auc `
  --max-iterations 3 --no-llm `
  --thread-id t-auto

# 원본 CSV로 시작 — profiling 노드가 그래프 안에서 카드를 만듭니다 (1번을 건너뛴 경로)
python -m automl_agent.main run `
  --data local\data.csv --target died `
  --metric roc_auc `
  --max-iterations 3 --no-llm `
  --thread-id t-raw
```

`--no-llm`은 **학습은 실제로 하고** 계획하는 쪽만 규칙 기반으로 돌립니다. 자격 증명 없이
전체 공정을 확인하는 데 쓸 수 있지만, **그것이 이 플래그의 용도는 아닙니다** — 학습·분할·
지표·홀드아웃 규칙이 전부 같고 계획만 결정론적인 실행이라서, LLM 팔과 **비교 가능한
대조군**입니다. 같은 카드·같은 시드면 같은 계획이 나오므로 두 번 돌려 차이를 재면 그 차이는
LLM 쪽에서 온 것입니다. LLM을 쓰려면 5번.

같은 데이터에서 모드만 바꾸면 종료 사유가 바뀝니다. 아래는 예시 코호트(8,000행,
양성 12.5%)에서 측정된 것입니다:

| 실행 | 결과 | 최종 test |
| --- | --- | --- |
| `auto`(→ 0.8083), 3회 | iter 1에서 roc_auc=0.8545 → **목표 달성으로 종료** | 0.8427 (`selection_gap` +0.0119) |
| `fixed` 0.90, 3회 | 3회 소진(0.8545 → 0.8648 → 0.8565) → **반복 예산 소진** | 0.8412 (`selection_gap` +0.0236) |

0.90은 이 데이터에서 도달 불가였고, `auto`의 0.8083은 logreg의 0.7444를 남은 여유의
25%만큼 이긴 지점입니다. 오른쪽 열이 test 채점이 있는 이유입니다 — 시도를 3개 놓고
최댓값을 고른 쪽에서 격차가 두 배가 됩니다.

---

## 3. 조각 골라 붙이기

2번의 본체에 아래에서 한 줄씩 골라 이어 붙이면 됩니다.

### 목표 임계값 — 하나만

| 조각 | 의미 |
| --- | --- |
| (생략) | `auto` 기본. 기준선 + 남은 여유의 25% |
| `--margin 0.5` | `auto`, 남은 여유의 50% (더 어렵게) |
| `--threshold 0.88` | 그 숫자를 그대로 (`fixed` 모드) |

모드를 이름으로 고르는 플래그는 없습니다 — `--threshold`를 줬는지가 모드입니다.
`--threshold` + `--margin` 은 **거부됩니다**(8번).

### 지표 — 분류 (`died` 같은 라벨 열)

| 조각 | 비고 |
| --- | --- |
| (생략) | `--metric f1` — 양성이 드물면 낮게 나옵니다 |
| `--metric roc_auc` | 랭킹 품질. 운영점과 무관합니다 |
| `--metric balanced_accuracy` | 불균형 데이터에서 accuracy보다 읽기 쉽습니다 |
| `--metric pr_auc` | 희소 양성에서 roc_auc보다 민감. chance는 0.5가 아니라 **양성 비율** |
| `--metric precision` / `--metric recall` | 한쪽만 보는 목표. 다른 쪽은 `result.json`에서 함께 확인하세요 |
| `--metric accuracy` | chance(= 다수 클래스 비율)를 먼저 보세요 — 고정 기본값 0.85보다 높을 수 있습니다 |

### 지표 — 회귀 (`los_days` 같은 연속 열)

| 조각 | 방향 | 비고 |
| --- | --- | --- |
| `--metric r2` | 높을수록 좋음 | 1이 상한이라 `auto` 여유 계산이 분류와 같습니다. 음수도 나옵니다 |
| `--metric mae` | **낮을수록 좋음** | 정답 열의 단위 그대로. 지표별 기본 바가 없어 `--threshold`나 측정된 기준선이 필요합니다 |
| `--metric rmse` | **낮을수록 좋음** | 큰 오차에 더 민감. `mae`와 같은 이유로 기본값 없음 |

받는 이름은 이 아홉 개뿐입니다(별칭: `average_precision`→`pr_auc`,
`mean_absolute_error`→`mae`, `root_mean_squared_error`→`rmse`, `r2_score`→`r2`).
없는 이름은 argparse가 바로 거부합니다.

**두 목록은 섞이지 않습니다.** task는 고르는 값이 아니라 정답 열에서 읽힙니다
(`targets.detect_task`). 연속 열에 `--metric f1`을 주면 거부가 아니라 `r2`로 **바뀌어
실행되고**, 바꿨다는 사실이 콘솔·Critic 프롬프트·`report.md`에 같은 한 줄로 남습니다
(8번). 방향은 지표가 정하므로 고를 수 없습니다 — `mae`는 낮을수록, `f1`은 높을수록 좋습니다.

### LLM — 하나만

| 조각 | 학습 | 계획 | 자격 증명 |
| --- | --- | --- | --- |
| `--no-llm` | 실제로 함 | **규칙 기반 (결정론적)** | 불필요 |
| `--dry-run` | 모킹 | 모킹 | 불필요 |
| (생략) | 실제로 함 | LLM 실제 호출 | 필요 |

### 예산

`--max-iterations 3` · `--time-budget-sec 3600` · `--seed 42` ·
`--force`(기존 `thread-id` 덮어쓰기) · `--keep-models all` ·
`--search-past-goal`(목표를 넘어도 예산을 다 씀 — 첫 시도가 바를 넘어 Critic이 0회 도는
실행을 피하려면 이것)

**`--time-budget-sec`은 실행 하나가 일하는 초의 상한입니다** — 학습 하나의 timeout이
아닙니다. 그래서 `--max-iterations 5 --time-budget-sec 3600`은 최악의 경우에도 5시간이
아니라 1시간에 가깝습니다: 10%(360초)는 holdout 몫으로 떼어 두고, 남은 3,240초를 적합들이
`남은 시간 ÷ 남은 반복 수`로 나눠 씁니다. 앞 반복이 빨리 끝나면 남은 시간은 뒤로 넘어갑니다.

- 몫이 이미 없으면 `subprocess`를 **띄우지 않고** `too_slow`로 기록합니다.
- 루프가 예산을 다 써서 끊기면 종료 사유가 `out_of_time`입니다 — 목표 달성·상한·정체를
  먼저 검사하므로, 이 이름은 **그 밖에도 계속 갈 수 있었던 루프**만 가리킵니다.
- 벽시계가 아니라 실행이 *일한* 초를 세므로 `resume`이 안전합니다. 프로세스가 꺼져 있던
  시간은 청구되지 않습니다.
- 상한은 근사치입니다. 이미 뜬 적합은 자기 몫을 조금 넘길 수 있고, LLM 호출은 시간을
  세기만 하고 중간에 끊지 않으며, 종료 검사는 반복 *사이*에만 돕니다.

디스크도 예산입니다. 기본값 `--keep-models best`는 `predict`가 실제로 쓰는 최고 시도의
`model.joblib`만 남깁니다 — 적합된 모델 하나가 수백 MB가 되고(개발 중 측정된 최댓값
486 MB) 루프에는 그것을 막는 것이 없기 때문입니다. 지운 목록은 콘솔과
`history.json`의 `models`에 남습니다.

### 정답 결측

| 조각 | 의미 |
| --- | --- |
| (생략) | `profile`은 `reject` — 빈 라벨이 있으면 개수를 알리고 멈춥니다. `run`은 카드의 정책을 따릅니다 |
| `--on-missing-target drop` | 그 행을 빼고 진행. 카드에 `target_missing.n_dropped`로 기록 |

`drop`으로 카드를 만들었으면 `run`에는 플래그를 **주지 마세요** — 카드의 정책을
그대로 따라야 기준선을 잰 데이터와 학습이 점수를 내는 데이터가 같습니다.

---

## 4. 데이터 없이 루프만 — `--dry-run`

학습과 LLM을 모두 모킹하므로 즉시 끝납니다. 저장소에 포함된 손으로 쓴 카드를 씁니다.

```powershell
python -m automl_agent.main run --dataset-card examples\dataset_card.json --dry-run --thread-id t-dry-success
python -m automl_agent.main run --dataset-card examples\dataset_card_hard.json --dry-run fail --thread-id t-dry-fail
python -m automl_agent.main run --dataset-card examples\dataset_card_oom.json --dry-run oom --thread-id t-dry-oom
python -m automl_agent.main run --dataset-card examples\dataset_card_crash.json --dry-run crash --thread-id t-dry-crash
```

이 카드들에는 `baseline`이 없어서 `auto`가 지표별 기본값으로 폴백하고, 그렇다고
경고합니다. 실제 도출을 보려면 1~2번 경로를 쓰세요.

프롬프트는 `--dry-run`에서도 **실제로 렌더링해서** 저장하므로
`artifacts\t-dry-success\llm\`을 읽으면 모델에게 보냈을 텍스트를 토큰 한 개 없이 감사할
수 있습니다.

---

## 5. LLM으로 실행

```powershell
# Anthropic API 직접
$env:ANTHROPIC_API_KEY = "sk-ant-..."

# 또는 Bedrock 경유
$env:AUTOML_USE_BEDROCK = "1"
$env:AWS_REGION = "us-west-2"

python -m automl_agent.main run `
  --dataset-card local\my_card.json `
  --metric roc_auc --margin 0.5 `
  --max-iterations 4 --thread-id t-llm
```

자격 증명이 없으면 **시작 전에** 안내와 함께 종료합니다. 프롬프트 전량이
`artifacts\t-llm\llm\`에 남으므로 원본 데이터가 실리지 않았음을 직접 확인할 수 있습니다.

```powershell
$files = Get-ChildItem artifacts\t-llm\llm\* -File
"검사 대상 $($files.Count)개"                 # 0개면 경로가 틀린 것
$files | Select-String -Pattern "data.csv"    # 출력 없음이 정상
```

확장자를 걸지 마세요. 실제 실행은 `.json`, `--dry-run`은 `.dryrun.md`로 남으므로
`*.md`로 좁히면 실제 실행에서 0개 파일을 검사하고도 "결과 없음"이 나옵니다.

---

## 6. 재개·조회

```powershell
# 재개 (다른 플래그 불필요 — run_config.json에서 복원)
python -m automl_agent.main resume --thread-id t-auto

# 상태 + 보고서 전문
python -m automl_agent.main show --thread-id t-auto --report

# 보고서만 / 최종 test 채점만 / 산출물 목록
Get-Content artifacts\t-auto\report.md
Get-Content artifacts\t-auto\holdout.json
Get-ChildItem artifacts\t-auto -Recurse -Name

# 그래프 구조
python -m automl_agent.main graph --out graph.png
```

---

## 7. 새 데이터에 적용

끝난 실행이 고른 모델을 새 행에 씁니다. iteration은 알아서 찾습니다 — 체크포인트를
보고, 없으면 `history.json`의 `best`를 봅니다.

```powershell
python -m automl_agent.main predict `
  --thread-id t-auto `
  --data local\new_rows.csv `
  --id-column patient_id `
  --out local\pred.csv
```

```
이 실행이 선택한 iteration 3의 모델을 사용합니다
500행을 예측해 local\pred.csv에 저장했습니다 (인코딩된 피처 12개, task=classification)
피처가 아닌 컬럼은 제외했습니다: died
학습 때의 인코딩과 어긋난 곳은 없었습니다
실행 요약: local\pred.report.json
출력 파일은 행 단위 예측이므로 원본 데이터와 같은 취급을 하십시오 — 커밋하지 말고, 프롬프트에 넣지 마십시오.
```

출력 CSV는 `--id-column`(주었다면), `prediction`, 분류면 `proba_<라벨>` 열입니다. 입력
열은 다시 쓰지 않습니다 — 이미 가진 파일을 한 부 더 복사하는 일이라서. `prediction`은
**학습 파일이 쓴 값 그대로**입니다(라벨이 `died`/`survived` 문자열이었다면 그 단어와
`proba_died`가 나옵니다). 회귀면 `prediction` 하나뿐입니다.

`--out`을 생략하면 `artifacts\<id>\predict\`에 씁니다. 실행 요약 JSON은 **항상**
`--out` 옆에 함께 나오고 `--report`는 그 위치만 바꿉니다. 어느 쪽이든 행 단위
데이터이므로 `local\` 또는 `artifacts\` 안에 두세요.

```powershell
# 정답이 이미 붙어 있는 배치라면 채점까지 (예측은 위와 한 행도 다르지 않습니다)
python -m automl_agent.main predict `
  --thread-id t-auto --data local\last_month.csv `
  --label-column outcome --out local\scored.csv

# 최고 시도가 아닌 특정 iteration
python -m automl_agent.main predict `
  --thread-id t-auto --data local\new_rows.csv --iteration 2 --out local\pred_i2.csv

# 고정 스크립트를 직접 (두 경로를 손으로 짝지어야 합니다)
python -m automl_agent.scripts.predict `
  --model artifacts\t-auto\train\iter_03\model.joblib `
  --schema artifacts\t-auto\train\iter_03\feature_schema.json `
  --data local\new_rows.csv --out local\pred.csv
```

`--label-column`을 주면 목표 지표와 그 신뢰구간, 다른 지표들, 확률 품질(`brier`,
구간별 실제 발생률)이 함께 나옵니다. 라벨이 빈 행과 학습 때 없던 라벨 값을 가진 행은
채점에서 빠지고 개수가 표시됩니다. **이 점수는 이 실행의 채점 프로토콜이
아닙니다** — 홀드아웃은 학습 전에 떼어 둔 행을 1회 채점한 값이고, 이 파일이 어떤
행으로 이루어졌는지는 여기서 알 수 없습니다.

`model.joblib` 하나만으로는 새 파일에 적용할 수 없습니다 — 어떤 level이 몇 번째 열인지가
그 파일에 없기 때문입니다. 그래서 학습이 옆에 `feature_schema.json`을 남기고
`predict`는 그 짝을 요구합니다. **거부하는 것**은 모델이 필요한 컬럼이 없는 파일과
스키마가 없는 실행뿐이고, 나머지는 예측을 낸 뒤 알립니다:

```
확인할 점 3건 — 예측은 나왔지만 아래를 읽으십시오:
  - 수치 컬럼 'age'의 41행이 숫자로 읽히지 않아 결측으로 처리했습니다
  - 'city'에 학습 때 없던 범주 2개(jeju, sejong) — 12행이 이 컬럼에서 전부 0으로 인코딩됩니다
  - 'age'의 결측률이 학습 때 2.1%에서 이 배치 11.4%로 바뀌었습니다 — 인코딩은 맞지만
    모델이 보는 값의 출처가 달라졌습니다
  - 'creatinine'에 결측 코드로 의심되는 -9999이 0.0%에서 3.2%로 늘었습니다 — 이 값은
    결측이 아니라 숫자 -9999로 모델에 들어갑니다
  - 학습 때와 라이브러리 버전이 다릅니다 (scikit-learn 1.5.2 → 1.7.0) — 저장된 모델은
    pickle이라 다른 버전에서 열면 예측이 조용히 달라질 수 있습니다
```

폭 검사로는 보이지 않는 것들입니다. 레이아웃이 맞는 것과 배치가 비교 가능한 것은 다른
이야기입니다. 결측률 문턱은 5%p **그리고** 2×표준오차, 결측 코드 비율은 1%p로 훨씬
낮습니다 — 결측률이 움직이는 것은 평범한 일이지만 **결측 코드의 비중이 움직이는 것은
규약이 바뀌었다는 신호**입니다.

---

## 8. 거부되는 조합

```powershell
# --threshold + --margin (바를 정하는 방식이 서로 다릅니다)
python -m automl_agent.main run --dataset-card local\my_card.json --threshold 0.9 --margin 0.5 --thread-id t-x

# 알 수 없는 시나리오 (argparse가 바로 거부)
python -m automl_agent.main run --dataset-card local\my_card.json --dry-run clever --thread-id t-x

# 이미 쓴 thread-id (--force 없이) — run_config.json은 그대로 남습니다
python -m automl_agent.main run --dataset-card local\my_card.json --thread-id t-auto

# 없는 지표 이름 / 바가 없는 지표를 바 없이 / 실행을 무의미하게 만드는 값
python -m automl_agent.main run --dataset-card local\my_card.json --metric auroc --thread-id t-x
python -m automl_agent.main run --dataset-card local\my_reg_card.json --metric mae --thread-id t-x
python -m automl_agent.main run --dataset-card local\my_card.json --max-iterations -1 --thread-id t-x
```

```
오류: --threshold 와 --margin 은 함께 쓸 수 없습니다 — 바를 정하는 방식이 서로 다릅니다.
  - 값을 못박으려면: --threshold 0.9
  - 기준선에서 도출하되 요구 수준만 조절하려면: --margin 0.5  (기준선에서 남은 여유의 50%)
```

조용히 무시된 플래그가 최악이라서 모순되는 조합은 실행 전에 종료합니다. 반면
**task가 어긋난 지표는 끊지 않고 바꿉니다** — 오타 하나로 긴 실행이 죽는 것보다,
그 task의 기본 지표로 돌리면서 바꿨다고 말하는 편이 낫습니다.

```
경고: 이 데이터의 task는 binary_classification인데 목표 지표 r2는 다른 task의 지표입니다 — 이
정답 열에서는 계산되지 않으므로, 그대로 두면 모든 시도가 '목표 미달'로 기록되고 반복 예산만
소모됩니다. f1로 바꿔 실행합니다
```

**괄호가 실행 내내 따라옵니다** — 콘솔 헤더, Critic 프롬프트, `report.md`가 같은 한
줄을 읽으므로 나중에 보고서만 본 사람도 이 실행이 요청과 다른 지표로 채점됐다는 것을
압니다. 함께 준 `--threshold`는 버립니다(`mae`의 단위로 준 3.5를 `f1`의 바로 쓰면
아무 모델도 닿지 못하는 목표가 되고, 그게 대체가 막으려던 결과입니다).
`run_config.json`에는 입력한 이름이 그대로 남습니다 — 물어본 것과 채점된 것을
구분하려고.

---

## 9. 자주 걸리는 것

| 증상 | 원인 / 대응 |
| --- | --- |
| `No module named 'automl_agent'` | 저장소 루트에서 실행하거나 `pip install -e .` |
| 이 파일 커밋해도 되나 | `local\` 안에 있으면 됩니다 — 그 디렉터리는 통째로 무시됩니다. `git status --short --ignored`로 확인 |
| 한글이 깨져 보임 | `$env:PYTHONIOENCODING = "utf-8"`. subprocess 쪽은 코드에서 고정했지만 콘솔은 별개입니다 |
| `이미 체크포인트가 있습니다` | `--thread-id`를 새로 잡거나 `--force`. 재사용하면 두 실행의 `history`가 섞입니다 |
| `Anthropic API 자격 증명이 없어…` | `--no-llm` 또는 `--dry-run`을 붙이거나 5번의 환경변수 설정 |
| xgboost 모델이 안 보임 | `python -m pip install -e ".[xgboost]"` |
| `auto`인데 목표가 지표 기본값 | 카드에 `baseline` 블록이 없습니다. `--no-baseline` 없이 `profile`을 다시 실행 |
| 목표에 `(도달 불가)`가 붙음 | chance가 상한 0.99를 밀어냈습니다 — 이 지표로는 모델과 다수 클래스를 구분할 수 없습니다. `--metric balanced_accuracy` 또는 `pr_auc` |
| 목표 줄에 `랭킹 … 상한을 넘습니다` | 지표의 상한이 아니라 **기준선 랭킹**의 상한입니다(카드의 `baseline.ks`). 더 좋은 모델은 넘을 수 있으므로 바는 낮추지 않습니다 — 넘기려면 랭킹을 올리세요(모델 family나 특성. 하이퍼파라미터로는 거의 안 움직입니다) |
| 목표 줄에 `기준선 랭킹의 최적 컷을 넘도록 상향` | `--margin`이 낸 바가 **같은 기준선을 최적 컷에서 자른 점수보다 낮았습니다.** 기준선은 0.5 컷에서 재므로 생기는 일이고, 그 바는 재자른 logreg가 이미 넘으므로 목표가 아닙니다. `balanced_accuracy`에서만 발동합니다. 같은 줄이 `--margin` 상한을 함께 출력하니, **더 어려운 목표를 원하면 그보다 크게** 주세요 — 그 아래 값은 모두 같은 바를 냅니다 |
| 목표 줄에 `기준선 자체의 95% CI … 안에 있습니다` | 바와 기준선의 차이가 val 슬라이스가 분해할 수 있는 폭보다 작습니다. `--margin`을 키우거나 최종 홀드아웃 점수를 판정 근거로 쓰세요. 실행은 막지 않습니다 |
| `has N missing values`로 멈춤 | 정답 컬럼이 빈 행이 있습니다. 라벨을 고치거나 `--on-missing-target drop` |
| 정답 열에 값이 하나뿐 | task 판정이 아니라 거부입니다(`TargetUnusableError`). 컬럼 이름이 맞는지, 행이 한 결과만 남게 필터되지 않았는지 보세요 |
| 라벨이 `0.0`/`1.0` float인데 회귀로 잡힐까 | 아닙니다 — 고유값이 2개면 dtype과 무관하게 분류입니다. 판정 순서는 [targets.py](automl_agent/dataset/targets.py)의 `detect_task` |
| `카드에 허용되지 않은 키가 있습니다` | 손으로 쓴 카드에 모르는 최상위 키가 있습니다. 메시지가 허용 목록을 함께 출력합니다 |
| `카드의 split 프로토콜이 이번 실행과 다릅니다` | 카드의 기준선이 **다른 행**에서 측정됐습니다. `--seed`를 카드와 맞추거나 `profile`을 다시 돌리세요 |
| 문자열 열이 `고유값 50개 초과로 제외`됨 | 자유 텍스트나 식별자 열입니다. 쓰고 싶으면 값을 구간으로 묶어(상위 20개 + `기타`) 다시 프로파일링하세요 |
| 카드의 `n_features`와 `encoding` 합이 다름 | 앞은 파일의 열 수, 뒤는 추정기가 실제로 받는 열 수입니다. 범주형 1열이 one-hot 6열이 되면 뒤가 큽니다 |
| 최종 test 점수가 검증 `best`보다 낮음 | **정상이고 그게 이 숫자의 용도입니다.** `best`는 검증 점수 N개의 최댓값이라 위로 치우쳐 있고 그 폭이 `selection_gap`입니다. 보고할 숫자는 test 쪽. 반대로 test가 더 높으면 분할 노이즈이므로 둘 다 인용하세요 |
| `검증 점수가 위 CI 안에 있어 …구분되지 않습니다` | `selection_gap`이 test 구간보다 작습니다. "선택 편향 0.004"로 인용하지 말고 두 점수를 구간과 함께 인용하세요 |
| `최종 테스트 채점 생략 — …` | 채점할 모델이 없습니다(`--dry-run`, 모든 시도 실패, 채점 자식 프로세스 실패). 보고서는 그대로 작성되고 사유가 실립니다. 로그는 `artifacts\<id>\holdout.log` |
| iteration 간 차이가 개선인지 노이즈인지 | `result.json`의 `<metric>_ci_low`/`_ci_high` 폭과 비교하세요. 폭보다 작은 차이는 이 행들이 구분해 낸 차이가 아니고, Critic도 같은 규칙으로 판단합니다 |
| `no confidence interval (split too small…)` | 재표집 단위가 20개 미만입니다. 정상 동작이고, 없는 구간을 지어내지 않은 것입니다 — 이 실행에서는 작은 차이를 개선으로 읽지 마세요 |
| `out_of_time`으로 끝남 | 루프가 `--time-budget-sec`을 다 썼습니다. 목표·상한·정체를 먼저 검사하므로 **이 이름은 계속 갈 수 있었던 루프만** 가리킵니다. `history.json`의 `budget`으로 어디에 들어갔는지 보세요 — LLM 호출 시간도 포함됩니다 |
| 적합이 `too_slow`인데 로그가 없음 | 그 시도의 몫이 이미 없어서 `subprocess`를 **띄우지 않았습니다.** 1초 뒤에 죽일 프로세스를 띄우면 기록에 남는 것이 적합이 아니라 spawn이 됩니다. 예산을 키우거나 `--max-iterations`를 줄여 몫을 늘리세요 |
| 바를 못 넘고 `stalled`로 끝남 | 지표의 상한이 아니라 **이 데이터의 랭킹 품질**이 상한일 수 있습니다. `roc_auc`는 그대로인데 `balanced_accuracy`만 낮으면 운영점 문제이고, 그때 `balanced_accuracy_cut_headroom`이 크면 `tune_threshold`가 가장 싼 레버입니다 — 작으면 컷에는 살 것이 없습니다. `--margin`을 낮추거나 `--metric roc_auc`로 재세요 |
| `recall`이 `specificity`보다 훨씬 낮음(또는 반대) | `balanced_accuracy`는 두 값의 평균이라 **둘이 같아지는 지점이 최적**입니다. 차이가 0.08을 넘으면 규칙 기반 진단이 낮은 쪽을 인용해 양성 가중치를 ×1.5 또는 ÷1.5 하라고 처방합니다 |
| 가중치를 올렸다 내렸다 하며 제자리 | 부호가 한 번 뒤집혔으면 그 두 점이 최적을 감싼 것이니 **사이를 재세요**. 모델을 바꾸면 그 짝은 무효입니다 |
| 운영점을 더 밀어볼 가치가 있나 | `result.json`의 `balanced_accuracy_cut_headroom`을 보세요 — **더 나은 컷이 이 행들에서 아직 값하는 정확한 양**입니다. 크면 계획에 `tune_threshold`를 주면 되고(학습 행 20%가 비용), 작으면 컷에 살 것이 없고 남은 격차는 랭킹에 있습니다. **도달한 점수로 인용하면 안 됩니다** — 그 시도가 쓰지 않은 컷에서 잰 값입니다. 목표 지표가 무엇이든 이 값은 `balanced_accuracy`로 재므로 `f1` 목표까지 남은 거리와 나눠서 비율로 읽으면 안 됩니다 |
| 보고서의 하이퍼파라미터가 제안과 다름 | 정상입니다 — 실제 적용된 `applied_hyperparams`를 인용합니다. 버려진 이름은 `result.json`의 `dropped_hyperparams`에 |
| `계획이 실행기에 없는 기능을 전제한 것으로 보입니다` | 산문이 threshold 튜닝·교차검증·특성 공학 같은 없는 기능에 의존하는 것으로 읽혔습니다. 실행은 계속되고 그 부분만 실행되지 않으며 `plan.unsupported_claims`로 남습니다. **부분 문자열 탐지라 오탐 가능** — 목록은 [capabilities.py](automl_agent/capabilities.py) |
| `predict`: `인코딩 스키마가 없습니다` | 합성 데이터 실행이거나 이 파일이 저장되기 전 버전입니다. 새 파일에서 인코딩을 다시 유도하는 건 폭이 우연히 맞으면 어긋난 컬럼으로 예측이 나오는 일이라 하지 않습니다 — 같은 데이터로 다시 학습하세요 |
| `predict`: `선택된 최고 시도가 없습니다` | 성공한 학습이 없거나 실행이 report까지 가지 않았습니다. `show`로 확인하고, 특정 시도를 쓰려면 `--iteration <번호>` |
| `predict --iteration N`: `모델 파일이 없습니다` | 최고 시도가 아니면 실행이 끝날 때 정리됩니다(기본 `--keep-models best`). 남은 것은 `history.json`의 `models`에 있고, 다음 실행부터 전부 남기려면 `--keep-models all` |
| `predict`: `FeatureSchemaMismatch` | 모델이 필요한 원본 컬럼이 새 파일에 없습니다(메시지가 전부 열거합니다). 이름을 맞추세요 — 빈 열을 만들어 넣으면 값을 발명하는 것이 됩니다 |
| `predict`: `report not written: …` | 예측 CSV는 이미 저장되었습니다 — `--report` 경로만 실패한 것이고 같은 내용이 콘솔에 나옵니다 |
| 예측 CSV를 커밋해도 되나 | 안 됩니다 — 행 단위 데이터입니다. `local\` 또는 `artifacts\` 안에 두세요 |
| `artifacts\`가 수 GB | 대부분 `model.joblib`입니다. 끝난 실행은 기본값이 이미 정리하지만 `--keep-models all`로 돌린 것은 손으로 지워야 합니다 — 남길 것은 `history.json`·`report.md`입니다 |
| `iteration N의 train_config.json을 쓸 수 없습니다` | 학습의 실패가 아니라 디스크나 권한입니다. 그 iteration은 `write_failed`로 기록되고 실행은 계속되지만 원인이 그대로면 다음도 같은 지점에서 멈춥니다 — 공간을 만들고 같은 `--thread-id`로 `resume` |
| `… database is locked` | 같은 `artifacts` 디렉터리에 다른 `run`/`resume`이 살아 있습니다. 60초를 기다린 뒤의 메시지이므로 순간 경합이 아닙니다. 나란히 돌려야 하면 한쪽에 `--artifacts-root`를 다르게 주세요 |
| `실행 설정이 올바른 JSON이 아닙니다` | `run_config.json`이 쓰이던 중에 끊긴 파일입니다. 같은 `--thread-id`로 `run`을 다시 실행하면 이 파일만 새로 쓰이고, 체크포인트가 남아 있으면 그 지점부터 이어집니다 |
