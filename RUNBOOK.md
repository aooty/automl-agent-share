# 실행 예시 모음 (PowerShell)

복사해서 붙여 넣는 용도입니다. 설계 설명은 [README.md](README.md)에 있습니다.

- 셸: PowerShell. 줄바꿈은 백틱(`` ` ``), 경로 구분은 `\`
- **저장소 루트에서 실행하세요.** `pip install -e .` 를 하지 않았다면 `python -m automl_agent.main` 이 패키지를 찾지 못합니다
- 데이터는 저장소에 포함되지 않습니다. 1번에서 합성 데이터를 만들어 씁니다
- **배포하면 안 되는 것은 전부 `local\` 안에 둡니다.** 이 디렉터리는 통째로
  [.gitignore](.gitignore)에 있어서, 아래 명령들의 `--out`도 기본적으로 여기를 가리킵니다

```powershell
python -m pip install -e ".[dev]"
```

## 차례

**처음이면 1 → 2 → 4 순서로 읽으세요.**

| | |
| --- | --- |
| [1. 데이터 준비 (합성)](#1-데이터-준비-합성) | 저장소에 데이터가 없으니 먼저 만듭니다 |
| [2. 조합 규칙](#2-조합-규칙) | 플래그를 조각으로 놓고 골라 붙이는 방법 |
| [3. 목표 미리보기](#3-목표-미리보기--실행을-쓰지-않고-바만-확인) | 실행을 쓰지 않고 바만 확인 |
| [4. auto 모드 실행](#4-auto-모드-실행) | 기본 경로 |
| [5. 같은 데이터, fixed 0.90](#5-같은-데이터-fixed-090) | 모드만 바꿨을 때 종료 사유가 어떻게 바뀌는지 |
| [6. auto를 더 밀어붙이기](#6-auto를-더-밀어붙이기) | `--margin` |
| [7. 원본 CSV에서 바로 실행](#7-원본-csv에서-바로-실행-카드-없이) | 카드 없이 (`profiling` 노드가 만듭니다) |
| [8. 연속 정답 열 (회귀)](#8-연속-정답-열-회귀) | 두 번째 모드 |
| [9. 거부되는 조합](#9-거부되는-조합) | 일부러 실패시켜 메시지를 확인 |
| [10. `--dry-run`](#10-카드-없이-루프만-돌려-보기---dry-run) | 카드도 학습도 없이 루프 구조만 |
| [11. LLM으로 실행](#11-llm으로-실행) | Anthropic API 직접 / Bedrock 경유 |
| [12. 중단·재개·조회](#12-중단재개조회) | `resume` · `show` |
| [13. 학습한 모델을 새 데이터에 적용](#13-학습한-모델을-새-데이터에-적용) | `predict`, 라벨 있으면 채점까지 |
| [14. 자주 걸리는 것](#14-자주-걸리는-것) | 막히면 여기 |

---

## 1. 데이터 준비 (합성)

**합성 데이터 생성기는 배포본에 없습니다** — 개발 저장소의 `examples\make_demo_data.py`입니다.
이미 자기 CSV가 있으면 그것을 쓰고(7번이 카드 없이 원본에서 바로 시작하는 경로입니다), 아래
숫자는 그 생성기가 내는 것이라 그대로 재현하려면 개발 저장소에서 가져오세요.

```powershell
python examples\make_demo_data.py
```

```
local\demo.csv — 8000행, 양성 비율 0.1255, target 컬럼 'died'
```

이 데이터는 개발 저장소의 `examples/make_demo_data.py`가 일부러 그렇게 만든
것입니다 — 선형 모델은 roc_auc 0.73 정도, 부스팅은 0.85 정도에서 멈추므로 `auto`가
내는 바는 넘고 고정 기본값 0.90은 못 넘습니다. 두 모드의 차이가 드러나는 난이도입니다.
양성 비율이 12.5%라서 `accuracy`의 chance 수준(0.8745)도 고정 기본값 0.85보다 높습니다.

연속 정답 열이 필요하면 `--task regression`을 줍니다 — `local\demo_reg.csv`에 `los_days`
열을 씁니다(8번).

자기 데이터로 하려면 이후 모든 명령의 `local\demo.csv`와 `died`를 바꾸면 됩니다.
원본 파일도 `local\` 안에 두는 것을 권합니다 — 그러면 "커밋해도 되나"를 파일마다 따질
필요가 없습니다. 어디에 두더라도 `*.csv` 자체가 무시 목록에 있긴 합니다.

---

## 2. 조합 규칙

명령은 **본체 + 조각들**입니다. 아래에서 한 줄씩 골라 이어 붙이면 됩니다.

### 본체

```powershell
# 카드 만들기 / 목표 미리보기
python -m automl_agent.main profile --data local\demo.csv --target died --out local\my_card.json

# 실행 (카드로 시작)
python -m automl_agent.main run --dataset-card local\my_card.json --thread-id <아이디>

# 실행 (원본 CSV로 시작 — profiling 노드가 그래프 안에서 카드를 만듭니다)
python -m automl_agent.main run --data local\demo.csv --target died --thread-id <아이디>
```

### 목표 임계값 조각 — 하나만 고릅니다

| 조각 | 의미 | demo.csv의 roc_auc에서 |
| --- | --- | --- |
| (생략) | `auto` 기본. 기준선 + 남은 여유의 25% | 0.8083 |
| `--margin 0.4` | `auto`, 남은 여유의 40% | 0.8466 |
| `--margin 0.5` | `auto`, 남은 여유의 50% | 0.8722 |
| `--goal-mode fixed --threshold 0.88` | 그 숫자를 그대로 | 0.88 |
| `--goal-mode fixed` | 지표별 기본값 | 0.90 |

`--goal-mode auto` + `--threshold`, `--goal-mode fixed` + `--margin` 은 **거부됩니다** (9번).

### 지표 조각 — 분류 (`died` 같은 라벨 열)

| 조각 | 비고 |
| --- | --- |
| (생략) | `--metric f1` — 12% 양성 비율에서는 낮게 나옵니다 |
| `--metric roc_auc` | 아래 예시의 기본 선택 |
| `--metric balanced_accuracy` | 불균형 데이터에서 accuracy보다 읽기 쉽습니다 |
| `--metric pr_auc` | 희소 양성에서 roc_auc보다 민감합니다. chance는 0.5가 아니라 양성 비율(demo.csv에서 0.1255) |
| `--metric precision` / `--metric recall` | 한쪽만 보는 목표. 다른 쪽은 `result.json`에서 함께 확인하세요 |
| `--metric accuracy` | chance가 0.8745임을 염두에 두세요 (고정 기본값 0.85보다 높습니다) |

### 지표 조각 — 회귀 (`los_days` 같은 연속 열)

| 조각 | 방향 | 비고 |
| --- | --- | --- |
| `--metric r2` | 높을수록 좋음 | 1이 상한이라 `auto` 여유 계산이 분류와 똑같이 동작합니다. 음수도 나옵니다(평균 예측보다 나쁨) |
| `--metric mae` | **낮을수록 좋음** | 정답 열의 단위 그대로. 고정 기본값이 없어 `--threshold`나 측정된 기준선이 필요합니다 |
| `--metric rmse` | **낮을수록 좋음** | 큰 오차에 더 민감. `mae`와 같은 이유로 기본값이 없습니다 |

받는 이름은 이 아홉 개뿐입니다(별칭: `average_precision`→`pr_auc`,
`mean_absolute_error`→`mae`, `root_mean_squared_error`→`rmse`, `r2_score`→`r2`). 목록은
`automl_agent/scoring/metrics.py` 한 곳에서 나오고, 학습기가 내는 지표와 같은 목록입니다.
없는 이름은 argparse가 바로 거부합니다 — 잴 수 없는 바로 실행이 끝까지 도는 것보다
낫습니다.

**두 목록은 섞이지 않습니다.** task는 정답 열에서 읽히므로(`targets.detect_task`) 고른
쪽이 아니라 데이터가 결정합니다. 연속 열에 `--metric f1`을 주면 그 지표로 도는 대신 회귀
기본 지표 `r2`로 바뀌고, 바꿨다는 사실이 실행 시작 시점과 `report.md`에 남습니다 — 라벨
열에 `--metric mae`도 같습니다(그쪽은 `f1`로). 자세한 것은 9번. 방향도 지표의 속성이라
`--direction`은 생략하거나 지표가 이미 함의하는 값을 그대로 줄 때만 받습니다.

### 정답 결측 조각

| 조각 | 의미 |
| --- | --- |
| (생략) | `profile`은 `reject` — 빈 라벨이 있으면 개수를 알리고 멈춥니다. `run`은 카드에 기록된 정책을 따릅니다 |
| `--on-missing-target drop` | 그 행을 빼고 진행. 카드에 `target_missing.n_dropped`로 기록됩니다 |

`drop`으로 카드를 만들었으면 `run`에는 플래그를 주지 마세요 — 카드의 정책을 그대로
따라야 기준선을 잰 데이터와 학습이 점수를 내는 데이터가 같습니다.

### LLM 조각 — 하나만 고릅니다

| 조각 | 학습 | LLM | 자격 증명 |
| --- | --- | --- | --- |
| `--no-llm` | 실제로 함 | 규칙 기반 폴백 | 불필요 |
| `--dry-run` | 모킹 | 모킹 | 불필요 |
| (생략) | 실제로 함 | 실제 호출 | 필요 |

### 예산 조각

`--max-iterations 3` · `--time-budget-sec 3600` · `--seed 42` · `--force`(기존 `thread-id` 덮어쓰기)

디스크도 예산입니다. `--keep-models all`은 모든 iteration의 `model.joblib`을 남기고,
기본값 `best`는 `predict`가 실제로 쓰는 최고 시도의 것만 남깁니다 — 적합된 모델 하나가
하이퍼파라미터에 따라 수백 MB가 되고(이 저장소에서 측정된 최댓값 486 MB) 루프에는 그것을
막는 것이 없기 때문입니다. 지운 목록은 콘솔과 `history.json`의 `models`에 남습니다.

---

## 3. 목표 미리보기 — 실행을 쓰지 않고 바만 확인

```powershell
python -m automl_agent.main profile `
  --data local\demo.csv --target died `
  --metric roc_auc `
  --out local\my_card.json
```

```
  rows=8000  features=12  dropped_non_numeric=0
  target=died  n_classes=2  balance=[0.8745, 0.1255]  imbalance_ratio=6.97
  split: train 60% / val 20% / test 20% (stratified, seed=42) — test는 반복 밖에서 저장된 모델로 1회만 채점합니다
  features: 수치 12개 + 범주형 0개를 one-hot 0열로
  기준선(logreg (median impute + standard scale), 8000행):
    f1=...  roc_auc=0.7444(chance 0.5)  ...
    95% CI (행 단위 부트스트랩 400회): f1=...  roc_auc=0.6994~0.7882  ...

카드를 저장했습니다: local\my_card.json
이 카드로 roc_auc 를 목표로 실행하면:
  auto 모드 — roc_auc 0.8083 이상 ← 기준선 0.7444 + 남은 여유의 25% (chance 0.5)
  fixed 모드 — roc_auc 0.9 이상 (지표 기본값)
```

`95% CI` 줄은 그 위 점수를 **같은 분할에서 400번 재표집**했을 때의 폭입니다. 바가 이
구간 안에 들어앉으면 `auto 모드` 줄이 그렇다고 말하면서 더 정직한 `--margin`을 가리킵니다
— 반복을 하나도 쓰지 않고 알 수 있는 것이라 여기서 확인하는 편이 쌉니다
([README의 부트스트랩 신뢰구간](README.md#다시-뽑으면-얼마나-흔들리는가--부트스트랩-신뢰구간)).
`--group-column`을 준 카드에서는 `그룹 단위 부트스트랩`으로 바뀌고 폭이 넓어집니다.

`split` 줄과 `features` 줄이 기준선 점수 **위**에 있는 이유는 그 점수가 그 분할과 그
행렬에 대한 값이기 때문입니다. 기준선은 train 60%에 적합해 val 20%에서 잰 것이고,
test 20%는 프로파일링에서도 읽지 않습니다 —
[README의 채점 프로토콜](README.md#채점-프로토콜--3분할과-반복-밖-1회-채점) 참조.

margin을 여러 개 훑어보기:

```powershell
foreach ($m in 0.25, 0.4, 0.5) {
  python -m automl_agent.main profile `
    --data local\demo.csv --target died `
    --metric roc_auc --margin $m --out local\my_card.json | Select-String "auto 모드"
}
```

고정값의 함정을 보려면 지표를 바꿔 봅니다:

```powershell
python -m automl_agent.main profile `
  --data local\demo.csv --target died `
  --metric accuracy --out local\my_card.json | Select-String "모드"
```

`fixed` 기본값 0.85가 chance 수준보다 **낮게** 나옵니다 — 다수 클래스만 찍어도 통과합니다.

---

## 4. auto 모드 실행

```powershell
python -m automl_agent.main run `
  --dataset-card local\my_card.json `
  --metric roc_auc `
  --max-iterations 3 --no-llm --thread-id t-auto
```

```
목표: auto 모드 — roc_auc 0.8083 이상 ← 기준선 0.7444 + 남은 여유의 25% (chance 0.5)
[iter 1] 계획: baseline: tabular 데이터에 대한 검증된 기본값(hist_gbdt)으로 기준선을 만든다
[iter 1] model=hist_gbdt roc_auc=0.8545 (goal 0.8083) [목표 달성]
  [holdout] 최종 테스트(20%, 반복 중 한 번도 쓰이지 않은 행): iteration 1의 hist_gbdt → roc_auc=0.8427 (검증 0.8545 대비 +0.0119 — 이 차이가 선택 편향의 크기입니다)
종료 사유: 목표 지표 달성
최고 성능: roc_auc=0.8545 (iteration 1, model=hist_gbdt)
```

마지막 `[holdout]` 줄이 루프 밖에서 1회만 나오는 숫자입니다. 0.8545는 목표 달성 판정에
쓴 검증 점수이고, 0.8427은 그 판정에 쓰이지 않은 test 20%의 점수입니다.

---

## 5. 같은 데이터, fixed 0.90

```powershell
python -m automl_agent.main run `
  --dataset-card local\my_card.json `
  --metric roc_auc --goal-mode fixed --threshold 0.90 `
  --max-iterations 3 --no-llm --thread-id t-fixed
```

```
목표: fixed 모드 — roc_auc 0.9 이상 (직접 지정)
[iter 1] model=hist_gbdt roc_auc=0.8545 (goal 0.9) → critic: hyperparam
          방향: 계열은 유지하고 learning_rate와 깊이 조합을 다르게 탐색한다.
[iter 2] 계획: 하이퍼파라미터 재탐색: learning_rate와 깊이 조합을 이전과 다른 지점에서 시도한다
[iter 2] model=hist_gbdt roc_auc=0.8648 (goal 0.9) → critic: hyperparam
[iter 3] 계획: 하이퍼파라미터 재탐색: learning_rate와 깊이 조합을 이전과 다른 지점에서 시도한다
[iter 3] model=hist_gbdt roc_auc=0.8565 (goal 0.9)
  [holdout] 최종 테스트(20%, 반복 중 한 번도 쓰이지 않은 행): iteration 2의 hist_gbdt → roc_auc=0.8412 (검증 0.8648 대비 +0.0236 — 이 차이가 선택 편향의 크기입니다)
종료 사유: 최대 반복 횟수 도달
최고 성능: roc_auc=0.8648 (iteration 2, model=hist_gbdt)
```

4번과 5번은 **같은 데이터·같은 모델·같은 점수**인데 종료 사유가 다릅니다. 두 모드를
나눈 이유가 이 두 출력의 차이입니다. 5번에서는 반복 루프도 실제로 돕니다 — 진단이
`hyperparam`으로 나오고 계획이 그에 따라 달라집니다.

**`selection_gap`도 두 배가 됩니다** (4번 +0.0119 → 5번 +0.0236). 시도 3개의 최댓값을
`best`로 고르는 쪽이 1개를 그대로 쓰는 쪽보다 더 위로 치우친다는 뜻이고, 검증 점수만
보고 있으면 이 폭은 보이지 않습니다.

---

## 6. auto를 더 밀어붙이기

```powershell
python -m automl_agent.main run `
  --dataset-card local\my_card.json `
  --metric roc_auc --margin 0.5 `
  --max-iterations 3 --no-llm --thread-id t-auto-05
```

0.8722를 요구하므로 5번처럼 반복이 돌게 됩니다. 고정값 0.90과 달리 이 숫자는 데이터셋을
바꾸면 따라 움직입니다.

---

## 7. 원본 CSV에서 바로 실행 (카드 없이)

`profiling` 노드가 그래프 안에서 카드를 만듭니다. 배너 시점에는 카드가 없으므로 목표는
`[profiling]` 줄에서 확정됩니다.

```powershell
python -m automl_agent.main run `
  --data local\demo.csv --target died `
  --metric roc_auc `
  --max-iterations 3 --no-llm --thread-id t-fromcsv
```

```
데이터셋 카드: profiling 노드가 local\demo.csv 에서 생성합니다
목표: auto 모드 — roc_auc 0.9 이상 (카드에 기준선이 없어 지표 기본값 사용)
  profiling이 기준선을 측정하면 다시 도출됩니다
최대 3회 반복
----------------------------------------------------------------------
  [profiling] 데이터셋 카드 생성 완료 → ...\t-fromcsv\dataset_card.json
  [profiling] 목표: auto 모드 — roc_auc 0.8083 이상 ← 기준선 0.7444 + 남은 여유의 25% (chance 0.5)
```

배너의 0.9는 아직 카드가 없어 나온 지표 기본값이고, 그 사실을 바로 아래 줄이 말합니다.
실제 목표는 기준선을 측정한 뒤 `[profiling]` 줄에서 확정됩니다.

---

## 8. 연속 정답 열 (회귀)

같은 CLI, 같은 그래프입니다. 바뀌는 것은 정답 열이고, task는 그 열에서 읽힙니다 —
`--task` 같은 플래그는 없습니다([README의 두 번째 모드](README.md#두-번째-모드--연속-정답-열-회귀)).

```powershell
# 회귀용 합성 데이터 (재원일수처럼 오른쪽으로 치우친 연속 열)
# 생성기는 개발 저장소에 있습니다 — 연속 정답 열이 있는 자기 CSV로 대체해도 됩니다.
python examples\make_demo_data.py --task regression
```

```
local\demo_reg.csv — 8000행, target 컬럼 'los_days' (일 단위, 중앙값 3.98, 최대 31.64 — 오른쪽으로 치우침)
```

```powershell
python -m automl_agent.main profile `
  --data local\demo_reg.csv --target los_days `
  --metric mae `
  --out local\demo_reg_card.json
```

```
dataset card: los_days-prediction (regression)
  target=los_days  kind=continuous  magnitude=unit  skew=moderate  outlier_rate=0.0382
  split: train 60% / val 20% / test 20% (not stratified, seed=42) — test는 반복 밖에서 저장된 모델로 1회만 채점합니다
  기준선(ridge (median impute + standard scale), 8000행):
    r2=0.8029(chance -0.0004)  mae=0.7803(chance 1.9137)  rmse=1.1466(chance 2.5833)
    95% CI (행 단위 부트스트랩 400회): r2=0.7845~0.8238  mae=0.7383~0.8219  rmse=1.0325~1.258

이 카드로 mae 를 목표로 실행하면:
  auto 모드 — mae 0.5852 이하 ← 기준선 0.7803에서 25% 감소 (chance 1.9137)
  fixed 모드 — mae 목표값 미정 (이 지표는 기본 바가 없습니다)
```

분류 카드와 다른 네 줄이 전부입니다. `n_classes`·`balance` 대신 정답 열의 **모양**
(`magnitude=unit` — 그래서 `mae` 0.78이 "0.78일"로 읽힙니다), 층화하지 않는 분할,
`ridge` 기준선, 그리고 `이하`입니다 — 바가 기준선에서 **25% 감소**로 나옵니다.
`fixed` 모드에 값이 없는 것도 정상입니다: 정답 열의 단위로 나오는 지표에 이식 가능한
기본 바는 없습니다.

```powershell
python -m automl_agent.main run `
  --dataset-card local\demo_reg_card.json `
  --metric mae --margin 0.45 `
  --max-iterations 3 --no-llm --thread-id t-reg
```

```
목표: auto 모드 — mae 0.4292 이하 ← 기준선 0.7803에서 45% 감소 (chance 1.9137)
[iter 1] 계획: baseline: tabular 데이터에 대한 검증된 기본값(hist_gbdt)으로 기준선을 만든다
[iter 1] model=hist_gbdt mae=0.5706 (goal 0.4292) → critic: overfitting
          방향: 정규화를 강화하고 용량을 줄인다: l2 증가, 깊이 축소, learning_rate 하향.
[iter 2] 계획: 과적합 대응: l2 정규화를 넣고 깊이와 learning_rate를 낮춘다
[iter 2] model=hist_gbdt mae=0.5697 (goal 0.4292) → critic: underfitting
          방향: 모델 용량과 학습량을 늘린다: 반복 수 증가, 트리 깊이·리프 수 확대, 정규화 완화.
[iter 3] model=hist_gbdt mae=0.5733 (goal 0.4292)
  [holdout] ... iteration 2의 hist_gbdt → mae=0.5956 (95% CI 0.5625~0.6343) (검증 0.5697 대비 -0.0258 ...)
종료 사유: 최대 반복 횟수 도달
최고 성능: mae=0.5697 (iteration 2, model=hist_gbdt)
```

`--margin 0.45`를 준 이유는 기본 25%(바 0.5852)가 **첫 시도에 달성**되어 Critic이 한 번도
돌지 않기 때문입니다. 그 두 진단이 회귀에서 어떻게 나오는지가 이 절의 요점입니다.

- **iteration 1 `overfitting`** — `train_mae=0.2950`, `mae=0.5706`, `train_val_gap=0.2756`.
  절대 0.15가 아니라 **학습 점수의 비율**로 판정합니다: 0.2950 × 0.25 = 0.0738을 넘습니다.
  검증 오차가 학습 오차의 거의 두 배라는 뜻이고, 이 비율은 달러든 일수든 같은 의미로
  옮겨집니다. 0.15는 안 됩니다.
- **iteration 2 `underfitting`** — `train_mae=0.4596`. 방향을 읽습니다: minimize에서는
  `train > 목표 × 1.05`(0.4292 × 1.05 = 0.4507)일 때, 즉 **학습 오차조차 바에 못 미칠 때**
  용량 부족입니다. 같은 시도의 gap 0.1101은 0.4596 × 0.25 = 0.1149보다 작아서 과적합
  분기를 통과했습니다 — 두 판정이 같은 숫자로 순서대로 걸러집니다.

`train_val_gap`이 양수인 것을 눈여겨보십시오. 오차 지표에서 단순한 `train - val`은
과적합일 때 정확히 **음수**가 되므로, 방향은 `train.py` 한 곳에서 적용되고 모든 소비자는
양수를 과적합으로 읽습니다.

`r2`로 바꾸면 분류와 완전히 같게 읽힙니다 — 1이 상한이라 "남은 여유의 25%"가 그대로
성립합니다:

```powershell
python -m automl_agent.main run `
  --dataset-card local\demo_reg_card.json `
  --metric r2 `
  --max-iterations 3 --no-llm --thread-id t-reg-r2
```

```
목표: auto 모드 — r2 0.8522 이상 ← 기준선 0.8029 + 남은 여유의 25% (chance -0.0004)
[iter 1] model=hist_gbdt r2=0.8914 (goal 0.8522) [목표 달성]
  [holdout] ... iteration 1의 hist_gbdt → r2=0.8953 (95% CI 0.8824~0.9078) (검증 0.8914 대비 -0.0039 ...)
```

`chance -0.0004`는 오타가 아닙니다 — 평균만 예측하는 `DummyRegressor`의 `r2`이고, 정의상
0 근처입니다. 음수도 나올 수 있습니다(평균 예측보다 나쁨).

---

## 9. 거부되는 조합

```powershell
# auto + --threshold
python -m automl_agent.main run --dataset-card local\my_card.json --goal-mode auto --threshold 0.9 --thread-id t-x

# fixed + --margin
python -m automl_agent.main run --dataset-card local\my_card.json --goal-mode fixed --margin 0.5 --thread-id t-x

# 알 수 없는 모드
python -m automl_agent.main run --dataset-card local\my_card.json --goal-mode clever --thread-id t-x

# 이미 쓴 thread-id (--force 없이) — run_config.json은 그대로 남습니다
python -m automl_agent.main run --dataset-card local\my_card.json --thread-id t-auto

# 없는 지표 이름 (argparse가 바로 거부)
python -m automl_agent.main run --dataset-card local\my_card.json --metric auroc --thread-id t-x

# 바가 없는 지표를 fixed 모드로 (mae·rmse는 지표별 기본값이 없습니다)
python -m automl_agent.main run --dataset-card local\my_reg_card.json --goal-mode fixed --metric mae --thread-id t-x

# 실행을 무의미하게 만드는 값
python -m automl_agent.main run --dataset-card local\my_card.json --max-iterations -1 --thread-id t-x
```

```
오류: auto 모드는 카드의 기준선에서 임계값을 도출하므로 --threshold 와 함께 쓸 수 없습니다.
  - 값을 못박으려면: --goal-mode fixed --threshold 0.9
  - 요구 수준만 조절하려면: --margin 0.5
```

조용히 무시된 플래그가 최악이라서, 모순되는 조합은 실행 전에 종료합니다.

반면 **task가 어긋난 지표는 끊지 않고 바꿉니다.** 오타 하나로 긴 실행이 죽는 것보다, 그
task의 기본 지표(분류 `f1`, 회귀 `r2`)로 돌리면서 바꿨다고 말하는 편이 낫다는 판단입니다:

```powershell
# 분류 카드에 회귀 지표 (그 반대도 같음) — 거부가 아니라 f1로 바꿔 실행됩니다
python -m automl_agent.main run --dataset-card local\my_clf_card.json --metric r2 --thread-id t-x
```

카드가 있으면 실행 시작 전에, `--data`로 주면 프로파일링 직후에 나옵니다:

```
경고: 이 데이터의 task는 binary_classification인데 목표 지표 r2는 다른 task의 지표입니다 — 이 정답 열에서는 계산되지 않으므로, 그대로 두면 모든 시도가 '목표 미달'로 기록되고 반복 예산만 소모됩니다. f1로 바꿔 실행합니다
목표: auto 모드 — f1 0.85 이상 (요청한 지표 r2는 이 task의 지표가 아니라 대체됨) ← ...
```

**괄호가 실행 내내 따라옵니다** — 콘솔 헤더, Critic 프롬프트, `report.md`가 같은 한 줄을
읽으므로, 나중에 보고서만 본 사람도 이 실행이 요청과 다른 지표로 채점됐다는 것을 압니다.
`run_config.json`에는 입력한 `r2`가 그대로 남습니다(물어본 것과 채점된 것을 구분하려고).

직접 준 `--threshold`는 **같이 버립니다.** 3.5는 `mae`의 단위로 준 오차라서 `f1`의 바로 쓰면
아무 모델도 닿지 못하는 목표가 되고, 그것은 대체가 막으려던 결과 그 자체입니다:

```
경고: ... f1로 바꿔 실행합니다. --threshold 3.5는 mae의 단위로 준 값이라 f1의 바로 쓸 수 없어 버립니다 — f1 기준으로 다시 지정하려면 --metric f1 --threshold <값>
```

`mae`·`rmse`를 `fixed`로 쓰면서 숫자를 주지 않은 경우도 같은 자리에서 끊습니다 — 바가
`None`이면 `goal_met`은 구조적으로 영원히 False입니다:

```
오류: mae는 정답 열의 단위로 나오는 지표라서 이식 가능한 기본 목표값이 없습니다 — 0.85 같은 숫자를 넣으면 모델이 아니라 그 열의 단위에 대한 바가 됩니다.
  - 요구 수준을 알고 있다면: --goal-mode fixed --threshold <mae 값>
  - 데이터에서 도출하려면: 기준선이 측정된 카드로 auto 모드를 쓰십시오 (`profile`을 --no-baseline 없이 실행)
```

---

## 10. 카드 없이 루프만 돌려 보기 (`--dry-run`)

학습과 LLM을 모두 모킹하므로 즉시 끝납니다. 저장소에 포함된 손으로 쓴 카드를 씁니다.

```powershell
python -m automl_agent.main run --dataset-card examples\dataset_card.json --dry-run --thread-id t-dry-success
python -m automl_agent.main run --dataset-card examples\dataset_card_hard.json --dry-run --scenario fail --thread-id t-dry-fail
python -m automl_agent.main run --dataset-card examples\dataset_card_oom.json --dry-run --scenario oom --thread-id t-dry-oom
python -m automl_agent.main run --dataset-card examples\dataset_card_crash.json --dry-run --scenario crash --thread-id t-dry-crash
```

이 카드들에는 `baseline`이 없어서 `auto`가 지표별 기본값으로 폴백하고, 그렇다고
경고합니다. 실제 도출을 보려면 1~4번 경로를 쓰세요.

---

## 11. LLM으로 실행

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

자격 증명이 없으면 **시작 전에** 안내와 함께 종료합니다. 프롬프트 전량은
`artifacts\t-llm\llm\` 에 남으므로, 원본 데이터가 실리지 않았음을 직접 확인할 수 있습니다.

```powershell
$files = Get-ChildItem artifacts\t-llm\llm\* -File
"검사 대상 $($files.Count)개"                    # 0개면 경로가 틀린 것
$files | Select-String -Pattern "demo.csv"       # 출력 없음이 정상
```

확장자를 걸지 마세요. 실제 실행은 `.json`, `--dry-run`은 `.dryrun.md`로 남으므로
`*.md`로 좁히면 실제 실행에서 0개 파일을 검사하고도 "결과 없음"이 나옵니다.

---

## 12. 중단·재개·조회

```powershell
# 재개 (다른 플래그 불필요 — run_config.json에서 복원)
python -m automl_agent.main resume --thread-id t-auto

# 상태 + 보고서 전문
python -m automl_agent.main show --thread-id t-auto --report

# 보고서만
Get-Content artifacts\t-auto\report.md

# 최종 test 채점만 (실행당 1회)
Get-Content artifacts\t-auto\holdout.json

# 산출물 목록
Get-ChildItem artifacts\t-auto -Recurse -Name

# 그래프 구조
python -m automl_agent.main graph --out graph.png
```

---

## 13. 학습한 모델을 새 데이터에 적용

끝난 실행이 고른 모델을 새 행에 씁니다. iteration은 알아서 찾습니다 — 체크포인트를
보고, 없으면 `history.json`의 `best`를 봅니다.

```powershell
# 예측할 새 파일 (여기서는 다른 시드로 만든 합성 데이터 500행 — 생성기는 개발 저장소)
python examples\make_demo_data.py --rows 500 --seed 7 --out local\demo_new.csv

# 적용
python -m automl_agent.main predict `
  --thread-id t-auto `
  --data local\demo_new.csv `
  --out local\demo_new_predictions.csv
```

```
이 실행이 선택한 iteration 3의 모델을 사용합니다
500행을 예측해 local\demo_new_predictions.csv에 저장했습니다 (인코딩된 피처 12개, task=classification)
피처가 아닌 컬럼은 제외했습니다: died
학습 때의 인코딩과 어긋난 곳은 없었습니다
실행 요약: local\demo_new_predictions.report.json
출력 파일은 행 단위 예측이므로 원본 데이터와 같은 취급을 하십시오 — 커밋하지 말고, 프롬프트에 넣지 마십시오.
```

`died`가 제외된 이유는 이 합성 파일에 정답 열이 아직 붙어 있기 때문입니다. **정답 열이
있는 파일을 줘도 됩니다** — 스키마가 적은 이름으로 특성에서 빼고 버립니다. 그 열 이름을
`--label-column`으로 주면 [채점까지 합니다](#131-라벨이-있는-배치-채점하기----label-column).

`--out`을 생략하면 `artifacts\t-auto\predict\<입력파일이름>_predictions.csv`에 씁니다.
실행 요약 JSON(행 수, 어긋난 곳, 두 아티팩트 경로)은 **항상** `--out` 옆에 함께 나오고,
`--report`는 그 위치만 바꿉니다. 어느 쪽이든 **행 단위 데이터**이므로 `local\` 또는
`artifacts\` 안에 두세요(둘 다 git이 무시합니다).

```powershell
# 결과를 원래 행과 키로 붙일 수 있게 식별자 열을 함께 냅니다
# (합성 demo.csv에는 식별자 열이 없습니다 — 자기 파일의 키 컬럼 이름을 주세요)
python -m automl_agent.main predict `
  --thread-id t-auto --data local\my_new_rows.csv `
  --id-column patient_id --out local\pred.csv

# 실행 요약 JSON을 다른 곳에 두고 싶을 때
python -m automl_agent.main predict `
  --thread-id t-auto --data local\demo_new.csv `
  --out local\pred.csv --report local\pred_summary.json

# 최고 시도가 아닌 특정 iteration을 쓰고 싶을 때 (이때는 위의 iteration 안내 줄이 없습니다)
python -m automl_agent.main predict `
  --thread-id t-auto --data local\demo_new.csv --iteration 2 --out local\pred_i2.csv

# 고정 스크립트를 직접 (두 경로를 손으로 짝지어야 합니다)
python -m automl_agent.scripts.predict `
  --model artifacts\t-auto\train\iter_03\model.joblib `
  --schema artifacts\t-auto\train\iter_03\feature_schema.json `
  --data local\demo_new.csv --out local\pred.csv
```

출력 CSV는 `--id-column`(주었다면), `prediction`, 그리고 분류라면 `proba_<라벨>`
열입니다. 입력 열은 다시 쓰지 않습니다 — 이미 가진 파일을 한 부 더 복사하는 일이라서.

```
prediction,proba_0,proba_1
0,0.9134,0.0866
```

**`prediction`은 학습 파일이 쓴 값 그대로입니다.** 위는 `died`가 0/1 정수인 합성
데이터이므로 0/1로 나오고, 라벨이 `died`/`survived` 문자열이었다면 그 단어와
`proba_died`/`proba_survived`가 나옵니다. 확률 열 이름도 코드가 아니라 라벨을 씁니다.
회귀면 `prediction` 하나뿐이고 실수입니다(확률 열은 만들지 않습니다).

### 13.1 라벨이 있는 배치 채점하기 — `--label-column`

지난달 마감된 배치처럼 **정답이 이미 붙어 있는 파일**이면 그 열 이름을 주십시오. 예측은
플래그 없을 때와 한 행도 다르지 않게 나오고, 거기에 점수가 붙습니다.

```powershell
python -m automl_agent.main predict `
  --thread-id t-auto `
  --data local\demo_new.csv `
  --label-column died `
  --out local\demo_new_scored.csv
```

아래는 실제 출력입니다(900행으로 학습한 모델, 뒤쪽 400행을 배치로 준 것 — 라벨 5행을
비우고 3행을 학습 때 없던 값으로 바꿔 두었습니다). 긴 줄은 읽기 좋게 접었습니다.

```
이 실행이 선택한 iteration 3의 모델을 사용합니다
400행을 예측해 local\demo_new_scored.csv에 저장했습니다 (인코딩된 피처 5개, task=classification)
피처가 아닌 컬럼은 제외했습니다: outcome
이 배치를 채점했습니다 — 라벨 컬럼 'outcome', 392행
  채점에서 제외: 라벨이 빈 행 5개, 학습 때 없던 라벨 값을 가진 행 3개
  f1=0.8165 (95% 구간 0.7634~0.8602, row 단위 재표집 400회) ← 이 실행이 목표로 삼았던 지표
  그 밖의 지표: accuracy=0.8750, balanced_accuracy=0.8673,
    balanced_accuracy_at_best_cut=0.8812, cut_headroom=0.0139, pr_auc=0.9008,
    precision=0.7899, recall=0.8450, roc_auc=0.9453, specificity=0.8897
  확률 품질(진단, 목표로 삼을 수 없음): brier=0.0883, 확률 오차=0.0343 — 예측 확률과 실제
    발생률의 차이가 평균 3.4%p입니다 (10개 구간, 개수 가중)
  확률 구간별 실제 발생률 (10구간, 빈 구간 제외):
    0.0~0.1: 166행, 예측 평균 0.017, 실제 0.024
    0.1~0.2: 30행, 예측 평균 0.153, 실제 0.067
    0.2~0.3: 28행, 예측 평균 0.249, 실제 0.179
    0.3~0.4: 19행, 예측 평균 0.358, 실제 0.210
    (중략)
    0.9~1.0: 60행, 예측 평균 0.960, 실제 0.967
  이 점수는 이 실행의 채점 프로토콜이 아닙니다 — result.json의 홀드아웃은 학습 전에 떼어 둔
    행을 한 번만 채점한 값이지만, 이 파일이 어떤 행으로 이루어졌는지는 여기서 알 수 없습니다.
    낮게 나오는 것이 정상일 수도 있고, 위의 '확인할 점'이 그 이유일 수도 있습니다. 이 행들이
    한 대상에서 여러 번 나온 것이라면 위 신뢰구간은 실제보다 좁습니다.
학습 때의 인코딩과 어긋난 곳은 없었습니다
```

위 표에서 0.1~0.4 구간의 `실제`가 `예측 평균`보다 일관되게 낮은 것이 **과신**의 모양입니다 —
그 구간에서는 모델이 말하는 것보다 실제로 덜 일어납니다. 전체로 뭉치면 3.4%p이고, 어디서
어긋나는지는 이 표만 말해 줍니다.

**마지막 문단이 이 기능의 핵심입니다.** 홀드아웃 점수와 이 점수를 나란히 놓고 싶어지는데,
그 비교가 뜻을 가지려면 이 파일이 어떤 행으로 이루어졌는지를 **당신이** 알아야 합니다 —
스크립트는 모릅니다. 그리고 채점은 홀드아웃을 채점한 것과 같은 `evaluate_split()`을 지나고,
지표는 실행이 목표로 삼았던 것(스키마의 `metric`)을 씁니다. 그래서 다른 것은 지표가 아니라
행입니다.

읽는 법:

| 줄 | 무엇을 보는가 |
| --- | --- |
| `← 이 실행이 목표로 삼았던 지표` | 실행이 최적화한 그 숫자입니다. 괄호의 95% 구간이 이 행 수에서의 폭 — 홀드아웃과 겹치면 "이 행들로는 차이를 말할 수 없다"가 정답입니다 |
| `확률 품질` | 랭킹이 아니라 **확률**이 쓸 만한지. 0.039면 "평균 3.9%p 어긋남". 임계값을 직접 골라 쓸 거라면 이 숫자가 `f1`보다 중요합니다 |
| `확률 구간별 실제 발생률` | 어느 구간에서 어긋나는지. 높은 확률 구간에서만 실제가 낮으면 과신입니다. **콘솔에만** 찍고 프롬프트에는 넣지 않습니다(구간마다 행 수가 들어갑니다) |
| `채점에서 제외` | 라벨이 빈 행과 **학습 때 없던 라벨 값**을 가진 행을 따로 셉니다. 뒤쪽이 0이 아니면 정답 열의 규약이 바뀐 것이고, 원인이 다른 문제라 따로 셉니다 |
| `이 배치에는 목표 지표가 없습니다` | 이 줄이 나오면 **신뢰구간도 없습니다.** 두 원인이 따로 적힙니다 — 스키마에 지표가 아예 없거나(지표를 대체해 기록하기 전에 만들어진 옛 스키마), 지표는 적혀 있는데 이 행들로 계산이 안 되는 경우(한 클래스만 있는 배치의 `roc_auc`) |

50행 미만이면 `확률 오차`도, 구간별 표도 나오지 않습니다:

```
  확률 품질(진단, 목표로 삼을 수 없음): brier=0.0966 — 행이 30개뿐이라 구간별 확률 오차는
    측정하지 않았습니다 (50행 이상 필요)
```

10구간에 40행이면 한 구간이 4행이라 실제 발생률이 0·0.25·0.5·0.75·1 중 하나밖에 될 수
없어서, 잘 맞는 모델도 몇 %p 어긋난 것으로 측정되기 때문입니다. 요약값을 보류하면서 그
보류의 근거인 구간 표를 찍는 것은 양쪽을 다 취하는 것이라, 표도 같은 문턱에서 함께
멈춥니다. `brier`는 그 편향이 없어 20행에서도 나옵니다 — **없음은 이상 없음이 아니라 측정하지
않았다는 뜻입니다.**

목표 지표를 쓸 수 없을 때도 같은 규칙입니다 — 줄이 사라지는 대신 없는 이유가 나옵니다:

```
  이 배치에는 목표 지표가 없습니다 — 스키마가 적은 지표(roc_auc)는 이 행들에서 계산할 수
    없었습니다. 아래 지표는 모두 같은 채점에서 나온 값이지만, 어느 것이 이 실행이 목표로
    삼았던 숫자인지는 여기서 알 수 없고 신뢰구간도 없습니다
```

헤드라인이 **유일한 신뢰구간을 함께 들고 갑니다.** 그래서 이 줄 없이 조용히 빠지면, 남은
지표 목록이 "그중 하나가 목표인 평범한 점수표"로 읽히는데 어느 것이 목표인지는 알 방법이
없습니다. 위 예시(`roc_auc`)는 지표 이름은 적혀 있고 이 행들에서 계산만 안 된 경우입니다.
이름이 아예 없는 쪽은 `train.py`가 지표를 대체해 스키마에 적기 전에 만들어진 모델에서만
나오고(그때는 task에 없는 지표를 `null`로 적었습니다), 그 모델을 다시 학습하면 사라집니다 —
그래서 이 줄은 "옛 모델"과 "이 배치로는 못 잰다"를 구분해 줍니다.

```powershell
# 정답 열 이름이 스키마의 정답 열과 다를 때도 됩니다 (특성으로 쓰이지 않고, 드리프트로도 안 셉니다)
python -m automl_agent.main predict `
  --thread-id t-auto --data local\last_month.csv `
  --label-column outcome_actual --id-column patient_id --out local\last_month_scored.csv
```

채점 결과는 `--report` JSON의 `score` 아래에도 들어갑니다(`batch_metrics`, `scored_rows`,
`excluded_rows`, `reliability`). 행 단위 데이터와 같은 등급으로 다루십시오.

**재캘리브레이션은 하지 않습니다.** 확률이 어긋난 것을 알려 주기만 하고 고치지는 않습니다 —
확률을 다시 맞추는 것은 라벨 있는 행에 무언가를 적합하는 모델 변경이고, 이 스크립트는 당신의
백테스트 파일에 그럴 권한이 없습니다.

### 어긋나면 어떻게 되는가

`model.joblib` 하나만으로는 새 파일에 적용할 수 없습니다. 인코딩(어떤 level이 몇 번째
열인지)이 그 파일에 없기 때문입니다. 그래서 학습이 옆에 `feature_schema.json`을 남기고,
`predict`는 그 짝을 요구합니다. 자세한 이유와 측정치는
[README](README.md#학습한-모델을-새-데이터에-쓰기--predict).

```powershell
# 모델이 필요한 열이 없는 파일 → 거부 (없는 열 이름을 한 번에 전부 알려 줍니다)
python -m automl_agent.main predict --thread-id t-auto --data local\few_columns.csv --out local\pred.csv

# --dry-run 으로 돌린 실행 → 저장된 모델이 없습니다
python -m automl_agent.main predict --thread-id t-dry --data local\demo_new.csv --out local\pred.csv

# 스키마가 저장되기 전 버전의 실행, 또는 합성 데이터 실행 → 다시 유도하지 않고 거부
python -m automl_agent.main predict --thread-id t-old --data local\demo_new.csv --out local\pred.csv
```

반대로 **막지 않고 알리기만** 하는 것들은 예측이 정상적으로 나오고 아래처럼 출력됩니다
(범주형 열이 있는 데이터의 예 — 합성 demo.csv는 전부 수치라 이 줄들이 나오지 않습니다).
전부 "학습 때 없던 것이 새 파일에 있다"는 신호입니다.

```
확인할 점 3건 — 예측은 나왔지만 아래를 읽으십시오:
  - 수치 컬럼 'age'의 41행이 숫자로 읽히지 않아 결측으로 처리했습니다
  - 'city'에 학습 때 없던 범주 2개(jeju, sejong) — 12행이 이 컬럼에서 전부 0으로 인코딩됩니다
  - 'grade'의 3행이 결측인데 학습 데이터에는 결측이 없어 결측 전용 열이 없습니다 — 이 행들도 전부 0입니다
```

`age`처럼 개수가 크면 열 전체가 텍스트로 export된 경우일 수 있습니다 — 한두 셀이면
원본의 오타이고, 파일 행 수와 비슷하면 파일 자체를 보세요. 같은 블록에 "학습 파일에
없던 컬럼"과 "학습 때도 제외된 컬럼"도 함께 나옵니다 — 후자는 어긋남이 아니라 **모델이
그 열을 쓰지 않는다**는 상시 사실이라, 이 블록의 제목이 "경고"가 아닌 이유입니다.

같은 블록에 **폭 검사로는 절대 보이지 않는** 세 가지도 나옵니다. 레이아웃이 맞는 것과
배치가 비교 가능한 것은 다른 이야기입니다.

```
확인할 점 3건 — 예측은 나왔지만 아래를 읽으십시오:
  - 'age'의 결측률이 학습 때 2.1%에서 이 배치 11.4%로 바뀌었습니다 — 인코딩은 맞지만 모델이
    보는 값의 출처가 달라졌습니다 (결측 대치를 쓰는 파이프라인이면 그만큼이 학습 때의 대치값입니다)
  - 'creatinine'에 결측 코드로 의심되는 -9999이 학습 때 0.0%에서 이 배치 3.2%로 늘었습니다 —
    이 값은 결측이 아니라 숫자 -9999로 모델에 들어갑니다
  - 학습 때와 라이브러리 버전이 다릅니다 (scikit-learn 1.5.2 → 1.7.0) — 저장된 모델은 pickle이라
    다른 버전에서 열면 예측이 조용히 달라질 수 있습니다. 같은 버전으로 맞추거나, 학습 때의
    홀드아웃 점수를 이 배치에서 다시 확인하십시오
```

| 줄 | 무엇을 의미하는가 |
| --- | --- |
| 라이브러리 버전 | 대개 아무 일도 없습니다. 거부하지 않는 이유가 그것이고, **조용한 실패의 후보 목록**으로 남기는 이유는 sklearn이 pickle 호환을 보장하지 않기 때문입니다. `numpy`·`pandas`·`scikit-learn`·`joblib`은 정확히, python은 major.minor까지 비교합니다 |
| 결측률 변화 | 문턱은 5%p **그리고** 그 배치 행 수에서의 2×표준오차입니다. 10행짜리 배치도 조용히 건너뛰지 않고 비교하되, 잡음으로 줄을 만들지 않기 위해서입니다 |
| 결측 코드(`-9999`) 비율 | 문턱이 1%p로 훨씬 낮습니다 — 결측률이 움직이는 것은 일부 평범한 일이지만, **결측 코드의 비중이 움직이는 것은 규약이 바뀌었다는 신호**입니다. 적합 때 기록된 코드는 값을 그대로 세고(짧은 배치에 "규약이 바뀌었다"고 말하지 않기 위해), 기록에 없던 코드만 새로 감지합니다 |

**옛 스키마로 만든 모델도 그대로 씁니다.** 검사가 추가되기 전(`version: 1`)에 학습한
모델이면 거부하지 않고, 대신 하지 못한 검사를 이름으로 적습니다:

```
  - 이 스키마는 version 1이라 라이브러리 버전 대조, 컬럼 결측률·결측 코드 대조를 하지
    못했습니다 (현재 version 2) — 위에 없는 항목은 이상이 없다는 뜻이 아니라 검사하지
    않았다는 뜻입니다. 같은 데이터로 다시 학습하면 켜집니다
```

거부는 **더 높은** version에만 남습니다 — 어제 학습한 모델을 오늘 못 쓰게 만드는 쪽이 더
나쁜 실패입니다.

---

## 14. 자주 걸리는 것

| 증상 | 원인 / 대응 |
| --- | --- |
| `No module named 'automl_agent'` | 저장소 루트에서 실행하거나 `pip install -e .` |
| 이 파일 커밋해도 되나 | `local\` 안에 있으면 안 됩니다 — 그 디렉터리는 통째로 무시됩니다. `git status --short --ignored` 로 확인 |
| `이미 체크포인트가 있습니다` | `--thread-id`를 새로 잡거나 `--force`. 재사용하면 두 실행의 `history`가 섞입니다 |
| 한글이 깨져 보임 | `$env:PYTHONIOENCODING = "utf-8"`. subprocess 쪽은 코드에서 고정했지만 콘솔은 별개입니다 |
| `Anthropic API 자격 증명이 없어...` | `--no-llm` 또는 `--dry-run`을 붙이거나 11번의 환경변수를 설정 |
| `auto` 모드인데 목표가 지표 기본값 | 카드에 `baseline` 블록이 없습니다. `--no-baseline` 없이 `profile`을 다시 실행하세요 |
| `… f1로 바꿔 실행합니다` | 준 지표가 이 정답 열의 task에 없어서 그 task의 기본 지표로 바뀐 것입니다 — 거부가 아니라 경고이고, 실행은 계속됩니다. 원래 지표로 재려면 정답 열을 보십시오(task는 고르는 값이 아니라 열에서 읽힙니다). 함께 준 `--threshold`는 버려집니다 (9번) |
| `mae는 정답 열의 단위로 나오는 지표라서 …` | `mae`·`rmse`에는 지표별 기본 바가 없습니다. `--goal-mode fixed --threshold <값>`을 주거나, 기준선이 측정된 카드로 `auto`를 쓰세요 |
| 라벨이 `0.0`/`1.0` float인데 회귀로 잡힐까 | 아닙니다 — 고유값이 2개면 dtype과 무관하게 분류입니다. 반대로 반올림된 3값 점수는 분류로 남습니다(그 열이 다른 말을 하지 않으므로). 판정 순서는 [targets.py](automl_agent/dataset/targets.py)의 `detect_task` |
| 정답 열에 값이 하나뿐 | task가 아니라 거부입니다(`TargetUnusableError`). 상수 정답에는 배울 것이 없습니다 — 컬럼 이름이 맞는지, 행이 한 결과만 남게 필터되지 않았는지 보세요 |
| 목표에 `(도달 불가)`가 붙음 | chance가 상한 0.99를 밀어냈습니다 — 이 지표로는 이 데이터에서 모델과 다수 클래스를 구분할 수 없습니다. `--metric balanced_accuracy` 또는 `--metric pr_auc` |
| 바를 못 넘고 `stalled`로 끝남 | 지표의 상한이 아니라 **이 데이터의 랭킹 품질**이 상한일 수 있습니다. `roc_auc`는 그대로인데 `balanced_accuracy`만 낮으면 운영점 문제이고, 그 경우 임계값 튜닝으로도 거의 오르지 않습니다 (m-llm9에서 측정: 홀드아웃 오라클 임계값조차 +0.0024). `--margin`을 낮추거나 `--metric roc_auc`로 재세요 |
| `has N missing values` 로 멈춤 | 정답 컬럼이 빈 행이 있습니다. 라벨을 고치거나 `--on-missing-target drop` |
| `카드에 허용되지 않은 키가 있습니다` | 손으로 쓴 카드에 모르는 최상위 키가 있습니다. 카드는 집계 요약이고, 모르는 키는 통과시키지 않습니다 (메시지가 허용 목록을 함께 출력합니다) |
| `recall`이 `specificity`보다 훨씬 낮음 (또는 반대) | `balanced_accuracy`는 두 값의 평균이라 **둘이 같아지는 지점이 최적**입니다. 차이가 0.08을 넘으면 규칙 기반 진단이 `hyperparam`으로 판정하고 낮은 쪽을 인용해 양성 가중치를 ×1.5 또는 ÷1.5 하라고 처방합니다 — 직접 돌릴 때도 같은 방향입니다. train/val 격차는 용량에 대한 증거이므로 방향의 근거가 아닙니다 |
| 가중치를 올렸다 내렸다 하며 제자리 | 부호가 한 번 뒤집혔으면 그 두 점이 최적을 감싼 것이니 **사이를 재세요**. 진단도 그렇게 합니다 — 같은 모델의 이전 시도에서 부호가 다른 짝을 찾으면 한 스텝 대신 보간합니다 (m-llm8: 12에서 +0.049, 8에서 −0.118 → 10.5가 최적). 모델을 바꾸면 짝은 무효입니다 |
| 차이가 0.08 아래인데 여전히 바를 못 넘음 | 가중치에는 남은 게 거의 없습니다. 최적점 근처는 평평해서, 차이 0.049를 0에 맞춰도 `balanced_accuracy`는 +0.0004였습니다 (m-llm8). 모델 family나 데이터를 보세요 |
| 운영점을 더 밀어볼 가치가 있나 | `result.json`의 `cut_headroom`을 보세요. `balanced_accuracy_at_best_cut`(= 이 랭킹의 `(1 + KS)/2` 상한) 에서 실제 점수를 뺀 값이고, **임계값을 골랐다면 얻었을 정확한 양**입니다. MIMIC 표본에서는 +0.0024 — 즉 남은 격차는 컷이 아니라 랭킹에 있습니다. 도달한 점수로 인용하면 안 됩니다: 실행기는 항상 `predict()`를 부릅니다 |
| 목표 줄에 `랭킹 … 상한을 넘습니다`가 붙음 | 지표의 상한이 아니라 **기준선 랭킹**의 상한을 넘은 바입니다 (카드의 `baseline.ks`에서 계산). 이건 기준선의 한계라 실행의 한계보다 낮습니다 — 더 좋은 모델이 넘을 수 있고, 실제로 m-llm7이 그렇게 통과했습니다. 그래서 바는 낮춰지지 않고 그대로 갑니다: 넘기려면 랭킹을 올리세요(모델 family나 특성 — 하이퍼파라미터로는 m-llm8의 11개 시도 동안 움직이지 않았습니다). 바를 낮추기로 하면 같은 줄이 알려 주는 `--margin` 숫자를 쓰세요 |
| 목표 줄에 `이 바는 기준선 자체의 95% CI … 안에 있습니다`가 붙음 | 바와 기준선의 차이가 **val 슬라이스가 분해할 수 있는 폭보다 작습니다**. 넘겨도 넘겼는지 이 행들로는 알 수 없다는 뜻이라, 같은 줄이 권하는 대로 `--margin`을 키우거나 최종 홀드아웃 점수를 판정 근거로 쓰세요. 도달 불가와 달리 실행을 막지 않습니다 — 밝히기만 하고 바는 그대로 갑니다 |
| 로그에 `f1: no confidence interval (split too small, or metric degenerate)` | 재표집 단위가 20개 미만입니다(그룹 분할이면 **그룹** 20개). 정상 동작이고, 없는 구간을 지어내는 대신 없다고 적은 것입니다. 이 실행에서는 작은 차이를 개선으로 읽지 마세요 |
| iteration 간 점수 차이가 개선인지 노이즈인지 | `result.json`의 `<metric>_ci_low`/`_ci_high` 폭과 비교하세요. 폭보다 작은 차이는 이 행들이 구분해 낸 차이가 아니고, Critic 프롬프트도 같은 규칙으로 판단합니다(`## What these rows can resolve`). 그룹 분할에서는 폭이 몇 배 넓어집니다 — 그게 정직한 폭입니다 |
| 보고서의 하이퍼파라미터가 제안과 다름 | 정상입니다 — 실제 적용된 `applied_hyperparams`를 인용합니다. 버려진 이름은 `result.json`의 `dropped_hyperparams`에 있습니다 |
| `계획이 실행기에 없는 기능을 전제한 것으로 보입니다` / `진단이 …처방한 것으로 보입니다` | 산문이 threshold 튜닝, 교차검증, 특성 공학 같은 없는 기능에 의존하는 것으로 읽혔습니다. 실행은 계속되고 그 부분만 실행되지 않으며, 이름이 `plan.unsupported_claims`로 `history.json`에 남습니다. 부분 문자열 탐지라 **오탐 가능** — 원인으로 인용하기 전에 그 iteration의 계획 본문을 확인하세요. 목록은 [capabilities.py](automl_agent/capabilities.py) |
| 최종 test 점수가 검증 `best`보다 낮음 | **정상이고, 그게 이 숫자의 용도입니다.** `best`는 검증 점수 N개의 최댓값이라 위로 치우쳐 있고, 그 폭이 `selection_gap`입니다(demo.csv 3회 반복에서 +0.0236). 보고할 숫자는 test 쪽입니다. 반대로 test가 더 높으면 분할 노이즈이므로 둘 다 인용하세요 |
| 최종 줄에 `검증 점수가 위 CI 안에 있어 이 행들로는 0과 구분되지 않습니다` | `selection_gap`이 test 분할의 신뢰구간보다 작습니다. test는 파일의 20%라 이 실행에서 가장 넓은 구간이므로, 그 폭보다 작은 격차는 **측정된 편향의 양이 아닙니다**. "선택 편향 0.004"로 인용하지 말고 두 점수를 구간과 함께 인용하세요 |
| `최종 테스트 채점 생략 — …` | 채점할 모델이 없습니다. `--dry-run`이거나(저장된 모델이 없음), 모든 시도가 실패했거나, 모델 저장 이전 버전의 실행이거나, 채점 자식 프로세스가 실패한 경우입니다. 보고서는 그대로 작성되고 사유가 함께 실립니다. 로그는 `artifacts\<id>\holdout.log` |
| `카드의 split 프로토콜이 이번 실행과 다릅니다` | 카드의 기준선이 **다른 행**에서 측정됐습니다 — 그 카드로 도출한 바와 이 실행의 점수는 서로 다른 규칙으로 매겨진 숫자입니다. `--seed`를 카드와 맞추거나 `profile`을 다시 돌려 카드를 새로 만드세요. `protocol` 블록이 없는 옛 카드는 그냥 통과합니다 |
| 문자열 열이 `고유값 50개 초과로 제외`됨 | 자유 텍스트나 식별자 열입니다. one-hot하면 수천 열이 되므로 제외하고 **이름을 알립니다**. 쓰고 싶으면 값을 구간으로 묶어(예: 상위 20개 + `기타`) 다시 프로파일링하세요 |
| 카드의 `n_features`와 `encoding` 합이 다름 | `n_features`는 파일의 열 수, `encoding`은 추정기가 실제로 받는 열 수입니다. 범주형 1열이 one-hot 6열이 되면 후자가 더 큽니다 |
| `predict`: `인코딩 스키마가 없습니다` | `model.joblib`은 있는데 `feature_schema.json`이 없습니다. 합성 데이터 실행(`--data` 없이 카드만)이거나 이 파일이 저장되기 전 버전의 실행입니다. 새 파일에서 인코딩을 다시 유도하는 건 폭이 우연히 맞으면 어긋난 컬럼으로 예측이 나오는 일이라 하지 않습니다 — 같은 데이터로 다시 학습하세요 (13번) |
| `predict`: `선택된 최고 시도가 없습니다` | 성공한 학습이 없거나 실행이 report까지 가지 않았습니다. `show --thread-id ...`로 확인하고, 특정 시도를 쓰려면 `--iteration <번호>` |
| `predict --iteration N`: `모델 파일이 없습니다` | 그 iteration이 최고 시도가 아니면 실행이 끝날 때 정리된 것입니다(기본 `--keep-models best`). 어느 것이 남았는지는 `history.json`의 `models` 블록에 있고, 다음 실행부터 전부 남기려면 `--keep-models all`. `--dry-run` 실행은 애초에 모델을 저장하지 않습니다 |
| `artifacts\`가 수 GB | 대부분 `model.joblib`입니다(이 저장소에서 1.9 GB 중 1.79 GB, 최대 한 개 486 MB). 끝난 실행은 기본값이 이미 정리하지만, `--keep-models all`로 돌린 것과 이 기능 이전 실행은 손으로 지워야 합니다 — 남길 것은 `history.json`·`report.md`이고, 지워도 되는 것은 진 iteration의 `model.joblib`입니다 |
| `predict`: `prediction failed: FeatureSchemaMismatch: ...` | 모델이 필요한 원본 컬럼이 새 파일에 없습니다(메시지가 없는 이름을 전부 열거합니다). 컬럼 이름이 바뀐 export이거나 파일이 잘린 경우입니다. 이름을 맞추세요 — 빈 열을 만들어 넣으면 값을 발명하는 것이 됩니다 |
| `predict`: `확인할 점 N건` | 거부가 아닙니다. 예측은 나왔고, 학습 때 없던 범주·결측·텍스트가 된 수치 컬럼을 알린 것입니다. 개수가 파일 행 수에 가까우면 데이터 파이프라인 쪽을 보세요 (13번) |
| 예측 CSV를 커밋해도 되나 | 안 됩니다 — 행 단위 데이터입니다. `local\` 또는 `artifacts\` 안에 두면 git이 통째로 무시합니다 |
| `iteration N의 train_config.json을 쓸 수 없습니다` | 학습의 실패가 아니라 디스크나 권한입니다. 그 iteration은 `write_failed`로 기록되고 실행은 계속되지만, 원인이 그대로면 다음 iteration도 같은 지점에서 멈춥니다 — 공간을 만들고 같은 `--thread-id`로 `resume` 하세요. Critic이 모델을 바꿔 봐야 소용없는 유일한 실패 종류입니다 |
| `iteration N의 train.log를 저장하지 못했습니다` | 그 시도의 점수와 `result.json`은 정상입니다 — 사람이 나중에 읽을 사본만 없습니다. 조용히 없으면 "돌지 않은 시도"처럼 보이므로 알리는 것입니다 |
| `체크포인트 데이터베이스를 쓸 수 없습니다 … database is locked` | 같은 `artifacts` 디렉터리에 다른 `run`/`resume`이 살아 있습니다. 60초를 기다린 뒤의 메시지이므로 순간 경합이 아닙니다. 그 프로세스를 끝내고 `resume` 하면 마지막 체크포인트부터 이어집니다. 나란히 돌려야 하면 한쪽에 `--artifacts-root`를 다르게 주세요 |
| `실행 설정이 올바른 JSON이 아닙니다` | `run_config.json`이 쓰이던 중에 끊긴 파일입니다(현재는 임시 파일로 쓰고 rename하므로 새로 생기지 않습니다). 같은 `--thread-id`로 `run`을 다시 실행하면 이 파일만 새로 쓰이고, 체크포인트가 남아 있으면 그 지점부터 이어집니다 |
| `predict`: `report not written: …` | 예측 CSV는 이미 저장되었습니다 — `--report` 경로만 실패한 것이고, 같은 내용이 콘솔에 그대로 출력됩니다. 예측 실패가 아닙니다 |
| xgboost 모델이 안 보임 | `python -m pip install -e ".[xgboost]"` |
