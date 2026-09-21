---
name: automl-results
description: >
  끝난(또는 도는) AutoML 실행을 읽고 보고한다 — 실행 목록, 한 실행의 시도 이력, 보고서 전문,
  그리고 그 숫자를 어떻게 인용해야 하는지. 사용자가 "결과 보여 줘", "실행 목록", "어떻게 됐어",
  "보고서 읽어 줘", "점수 얼마야" 라고 할 때 쓴다. 새 실행을 띄우는 것은 automl-run 쪽이다.
---

저장소 루트(`${CLAUDE_PLUGIN_ROOT}`)에서 돌린다.

```bash
python -m automl_agent.main list                          # 어떤 실행이 있나 (최근 순)
python -m automl_agent.main show --thread-id <아이디>       # 시도 이력·최고·홀드아웃
python -m automl_agent.main show --thread-id <아이디> --report   # 보고서 전문까지
```

`list`가 방식·지표·최고(val)·test·반복·종료 사유를 한 줄씩 낸다. 아티팩트는
`artifacts/<thread_id>/`에 있고 `history.json`이 기계가 읽는 형태다.

## 보고가 끝이다

정리해 말해 주는 데서 멈춘다. `report.md`의 다음 레버 목록은 **사용자가 추가로 해 볼 수 있는
것으로 전하고**, 그중 하나를 지금 해 보자고 나서지 마라. 다음 실행을 띄울지 묻지 말고, 코드를
고치겠다고 하지 마라 — 사용자가 고르면 그때 하는 일이다.

## 인용할 때 지켜야 하는 것

**보고할 숫자는 holdout이다.** `best`는 val에서 *고른* 점수라 낙관적이다. holdout은 반복이 한
번도 보지 않은 test 20%를 끝에 딱 한 번 채점한 값이다. 둘을 나란히 놓으면 안 되고, 하나만
말해야 하면 holdout이다([README.md](README.md)의 「보고하는 점수는 고르지 않은 점수다」).

**제안자가 실제로 결정했는지 매번 확인해라.** 각 시도의 `plan_source`·`selection_source`가
`llm`/`fallback`/`rules` 중 하나다. `fallback`은 호출값을 냈는데 답이 안 쓰인 반복이다.
이 플러그인의 실행은 제안자가 항상 로컬 `ollama:gemma4:12b`이므로 **이것을 가장 먼저 봐라** —
ollama가 죽어 있거나 모델이 형식을 어기면 실행은 계속 돌고 조용히 규칙 기반이 된다. 절반이
`fallback`이면 그 실행은 모델이 아니라 규칙을 잰 것이다.

```bash
python -c "import json;d=json.load(open('artifacts/<아이디>/history.json',encoding='utf-8'));[print(a['iteration'],a.get('plan_source'),a.get('selection_source')) for a in d['history']]"
```

**`--dry-run` 실행의 점수는 가짜다.** `list`의 방식 열이 `dry-run`이면 인용하지 마라.

**제안한 것과 실제로 돌아간 것을 구분해라.** `hyperparams`는 적용된 값이고,
`dropped_hyperparams`는 제안됐지만 버려진 키다. `applied_pipeline`은 전처리가 실제로 무엇이었나.
계획이 요청한 것을 성과로 돌리지 마라 — 그 구분이 이 저장소가 기록하는 이유다.

**차이 하나를 증거로 읽지 마라.** 같은 설정을 반복해도 짝지은 Δ가 12쌍 중 4쌍에서 0을
4쌍에서 0을 벗어나는 것을 관측했다. Δ 하나는 처치의 증거로 부족하다.

## 저장소의 판정을 인용할 때

**이 배포본에는 측정 문서도 `bench/`도 없다.** 판정을 인용해야 하면 개발 저장소로 가라 —
URL은 [README.md](README.md)의 「더 자세한 것」에 있다. 거기 측정 문서 15개와, 판정 전체를
저장된 예측 배열에서 재계산하는 `python -m bench.recheck`가 있다.

핵심 판정을 정직하게 말해야 할 때:

- `llm` 팔이 **random search**를 이긴다 — 이진 4개 중 3개, 기준 충족
- `llm` 팔이 **`no_llm`(규칙 기반)**을 이기는지는 **구분되지 않는다** — 2승 1무 1패, 사전 등록 미달
- 측정된 개선 둘은 **코드 레버**다 — 결정 컷 +0.0762, 정렬된 spec +0.0046. LLM의 기여로 돌리지 마라
- 시드가 하나이고 시드 잡음이 평균 0.1472다

`python -m bench.recheck`가 판정 전체를 예측 배열에서 재계산한다. 통과가 뜻하는 것은 **산수가
맞다**는 것이고, 그 예측이 config가 이름 붙인 모델에서 나왔다는 것은 아니다.
