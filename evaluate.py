"""학습된 모델(run 폴더)을 평가하고 결과 CSV/JSON 을 run 폴더에 저장하는 파일.

model.train() 이 마지막에 run_evaluation(run) 을 호출한다.

평가 흐름
---------
1. 평가 사용자 500명을 시드 7로 뽑는다.
2. [validation] 하이브리드 가중치 alpha 를 5개 후보 중에서 NDCG 최고인 값으로 고른다.
3. [validation] 고정(fixed)/개인화(adaptive) 인기도 페널티 강도를 grid 에서 고른다.
   (NDCG 가 최고치의 95% 이상인 후보 중 롱테일 비율(tail)이 가장 높은 것)
4. [test] content / cf / hybrid / fixed / adaptive 를 비교한다.
5. 후보 풀의 콘텐츠 슬롯 수(0/10/25/50)를 validation 으로 고르고 test 로 확인한다.
6. 신규 사용자(콜드스타트): 평점 1/3/5개만 알 때의 추천 성능을 측정한다.

지표 정의
---------
    NDCG@10   = DCG / IDCG
                DCG  = Σ_{k=1..10} hit_k / log2(k+1)        (hit_k: k번째 추천이 정답이면 1)
                IDCG = Σ_{k=1..min(10,|정답|)} 1 / log2(k+1)   (정답을 모두 앞에 놓았을 때의 최댓값)
    Recall@10 = (추천 10개 중 정답 수) / |정답|
    정답(truth) = 해당 분할에서 평점 ≥ 4 인 영화 중 이미 본 영화를 뺀 것
    tail      = 추천 10개 중 롱테일 영화 비율,  popularity = 추천 10개의 평균 인기도(pop)
    tail_recall100 / tail_recall10 = 롱테일 정답 중 후보 풀(100) / 추천(10) 에 들어간 비율
    coverage  = 추천된 서로 다른 영화 수 / 전체 영화 수
    개인화 페널티 = λ_max · (1 - P_u)   (P_u: 사용자의 선호 인기도 pref, λ_max: 선택된 penalty)

함수 목록 (흐름 순서)
--------------------
    run_evaluation()      전체 평가 진입 함수 (아래 함수들을 순서대로 호출).
    load_eval_groups()    평가 사용자 500명의 평점을 사용자별로 묶는다.
    user_group()          선호 인기도 pref 로 explorer/balanced/mainstream 구분.
    ndcg_at_10()          한 사용자의 NDCG@10.
    tail_recalls()        롱테일 정답에 대한 풀/추천 재현율.
    user_metrics()        한 사용자·한 설정의 [ndcg, recall, tail, popularity].
    evaluate()            사용자 전체에 대해 설정(configs)별 지표를 집계한다.
    summarize()           누적 결과(out)를 CSV 행(dict) 목록으로 바꾼다.
    choose_penalties()    validation 에서 fixed/adaptive 페널티 강도를 고른다.
    compare_slots()       콘텐츠 슬롯 수별 비교 행을 계산해 comparison 에 추가한다.
    choose_slots()        validation 비교에서 콘텐츠 슬롯 수를 고른다.
    cold_start_report()   신규 사용자 평가 결과 표를 만든다.
"""
import json
import numpy as np
import pandas as pd
from model import Engine, active_model

# 페널티 후보 grid (참고: 간격이 거칠어서 0 이 선택되는 경우가 있다 -> outputs/settings.json 의 0)
GRID = [0, .02, .05, .1, .2]


# ---------------------------------------------------------------------------
# 사용자 단위 계산
# ---------------------------------------------------------------------------
def load_eval_groups(rows):
    """평가에 쓸 사용자 500명을 뽑아 (u, 해당 사용자 평점 DataFrame) 목록으로 돌려준다.

    np.random.default_rng(7).choice(a, n, replace=False): 시드 7 로 고정된 난수 생성기로
    a 에서 중복 없이 n 개를 뽑는다. groupby('u') 를 list 로 만들면 (u, DataFrame) 쌍 목록이 된다.
    """
    users = np.random.default_rng(7).choice(rows.u.unique(), min(500, rows.u.nunique()), replace=False)
    return list(rows[rows.u.isin(users)].groupby('u'))


def user_group(pref):
    """선호 인기도 pref 로 사용자 그룹 이름을 정한다.

    pref ≤ 0.33 explorer(탐색형), 0.33 < pref ≤ 0.66 balanced, 그 위 mainstream.
    참고: 경계가 0.33/0.66 인데 pref 는 log 인기도 평균이라 대부분 높게 나와 explorer 가
    0명인 경우가 많다 (CLAUDE.md 의 미해결 사항). 동작은 바꾸지 않았다.
    """
    return ['explorer', 'balanced', 'mainstream'][int(pref > .33) + int(pref > .66)]


def ndcg_at_10(hits, n_truth):
    """NDCG@10 = DCG / IDCG (정의는 모듈 docstring 참고).

    hits: 추천 목록 순서대로의 정답 여부(bool 배열). n_truth: 정답 개수.
    일부러 파이썬 내장 sum 을 쓴다 (원래 계산과 부동소수점 덧셈 순서를 똑같이 유지하기 위해).
    """
    dcg = sum(hits / np.log2(np.arange(len(hits)) + 2))
    idcg = sum(1 / np.log2(np.arange(min(10, n_truth)) + 2))
    return dcg / idcg


def tail_recalls(e, truth, pool, rec):
    """롱테일 정답 중 후보 풀(pool)과 추천(rec)에 들어간 비율 [풀 재현율, 추천 재현율].

    롱테일 정답이 하나도 없으면 빈 리스트를 돌려준다 (그 사용자는 tail 통계에서 제외).
    """
    tail_truth = {i for i in truth if e.tail[i]}
    if not tail_truth:
        return []
    return [len(tail_truth.intersection(items)) / len(tail_truth) for items in (pool, rec)]


def user_metrics(e, rec, truth):
    """한 사용자의 추천 rec 에 대한 [ndcg, recall, tail, popularity] 를 돌려준다.

    np.isin(rec, list(truth)): 추천 각 항목이 정답에 포함되는지 불리언 배열.
    e.tail[rec].mean(): 추천 중 롱테일 비율, e.pop[rec].mean(): 평균 인기도.
    """
    hits = np.isin(rec, list(truth))
    return [ndcg_at_10(hits, len(truth)),
            hits.sum() / len(truth),
            e.tail[rec].mean(),
            e.pop[rec].mean()]


# ---------------------------------------------------------------------------
# 전체 사용자 평가
# ---------------------------------------------------------------------------
def evaluate(e, groups, split, configs, slots=0):
    """모든 평가 사용자에 대해 configs 의 각 설정을 평가해 결과 행(dict) 목록을 돌려준다.

    split: 1 이면 validation 을, 2 이면 test 를 정답으로 삼는다.
           split 번호보다 작은 분할의 평점은 모두 "이미 본 영화(seen)" 이다.
    configs: (alpha, method, k) 목록. method 가 'adaptive' 면 페널티 = k·(1 - pref),
             그 외에는 페널티 = k. ('content'/'cf'/'hybrid' 는 k=0, 'fixed' 는 k 고정.)
    slots: 후보 풀에 넣을 콘텐츠 슬롯 수 (Engine.pool 의 content_slots).

    out 의 삽입 순서가 결과 행 순서(= CSV 행 순서)를 결정한다. 순서를 바꾸면 안 된다.
    out[(alpha, method, k, group)] = (사용자별 지표 목록, 추천된 영화 집합, 롱테일 재현율 목록)
    group 은 'all' 과 사용자 자신의 그룹(explorer/balanced/mainstream) 두 곳에 모두 누적한다.
    """
    out = {}
    for u, g in groups:
        train = g[g.split == 0]              # 사용자 요인을 만드는 데 쓴 평점
        seen = g.loc[g.split < split, 'i']   # 추천에서 제외할 이미 본 영화
        truth = set(g.loc[(g.split == split) & (g.rating >= 4) & (g.i >= 0), 'i']) - set(seen)
        if not truth:
            continue                         # 정답이 없는 사용자는 평가에서 제외
        cf, cb, pref = e.scores(train.i, train.rating, u)
        pools = {}                           # alpha 가 같으면 후보 풀을 재사용
        for alpha, method, k in configs:
            if alpha not in pools:
                pools[alpha] = e.pool(cf, cb, alpha, seen, slots)
            penalty = k * (1 - pref) if method == 'adaptive' else k
            rec = e.rank(*pools[alpha], penalty)
            tail_scores = tail_recalls(e, truth, pools[alpha][0], rec)
            metrics = user_metrics(e, rec, truth)
            for group in ['all', user_group(pref)]:
                vals, items, tails = out.setdefault((alpha, method, k, group), ([], set(), []))
                if tail_scores:
                    tails.append(tail_scores)
                vals.append(metrics)
                items.update(rec)
    return summarize(e, out)


def summarize(e, out):
    """누적 결과 out 을 설정별 평균 지표 행(dict) 목록으로 바꾼다.

    열 순서(= CSV 열 순서): alpha, method, penalty, group, users, tail_users,
    tail_recall100, tail_recall10, coverage, ndcg, recall, tail, popularity.
    np.mean(v, axis=0): 사용자 축(0번 축)으로 평균을 내 지표별 평균을 얻는다.
    """
    rows = []
    for (a, m, k, g), (v, items, tails) in out.items():
        rows.append(dict(
            alpha=a, method=m, penalty=k, group=g,
            users=len(v),
            tail_users=len(tails),
            tail_recall100=np.mean([x[0] for x in tails]) if tails else None,
            tail_recall10=np.mean([x[1] for x in tails]) if tails else None,
            coverage=len(items) / len(e.movies),
            **dict(zip(['ndcg', 'recall', 'tail', 'popularity'], np.mean(v, axis=0)))))
    return rows


# ---------------------------------------------------------------------------
# 하이퍼파라미터 선택
# ---------------------------------------------------------------------------
def choose_penalties(val, best_ndcg):
    """validation 결과에서 방식별(fixed/adaptive) 페널티 강도를 고른다.

    조건: 전체('all') 그룹에서 NDCG ≥ 최고 NDCG 의 95%.
    그 중 롱테일 비율(tail)이 가장 높은 것, 동점이면 NDCG 가 높은 것을 선택한다.
    """
    chosen = {}
    for m in ['fixed', 'adaptive']:
        candidates = (v for v in val
                      if v['method'] == m and v['group'] == 'all' and v['ndcg'] >= .95 * best_ndcg)
        chosen[m] = max(candidates, key=lambda v: (v['tail'], v['ndcg']))['penalty']
    return chosen


def compare_slots(e, groups, comparison, a, chosen, split, slots):
    """adaptive 설정으로 콘텐츠 슬롯 수 slots 의 전체('all') 그룹 결과를 comparison 에 추가한다.

    열 순서: split, content_slots, 그 뒤에 evaluate() 결과 열들.
    """
    rows = evaluate(e, groups, split, [(a, 'adaptive', chosen['adaptive'])], slots)
    row = next(row for row in rows if row['group'] == 'all')
    comparison.append(dict(split=split, content_slots=slots, **row))
    return row


def choose_slots(comparison):
    """validation 비교(comparison)에서 콘텐츠 슬롯 수를 고른다.

    슬롯 0 의 NDCG 의 95% 이상인 후보 중 tail_recall100 이 가장 높은 것,
    동점이면 NDCG 가 높은 것, 그래도 같으면 슬롯 수가 작은 것을 고른다.
    (tail_recall100 이 None 이면 0 으로 간주한다.)
    """
    floor = .95 * comparison[0]['ndcg']
    eligible = (v for v in comparison if v['ndcg'] >= floor)
    best = max(eligible, key=lambda v: (v['tail_recall100'] or 0, v['ndcg'], -v['content_slots']))
    return best['content_slots']


# ---------------------------------------------------------------------------
# 콜드스타트
# ---------------------------------------------------------------------------
def cold_start_report(e, cold, a, chosen, slots):
    """신규 사용자가 평점을 size(1, 3, 5)개만 남겼을 때의 추천 성능 표를 만든다.

    사용자별 이력의 앞 90%를 입력 후보, 뒤 10% 를 정답 구간으로 본다.
    입력은 앞 90% 중 학습 영화(i ≥ 0)인 처음 size 개. 정답은 뒤 10% 의 평점 ≥ 4 영화.
    사용자 요인은 학습 때 본 적 없으므로 Engine.scores(user=None) 의 ridge fold-in 을 쓴다.
    페널티는 adaptive: chosen['adaptive'] · (1 - pref).
    groupby('userId'): 사용자별 이력 묶음. DataFrame.query('식'): 문자열 조건으로 행을 거른다.
    """
    reports = []
    for size in (1, 3, 5):
        values = []
        for _, history in cold.groupby('userId'):
            boundary = int(len(history) * .9)
            initial = history.iloc[:boundary].query('i>=0').head(size)
            truth = set(history.iloc[boundary:].query('i>=0 and rating>=4').i) - set(initial.i)
            if len(initial) != size or not truth:
                continue
            cf, cb, pref = e.scores(initial.i, initial.rating)
            pool, score = e.pool(cf, cb, a, initial.i, slots)
            rec = e.rank(pool, score, chosen['adaptive'] * (1 - pref))
            tail = {i for i in truth if e.tail[i]}
            values.append(dict(
                recall10=len(truth.intersection(rec)) / len(truth),
                tail_recall100=len(tail.intersection(pool)) / len(tail) if tail else np.nan,
                tail_recall10=len(tail.intersection(rec)) / len(tail) if tail else np.nan))
        frame = pd.DataFrame(values)
        reports.append(dict(
            ratings=size,
            users=len(frame),
            tail_users=frame.tail_recall10.count() if values else 0,
            **frame.mean().to_dict()))
    return pd.DataFrame(reports)


# ---------------------------------------------------------------------------
# 진입 함수
# ---------------------------------------------------------------------------
def run_evaluation(run):
    """run 폴더의 모델을 평가해 validation.csv, test.csv, settings.json,
    candidate_comparison.csv, cold_start.csv 를 같은 폴더에 쓰고 test 결과를 출력한다."""
    e = Engine(run)
    rows = pd.read_csv(run / 'split.csv.gz')
    groups = load_eval_groups(rows)

    # 1) validation: alpha 선택 (CF 가중치 0, .25, .5, .75, 1 을 비교)
    base = evaluate(e, groups, 1, [(a, 'hybrid', 0) for a in [0, .25, .5, .75, 1]])
    best = max((v for v in base if v['group'] == 'all'), key=lambda v: v['ndcg'])
    a = best['alpha']

    # 2) validation: 페널티 강도 선택
    val = evaluate(e, groups, 1, [(a, m, k) for m in ['fixed', 'adaptive'] for k in GRID])
    chosen = choose_penalties(val, best['ndcg'])

    # 3) test: 방법별 비교와 저장
    test = evaluate(e, groups, 2,
                    [(0, 'content', 0), (1, 'cf', 0), (a, 'hybrid', 0)]
                    + [(a, m, k) for m, k in chosen.items()])
    pd.DataFrame(base + val).to_csv(run / 'validation.csv', index=False)
    pd.DataFrame(test).to_csv(run / 'test.csv', index=False)
    (run / 'settings.json').write_text(json.dumps(dict(alpha=a, **chosen)))

    # 4) 콘텐츠 슬롯 수 선택 (validation) 과 확인 (test)
    comparison = []
    for slots in [0, 10, 25, 50]:
        compare_slots(e, groups, comparison, a, chosen, 1, slots)
    slots = choose_slots(comparison)
    for test_slots in sorted({0, slots}):    # set -> 중복 제거 후 정렬 (slots 가 0 이면 한 번만)
        compare_slots(e, groups, comparison, a, chosen, 2, test_slots)
    pd.DataFrame(comparison).to_csv(run / 'candidate_comparison.csv', index=False)
    (run / 'settings.json').write_text(json.dumps(dict(alpha=a, content_slots=slots, **chosen)))

    # 5) 콜드스타트 평가
    cold = pd.read_csv(run / 'new_users.csv.gz')
    cold_start_report(e, cold, a, chosen, slots).to_csv(run / 'cold_start.csv', index=False)

    print(pd.DataFrame(test).query("group=='all'").to_string(index=False))
