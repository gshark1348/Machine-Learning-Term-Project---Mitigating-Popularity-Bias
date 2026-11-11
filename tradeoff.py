"""평균 실효 강도를 맞춘 정확도-롱테일 trade-off 곡선 (fixed vs adaptive vs adaptive_rank).

연구 질문: 개인화 인기도 페널티가 고정 페널티보다 나은 정확도-롱테일 trade-off 를 주는가.

실험 설계
---------
재정렬 점수 = S(i) - penalty·pop(i).  방식별 penalty:
    fixed          : λ
    adaptive       : λ·(1 - P_u)            (P_u: 학습 구간 4점 이상 영화의 평균 pop)
    adaptive_rank  : λ·(1 - rank(P_u))      (rank: 학습 사용자 분포 안의 분위수)
같은 λ 로 비교하면 평균 실효 강도가 달라 불공정하다 (평균 1-P_u ≈ 0.26, 평균 1-rank ≈ 0.5).
그래서 목표 평균 실효 강도 s 를 정하고 λ = s / m 으로 맞춘다.
    m = 평가 대상 사용자에 대한 평균 배율 (fixed 1, adaptive 평균 1-P_u, adaptive_rank 평균 1-rank P_u)
이러면 같은 s 에서 세 방식의 "사용자 평균 페널티"가 같고, 사용자 사이 분배만 달라진다.
- 검증(validation, split=1) 구간만 쓴다. 테스트 구간은 건드리지 않는다.
- 평가 사용자는 evaluate.py 와 같은 500명(시드 7), 정답/제외 규칙도 evaluate() 와 같다.
- 같은 롱테일 노출에서 NDCG 를 비교한다 (곡선 보간: tradeoff_at_exposure.csv).
- 방식 간 차이는 사용자 단위로 짝지은 부트스트랩(시드 11)으로 95% CI 를 구한다.

출력 (out-dir): tradeoff.csv, tradeoff_matched.csv, tradeoff_at_exposure.csv,
               tradeoff_groups.csv, preference.csv

실행:  python tradeoff.py [--run RUN_DIR] [--split 1] [--out-dir scratch/tradeoff] [--boot 1000]
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from evaluate import (evaluate, load_eval_groups, load_preference_reference,
                      tail_recalls, user_metrics)
from model import Engine, active_model, rank_preference

# 목표 평균 실효 강도 grid
STRENGTHS = [0, .0025, .005, .01, .015, .02, .03, .05, .075, .1, .15]
METHODS = ['fixed', 'adaptive', 'adaptive_rank']
# 공통 롱테일 노출 수준 (추천 중 롱테일 비율)
EXPOSURES = [.15, .20, .25, .30, .35, .40]
METRICS = ['ndcg', 'recall', 'tail', 'popularity']
ROOT = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# 준비: 사용자별 후보 풀/점수/선호도를 한 번만 계산
# ---------------------------------------------------------------------------
def prepare_users(e, groups, split, alpha, slots, reference):
    """평가 사용자별로 (후보 풀, 점수, 정답, raw P_u, rank P_u) 를 계산해 dict 목록으로 돌려준다.

    무엇을/왜: 페널티 강도(s)와 방식을 바꿔도 풀과 점수는 같으므로 한 번만 계산해 재사용한다.
    규칙은 evaluate() 와 같다: seen = split 이전 구간 전체, truth = 해당 split 의 4점 이상 &
    i>=0 영화 - seen, 정답이 없는 사용자는 제외.
    """
    users = []
    for u, g in groups:
        train = g[g.split == 0]
        seen = g.loc[g.split < split, 'i']
        truth = set(g.loc[(g.split == split) & (g.rating >= 4) & (g.i >= 0), 'i']) - set(seen)
        if not truth:
            continue
        cf, cb, pref = e.scores(train.i, train.rating, u)
        pool, score = e.pool(cf, cb, alpha, seen, slots)
        users.append(dict(u=u, pool=pool, score=score, truth=truth, pref=pref,
                          rank=rank_preference(pref, reference)))
    return users


def multipliers(users):
    """방식별 평균 배율 m: fixed 1, adaptive 평균(1-P_u), adaptive_rank 평균(1-rank P_u)."""
    return {'fixed': 1.0,
            'adaptive': float(np.mean([1 - x['pref'] for x in users])),
            'adaptive_rank': float(np.mean([1 - x['rank'] for x in users]))}


def user_penalty(x, method, lam):
    """사용자 x 에 적용되는 페널티 (evaluate() 의 분기와 같은 식)."""
    if method == 'adaptive':
        return lam * (1 - x['pref'])
    if method == 'adaptive_rank':
        return lam * (1 - x['rank'])
    return lam


# ---------------------------------------------------------------------------
# 한 설정의 사용자 단위 결과와 집계
# ---------------------------------------------------------------------------
def run_config(e, users, method, lam):
    """한 (방식, λ) 설정으로 모든 사용자를 추천해 사용자 단위 결과를 돌려준다.

    반환: dict(metrics=(n,4) 배열 [ndcg, recall, tail, popularity], penalty=(n,) 배열,
              tails=사용자별 [풀 재현율, 추천 재현율] (롱테일 정답 없으면 None), items=추천된 영화 집합)
    """
    metrics, penalties, tails, items = [], [], [], set()
    for x in users:
        penalty = user_penalty(x, method, lam)
        rec = e.rank(x['pool'], x['score'], penalty)
        metrics.append(user_metrics(e, rec, x['truth']))
        tail_scores = tail_recalls(e, x['truth'], x['pool'], rec)
        tails.append(tail_scores or None)
        penalties.append(penalty)
        items.update(rec)
    return dict(metrics=np.array(metrics), penalty=np.array(penalties), tails=tails, items=items)


def aggregate(e, res):
    """사용자 단위 결과를 evaluate() 의 'all' 행과 같은 방식으로 평균낸다 (dict)."""
    tails = [t for t in res['tails'] if t]
    return dict(
        users=len(res['metrics']),
        tail_users=len(tails),
        tail_recall100=np.mean([t[0] for t in tails]) if tails else None,
        tail_recall10=np.mean([t[1] for t in tails]) if tails else None,
        coverage=len(res['items']) / len(e.movies),
        **dict(zip(METRICS, np.mean(res['metrics'], axis=0))))


def check_against_evaluate(e, users, groups, split, alpha, slots, reference):
    """tradeoff.py 의 사용자 평균이 evaluate() 의 'all' 행과 같은지 확인한다 (1e-12 이내 assert).

    hybrid 0 은 방식과 무관한 기준선(페널티 0) 이므로 fixed λ=0 으로 계산한다.
    """
    checks = [('fixed', .02), ('adaptive', .05), ('adaptive_rank', .05), ('hybrid', 0)]
    configs = [(alpha, method, k) for method, k in checks]
    rows = evaluate(e, groups, split, configs, slots, reference)
    print('[check_against_evaluate] evaluate() vs tradeoff.py')
    for (method, k), row in zip(checks, [r for r in rows if r['group'] == 'all']):
        mine = aggregate(e, run_config(e, users, 'fixed' if method == 'hybrid' else method, k))
        worst = 0.0
        for key in ['users', 'tail_users', 'tail_recall100', 'tail_recall10', 'coverage'] + METRICS:
            diff = abs(float(mine[key]) - float(row[key]))
            assert diff <= 1e-12, f'{method} {k} {key}: {mine[key]} != {row[key]}'
            worst = max(worst, diff)
        print(f'  {method:14s} k={k:<5} 일치 (최대 차이 {worst:.1e})')


# ---------------------------------------------------------------------------
# 부트스트랩 / 보간
# ---------------------------------------------------------------------------
def paired_bootstrap(delta, idx):
    """사용자별 차이 delta 의 평균과 부트스트랩 95% CI (2.5%, 97.5% 분위수) 를 돌려준다.

    idx: (boot, n) 정수 배열 = 복원추출한 사용자 번호. delta[idx].mean(axis=1) 이 재표본 평균들.
    같은 idx 를 방식·지표에 모두 써서 사용자 짝이 유지된다.
    np.quantile(a, q): 배열 a 의 q 분위수 (q=[.025,.975] 이면 95% 구간).
    """
    means = delta[idx].mean(axis=1)
    lo, hi = np.quantile(means, [.025, .975])
    return float(delta.mean()), float(lo), float(hi)


def interpolate(curve, level):
    """곡선(tail 오름차순 정렬된 (tail, ndcg) 쌍) 에서 노출 level 의 NDCG 를 선형 보간한다.

    np.interp(x, xp, fp): xp(증가하는 x좌표), fp(값) 를 직선으로 이어 x 에서의 값을 읽는다.
    곡선 범위 밖이면 None (외삽하지 않는다).
    """
    xs, ys = curve[:, 0], curve[:, 1]
    if level < xs.min() or level > xs.max():
        return None
    return float(np.interp(level, xs, ys))


# ---------------------------------------------------------------------------
# 메인
# ---------------------------------------------------------------------------
def main():
    # argparse: 명령줄 인자 정의. type 은 변환 함수, default 는 생략 시 값.
    cli = argparse.ArgumentParser(description='평균 실효 강도를 맞춘 trade-off 곡선')
    cli.add_argument('--run', type=Path, default=None, help='run 폴더 (기본: active_model())')
    cli.add_argument('--split', type=int, default=1, help='평가 구간 (1=검증, 기본)')
    cli.add_argument('--out-dir', type=Path, default=ROOT / 'scratch' / 'tradeoff')
    cli.add_argument('--boot', type=int, default=1000, help='부트스트랩 반복 횟수')
    args = cli.parse_args()

    run = (args.run or active_model()).resolve()
    settings = json.loads((run / 'settings.json').read_text())
    alpha, slots = settings['alpha'], settings['content_slots']
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    e = Engine(run)
    rows = pd.read_csv(run / 'split.csv.gz')
    groups = load_eval_groups(rows)
    reference = load_preference_reference(e, rows)
    users = prepare_users(e, groups, args.split, alpha, slots, reference)
    n = len(users)
    m = multipliers(users)
    print(f'run={run.name} alpha={alpha} content_slots={slots} split={args.split} users={n}')
    print('평균 배율 m:', {k: round(v, 4) for k, v in m.items()})

    check_against_evaluate(e, users, groups, args.split, alpha, slots, reference)

    # 1) 모든 (방식, s) 계산
    results, table = {}, []
    for method in METHODS:
        for s in STRENGTHS:
            lam = s / m[method]
            res = run_config(e, users, method, lam)
            results[(method, s)] = res
            table.append(dict(method=method, target_strength=s, lam=lam,
                              mean_penalty=res['penalty'].mean(), **aggregate(e, res)))
    tradeoff = pd.DataFrame(table)
    tradeoff.to_csv(out / 'tradeoff.csv', index=False)

    # 2) fixed 대비 짝지은 차이 + 부트스트랩
    # rng.integers(low, high, size): [low, high) 정수를 size 모양으로 무작위 추출 (복원추출 번호).
    idx = np.random.default_rng(11).integers(0, n, size=(args.boot, n))
    matched = []
    for s in STRENGTHS:
        base = results[('fixed', s)]['metrics']
        for method in ['adaptive', 'adaptive_rank']:
            cur = results[(method, s)]['metrics']
            d_ndcg, n_lo, n_hi = paired_bootstrap(cur[:, 0] - base[:, 0], idx)
            d_tail, t_lo, t_hi = paired_bootstrap(cur[:, 2] - base[:, 2], idx)
            matched.append(dict(
                method=method, target_strength=s, users=n,
                ndcg_diff=d_ndcg, ndcg_lo=n_lo, ndcg_hi=n_hi, ndcg_ci_has_zero=n_lo <= 0 <= n_hi,
                tail_diff=d_tail, tail_lo=t_lo, tail_hi=t_hi, tail_ci_has_zero=t_lo <= 0 <= t_hi,
                tail_slots=d_tail * n * 10, tail_slots_lo=t_lo * n * 10, tail_slots_hi=t_hi * n * 10))
    matched = pd.DataFrame(matched)
    matched.to_csv(out / 'tradeoff_matched.csv', index=False)

    # 3) 공통 롱테일 노출에서 NDCG 보간
    interp = []
    curves = {}
    for method in METHODS:
        sub = tradeoff[tradeoff.method == method].sort_values('tail')
        curves[method] = sub[['tail', 'ndcg']].to_numpy()
    for level in EXPOSURES:
        base = interpolate(curves['fixed'], level)
        for method in METHODS:
            v = interpolate(curves[method], level)
            interp.append(dict(method=method, tail_level=level, ndcg=v,
                               diff_vs_fixed=None if v is None or base is None else v - base))
    pd.DataFrame(interp).to_csv(out / 'tradeoff_at_exposure.csv', index=False)

    # 4) rank P_u 3분위 그룹별
    ranks = np.array([x['rank'] for x in users])
    cuts = np.quantile(ranks, [1 / 3, 2 / 3])
    tercile = np.digitize(ranks, cuts)          # np.digitize(x, bins): x 가 속한 구간 번호(0,1,2)
    names = ['low', 'mid', 'high']
    grp = []
    for (method, s), res in results.items():
        for t in range(3):
            mask = tercile == t
            grp.append(dict(method=method, target_strength=s, group=names[t], users=int(mask.sum()),
                            mean_penalty=res['penalty'][mask].mean(),
                            ndcg=res['metrics'][mask, 0].mean(), tail=res['metrics'][mask, 2].mean()))
    pd.DataFrame(grp).to_csv(out / 'tradeoff_groups.csv', index=False)

    # 5) 사용자별 선호도
    pd.DataFrame(dict(u=[x['u'] for x in users], raw_pu=[x['pref'] for x in users],
                      rank_pu=ranks)).to_csv(out / 'preference.csv', index=False)

    # 콘솔 요약
    pd.set_option('display.width', 200)
    print('\n[맞춤 확인] s 별 방식 간 mean_penalty 최대 차이:',
          f"{tradeoff.groupby('target_strength').mean_penalty.agg(np.ptp).max():.2e}")
    wide = tradeoff.pivot(index='target_strength', columns='method', values=['ndcg', 'tail'])
    print('\n[tradeoff] s 별 ndcg / tail'); print(wide.round(4).to_string())
    print('\n[matched] fixed 대비 차이 (95% CI)')
    print(matched[['method', 'target_strength', 'ndcg_diff', 'ndcg_lo', 'ndcg_hi',
                   'tail_diff', 'tail_lo', 'tail_hi', 'tail_slots']].round(5).to_string(index=False))
    print('\n[at_exposure]'); print(pd.DataFrame(interp).round(4).to_string(index=False))
    print(f'\n저장 위치: {out}')


if __name__ == '__main__':
    main()
