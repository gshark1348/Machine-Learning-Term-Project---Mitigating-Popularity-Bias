"""MovieLens 20M 학습(train)과 공용 추천 로직(Engine)을 담은 파일.

역할
----
1. train(): ml-20m.zip 에서 사용자 일부를 뽑아 행렬분해(MF) 모델을 SGD로 학습하고,
   결과물(model.npz, movies.json, split.csv.gz, ...)을 artifacts/runs/<uuid>/ 에 저장한다.
   마지막에 evaluate.run_evaluation() 으로 평가한 뒤 해당 run 을 'active' 로 지정한다.
2. Engine: 저장된 모델을 읽어 (a) 협업필터링(CF) 점수, (b) 콘텐츠기반(CB) 점수,
   (c) 후보 풀(pool), (d) 인기도 페널티 재순위(rank)를 계산한다. app.py, evaluate.py 등이 공유한다.

모델 수식
---------
    예측 평점   r̂(u,i) = μ + b_u + b_i + p_u · q_i
    SGD 손실    L = Σ (r - r̂)^2 + λ(|p|^2 + |q|^2 + b_u^2 + b_i^2)
    업데이트    θ ← θ + lr · (err · ∂r̂/∂θ - reg · θ),   err = r - r̂,  lr=0.015, reg=0.04
    (코드에서는 모든 파라미터에 같은 형태 `.015*(grad - .04*param)` 을 쓴다.)

함수 목록 (흐름 순서)
--------------------
    active_model()        현재 활성(active) run 폴더 경로를 돌려준다.
    publish(run)          active 포인터 파일을 원자적으로 갱신한다.
    unit(x)               행 벡터를 길이 1로 정규화한다.
    ── train() 의 하위 단계 ──
    read_zip_csv()        zip 안의 CSV 하나를 pandas 로 읽는다.
    sample_users()        학습용 표본 사용자와 콜드스타트용 holdout 사용자를 뽑는다.
    load_ratings()        두 집단의 평점을 청크 단위로 읽어 정렬한다.
    assign_split()        사용자별 시간순 80/10/10 (train/validation/test) 분할.
    select_movies()       학습 분할에 등장하는 영화만 남겨 영화 색인을 만든다.
    encode_ids()          userId/movieId 를 0부터 시작하는 행 번호(u, i)로 바꾼다.
    build_features()      장르 + 태그 게놈으로 콘텐츠 특징 행렬을 만든다.
    popularity_stats()    영화별 인기도(pop)와 롱테일 표시(tail)를 계산한다.
    init_factors()        요인 행렬 p, q 와 편향 bu, bi 를 초기화한다.
    run_sgd()             미니배치 SGD 로 p, q, bu, bi 를 학습한다.
    save_run()            학습 산출물을 run 폴더에 저장한다.
    train()               위 단계를 순서대로 호출하는 진입 함수.
    ── 추천 엔진 ──
    Engine.scores()       CF 점수, CB 점수, 사용자 선호 인기도를 계산한다.
    Engine.pool()         하이브리드 점수로 상위 100개 후보 풀을 만든다.
    Engine.rank()         인기도 페널티를 적용해 최종 10개를 고른다.
"""
import argparse
import os
import tempfile
from uuid import uuid4
import json
import zipfile
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
MODEL = ROOT / 'artifacts'   # 산출물 루트 (테스트 시 이 전역을 바꿔 다른 폴더에 쓸 수 있다)
THRESHOLD = 4.0              # 평점 4.0 이상이면 "좋아함(liked)" 으로 본다


# ---------------------------------------------------------------------------
# 공용 유틸
# ---------------------------------------------------------------------------
def active_model():
    """현재 활성 run 폴더를 돌려준다.

    artifacts/active 파일에 run 폴더 이름(uuid)이 적혀 있으면 artifacts/runs/<이름>,
    파일이 없으면 artifacts 폴더 자체를 쓴다.
    """
    pointer = MODEL / 'active'
    return MODEL / 'runs' / pointer.read_text().strip() if pointer.exists() else MODEL


def publish(run):
    """run 을 활성 모델로 지정한다 (artifacts/active 파일에 run 이름을 기록).

    임시 파일에 먼저 쓴 뒤 os.replace 로 바꿔치기해서, 읽는 쪽이 절반만 쓰인 파일을
    보는 일이 없게 한다.
    - tempfile.NamedTemporaryFile(mode='w', dir=MODEL, delete=False):
        이름이 있는 임시 파일을 만든다. mode='w' 는 텍스트 쓰기, dir=MODEL 은 최종 파일과
        같은 폴더(같은 디스크)에 만들라는 뜻(다른 디스크면 원자적 교체가 안 된다),
        delete=False 는 with 블록이 끝나도 파일을 지우지 말라는 뜻이다.
    - os.replace(src, dst): src 를 dst 로 "원자적으로" 교체한다. 이미 dst 가 있어도 덮어쓰며,
        중간 상태가 외부에 보이지 않는다.
    """
    with tempfile.NamedTemporaryFile(mode='w', dir=MODEL, delete=False) as tmp:
        tmp.write(run.name)
    os.replace(tmp.name, MODEL / 'active')


def unit(x):
    """마지막 축 기준으로 각 벡터를 길이 1(단위 벡터)로 정규화한다.

        unit(x) = x / max(‖x‖, 1e-8)

    영벡터는 0으로 나누지 않도록 분모를 1e-8 이상으로 막는다.
    np.linalg.norm(x, axis=-1, keepdims=True): 마지막 축을 따라 L2 노름을 구하고,
    keepdims=True 로 차원을 유지해(예: (N,1)) 나눗셈이 브로드캐스트되게 한다.
    """
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-8)


# ---------------------------------------------------------------------------
# train() 의 하위 단계들
# ---------------------------------------------------------------------------
def read_zip_csv(z, name, **kw):
    """열려 있는 zip(z) 안의 'ml-20m/<name>.csv' 를 pandas 로 읽는다.

    zipfile.ZipFile.open(이름): 압축을 풀지 않고 zip 내부 파일을 스트림으로 연다.
    kw 는 pd.read_csv 로 그대로 전달된다 (usecols, dtype, chunksize 등).
    """
    return pd.read_csv(z.open('ml-20m/' + name + '.csv'), **kw)


def sample_users(z, users, rng):
    """학습용 표본 사용자(sample)와 콜드스타트 평가용 holdout 사용자를 뽑는다.

    - 표본: 전체 사용자 중 users 명을 비복원 추출 (users=0 이면 전원).
      ratings.csv 는 사용자별로 묶여 있으므로 "한 사용자의 전체 이력"을 통째로 쓴다.
    - holdout: 표본에 속하지 않은 사용자 중 최대 500명. 학습에 전혀 쓰이지 않으며,
      신규 사용자(콜드스타트) 평가에만 쓰인다.

    난수 소비 순서 주의: 여기서 rng(seed 42)를 가장 먼저 사용한다 (이후 p, q 초기화가
    같은 rng 를 이어서 쓴다). holdout 은 별도 시드 8의 독립 생성기를 쓴다.
    """
    # usecols=['userId']: 해당 열만 읽어 메모리 절약. dtype='int32': 정수 크기를 4바이트로 지정.
    ids = read_zip_csv(z, 'ratings', usecols=['userId'], dtype='int32').userId.unique()
    sample = rng.choice(ids, min(users, len(ids)), replace=False) if users else ids
    # np.setdiff1d(a, b): a 에는 있고 b 에는 없는 값(정렬된 고유값)을 돌려준다.
    others = np.setdiff1d(ids, sample)
    # np.random.default_rng(seed): 시드를 고정한 난수 생성기. .choice(a, n, replace=False)는
    # a 에서 중복 없이 n 개를 뽑는다.
    holdout = np.random.default_rng(8).choice(others, min(500, len(others)), replace=False)
    return sample, holdout


def load_ratings(z, sample, holdout):
    """ratings.csv 를 청크로 읽어 표본/holdout 사용자의 평점만 모은다.

    반환: (r, cold)
      r    : 표본 사용자의 평점 (userId, movieId 순 시간정렬, 인덱스는 정렬 전 값 유지)
      cold : holdout 사용자의 평점 (같은 기준으로 정렬)

    read_csv(chunksize=500000): 2천만 행을 한 번에 올리지 않고 50만 행씩 DataFrame 조각으로
    나눠 읽는 반복자(iterator)를 돌려준다. 메모리를 아끼기 위한 것이다.
    정렬 키 (userId, timestamp, movieId): 시간순 분할을 재현 가능하게 하려고 동점도 movieId 로 푼다.
    """
    chunks = read_zip_csv(z, 'ratings', chunksize=500000,
                          dtype={'userId': 'int32', 'movieId': 'int32',
                                 'rating': 'float32', 'timestamp': 'int64'})
    training, test_users = [], []
    for chunk in chunks:
        training.append(chunk[chunk.userId.isin(sample)])
        test_users.append(chunk[chunk.userId.isin(holdout)])
    r = pd.concat(training, ignore_index=True)
    cold = pd.concat(test_users).sort_values(['userId', 'timestamp', 'movieId'])
    r.sort_values(['userId', 'timestamp', 'movieId'], inplace=True)
    return r, cold


def assign_split(r):
    """사용자별 시간순으로 r['split'] 열을 채운다 (r 을 제자리 수정).

        위치 pos < 0.8n  -> 0 (train)
        위치 pos < 0.9n  -> 1 (validation)
        나머지           -> 2 (test)

    - groupby('userId').transform('size'): 각 행에 "그 사용자의 평점 개수 n" 을 붙인다
      (그룹별 결과를 원래 행 개수·순서에 맞춰 돌려주는 방식).
    - groupby('userId').cumcount(): 그룹 안에서 0,1,2,... 순번을 매긴다 (이미 시간순으로
      정렬되어 있으므로 순번 = 시간 순서).
    참고: 경계는 int(n*.8), int(n*.9) 로 내림한다.
    """
    group = r.groupby('userId')
    n = group.movieId.transform('size')
    pos = group.cumcount()
    r['split'] = np.where(pos < (n * .8).astype(int), 0,
                          np.where(pos < (n * .9).astype(int), 1, 2))


def select_movies(read, r):
    """학습(split==0) 평점에 한 번이라도 등장한 영화만 남겨 영화표와 색인을 만든다.

    반환: (movies, index)
      movies : movieId 오름차순 DataFrame (행 번호가 곧 모델의 영화 번호 i)
      index  : movieId -> 행 번호 변환용 pd.Index
    학습에 없는 영화는 요인 q 를 학습할 수 없으므로 제외한다.
    """
    t = r[r.split == 0]
    movies = read('movies')
    movies = movies[movies.movieId.isin(t.movieId)].sort_values('movieId').reset_index(drop=True)
    index = pd.Index(movies.movieId)
    return movies, index


def encode_ids(r, cold, index, run):
    """movieId/userId 를 모델 내부의 행 번호(i, u)로 바꾸고 콜드스타트 이력을 저장한다.

    - Index.get_indexer(값들): 각 값이 index 의 몇 번째 위치인지 정수 배열로 돌려준다.
      index 에 없는 값(학습에 안 나온 영화)은 -1 이 된다. 이후 코드가 i>=0 로 걸러낸다.
    - pd.factorize(x, sort=True): x 의 고유값을 정렬한 뒤 0,1,2,... 코드로 바꾼다.
      반환 (codes, uniques) 중 [0] 이 코드 배열이다. 여기서는 userId -> u (0..사용자수-1),
      p 행렬의 행 번호로 쓴다.
    r 을 제자리 수정하고, cold 는 new_users.csv.gz 로 저장한다 (평가 시 신규 사용자 이력으로 쓴다).
    """
    cold['i'] = index.get_indexer(cold.movieId)
    cold[['userId', 'i', 'rating']].to_csv(run / 'new_users.csv.gz', index=False)
    r['i'] = index.get_indexer(r.movieId)
    r['u'] = pd.factorize(r.userId, sort=True)[0]


def build_features(read, movies, index):
    """콘텐츠 기반(CB) 추천용 영화 특징 행렬을 만든다. 모양: (영화수, 장르수 + 태그수).

        features = unit([ unit(장르 원-핫) , unit(태그 게놈 관련도) ])

    - Series.str.get_dummies('|'): 'Action|Comedy' 처럼 '|' 로 구분된 문자열을 쪼개 장르별
      0/1 열(원-핫)로 만든다. sep 인자가 구분자다.
    - genome-scores.csv 는 매우 커서 chunksize=500000 으로 나눠 읽으며, 각 조각에서
      (영화, 태그) 위치에 관련도(relevance)를 채운다. 태그 번호는 1부터라서 -1 해 0부터로 맞춘다.
    - keep = (i>=0): 학습 영화 색인에 없는 영화(-1)는 버린다.
    """
    genres = unit(movies.genres.str.get_dummies('|').values.astype('float32'))
    tags = read('genome-tags')
    genome = np.zeros((len(index), int(tags.tagId.max())), 'float32')
    for c in read('genome-scores', chunksize=500000):
        i = index.get_indexer(c.movieId)
        keep = i >= 0
        genome[i[keep], c.tagId[keep] - 1] = c.relevance[keep]
    return unit(np.column_stack((genres, unit(genome))))


def popularity_stats(i, n_movies):
    """학습 평점 수(count), 정규화 인기도(pop), 롱테일 표시(tail)를 계산한다.

        pop_i = log(1 + count_i) / log(1 + max count)      (0~1)
        tail_i = 평점 수 상위 20% 영화가 아니면 True (롱테일)

    - np.bincount(i): 정수 배열 i 에서 각 값 0,1,2,...이 몇 번 나왔는지 센다 (영화별 평점 수).
    - np.log1p(x) = log(1+x): 0 근처에서도 정확한 로그. 인기도의 극단적 편차를 줄인다.
    - np.argsort(-count, kind='stable'): 평점 수 내림차순 정렬 순서. kind='stable' 은 같은 값끼리
      원래 순서를 유지하는 안정 정렬이라 동점 처리가 항상 같다(재현성).
    """
    count = np.bincount(i)
    pop = np.log1p(count) / np.log1p(count.max())
    tail = np.ones(n_movies, bool)
    tail[np.argsort(-count, kind='stable')[:int(np.ceil(n_movies * .2))]] = False
    return count, pop, tail


def init_factors(rng, n_users, n_movies, rank):
    """요인 행렬과 편향을 초기화한다.

    p (사용자 요인), q (영화 요인): 평균 0, 표준편차 0.07 정규분포 난수 (float32).
    bu, bi (사용자/영화 편향): 0.
    난수 소비 순서: p 먼저, q 나중. 순서를 바꾸면 결과가 달라진다.
    """
    p = rng.normal(0, .07, (n_users, rank)).astype('float32')
    q = rng.normal(0, .07, (n_movies, rank)).astype('float32')
    bu, bi = np.zeros(len(p), 'float32'), np.zeros(len(q), 'float32')
    return p, q, bu, bi


def run_sgd(rng, u, i, y, p, q, bu, bi, mean, epochs):
    """미니배치(256) SGD 로 p, q, bu, bi 를 제자리 갱신한다.

    각 epoch 마다 학습 평점 순서를 rng.permutation 으로 섞고 256개씩 처리한다.
        err = y - μ - b_u - b_i - p_u·q_i
        p_u ← p_u + lr(err·q_i - reg·p_u)      q_i ← q_i + lr(err·p_u - reg·q_i)
        b_u ← b_u + lr(err - reg·b_u)          b_i ← b_i + lr(err - reg·b_i)
        (lr = 0.015, reg = 0.04)
    중요: 한 배치 안에서 p, q 갱신은 "갱신 전 값" 을 써야 하므로 pa, qj 를 먼저 복사해 둔다
    (p[a] 같은 정수 배열 인덱싱은 복사본을 돌려준다). 갱신 순서는 p, q, bu, bi 로 고정이다.

    np.add.at(param, idx, vals): param[idx] += vals 와 비슷하지만, idx 에 같은 번호가 여러 번
    나와도 모두 누적해 더해 준다. (그냥 param[idx] += vals 는 중복 인덱스를 한 번만 반영한다.)
    같은 배치에 한 사용자/영화가 여러 번 나오기 때문에 반드시 add.at 이 필요하다.
    """
    for _ in range(epochs):
        order = rng.permutation(len(y))
        for start in range(0, len(order), 256):
            b = order[start:start + 256]       # 이번 미니배치의 평점 번호들
            a, j = u[b], i[b]                  # a: 사용자 번호들, j: 영화 번호들
            pa, qj = p[a], q[j]                # 갱신 전 요인 복사본
            err = y[b] - mean - bu[a] - bi[j] - (pa * qj).sum(1)
            for param, idx, grad in [(p, a, err[:, None] * qj), (q, j, err[:, None] * pa),
                                     (bu, a, err), (bi, j, err)]:
                np.add.at(param, idx, .015 * (grad - .04 * param[idx]))


def save_run(run, movies, r, t, sample, index, epochs, rank, arrays):
    """학습 산출물을 run 폴더에 저장한다.

    model.npz     : p, q, bu, bi, mean, pop, tail, features, count (arrays 딕셔너리)
    movies.json   : 영화표 (행 번호 = 모델의 영화 번호)
    split.csv.gz  : 평점별 (u, i, rating, split) -- 평가 단계가 읽는다
    training.json : 학습 설정 요약
    np.savez(파일, **배열들): 여러 배열을 이름을 붙여 하나의 .npz(압축 안 한 zip)로 저장한다.
    """
    MODEL.mkdir(exist_ok=True)
    np.savez(run / 'model.npz', **arrays)
    movies.to_json(run / 'movies.json', orient='records', force_ascii=False)
    r[['u', 'i', 'rating', 'split']].to_csv(run / 'split.csv.gz', index=False)
    (run / 'training.json').write_text(json.dumps(dict(
        users=len(sample), movies=len(index), ratings=len(r), train=len(t),
        epochs=epochs, rank=rank, seed=42)))


def train(archive, users, epochs, rank):
    """전체 학습 파이프라인. 데이터 읽기 -> 분할 -> 특징 -> SGD -> 저장 -> 평가 -> 활성화.

    archive: ml-20m.zip 경로, users: 표본 사용자 수(0=전체), epochs: SGD 반복 수, rank: 요인 차원.
    uuid.uuid4().hex: 무작위 128비트 식별자를 32자 16진수 문자열로 만든다. 실행마다 새 폴더 이름이 된다.
    난수 소비 순서(rng = seed 42): ① 표본 사용자 choice ② p 초기화 ③ q 초기화 ④ epoch 별 permutation.
    """
    rng = np.random.default_rng(42)
    run = MODEL / "runs" / uuid4().hex
    run.mkdir(parents=True)

    # --- 1) 데이터 읽기, 분할, 영화 색인, 특징 (zip 이 열려 있는 동안 수행) ---
    with zipfile.ZipFile(archive) as z:   # zip 을 압축 해제 없이 읽기 모드로 연다
        def read(name, **kw):
            return read_zip_csv(z, name, **kw)
        sample, holdout = sample_users(z, users, rng)
        r, cold = load_ratings(z, sample, holdout)
        assign_split(r)
        movies, index = select_movies(read, r)
        encode_ids(r, cold, index, run)
        features = build_features(read, movies, index)

    # --- 2) 학습 데이터(split==0) 추출 및 통계 ---
    t = r[r.split == 0]
    u, i, y = t.u.to_numpy(), t.i.to_numpy(), t.rating.to_numpy()
    count, pop, tail = popularity_stats(i, len(index))

    # --- 3) 초기화와 SGD 학습 ---
    p, q, bu, bi = init_factors(rng, len(sample), len(index), rank)
    mean = y.mean()   # 전체 평균 평점 μ (float32)
    run_sgd(rng, u, i, y, p, q, bu, bi, mean, epochs)

    # --- 4) 저장, 평가, 활성화 ---
    save_run(run, movies, r, t, sample, index, epochs, rank,
             dict(p=p, q=q, bu=bu, bi=bi, mean=mean, pop=pop, tail=tail,
                  features=features, count=count))
    from evaluate import run_evaluation   # 순환 import 를 피하려고 함수 안에서 import
    run_evaluation(run)
    publish(run)
    print('Activated', run)


# ---------------------------------------------------------------------------
# 사용자 선호 인기도 (순위 변환 실험용)
# ---------------------------------------------------------------------------
def user_preferences(pop, train_rows):
    """학습 구간 평점으로 사용자별 원본 선호 인기도 raw P_u 를 계산한다.

    무엇을/왜: 개인화 페널티 λ_u = λ_max·(1 - P_u) 의 P_u 를 사용자 전체에 대해 한 번에 구한다.
    순위 변환(rank_preference)의 참조 분포를 만들려면 학습 사용자 전원의 raw P_u 가 필요하다.
    Engine.scores 의 pref 와 같은 정의다:
        P_u = mean(pop[i])  (i: 사용자가 평점 ≥ THRESHOLD 로 평가한 영화), 그런 영화가 없으면 0.5
    데이터 누수: train_rows 는 학습 구간(split == 0) 행만 넣어야 한다. validation/test 평점을
    섞으면 평가 구간 정보가 P_u 에 새어 들어간다 (호출하는 쪽이 split == 0 만 넘길 책임).

    train_rows: u, i, rating 열을 가진 DataFrame.   pop: 영화별 정규화 인기도 배열.
    반환: 인덱스가 사용자 번호 u(오름차순)인 pd.Series (값은 float raw P_u).
    """
    liked = train_rows[train_rows.rating >= THRESHOLD]
    # 좋아한 영화의 pop 을 사용자별로 평균. 좋아한 영화가 없는 사용자는 reindex 로 NaN -> 0.5
    mean_pop = pd.Series(pop[liked.i.to_numpy(int)], index=liked.u.to_numpy()).groupby(level=0).mean()
    return mean_pop.reindex(np.sort(train_rows.u.unique())).fillna(.5)


def rank_preference(p, reference):
    """raw P_u 를 학습 사용자 분포 안에서의 분위수(0~1)로 바꾼다 (중간 순위 경험적 CDF).

    무엇을/왜: raw P_u 는 0.7 근처에 몰려 (1 - P_u) 가 사용자마다 거의 같다. 분위수로 바꾸면
    0~1 에 고르게 펴져 페널티가 사용자 사이에서 실제로 달라진다.
    수식:  rank(p) = (#{ref < p} + #{ref <= p}) / (2·|ref|)
        동점(ref == p)은 구간의 중간값을 받는다. 결과는 0~1.
    np.searchsorted(a, v, side): 오름차순 정렬된 배열 a 에서 v 를 넣어도 정렬이 유지되는 위치(=앞쪽 원소 수).
        a: 정렬된 참조 배열, v: 찾을 값(스칼라 또는 배열),
        side='left'  -> a 에서 v 보다 작은 원소 수 (v 와 같은 값 앞쪽 위치),
        side='right' -> a 에서 v 이하인 원소 수 (v 와 같은 값 뒤쪽 위치).
    데이터 누수: reference 는 학습 구간(split == 0)에서 만든 학습 사용자들의 raw P_u 만 쓴다.

    p: 스칼라 또는 배열.   reference: 학습 사용자들의 raw P_u (정렬되지 않아도 됨).
    반환: p 와 같은 모양의 0~1 값 (스칼라 입력이면 float).
    """
    ref = np.sort(np.asarray(reference, float))
    p = np.asarray(p, float)
    rank = (np.searchsorted(ref, p, 'left') + np.searchsorted(ref, p, 'right')) / (2 * len(ref))
    return float(rank) if rank.ndim == 0 else rank


# ---------------------------------------------------------------------------
# 추천 엔진
# ---------------------------------------------------------------------------
class Engine:
    """저장된 run 을 읽어 점수 계산, 후보 풀 생성, 재순위를 수행한다.

    model.npz 의 모든 배열(p, q, bu, bi, mean, pop, tail, features, count ...)이 같은 이름의
    속성으로 붙고, 영화표는 self.movies (dict 목록)로 붙는다.
    """

    def __init__(self, run=None):
        run = run or active_model()
        with np.load(run / 'model.npz') as data:
            self.__dict__.update({k: data[k] for k in data.files})
        self.movies = json.loads((run / 'movies.json').read_text())

    def scores(self, items, ratings, user=None):
        """한 사용자에 대해 (cf, cb, pref) 를 돌려준다.

        items/ratings: 그 사용자가 이미 평가한 영화 번호와 평점.
        user: 학습에 쓰인 사용자면 번호(u) -> 저장된 bu, p 사용.
              None 이면 신규 사용자 -> 아래 ridge fold-in 으로 즉석 계산.
        반환:
          cf   : 전체 영화에 대한 CF 점수 (0~1)
          cb   : 전체 영화에 대한 CB 점수 (좋아한 영화 평균 특징과의 코사인 유사도)
          pref : 좋아한 영화들의 평균 인기도 (0=비주류 취향, 1=주류 취향; 없으면 0.5)

        [ridge fold-in] 영화 요인 q 를 고정한 채 새 사용자의 편향 b 와 요인 p 만 구한다.
            x = [1, q_items]           (평가한 영화마다 한 행)
            target = r - μ - b_i
            [b, p] = argmin ‖x·w - target‖² + 0.5‖w‖²  =  (xᵀx + 0.5·I)⁻¹ xᵀ target
          np.linalg.solve(A, b): 역행렬을 직접 만들지 않고 연립방정식 A·w = b 의 해 w 를 구한다
          (역행렬보다 빠르고 수치적으로 안정). 0.5·I 항이 정규화(ridge)다.
          np.eye(n): n×n 단위행렬.

        CF 점수: cf = clip((μ + b + b_i + q·p - 0.5) / 4.5, 0, 1)
          평점 범위 0.5~5.0 을 0~1 로 선형 변환하고 np.clip 으로 범위를 벗어난 값을 잘라낸다.
        CB 점수: cb = features · unit(mean(features[liked]))   (liked: 평점 ≥ THRESHOLD)
        """
        items, ratings = np.asarray(items, int), np.asarray(ratings, float)
        liked = items[ratings >= THRESHOLD]

        if user is None:
            # 신규 사용자: ridge fold-in (설명은 docstring 참고). 열 0 이 사용자 편향용 상수 1.
            x = np.column_stack((np.ones(len(items)), self.q[items]))
            gram = x.T @ x + np.eye(x.shape[1]) * .5
            target = x.T @ (ratings - self.mean - self.bi[items])
            coef = np.linalg.solve(gram, target)
            bias, factors = coef[0], coef[1:]
        else:
            # 학습에 쓰인 사용자: 저장된 값을 그대로 사용
            bias, factors = self.bu[user], self.p[user]

        # CF 점수: 평점 예측 -> 0~1 로 변환
        raw = self.mean + bias + self.bi + self.q @ factors
        cf = np.clip((raw - .5) / 4.5, 0, 1)

        # CB 점수: 좋아한 영화가 없으면 중립값 0.5
        if len(liked):
            cb = self.features @ unit(self.features[liked].mean(0))
        else:
            cb = np.full(len(cf), .5)

        pref = float(self.pop[liked].mean()) if len(liked) else .5
        return cf, cb, pref

    def pool(self, cf, cb, alpha, seen, content_slots=0):
        """후보 풀(최대 100개)과 하이브리드 점수를 돌려준다. 반환: (pool, score).

            score = alpha·cf + (1 - alpha)·cb      (alpha=1 이면 CF만, 0 이면 CB만)

        이미 본 영화(seen)를 뺀 상위 100개가 기본 풀이다.
        content_slots > 0 이면 풀 끝의 content_slots 자리를 "CB 점수가 높은 영화" 로 바꿔 넣어
        롱테일 후보를 늘린다 (core = 하이브리드 상위 100-content_slots 개, 나머지는 CB 상위).
        - np.argsort(-score, kind='stable'): 점수 내림차순 순서, 동점은 원래 번호 순(안정 정렬).
        - np.isin(a, b): a 의 각 원소가 b 에 들어 있는지 불리언 배열로 돌려준다.
        - np.ptp(cb): 최댓값 - 최솟값. 0 이면 CB 점수가 전부 같아 쓸모 없으므로 교체를 건너뛴다.
        - np.flatnonzero(mask): True 인 위치의 번호들.
        """
        score = alpha * cf + (1 - alpha) * cb
        order = np.argsort(-score, kind='stable')
        pool = order[~np.isin(order, seen)][:100]
        if content_slots and len(pool) == 100 and np.ptp(cb) > 0:
            core = pool[:100 - content_slots]
            # 교체 후보: 본 적 없고 core 에도 없는 영화 (seen 의 -1 은 무시)
            eligible = np.ones(len(cb), dtype=bool)
            observed = np.asarray(seen, dtype=int)
            eligible[observed[observed >= 0]] = False
            eligible[core] = False
            candidates = np.flatnonzero(eligible)
            similar = candidates[np.argsort(-cb[candidates], kind='stable')[:content_slots]]
            pool = np.concatenate((core, similar))
        return pool, score

    def rank(self, pool, score, penalty):
        """후보 풀을 인기도 페널티로 재순위해 최종 상위 10개를 돌려준다.

            최종 점수 = score - penalty · pop     (pop: 0~1 정규화 인기도)

        penalty 가 클수록 인기 영화가 밀려나 롱테일 영화가 올라온다.
        적응형(adaptive) 방식에서는 호출하는 쪽이 penalty = λ_max · (1 - P_u) 를 넘겨준다
        (P_u: 사용자의 평균 선호 인기도, scores() 의 pref). 주류 취향일수록 페널티가 작다.
        """
        final = score[pool] - penalty * self.pop[pool]
        return pool[np.argsort(-final, kind='stable')[:10]]


if __name__ == '__main__':
    cli = argparse.ArgumentParser()
    cli.add_argument('archive')
    cli.add_argument('--users', type=int, default=2000, help='0 = all users')
    cli.add_argument('--epochs', type=int, default=10)
    cli.add_argument('--rank', type=int, default=24)
    args = cli.parse_args()
    train(args.archive, args.users, args.epochs, args.rank)
