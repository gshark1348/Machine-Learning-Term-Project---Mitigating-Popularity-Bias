"""학습 데이터만 이용한 '추천 근거' 계산 모듈 (영화 카드 정보 + 이웃 기반 추천).

역할
    1) 영화마다 평균 평점·투표 수·대표 태그·롱테일 여부를 계산해 영화 dict에 붙인다.
    2) evidence(): 사용자가 준 평점과 비슷하게 평가한 학습 사용자(이웃)를 찾아,
       그 이웃들이 좋아한 영화의 지지도를 계산한다.
    3) explain(): 추천 영화마다 사람이 읽을 수 있는 추천 이유 문장을 만든다.
    4) neighbor_picks(): 이웃 지지도를 반영한 별도 추천 목록(10개)을 만든다.

'학습 데이터만' 쓰는 이유: 검증/테스트 구간(split != 0)의 정보가 근거에 섞이면
평가가 오염되므로 split == 0 인 행만 사용한다.

다른 스크립트가 `from discovery import Discovery, THRESHOLD` 로 가져다 쓰므로
클래스/메서드 이름과 THRESHOLD(model.py에서 가져온 값)를 바꾸면 안 된다.
"""
import numpy as np
import pandas as pd
from model import ROOT, THRESHOLD

# ---- 상수 (원래 코드에 숫자로 박혀 있던 값들. 값은 그대로) ----
TOP_TAGS = 5            # 영화당 보여줄 대표 태그 수
POPULAR_COUNT = 10      # /popular 에 보여줄 영화 수
MIN_AGREEMENT = .75     # 이웃 후보가 되기 위한 평균 일치도 하한
SHRINK = 3              # 축소(shrinkage) 상수 (evidence 설명 참고)
NEIGHBORS = 30          # 사용하는 이웃 수
MAX_RATING_GAP = 4.5    # 평점 차이의 최댓값 (0.5점 ~ 5점 이므로 5 - 0.5)
SUPPORT_BONUS = .12     # 이웃 지지도가 점수에 더해줄 수 있는 최대 가산점
SUPPORT_HALF = 2        # 지지도 가산이 포화되는 속도 (weight/(weight+2))
PICKS = 10              # 이웃 기반 목록 크기
TAIL_SLOTS = 3          # 이웃 기반 목록에 최소한 넣으려 하는 롱테일 영화 수


class Discovery:
    def __init__(self, engine, run):
        """영화별 통계를 미리 계산해 engine.movies 의 각 dict 에 직접 채워 넣는다.

        engine: model.Engine (영화 목록, 특징 행렬, 롱테일 표시 등을 가짐)
        run   : 학습 결과 폴더 경로 (split.csv.gz 가 들어 있음)
        """
        self.e = engine

        # split.csv.gz: 컬럼 u(사용자), i(영화 행 번호), rating, split(0=학습).
        # 학습용(split == 0) 평점만 남긴다. .loc[행조건, 열목록] 은 pandas 의 행/열 선택.
        split = pd.read_csv(run / 'split.csv.gz')
        self.train = split.loc[split.split == 0, ['u', 'i', 'rating']]

        # 영화(i)별 평점 평균(mean)과 평점 개수(count).
        # groupby('i').rating.agg(['mean','count']): i 가 같은 행끼리 묶어
        # rating 열에 평균과 개수를 한꺼번에 계산 -> index=i, 열=['mean','count'] 인 표.
        stats = self.train.groupby('i').rating.agg(['mean', 'count'])

        # genome-tags.csv(prepare_metadata.py 로 생성)의 태그 이름들을 tagId 순서로 배열화.
        # 영화 특징 행렬의 '마지막 태그 수'개 열이 곧 각 태그의 점수(semantic)라는 약속이다.
        tags = pd.read_csv(ROOT / 'genome-tags.csv').sort_values('tagId').tag.to_numpy()
        self.semantic = engine.features[:, -len(tags):]
        self.tags = tags

        for i, movie in enumerate(engine.movies):
            movie.update(average=float(stats.at[i, 'mean']),   # DataFrame.at[행, 열]: 단일 값 빠른 조회
                         voters=int(stats.at[i, 'count']),
                         tail=bool(engine.tail[i]),            # 롱테일(비인기) 영화 여부
                         tags=self._top_tags(i))

        # 투표 수가 많은 순, 같으면 movieId 가 작은 순으로 상위 10개
        self.popular = sorted(engine.movies, key=lambda m: (-m['voters'], m['movieId']))[:POPULAR_COUNT]

    def _top_tags(self, i):
        """영화 i 에서 점수가 가장 높은 태그 이름 TOP_TAGS 개 (태그 정보가 전혀 없으면 빈 목록)."""
        vector = self.semantic[i]
        if vector.any():
            # np.argsort(x, kind='stable'): x 를 오름차순으로 정렬했을 때의 '원래 위치 번호'를 반환.
            # -vector 로 부호를 뒤집어 큰 점수가 앞에 오게 하고, kind='stable' 은 동점일 때
            # 원래 순서(태그 번호 순)를 유지해 결과가 항상 같게 만든다.
            best = np.argsort(-vector, kind='stable')[:TOP_TAGS]
        else:
            best = []
        return [self.tags[j] for j in best]

    # ------------------------------------------------------------------
    # 이웃 찾기: 사실상 '사용자 기반 memory-based 협업필터링(user-based CF)'
    # ------------------------------------------------------------------
    def evidence(self, seen, ratings):
        """사용자와 취향이 비슷한 학습 사용자 30명을 찾고, 그들이 좋아한 영화의 지지도를 구한다.

        seen   : 사용자가 평점을 준 영화들의 행 번호 리스트
        ratings: 같은 순서의 평점 리스트
        반환   : index=영화 행 번호, 열 ['sum','count'] 인 표 (support)
                 sum   = 그 영화를 좋아한(4점 이상) 이웃들의 유사도 합
                 count = 그 영화를 좋아한 이웃 수

        학습이 필요 없고 평점 데이터를 요청 시점에 직접 비교하므로 memory-based CF 이다.

        참고: 요청마다 self.train 전체를 isin 으로 필터링한다. 데이터가 크면(전체 규모)
              요청당 시간이 길어질 수 있다. (동작은 그대로 두었음)
        """
        # 1) 사용자가 준 평점을 {영화: 평점} 사전으로
        supplied = dict(zip(seen, ratings))

        # 2) 사용자가 평점 준 영화를 '같이' 평가한 학습 평점만 추림 (공통 영화)
        overlap = self._overlapping_ratings(seen)

        # 3) 공통 영화마다 일치도: 1 - |남의 평점 - 내 평점| / 4.5
        #    같은 점수면 1, 최대로 다르면(0.5 vs 5) 0.
        #    Series.map(dict): 각 영화 번호를 사전에서 찾아 내 평점으로 바꿔 준다.
        overlap['agreement'] = 1 - (overlap.rating - overlap.i.map(supplied)).abs() / MAX_RATING_GAP

        # 4) 사용자(u)별로 평균 일치도와 공통 영화 수를 계산하고, 평균 일치도 0.75 이상만 남긴다.
        scores = overlap.groupby('u').agreement.agg(['mean', 'count'])
        scores = scores[scores['mean'] >= MIN_AGREEMENT]

        # 5) 유사도 = mean * count / (count + 3)   <- 축소(shrinkage)
        #    공통 영화가 1~2편뿐인 사용자는 우연히 평균 일치도가 높을 수 있다.
        #    count/(count+3) 은 count 가 작으면 0 에 가깝고(신뢰 낮음) 커질수록 1 에 가까워져
        #    공통 영화가 적은 사용자의 유사도를 깎아 준다. 예) count=1 -> 0.25, count=9 -> 0.75.
        similarity = scores['mean'] * scores['count'] / (scores['count'] + SHRINK)

        # 6) 유사도 상위 30명 = 이웃. Series.nlargest(n): 값이 큰 n개를 큰 순으로 반환.
        neighbors = similarity.nlargest(NEIGHBORS)

        # 7) 이웃들이 4점 이상(THRESHOLD)으로 평가한 영화 -> 영화별로 유사도 합과 인원 수 집계
        return self._neighbor_support(neighbors)

    def _overlapping_ratings(self, seen):
        """학습 평점 중 사용자가 평점을 준 영화(seen)에 대한 행만 복사해서 반환."""
        return self.train[self.train.i.isin(seen)].copy()

    def _neighbor_support(self, neighbors):
        """이웃(neighbors: index=사용자, 값=유사도)이 좋아한 영화별 지지도 표를 만든다."""
        liked = self.train[self.train.u.isin(neighbors.index) & (self.train.rating >= THRESHOLD)].copy()
        liked['weight'] = liked.u.map(neighbors)  # 각 행에 그 사용자의 유사도를 붙임
        return liked.groupby('i').weight.agg(['sum', 'count'])

    # ------------------------------------------------------------------
    # 추천 이유 문장
    # ------------------------------------------------------------------
    def explain(self, i, liked, support):
        """영화 i 의 정보 dict 에 추천 이유(reasons) 문장 리스트를 붙여 반환한다.

        i      : 추천 영화의 행 번호
        liked  : 사용자가 좋아한(4점 이상) 영화들의 행 번호 배열
        support: evidence() 의 반환 표
        이유는 최대 두 종류: (a) 이웃 지지, (b) 좋아한 영화와의 유사성(태그 또는 장르).
        둘 다 없으면 기본 문장 하나.
        """
        movie = self.e.movies[i]
        reasons = []

        # (a) 이웃 지지: 이 영화를 4점 이상 준 이웃 수
        if i in support.index:
            reasons.append(f"공통 영화에 비슷한 평점을 준 사용자 중 {int(support.at[i,'count'])}명이 이 영화에 4점 이상을 줬어요.")

        # (b) 좋아한 영화 중 특징이 가장 비슷한 한 편을 골라 이유로 쓴다.
        if len(liked):
            closest = self._closest_liked(i, liked)
            reason = self._similarity_reason(i, closest)
            if reason:
                reasons.append(reason)

        if not reasons:
            reasons.append("입력한 별점으로 추정한 취향과 영화 특성을 조합해 추천한 작품이에요.")
        return dict(movie, reasons=reasons)

    def _closest_liked(self, i, liked):
        """좋아한 영화들 중 영화 i 와 특징 벡터의 내적(@)이 가장 큰 영화의 행 번호.
        np.argmax 는 동점이면 가장 앞의 것을 고른다."""
        return liked[np.argmax(self.e.features[liked] @ self.e.features[i])]

    def _similarity_reason(self, i, closest):
        """영화 i 와 좋아한 영화 closest 가 비슷한 이유 문장 (없으면 None).

        우선 공통으로 두드러진 태그, 태그가 겹치지 않으면 공통 장르를 쓴다.
        """
        movie = self.e.movies[i]
        closest_movie = self.e.movies[closest]
        # np.minimum(a, b): 원소별 최솟값 -> 두 영화 모두에서 점수가 높은 태그가 큰 값을 가진다.
        shared = np.minimum(self.semantic[i], self.semantic[closest])
        if shared.any():
            tag = self.tags[np.argmax(shared)]
            return f"좋아한 〈{closest_movie['title']}〉와 ‘{tag}’ 특성이 두드러져요."
        # 장르는 'A|B|C' 형태 문자열. 집합 교집합(&)으로 공통 장르를 구하고 자리표시 값은 뺀다.
        genres = sorted(set(movie['genres'].split('|')) & set(closest_movie['genres'].split('|')) - {'(no genres listed)'})
        if genres:
            return f"좋아한 〈{closest_movie['title']}〉와 {', '.join(genres)} 장르가 같아요."
        return None

    # ------------------------------------------------------------------
    # 이웃 기반 추천 목록
    # ------------------------------------------------------------------
    def neighbor_picks(self, pool, score, support):
        """이웃 지지도를 반영해 추천 10개를 고르고, 롱테일 영화를 최대 3개 보장한다.

        pool   : 추천 후보 영화 행 번호들
        score  : 영화별 모델 점수 (행 번호로 인덱싱)
        support: evidence() 의 반환 표
        반환   : 영화 행 번호 리스트 (정렬 순서는 아래 ordered 의 순서)
        """
        # 1) 이웃이 한 명이라도 좋아한 후보만 남긴다.
        candidates = [i for i in pool if i in support.index]
        if not candidates:
            return []

        # 2) 점수 = 모델 점수 + 0.12 * w/(w+2).  w/(w+2) 는 0~1 로 포화되므로
        #    투표(지지)가 아주 많은 인기 영화가 가산점을 독식하지 못한다(영향 상한).
        #    sorted(key=-점수): 점수 내림차순. 동점이면 sorted 가 안정 정렬이라 candidates 순서 유지.
        weight = support['sum']
        ordered = sorted(candidates, key=lambda i: -(score[i] + SUPPORT_BONUS * weight[i] / (weight[i] + SUPPORT_HALF)))

        # 3) 상위 10개를 기본으로 하되, 롱테일 영화를 최대 3개까지 자리를 보장한다.
        picks = ordered[:PICKS]
        tail = [i for i in ordered if self.e.tail[i]][:TAIL_SLOTS]
        for i in tail:
            if i not in picks:
                # 뒤(낮은 순위)에서부터 롱테일이 아닌 자리를 찾아 교체
                # 참고: picks 가 모두 롱테일이면 next() 가 StopIteration 을 낼 수 있다(방어 코드 없음).
                #       현재는 롱테일을 최대 3개만 넣고 picks 가 보통 10개라 사실상 발생하지 않는다.
                #       동작은 그대로 두었음.
                replace = next(j for j in reversed(range(len(picks))) if not self.e.tail[picks[j]])
                picks[replace] = i

        # 4) 최종 표시 순서는 ordered(점수) 순서로 되돌린다. ordered.index = 그 영화의 순위.
        return sorted(picks, key=ordered.index)
