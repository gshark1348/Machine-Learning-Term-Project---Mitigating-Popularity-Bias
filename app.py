"""영화 추천 데모 웹 서버 (FastAPI).

역할
    학습된 추천 모델(model.Engine)과 설명 근거(discovery.Discovery)를
    HTTP API로 노출하고, 화면(index.html)을 함께 서빙한다.

실행 방법 (uvicorn은 FastAPI 앱을 실제로 띄워 주는 ASGI 서버)
    uvicorn app:api --host 127.0.0.1 --port 18765
    - "app:api"  : app.py 모듈 안의 `api` 객체를 실행하라는 뜻
    - --host     : 접속을 받을 주소 (127.0.0.1 = 내 컴퓨터에서만 접속)
    - --port     : 접속 포트 번호

전체 흐름
    1) 서버 시작 시 모델·영화 목록·포스터 주소를 한 번만 메모리에 올린다.
    2) 사용자가 영화에 매긴 평점을 POST /recommend 로 보낸다.
    3) 평점 -> (협업/콘텐츠) 점수 -> 후보 pool -> 방식별(hybrid/fixed/adaptive)
       인기도 페널티로 재정렬 + 이웃 기반(neighbors) 목록 -> 설명 문구 붙여 응답.
"""
import json
import math
from itertools import islice  # islice(iterable, start, stop): 제너레이터를 통째로 만들지 않고 [start:stop] 구간만 꺼냄

import numpy as np
from discovery import Discovery, THRESHOLD
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from model import Engine, active_model, ROOT

# ---------------------------------------------------------------------------
# 서버 시작 시 1회 실행되는 초기화
# ---------------------------------------------------------------------------

# FastAPI(title=...): 웹 앱 객체를 만든다. title은 자동 문서(/docs)에 표시되는 이름.
# uvicorn이 `app:api`로 이 객체를 찾으므로 이름 `api`를 바꾸면 안 된다.
api = FastAPI(title='Beyond Popularity')

run = active_model()                 # 현재 사용할 학습 결과 폴더 (artifacts/active)
engine = Engine(run)                 # 점수 계산·재정렬을 담당하는 모델
settings = json.loads((run / 'settings.json').read_text())  # alpha, fixed, adaptive 등 학습 때 정한 설정
poster_ids = json.loads((ROOT / 'poster_ids.json').read_text())  # movieId -> IMDb id (prepare_metadata.py가 생성)

# 영화마다 포스터 URL을 붙인다.
# 포스터 이미지 출처: 외부 서비스 images.metahub.space (IMDb id로 포스터 제공).
# 우리가 이미지를 저장하지 않고 브라우저가 이 주소에서 직접 불러온다. 없으면 None.
for movie in engine.movies:
    imdb = poster_ids.get(str(movie['movieId']))
    movie['poster'] = f'https://images.metahub.space/poster/small/{imdb}/img' if imdb else None

# movieId -> engine.movies 안의 위치(행 번호). 요청마다 빠르게 찾기 위한 색인.
indices = {m['movieId']: i for i, m in enumerate(engine.movies)}

# 영화별 평균 평점·투표 수·태그와 "비슷한 취향 사용자" 근거를 계산해 주는 객체 (discovery.py)
discovery = Discovery(engine, run)


# ---------------------------------------------------------------------------
# 엔드포인트
# 데코레이터 @api.get(경로) / @api.post(경로): 아래 함수를 해당 HTTP 메서드·경로의
# 처리기로 등록한다. 함수의 반환값(dict/list)은 FastAPI가 자동으로 JSON으로 바꿔 응답한다.
# ---------------------------------------------------------------------------

@api.get('/popular')
def popular():
    """GET /popular
    입력: 없음
    출력: 학습 데이터에서 투표 수가 가장 많은 영화 10개 (영화 정보 dict의 리스트).
    """
    return discovery.popular


@api.get('/movie/{movie_id}')
def detail(movie_id: int):
    """GET /movie/{movie_id}
    입력: 경로 파라미터 movie_id. 타입이 int로 선언돼 있어 FastAPI가 URL 문자열을
          정수로 자동 변환·검증한다 (정수가 아니면 422 오류).
    출력: 해당 영화 정보 dict. 없는 영화면 404.
    """
    if movie_id not in indices:
        # HTTPException(status_code, detail): 이 상태 코드와 메시지로 오류 응답을 즉시 반환
        raise HTTPException(404, '영화를 찾을 수 없습니다.')
    return engine.movies[indices[movie_id]]


@api.get('/')
def home():
    """GET /
    출력: 화면 파일 index.html (JSON이 아니라 파일 그대로).
    FileResponse(path): 해당 경로의 파일 내용을 응답 본문으로 보낸다.
    """
    return FileResponse(ROOT / 'index.html')


@api.get('/movies')
def movies(q: str = '', offset: int = Query(0, ge=0)):
    """GET /movies?q=검색어&offset=시작위치
    입력: q      - 제목에 포함된 글자 (대소문자 무시, 빈 문자열이면 전체)
          offset - 건너뛸 개수. Query(기본값, ge=0): 쿼리스트링 파라미터이며
                   ge=0 (greater or equal) 이라 0 미만이면 FastAPI가 422로 거부한다.
    출력: 제목이 맞는 영화를 최대 30개 (offset부터). 다음 페이지는 offset+30 으로 요청.
    """
    # casefold(): lower()보다 강한 대소문자 무시 비교용 변환
    found = (m for m in engine.movies if q.casefold() in m['title'].casefold())
    return list(islice(found, offset, offset + 30))


def is_valid_rating_entry(movie_id, rating):
    """평점 한 건이 올바른지 검사한다 (True면 정상).

    조건 (왼쪽부터 차례로 검사, 하나라도 어기면 정상이 아님):
      - 모델이 아는 영화여야 함
      - 유한한 숫자여야 함 (math.isfinite: nan, inf는 False)
      - 0.5 이상 5 이하
      - 0.5 단위여야 함: 2배 했을 때 정수 (예: 3.5*2=7.0 OK, 3.3*2=6.6 불가)
    """
    if movie_id not in indices:
        return False
    if not math.isfinite(rating):
        return False
    if not .5 <= rating <= 5:
        return False
    return rating * 2 == int(rating * 2)


def ratings_are_invalid(ratings):
    """평점 dict가 비었거나 한 건이라도 잘못되면 True."""
    return not ratings or any(not is_valid_rating_entry(i, r) for i, r in ratings.items())


def compute_penalties(strength, preference):
    """추천 방식별 인기도 페널티 세기를 정한다.

    hybrid  : 페널티 없음 (0). 기준선.
    fixed   : 모든 사용자에게 같은 세기. strength를 주면 그 값, 아니면 학습 설정값.
    adaptive: 사용자의 인기작 선호도(preference, 0~1)가 높을수록 페널티를 줄인다
              -> 세기 * (1 - preference).
    반환 dict의 순서(hybrid, fixed, adaptive)가 응답 순서가 되므로 바꾸지 말 것.
    """
    return dict(
        hybrid=0,
        fixed=settings['fixed'] if strength is None else strength,
        adaptive=(settings['adaptive'] if strength is None else strength) * (1 - preference),
    )


def describe_lists(ranked, liked, support):
    """영화 번호 목록들을 '영화 정보 + 추천 이유' 목록으로 바꾼다.

    여러 방식에 같은 영화가 나올 수 있으므로, 중복을 제거(set)해 설명을 한 번씩만 만든 뒤
    각 목록에서 재사용한다.
    """
    all_items = set().union(*ranked.values())  # 모든 목록의 영화 번호 합집합
    descriptions = {i: discovery.explain(i, liked, support) for i in all_items}
    return {method: [descriptions[i] for i in items] for method, items in ranked.items()}


@api.post('/recommend')
def recommend(ratings: dict[int, float], strength: float = Query(default=None, ge=0, le=1),
              watched: list[int] = Query(default=[])):
    """POST /recommend?strength=0.3&watched=10&watched=20
    입력:
      - 요청 본문(JSON) ratings: {"영화ID": 평점} 형태. 타입이 dict[int, float]로 선언돼
        있어 FastAPI가 본문 JSON을 읽어 키는 int, 값은 float로 자동 변환한다.
        예) {"1": 5, "260": 4.5}
      - 쿼리 strength (선택, 0~1): 페널티 세기를 직접 지정. 없으면 학습 설정값 사용.
        Query(default=None, ge=0, le=1) 이므로 범위를 벗어나면 422.
      - 쿼리 watched (선택, 반복 가능): 평점은 안 줬지만 이미 본 영화 ID. 추천에서 제외.
    출력(JSON):
      - preference: 모델이 추정한 사용자의 인기작 선호도 (0~1)
      - penalties : 방식별 페널티 세기 {hybrid, fixed, adaptive}
      - lists     : 방식별(hybrid/fixed/adaptive/neighbors) 추천 목록.
                    각 항목은 영화 정보 + reasons(추천 이유 문장 리스트)
    오류: 평점·영화 ID가 잘못되면 400.
    """
    # 1) 입력 검증
    if ratings_are_invalid(ratings):
        raise HTTPException(400, '영화와 0.5 단위 평점을 확인해 주세요.')
    if any(i not in indices for i in watched):
        raise HTTPException(400, "알 수 없는 영화입니다.")

    # 2) 평점 -> 점수·후보 계산
    seen = [indices[i] for i in ratings]  # 평점 준 영화의 내부 행 번호
    rating_values = list(ratings.values())
    # cf: 협업필터링 점수, cb: 콘텐츠 기반 점수, p: 사용자의 인기작 선호도
    cf, cb, p = engine.scores(seen, rating_values)
    # 이미 평점 준 영화 + watched 영화는 제외하고, 두 점수를 alpha로 섞어 후보(pool)와 점수 생성
    pool, score = engine.pool(cf, cb, settings['alpha'], seen + [indices[i] for i in watched],
                              settings.get("content_slots", 0))

    # 3) 방식별 페널티와 재정렬
    penalties = compute_penalties(strength, p)
    ranked = {m: engine.rank(pool, score, k) for m, k in penalties.items()}

    # 4) 이웃 기반 목록 + 설명 근거
    # support: 비슷한 취향의 이웃들이 좋아한 영화별 지지 정도 (discovery.evidence 참고)
    support = discovery.evidence(seen, rating_values)
    # liked: 사용자가 THRESHOLD(4점) 이상 준 영화의 행 번호. 설명 문구에서 "좋아한 영화"로 쓴다.
    liked = np.asarray(seen)[np.asarray(rating_values) >= THRESHOLD]
    ranked['neighbors'] = discovery.neighbor_picks(pool, score, support)

    # 5) 설명 붙여 응답
    return dict(preference=p, penalties=penalties, lists=describe_lists(ranked, liked, support))
