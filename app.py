import json
import math
import numpy as np
from discovery import Discovery, THRESHOLD
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from model import Engine, active_model, ROOT
api = FastAPI(title='Beyond Popularity')
run = active_model()
engine = Engine(run)
settings=json.loads((run/'settings.json').read_text())
poster_ids=json.loads((ROOT/'poster_ids.json').read_text())
for movie in engine.movies:
    imdb = poster_ids.get(str(movie['movieId']))
    movie['poster'] = f'https://images.metahub.space/poster/small/{imdb}/img' if imdb else None
indices={m['movieId']:i for i,m in enumerate(engine.movies)}

discovery = Discovery(engine,run)

@api.get('/popular')
def popular():
    return discovery.popular

@api.get('/movie/{movie_id}')
def detail(movie_id:int):
    if movie_id not in indices:
        raise HTTPException(404,'영화를 찾을 수 없습니다.')
    return engine.movies[indices[movie_id]]

@api.get('/')
def home():
    return FileResponse(ROOT/'index.html')

@api.get('/movies')
def movies(q:str='',offset:int=Query(0,ge=0)):
    found=(m for m in engine.movies if q.casefold() in m['title'].casefold())
    from itertools import islice
    return list(islice(found,offset,offset+30))

@api.post('/recommend')
def recommend(ratings:dict[int,float], strength:float=Query(default=None,ge=0,le=1), watched:list[int]=Query(default=[])):
    if not ratings or any(i not in indices or not math.isfinite(r) or not .5<=r<=5 or r*2!=int(r*2) for i,r in ratings.items()):
        raise HTTPException(400,'영화와 0.5 단위 평점을 확인해 주세요.')
    if any(i not in indices for i in watched):
        raise HTTPException(400, "알 수 없는 영화입니다.")
    seen=[indices[i] for i in ratings]
    cf,cb,p=engine.scores(seen,list(ratings.values()))
    pool,score=engine.pool(cf,cb,settings['alpha'],seen+[indices[i] for i in watched],settings.get("content_slots",0))
    penalties=dict(hybrid=0,fixed=settings['fixed'] if strength is None else strength,adaptive=(settings['adaptive'] if strength is None else strength)*(1-p))
    support = discovery.evidence(seen,list(ratings.values()))
    liked = np.asarray(seen)[np.asarray(list(ratings.values())) >= THRESHOLD]
    ranked = {m:engine.rank(pool,score,k) for m,k in penalties.items()}
    ranked['neighbors'] = discovery.neighbor_picks(pool,score,support)
    descriptions = {i:discovery.explain(i,liked,support) for i in set().union(*ranked.values())}
    return dict(preference=p,penalties=penalties,lists={m:[descriptions[i] for i in items] for m,items in ranked.items()})
