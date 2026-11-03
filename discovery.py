"""Training-only evidence behind movie cards and neighbor recommendations."""
import numpy as np
import pandas as pd
from model import ROOT, THRESHOLD

class Discovery:
    def __init__(self, engine, run):
        self.e = engine
        split = pd.read_csv(run/'split.csv.gz')
        self.train = split.loc[split.split == 0, ['u','i','rating']]
        stats = self.train.groupby('i').rating.agg(['mean','count'])
        tags = pd.read_csv(ROOT/'genome-tags.csv').sort_values('tagId').tag.to_numpy()
        self.semantic = engine.features[:, -len(tags):]
        self.tags = tags
        for i, movie in enumerate(engine.movies):
            vector = self.semantic[i]
            best = np.argsort(-vector, kind='stable')[:5] if vector.any() else []
            movie.update(average=float(stats.at[i,'mean']), voters=int(stats.at[i,'count']),
                         tail=bool(engine.tail[i]), tags=[tags[j] for j in best])
        self.popular = sorted(engine.movies,key=lambda m:(-m['voters'],m['movieId']))[:10]

    def evidence(self, seen, ratings):
        supplied = dict(zip(seen,ratings))
        overlap = self.train[self.train.i.isin(seen)].copy()
        overlap['agreement'] = 1-(overlap.rating-overlap.i.map(supplied)).abs()/4.5
        scores = overlap.groupby('u').agreement.agg(['mean','count'])
        scores = scores[scores['mean']>=.75]
        similarity = scores['mean']*scores['count']/(scores['count']+3)
        neighbors = similarity.nlargest(30)
        liked = self.train[self.train.u.isin(neighbors.index) & (self.train.rating>=THRESHOLD)].copy()
        liked['weight'] = liked.u.map(neighbors)
        support = liked.groupby('i').weight.agg(['sum','count'])
        return support

    def explain(self, i, liked, support):
        movie = self.e.movies[i]
        reasons = []
        if i in support.index:
            reasons.append(f"공통 영화에 비슷한 평점을 준 사용자 중 {int(support.at[i,'count'])}명이 이 영화에 4점 이상을 줬어요.")
        if len(liked):
            closest = liked[np.argmax(self.e.features[liked] @ self.e.features[i])]
            shared = np.minimum(self.semantic[i],self.semantic[closest])
            if shared.any():
                tag = self.tags[np.argmax(shared)]
                reasons.append(f"좋아한 〈{self.e.movies[closest]['title']}〉와 ‘{tag}’ 특성이 두드러져요.")
            else:
                genres = sorted(set(movie['genres'].split('|')) & set(self.e.movies[closest]['genres'].split('|')) - {'(no genres listed)'})
                if genres:
                    reasons.append(f"좋아한 〈{self.e.movies[closest]['title']}〉와 {', '.join(genres)} 장르가 같아요.")
        if not reasons:
            reasons.append("입력한 별점으로 추정한 취향과 영화 특성을 조합해 추천한 작품이에요.")
        return dict(movie, reasons=reasons)

    def neighbor_picks(self, pool, score, support):
        candidates = [i for i in pool if i in support.index]
        if not candidates:
            return []
        # Cap the influence of frequent votes and reserve up to three discovery slots.
        weight = support['sum']
        ordered = sorted(candidates,key=lambda i:-(score[i]+.12*weight[i]/(weight[i]+2)))
        picks = ordered[:10]
        tail = [i for i in ordered if self.e.tail[i]][:3]
        for i in tail:
            if i not in picks:
                replace = next(j for j in reversed(range(len(picks))) if not self.e.tail[picks[j]])
                picks[replace] = i
        return sorted(picks,key=ordered.index)
