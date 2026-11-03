"""MovieLens 20M training and shared ranking logic."""
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
MODEL = ROOT / 'artifacts'
THRESHOLD = 4.0


def active_model():
    pointer = MODEL / 'active'
    return MODEL / 'runs' / pointer.read_text().strip() if pointer.exists() else MODEL


def publish(run):
    with tempfile.NamedTemporaryFile(mode='w', dir=MODEL, delete=False) as tmp:
        tmp.write(run.name)
    os.replace(tmp.name, MODEL / 'active')


def unit(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-8)


def train(archive, users, epochs, rank):
    rng = np.random.default_rng(42)
    run = MODEL / "runs" / uuid4().hex
    run.mkdir(parents=True)
    with zipfile.ZipFile(archive) as z:
        def read(name, **kw):
            return pd.read_csv(z.open('ml-20m/' + name + '.csv'), **kw)
        # The archive is grouped by user. Uniformly sample complete histories.
        ids = read('ratings', usecols=['userId'], dtype='int32').userId.unique()
        sample = rng.choice(ids, min(users, len(ids)), replace=False) if users else ids
        others = np.setdiff1d(ids,sample)
        holdout = np.random.default_rng(8).choice(others,min(500,len(others)),replace=False)
        chunks = read('ratings', chunksize=500000, dtype={'userId':'int32','movieId':'int32','rating':'float32','timestamp':'int64'})
        training, test_users = [], []
        for chunk in chunks:
            training.append(chunk[chunk.userId.isin(sample)])
            test_users.append(chunk[chunk.userId.isin(holdout)])
        r = pd.concat(training, ignore_index=True)
        cold = pd.concat(test_users).sort_values(['userId','timestamp','movieId'])
        r.sort_values(['userId','timestamp','movieId'], inplace=True)
        group = r.groupby('userId')
        n = group.movieId.transform('size')
        pos = group.cumcount()
        r['split'] = np.where(pos < (n*.8).astype(int), 0, np.where(pos < (n*.9).astype(int), 1, 2))
        t = r[r.split == 0]
        movies = read('movies')
        movies = movies[movies.movieId.isin(t.movieId)].sort_values('movieId').reset_index(drop=True)
        index = pd.Index(movies.movieId)
        cold['i'] = index.get_indexer(cold.movieId)
        cold[['userId','i','rating']].to_csv(run/'new_users.csv.gz',index=False)
        r['i'] = index.get_indexer(r.movieId)
        r['u'] = pd.factorize(r.userId, sort=True)[0]
        genres = unit(movies.genres.str.get_dummies('|').values.astype('float32'))
        tags = read('genome-tags')
        genome = np.zeros((len(index), int(tags.tagId.max())), 'float32')
        for c in read('genome-scores', chunksize=500000):
            i = index.get_indexer(c.movieId)
            keep = i >= 0
            genome[i[keep], c.tagId[keep]-1] = c.relevance[keep]
    features = unit(np.column_stack((genres, unit(genome))))
    t = r[r.split == 0]
    u, i, y = t.u.to_numpy(), t.i.to_numpy(), t.rating.to_numpy()
    count = np.bincount(i)
    pop = np.log1p(count) / np.log1p(count.max())
    tail = np.ones(len(index), bool)
    tail[np.argsort(-count, kind='stable')[:int(np.ceil(len(index)*.2))]] = False
    p = rng.normal(0,.07,(len(sample),rank)).astype('float32')
    q = rng.normal(0,.07,(len(index),rank)).astype('float32')
    bu, bi = np.zeros(len(p),'float32'), np.zeros(len(q),'float32')
    mean = y.mean()
    for _ in range(epochs):
        order = rng.permutation(len(y))
        for start in range(0,len(order),256):
            b = order[start:start+256]
            a, j = u[b], i[b]
            pa, qj = p[a], q[j]
            err = y[b]-mean-bu[a]-bi[j]-(pa*qj).sum(1)
            for param, idx, grad in [(p,a,err[:,None]*qj),(q,j,err[:,None]*pa),(bu,a,err),(bi,j,err)]:
                np.add.at(param,idx,.015*(grad-.04*param[idx]))
    MODEL.mkdir(exist_ok=True)
    np.savez(run/'model.npz', p=p,q=q,bu=bu,bi=bi,mean=mean,pop=pop,tail=tail,features=features,count=count)
    movies.to_json(run/'movies.json', orient='records',force_ascii=False)
    r[['u','i','rating','split']].to_csv(run/'split.csv.gz',index=False)
    (run/'training.json').write_text(json.dumps(dict(users=len(sample),movies=len(index),ratings=len(r),train=len(t),epochs=epochs,rank=rank,seed=42)))
    from evaluate import run_evaluation
    run_evaluation(run)
    publish(run)
    print('Activated',run)


class Engine:
    def __init__(self,run=None):
        run = run or active_model()
        with np.load(run/'model.npz') as data:
            self.__dict__.update({k:data[k] for k in data.files})
        self.movies = json.loads((run/'movies.json').read_text())

    def scores(self, items, ratings, user=None):
        items, ratings = np.asarray(items,int), np.asarray(ratings,float)
        liked = items[ratings >= THRESHOLD]
        if user is None:
            # Ridge fold-in against frozen movie factors, including a user bias.
            x = np.column_stack((np.ones(len(items)),self.q[items]))
            coef = np.linalg.solve(x.T@x + np.eye(x.shape[1])*.5,x.T@(ratings-self.mean-self.bi[items]))
            bias, factors = coef[0],coef[1:]
        else:
            bias, factors = self.bu[user],self.p[user]
        cf = np.clip((self.mean+bias+self.bi+self.q@factors-.5)/4.5,0,1)
        cb = self.features @ unit(self.features[liked].mean(0)) if len(liked) else np.full(len(cf),.5)
        return cf, cb, float(self.pop[liked].mean()) if len(liked) else .5

    def pool(self, cf, cb, alpha, seen, content_slots=0):
        score = alpha*cf+(1-alpha)*cb
        order = np.argsort(-score,kind='stable')
        pool = order[~np.isin(order,seen)][:100]
        if content_slots and len(pool)==100 and np.ptp(cb)>0:
            core = pool[:100-content_slots]
            eligible = np.ones(len(cb), dtype=bool)
            observed = np.asarray(seen,dtype=int)
            eligible[observed[observed >= 0]] = False
            eligible[core] = False
            candidates = np.flatnonzero(eligible)
            similar = candidates[np.argsort(-cb[candidates],kind='stable')[:content_slots]]
            pool = np.concatenate((core, similar))
        return pool,score

    def rank(self,pool,score,penalty):
        return pool[np.argsort(-(score[pool]-penalty*self.pop[pool]),kind='stable')[:10]]


if __name__ == '__main__':
    cli = argparse.ArgumentParser()
    cli.add_argument('archive')
    cli.add_argument('--users',type=int,default=2000,help='0 = all users')
    cli.add_argument('--epochs',type=int,default=10)
    cli.add_argument('--rank',type=int,default=24)
    args = cli.parse_args()
    train(args.archive,args.users,args.epochs,args.rank)
