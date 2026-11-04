"""Paired comparison at equal mean penalty on the seed-19 retest cohort."""
import numpy as np
import pandas as pd
from model import Engine, active_model

run = active_model()
engine = Engine(run)
rows = pd.read_csv(run / 'split.csv.gz')
ids = rows.u.unique()
previous = np.random.default_rng(7).choice(ids, 500, replace=False)
sample = np.random.default_rng(19).choice(np.setdiff1d(ids, previous), 500, replace=False)
profiles = []
for user, history in rows[rows.u.isin(sample)].groupby('u'):
    train = history[history.split == 0]
    seen = history.loc[history.split < 2, 'i']
    test = history[(history.split == 2) & (history.rating >= 4) & (history.i >= 0)]
    truth = set(test.i) - set(seen)
    if not truth:
        continue
    cf, content, preference = engine.scores(train.i, train.rating, user)
    pool, scores = engine.pool(cf, content, .75, seen, 50)
    profiles.append((user, pool, scores[pool], truth, .005 * (1 - preference)))

fixed = np.mean([profile[-1] for profile in profiles])
records = []
for user, pool, scores, truth, personal in profiles:
    tail_truth = {item for item in truth if engine.tail[item]}
    for arm, strength in [('baseline', 0), ('fixed', fixed), ('personal', personal)]:
        order = np.argsort(-(scores - strength * engine.pop[pool]), kind='stable')
        rec = pool[order[:10]]
        hits = np.isin(rec, list(truth))
        ideal = sum(1 / np.log2(np.arange(min(10, len(truth))) + 2))
        records.append(dict(user=user, arm=arm, strength=strength,
                            ndcg=sum(hits / np.log2(np.arange(len(rec)) + 2)) / ideal,
                            recall=hits.sum() / len(truth), exposure=engine.tail[rec].mean(),
                            tail_recall=len(tail_truth.intersection(rec)) / len(tail_truth)
                            if tail_truth else np.nan))
df = pd.DataFrame(records)
summary = df.drop(columns='user').groupby('arm').mean()
assert np.isclose(summary.at['fixed', 'strength'], summary.at['personal', 'strength'])
summary.to_csv('outputs/matched_summary.csv')
contrasts = []
for metric in ['ndcg', 'recall', 'exposure', 'tail_recall']:
    wide = df.pivot(index='user', columns='arm', values=metric)
    delta = (wide.personal - wide.fixed).dropna().to_numpy()
    draws = np.random.default_rng(11).choice(delta, (10000, len(delta)), replace=True)
    lo, hi = np.quantile(draws.mean(axis=1), [.025, .975])
    contrasts.append(dict(metric=metric, users=len(delta), difference=delta.mean(),
                          lower=lo, upper=hi, improved=sum(delta>0), declined=sum(delta<0)))
pd.DataFrame(contrasts).to_csv('outputs/matched_contrasts.csv', index=False)
print(summary, pd.DataFrame(contrasts), sep='\n')
