import numpy as np
import pandas as pd
from model import active_model,Engine
e=Engine()
rows=pd.read_csv(active_model()/'split.csv.gz')
ids=rows.u.unique()
previous=np.random.default_rng(7).choice(ids,500,replace=False)
users=np.random.default_rng(19).choice(np.setdiff1d(ids,previous),500,replace=False)
assert not set(users).intersection(previous)
results=[]
for u,g in rows[rows.u.isin(users)].groupby('u'):
    train=g[g.split==0]
    seen=g.loc[g.split<2,'i']
    truth=set(g.loc[(g.split==2)&(g.rating>=4)&(g.i>=0),'i'])-set(seen)
    if not truth: continue
    cf,cb,p=e.scores(train.i,train.rating,u)
    pool,score=e.pool(cf,cb,.75,seen,50)
    tail={i for i in truth if e.tail[i]}
    for k in [0,.005]:
        rec=e.rank(pool,score,k*(1-p))
        hit=np.isin(rec,list(truth))
        results.append(dict(user=u,k=k,ndcg=sum(hit/np.log2(np.arange(2,12)))/sum(1/np.log2(np.arange(2,2+min(10,len(truth))))),recall=hit.sum()/len(truth),exposure=e.tail[rec].mean(),tail_recall=len(tail.intersection(rec))/len(tail) if tail else np.nan))
df=pd.DataFrame(results)
df.to_csv('outputs/independent_retest.csv',index=False)
print(df.groupby('k').mean().to_string(),df.groupby('k').tail_recall.count())
v=df.pivot(index='user',columns='k',values='ndcg').to_numpy()
sample=np.random.default_rng(11).integers(len(v),size=(10000,len(v)))
m=v[sample].mean(axis=1)
print('ratio',v[:,1].mean()/v[:,0].mean(),'CI',np.quantile(m[:,1]/m[:,0],[.025,.975]))
