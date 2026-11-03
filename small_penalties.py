import numpy as np
import pandas as pd
from model import Engine,active_model
from discovery import Discovery
run=active_model()
e=Engine(run)
d=Discovery(e,run)
points=[3.5,3,2.5,0,0,5,2.5,0,3.5,4,3,2.5,0,3.5,3,5,3,3.5,2.5,3.5,4.5,4,4.5,0,3.5,3,0,0,4,4.5]
indices={m['movieId']:i for i,m in enumerate(e.movies)}
history=dict(enumerate(points))
for mid,r in [(296,5),(268,5),(746,4.5),(1069,4),(2066,5),(149,4.5),(3966,4),(1068,0),(320,0),(2940,0)]:
    history[indices[mid]]=r
rated=[i for i,r in history.items() if r]
values=[history[i] for i in rated]
cf,cb,p=e.scores(rated,values)
pool,score=e.pool(cf,cb,.75,list(history),50)
top={indices[m['movieId']] for m in d.popular}
print('profile',len(history),len(rated),p,'top in pool',[e.movies[i]['title'] for i in pool if i in top])
def metrics(rec):
    return dict(tail=e.tail[rec].mean(),pop=e.pop[rec].mean(),top=len(set(rec)&top))
for method,k in [('baseline',0),('fixed',.2),('adaptive',.2*(1-p))]:
    rec=e.rank(pool,score,k)
    assert not set(rec)&history.keys()
    print(method,metrics(rec),[e.movies[i]['title'] for i in rec if i in top])
support=d.evidence(rated,values)
rec=d.neighbor_picks(pool,score,support)
print('neighbors',metrics(rec),[e.movies[i]['title'] for i in rec if e.tail[i]])
rows=pd.read_csv(run/'split.csv.gz')
users=np.random.default_rng(7).choice(rows.u.unique(),500,replace=False)
out=[]
for split in [1,2]:
 for u,g in rows[rows.u.isin(users)].groupby('u'):
    train=g[g.split==0]; seen=g.loc[g.split<split,'i']
    truth=set(g.loc[(g.split==split)&(g.rating>=4)&(g.i>=0),'i'])-set(seen)
    if not truth:continue
    cf,cb,p=e.scores(train.i,train.rating,u)
    pool,score=e.pool(cf,cb,.75,seen,50)
    for method in [0,.001,.002,.005,.01,.015,.02,.03,.04,.05]:
        k=method*(1-p)
        rec=e.rank(pool,score,k)
        tail={i for i in truth if e.tail[i]}
        hits=np.isin(rec,list(truth))
        out.append(dict(split=split,user=u,method=method,**metrics(rec),recall=hits.sum()/len(truth),tail_recall=len(tail.intersection(rec))/len(tail) if tail else np.nan,ndcg=sum(hits/np.log2(np.arange(2,12)))/sum(1/np.log2(np.arange(2,2+min(10,len(truth)))))))
df=pd.DataFrame(out)
avg=df.groupby(['split','method']).mean(numeric_only=True)
v=avg.loc[1]
k=v[v.ndcg>=.95*v.loc[0,'ndcg']].sort_values(['tail_recall','tail'],ascending=False).index[0]
print('selected',k,avg.loc[2].loc[[0,k]])
avg.to_csv('outputs/small_penalties.csv')
