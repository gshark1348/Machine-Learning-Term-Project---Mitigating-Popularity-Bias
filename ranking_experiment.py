"""Compare final content weights on a frozen candidate pool for each user."""
import json
import numpy as np
import pandas as pd
from model import Engine, active_model

def compare(run):
    e = Engine(run)
    settings = json.loads((run/'settings.json').read_text())
    rows = pd.read_csv(run/'split.csv.gz')
    users = np.random.default_rng(7).choice(rows.u.unique(),min(500,rows.u.nunique()),replace=False)
    groups = list(rows[rows.u.isin(users)].groupby('u'))
    baseline = 1-settings['alpha']

    def measure(split, weights):
        records = []
        for user, history in groups:
            train = history[history.split==0]
            seen = history.loc[history.split<split,'i']
            truth = set(history.loc[(history.split==split)&(history.rating>=4)&(history.i>=0),'i'])-set(seen)
            if not truth: continue
            tail = {i for i in truth if e.tail[i]}
            cf, cb, pref = e.scores(train.i,train.rating,user)
            pool,_ = e.pool(cf,cb,settings['alpha'],seen,settings['content_slots'])
            for weight in weights:
                score=(1-weight)*cf+weight*cb
                rec=e.rank(pool,score,settings['adaptive']*(1-pref))
                hits=np.isin(rec,list(truth))
                ideal=sum(1/np.log2(np.arange(min(10,len(truth)))+2))
                records.append(dict(weight=weight, recall=hits.sum()/len(truth),
                    ndcg=sum(hits/np.log2(np.arange(len(rec))+2))/ideal,
                    tail_recall=len(tail.intersection(rec))/len(tail) if tail else np.nan))
        grouped=pd.DataFrame(records).groupby('weight')
        result=grouped.mean()
        result['users']=grouped.size()
        result['tail_users']=grouped.tail_recall.count()
        result['split']=split
        return result

    val=measure(1,sorted(set([baseline,*np.linspace(0,1,21).round(2)])))
    eligible=val[val.ndcg>=.95*val.loc[baseline,'ndcg']].copy()
    eligible['distance']=abs(eligible.index-baseline)
    best=eligible.sort_values(['tail_recall','ndcg','distance'],ascending=[False,False,True]).index[0]
    test=measure(2,sorted({baseline,best}))
    pd.concat([val,test]).to_csv(run/'ranking_weights.csv')
    print('selected',best,'baseline',baseline)
    print(test.to_string())
    return float(best)

if __name__=='__main__':
    compare(active_model())
