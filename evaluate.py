import json
import numpy as np
import pandas as pd
from model import Engine, active_model

def run_evaluation(run):
    e = Engine(run)
    rows = pd.read_csv(run/'split.csv.gz')
    users = np.random.default_rng(7).choice(rows.u.unique(), min(500,rows.u.nunique()),replace=False)
    groups = list(rows[rows.u.isin(users)].groupby('u'))
    grid = [0,.02,.05,.1,.2]

    def evaluate(split,configs,slots=0):
        out = {}
        for u,g in groups:
            train = g[g.split==0]
            seen = g.loc[g.split<split,'i']
            truth = set(g.loc[(g.split==split)&(g.rating>=4)&(g.i>=0),'i'])-set(seen)
            if not truth: continue
            cf,cb,pref = e.scores(train.i,train.rating,u)
            pools = {}
            for alpha,method,k in configs:
                if alpha not in pools: pools[alpha]=e.pool(cf,cb,alpha,seen,slots)
                rec=e.rank(*pools[alpha],k*(1-pref) if method=='adaptive' else k)
                hits=np.isin(rec,list(truth))
                tail_truth={i for i in truth if e.tail[i]}
                tail_scores=[len(tail_truth.intersection(items))/len(tail_truth) for items in (pools[alpha][0],rec)] if tail_truth else []
                metrics=[sum(hits/np.log2(np.arange(len(rec))+2))/sum(1/np.log2(np.arange(min(10,len(truth)))+2)),hits.sum()/len(truth),e.tail[rec].mean(),e.pop[rec].mean()]
                for group in ['all',['explorer','balanced','mainstream'][int(pref>.33)+int(pref>.66)]]:
                    vals,items,tails=out.setdefault((alpha,method,k,group),([],set(),[]))
                    if tail_scores: tails.append(tail_scores)
                    vals.append(metrics); items.update(rec)
        return [dict(alpha=a,method=m,penalty=k,group=g,users=len(v),tail_users=len(tails),tail_recall100=np.mean([x[0] for x in tails]) if tails else None,tail_recall10=np.mean([x[1] for x in tails]) if tails else None,coverage=len(items)/len(e.movies),**dict(zip(['ndcg','recall','tail','popularity'],np.mean(v,axis=0)))) for (a,m,k,g),(v,items,tails) in out.items()]

    base=evaluate(1,[(a,'hybrid',0) for a in [0,.25,.5,.75,1]])
    best=max((v for v in base if v['group']=='all'),key=lambda v:v['ndcg'])
    a=best['alpha']
    val=evaluate(1,[(a,m,k) for m in ['fixed','adaptive'] for k in grid])
    chosen={m:max((v for v in val if v['method']==m and v['group']=='all' and v['ndcg']>=.95*best['ndcg']),key=lambda v:(v['tail'],v['ndcg']))['penalty'] for m in ['fixed','adaptive']}
    test=evaluate(2,[(0,'content',0),(1,'cf',0),(a,'hybrid',0)]+[(a,m,k) for m,k in chosen.items()])
    pd.DataFrame(base+val).to_csv(run/'validation.csv',index=False)
    pd.DataFrame(test).to_csv(run/'test.csv',index=False)
    (run/'settings.json').write_text(json.dumps(dict(alpha=a,**chosen)))
    comparison=[]
    def compare(split, slots):
        row=next(row for row in evaluate(split,[(a,'adaptive',chosen['adaptive'])],slots)
                 if row['group']=='all')
        comparison.append(dict(split=split,content_slots=slots,**row))
        return row
    for slots in [0,10,25,50]:
        compare(1,slots)
    floor=.95*comparison[0]['ndcg']
    slots=max((v for v in comparison if v['ndcg']>=floor),
              key=lambda v:(v['tail_recall100'] or 0,v['ndcg'],-v['content_slots']))['content_slots']
    for test_slots in sorted({0,slots}):
        compare(2,test_slots)
    pd.DataFrame(comparison).to_csv(run/'candidate_comparison.csv',index=False)
    (run/'settings.json').write_text(json.dumps(dict(alpha=a,content_slots=slots,**chosen)))
    cold = pd.read_csv(run/'new_users.csv.gz')
    reports=[]
    for size in (1,3,5):
        values=[]
        for _,history in cold.groupby('userId'):
            boundary=int(len(history)*.9)
            initial=history.iloc[:boundary].query('i>=0').head(size)
            truth=set(history.iloc[boundary:].query('i>=0 and rating>=4').i)-set(initial.i)
            if len(initial)!=size or not truth: continue
            cf,cb,pref=e.scores(initial.i,initial.rating)
            pool,score=e.pool(cf,cb,a,initial.i,slots)
            rec=e.rank(pool,score,chosen['adaptive']*(1-pref))
            tail={i for i in truth if e.tail[i]}
            values.append(dict(recall10=len(truth.intersection(rec))/len(truth),
                tail_recall100=len(tail.intersection(pool))/len(tail) if tail else np.nan,
                tail_recall10=len(tail.intersection(rec))/len(tail) if tail else np.nan))
        frame=pd.DataFrame(values)
        reports.append(dict(ratings=size,users=len(frame),tail_users=frame.tail_recall10.count() if values else 0,**frame.mean().to_dict()))
    pd.DataFrame(reports).to_csv(run/'cold_start.csv',index=False)
    print(pd.DataFrame(test).query("group=='all'").to_string(index=False))
