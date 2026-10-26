import pandas as pd, numpy as np
from sklearn.metrics import average_precision_score
p='PATH-Mat_Evidence_Strengthening_V211/results/AIRES_UNSEEN_LINKER_PREDICTIONS_V211.csv'
d=pd.read_csv(p); y=d.Crystal_clf.to_numpy(); g=d.LinkerCanonicalSmiles.to_numpy()
models=['score_chemistry_only','score_conditions_only','score_chem_plus_conditions']
rng=np.random.default_rng(20261017)
def replay(scores,budget=10,tie=None):
    tie=np.random.default_rng().random(len(d)) if tie is None else tie
    ix=np.lexsort((tie,-scores)); seen=set(); success=[]; attempts=0
    for i in ix:
        if g[i] in seen: continue
        attempts+=1
        if y[i]==1:
            seen.add(g[i]); success.append(g[i])
        if attempts>=budget: break
    return len(success),success
for m in models:
    rs=[replay(d[m].to_numpy(),10,rng.random(len(d)))[0] for _ in range(10000)]
    print(m, 'AP',average_precision_score(y,d[m]),'top10 successes',replay(d[m].to_numpy(),10,np.zeros(len(d)))[0], 'distribution',pd.Series(rs).value_counts().sort_index().to_dict(), 'mean',np.mean(rs))
