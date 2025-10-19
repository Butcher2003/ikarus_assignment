# file: train_ranker.py
# Train a simple LightGBM ranker using embedding similarity + structured features.
# Requires: data_prep_output/meta.json, data_prep_output/text_embeddings.npy
# Usage: python train_ranker.py
import json, random
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from sklearn.metrics import roc_auc_score

def load_meta(meta_path):
    return json.load(open(meta_path,'r',encoding='utf-8'))

def cosine_sim(a,b):
    if a is None or b is None: return 0.0
    a = a / (np.linalg.norm(a)+1e-10)
    b = b / (np.linalg.norm(b)+1e-10)
    return float(np.dot(a,b))

def main():
    out = Path("data_prep_output")
    meta = load_meta(out/"meta.json")
    id_to_idx = {m['uniq_id']: i for i,m in enumerate(meta)}
    text_emb = np.load(out/"text_embeddings.npy")
    # synthetic interactions: sample product as query, positive = nearest neighbor by embedding
    pairs = []
    prod_ids = [m['uniq_id'] for m in meta]
    for q in prod_ids:
        qi = id_to_idx[q]
        sims = (text_emb @ text_emb[qi]).tolist()
        # get top 10 excluding self
        ranked = sorted(enumerate(sims), key=lambda x:-x[1])
        pos_idx = ranked[1][0] if len(ranked)>1 else ranked[0][0]
        pos_id = meta[pos_idx]['uniq_id']
        # negatives: random sample
        negs = random.sample([pid for pid in prod_ids if pid!=pos_id and pid!=q], k=min(5, len(prod_ids)-2))
        pairs.append((q, pos_id, 1))
        for n in negs:
            pairs.append((q, n, 0))
    # featurize
    rows=[]
    for q,c,label in pairs:
        qi = id_to_idx[q]; ci = id_to_idx[c]
        sim = cosine_sim(text_emb[qi], text_emb[ci])
        rows.append({"query":q,"candidate":c,"sim":sim,"label":label})
    df = pd.DataFrame(rows)
    train, test = train_test_split(df, test_size=0.2, random_state=42)
    features = ["sim"]
    dtrain = lgb.Dataset(train[features], label=train['label'])
    dtest = lgb.Dataset(test[features], label=test['label'])
    params = {"objective":"binary","metric":"auc","verbosity":-1}
    callbacks = [lgb.early_stopping(stopping_rounds=20, verbose=False)]
    model = lgb.train(params, dtrain, valid_sets=[dtest], num_boost_round=200, callbacks=callbacks)
    model.save_model(str(out/"lightgbm_ranker.txt"))
    best_iter = model.best_iteration if model.best_iteration else model.num_trees()
    preds = model.predict(test[features], num_iteration=best_iter)
    print("AUC:", roc_auc_score(test['label'], preds))

if __name__ == "__main__":
    main()
