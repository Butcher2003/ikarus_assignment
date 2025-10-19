# file: clustering_nlp.py
# Use UMAP + HDBSCAN to cluster product embeddings for grouping / taxonomy suggestions
# Usage: python clustering_nlp.py --emb_npy data_prep_output/text_embeddings.npy
import argparse
import numpy as np
import umap
import hdbscan
import json
from pathlib import Path

def main(emb_npy, meta_json, out_dir):
    X = np.load(emb_npy)
    reducer = umap.UMAP(n_components=32, random_state=42)
    Xr = reducer.fit_transform(X)
    clu = hdbscan.HDBSCAN(min_cluster_size=5, metric='euclidean')
    labels = clu.fit_predict(Xr)
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    np.save(out/"umap_projection.npy", Xr)
    np.save(out/"clusters.npy", labels)
    meta = json.load(open(meta_json,'r',encoding='utf-8'))
    for i, m in enumerate(meta):
        m['cluster'] = int(labels[i])
    json.dump(meta, open(out/"meta_with_clusters.json",'w',encoding='utf-8'), ensure_ascii=False, indent=2)
    print("Saved clusters and enriched metadata")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--emb_npy", required=True)
    p.add_argument("--meta", default="data_prep_output/meta.json")
    p.add_argument("--out", default="data_prep_output")
    args = p.parse_args()
    main(args.emb_npy, args.meta, args.out)
