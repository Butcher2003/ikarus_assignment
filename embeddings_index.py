# file: embeddings_index.py
# Generates text embeddings (sentence-transformers) and image embeddings (CLIP), saves numpy arrays and a FAISS index.
# Usage: python embeddings_index.py --jsonl data_prep_output/products_clean.jsonl
import argparse, json, os
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch
import faiss

def build_text(prod):
    parts = [prod.get('title',''), prod.get('brand',''), prod.get('description','')] + prod.get('categories',[])
    return " \n ".join([p for p in parts if p])

def load_products(jsonl_path):
    prods = [json.loads(l) for l in open(jsonl_path, 'r', encoding='utf-8').read().splitlines()]
    return prods

def compute_text_embeddings(products, model_name="sentence-transformers/all-MiniLM-L6-v2", batch=64):
    model = SentenceTransformer(model_name)
    texts = [build_text(p) for p in products]
    emb = model.encode(texts, batch_size=batch, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True)
    return emb

def compute_image_embeddings(products, images_dir, device="cpu", clip_model_name="openai/clip-vit-base-patch32"):
    device = "cuda" if torch.cuda.is_available() and device!="cpu" else "cpu"
    clip = CLIPModel.from_pretrained(clip_model_name).to(device)
    proc = CLIPProcessor.from_pretrained(clip_model_name)
    embs = []
    for p in products:
        uid = p.get('uniq_id')
        img_paths = list(Path(images_dir).glob(f"{uid}_*.jpg"))
        if not img_paths:
            embs.append(None)
            continue
        vecs = []
        for ip in img_paths[:3]:
            img = Image.open(ip).convert('RGB').resize((224,224))
            inputs = proc(images=img, return_tensors="pt").to(device)
            with torch.no_grad():
                v = clip.get_image_features(**inputs).cpu().numpy()[0]
            vecs.append(v / (np.linalg.norm(v)+1e-10))
        meanv = np.mean(vecs, axis=0)
        meanv = meanv / (np.linalg.norm(meanv)+1e-10)
        embs.append(meanv)
    return embs

def build_faiss(text_embeddings, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    d = text_embeddings.shape[1]
    index = faiss.IndexFlatIP(d)
    xb = np.array(text_embeddings).astype('float32')
    faiss.normalize_L2(xb)
    index.add(xb)
    faiss.write_index(index, str(out_dir / "faiss.index"))
    np.save(out_dir / "text_embeddings.npy", xb)
    print("Saved FAISS and text embeddings")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--jsonl", required=True)
    p.add_argument("--out", default="data_prep_output")
    p.add_argument("--images_dir", default="data_prep_output/images")
    args = p.parse_args()
    products = load_products(args.jsonl)
    text_emb = compute_text_embeddings(products)
    build_faiss(text_emb, args.out)
    # image embeddings optional (saves list of arrays with None entries)
    img_embs = compute_image_embeddings(products, args.images_dir)
    # save image embeddings (object array)
    np.save(Path(args.out)/"image_embeddings.npy", np.array([e if e is not None else np.zeros((img_embs[0].shape)) for e in img_embs]))
    # save metadata list
    with open(Path(args.out)/"meta.json", "w", encoding="utf-8") as f:
        json.dump([{ "uniq_id":p.get("uniq_id"), "title": p.get("title"), "brand": p.get("brand") } for p in products], f, ensure_ascii=False, indent=2)
    print("Done")
