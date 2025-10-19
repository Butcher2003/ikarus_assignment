# file: gen_description.py
# Retrieve top-k similar products from FAISS and generate a short creative product description using a local small LLM (Flan-T5 via transformers).
# Usage: python gen_description.py --query "shoe rack metal hooks" --k 5
import argparse, json, numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

def load_index(index_path, emb_npy):
    index = faiss.read_index(index_path)
    mat = np.load(emb_npy)
    return index, mat

def retrieve(query, s_model, index, meta, top_k=5):
    q_emb = s_model.encode(query, convert_to_numpy=True, normalize_embeddings=True)
    D,I = index.search(np.array([q_emb.astype('float32')]), top_k)
    results = []
    for idx, score in zip(I[0], D[0]):
        if idx < 0: continue
        results.append((meta[idx], float(score)))
    return results, q_emb

def generate_description(query, context_items, gen_model_name="google/flan-t5-base"):
    tok = AutoTokenizer.from_pretrained(gen_model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(gen_model_name)
    # build prompt
    ctx_lines = []
    for it,sc in context_items:
        ctx_lines.append(f"- {it.get('title')[:120]} (brand: {it.get('brand','')})")
    prompt = f"Write a concise, creative 2-line product marketing description for: {query}\nContext (similar products):\n" + "\n".join(ctx_lines) + "\nTone: friendly, concise."
    inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=1024)
    out = model.generate(**inputs, max_new_tokens=120, do_sample=True, top_p=0.95, temperature=0.8)
    txt = tok.decode(out[0], skip_special_tokens=True)
    return txt

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--query", required=True)
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--index", default="data_prep_output/faiss.index")
    p.add_argument("--emb", default="data_prep_output/text_embeddings.npy")
    p.add_argument("--meta", default="data_prep_output/meta.json")
    args = p.parse_args()
    s_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    index, emb = load_index(args.index, args.emb)
    meta = json.load(open(args.meta,'r',encoding='utf-8'))
    results, qemb = retrieve(args.query, s_model, index, meta, args.k)
    print("Top context:")
    for r,s in results:
        print(r['title'][:120], "score:", s)
    desc = generate_description(args.query, results)
    print("\nGenerated description:\n", desc)
