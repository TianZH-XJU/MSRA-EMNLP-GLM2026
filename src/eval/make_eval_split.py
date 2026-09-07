"""
② fixed eval split manifest — drawn ONCE, never re-drawn (PHASE1_EXPERIMENT_PLAN step ②).

Books are taken from OUTSIDE the 64-doc training pool (parquet rows >= max_docs) so no
document can appear in both training (old per-seed 75/25 over head(64)) and this eval set.
Deterministic under seed=0; the manifest records the rule + book list; the sha256 of this
file is the split identity (goes into every result JSON's provenance block).

Usage: python -m src.eval.make_eval_split --n_books 256 --out results/eval_split_256.json
"""

import argparse, hashlib, json, os
import numpy as np
import pandas as pd

PG19 = "/data1/zhoujun/Auto-claude-code-research-in-sleep/dataset/pg19"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_books", type=int, default=256)
    ap.add_argument("--train_pool_docs", type=int, default=64,
                    help="old training protocol uses head(64) — manifest books start AFTER this row")
    ap.add_argument("--min_tokens", type=int, default=1025)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/eval_split_256.json")
    ap.add_argument("--model_path", default="/data1/zhoujun/Auto-claude-code-research-in-sleep/basemodel/llama3.2-1b")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model_path)

    files = sorted(f for f in os.listdir(os.path.join(PG19, "data")) if f.endswith(".parquet"))
    # global row ordering: concatenated sorted parquets (same as load_pg19_texts), skip head(train_pool_docs)
    ordered = []
    for f in files:
        df = pd.read_parquet(os.path.join(PG19, "data", f))
        for i, text in enumerate(df["text"].tolist()):
            ordered.append((f, i, text))
    pool = ordered[args.train_pool_docs:]
    print(f"pool beyond training docs: {len(pool)} candidates")

    rng = np.random.RandomState(args.seed)
    idx = rng.permutation(len(pool))
    books = []
    for j in idx:
        f, ri, text = pool[j]
        n_tok = len(tok.encode(text, add_special_tokens=False, truncation=True, max_length=8192))
        if n_tok >= args.min_tokens:
            books.append({"file": f, "row": int(ri), "n_tokens": int(n_tok)})
        if len(books) >= args.n_books:
            break
    print(f"selected {len(books)} books (>= {args.min_tokens} tokens)")

    manifest = {
        "rule": "PG-19 parquet (sorted filenames, concatenated rows); skip first "
                f"{args.train_pool_docs} docs (training pool); RNG(np.RandomState({args.seed})).permutation; "
                f"take first {args.n_books} with >= {args.min_tokens} tokens",
        "model_path": args.model_path,
        "seed": args.seed,
        "train_pool_docs": args.train_pool_docs,
        "n_books": len(books),
        "books": books,
    }
    blob = json.dumps(manifest, sort_keys=True).encode()
    manifest["sha256"] = hashlib.sha256(blob).hexdigest()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(manifest, open(args.out, "w"))
    print(f"saved {args.out}  sha256={manifest['sha256'][:16]}")


if __name__ == "__main__":
    main()
