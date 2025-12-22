
"""Kaleb J. Wasierski – Non-Commercial Use License (2025)

Copyright (c) 2025 Kaleb J. Wasierski

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to use
the Software for personal, educational, research, or non-commercial purposes only,
including modification and experimentation, subject to the following conditions:

1. Attribution: You must give appropriate credit to the original author, 
   including a link to the original repository if applicable.

2. Non-Commercial: You may NOT use, sell, or redistribute the Software, 
   in whole or in part, for any commercial purpose without express written permission 
   from the author.

3. Distribution: You may share the Software with others for personal, educational,
   or research purposes only, provided that this license and copyright notice 
   remain intact.

4. Warranty Disclaimer: The Software is provided "as is", without warranty 
   of any kind, express or implied. The author is not liable for any damages 
   arising from the use of the Software.

By using this Software, you agree to these terms."""
### #### Inference Stability Benchmark (ISB) ### ###
#   This is a benchmarking template / test you can use to 
#   benchmark your replies or conversations to capture drift and incoherence. 
#   You can input previous / persistant conversations in
# the "__main__", prompt = "xxx", within, # MAIN
###
import ollama
import numpy as np

# CONFIG 
LLM_MODEL = "gemma3:270m"
EMBED_MODEL = "embeddinggemma:latest"
REPEATS = 3
STEP_THRESHOLD = 0.80

# GLOBAL CACHE
EMBED_CACHE = {}

# UTILS
def cosine(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def embed_single(text):
    if text in EMBED_CACHE:
        return EMBED_CACHE[text]
    e = np.array(ollama.embeddings(model=EMBED_MODEL, prompt=text)["embedding"])
    EMBED_CACHE[text] = e
    return e

def perturb_prompt(prompt):
    """
    Minimal perturbation for PS:
    - swap some punctuation
    - add extra whitespace
    """
    perturbed = prompt.replace("=", " = ").replace("\n", " \n")
    return perturbed

# SIGNAL FUNCTIONS
def compute_novelty(e, history):
    if not history:
        return 1.0
    return 1 - max(cosine(e, h) for h in history)

def compute_constraint(token_counts):
    return 1.0 - (np.std(token_counts) / (np.mean(token_counts) + 1e-6))

def compute_cognitive_load(e, token_count):
    return np.log1p(token_count) * np.var(e)

def extract_steps(text):
    sentences = [s.strip() for s in text.split(".") if s.strip()]
    if len(sentences) < 2:
        return 1
    embs = [embed_single(s) for s in sentences]
    steps = 1
    for i in range(len(embs) - 1):
        if cosine(embs[i], embs[i + 1]) < STEP_THRESHOLD:
            steps += 1
    return steps

# RUN PROMPT
def run_prompt(prompt):
    novelty = []
    constraint_vals = []
    load = []
    steps = []

    history_embs = []
    token_counts = []

    for _ in range(REPEATS):
        r = ollama.chat(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}]
        )
        text = r["message"]["content"]
        tokens = len(text.split())
        token_counts.append(tokens)

        e = embed_single(text)
        novelty.append(compute_novelty(e, history_embs))
        history_embs.append(e)
        load.append(compute_cognitive_load(e, tokens))
        steps.append(extract_steps(text))

    # Constraint across runs
    C = [compute_constraint(token_counts)] * REPEATS

    return {
        "novelty": np.array(novelty),
        "constraint": np.array(C),
        "load": np.array(load),
        "steps": np.array(steps),
        "avg_embedding": np.mean(history_embs, axis=0)
    }

# ISM SCORE
def compute_ism(metrics, ps_metric):
    OC = 1 - np.var(metrics["novelty"])
    RPS = 1 - np.var(metrics["steps"])
    CR = 1 - np.var(metrics["constraint"])
    TC = 1 - np.var(metrics["load"])
    PS = ps_metric

    ISM = (OC + RPS + CR + TC + (1 - PS)) / 5

    return {
        "OC": OC,
        "RPS": RPS,
        "CR": CR,
        "TC": TC,
        "PS": PS,
        "ISM": ISM
    }

# MAIN
if __name__ == "__main__":
    prompt = """
    Given:
    A = 3
    B = A * 2
    C = B + 5
    D = C / A

    Explain each step and give D.
    """

    # Base run
    metrics = run_prompt(prompt)

    # Perturbation sensitivity
    perturbed_prompt = perturb_prompt(prompt)
    r = ollama.chat(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": perturbed_prompt}]
    )
    pert_text = r["message"]["content"]
    pert_emb = embed_single(pert_text)

    # PS = 1 - cosine similarity to average base embedding
    ps_metric = 1 - cosine(metrics["avg_embedding"], pert_emb)

    # ISM with PS
    scores = compute_ism(metrics, ps_metric)

    print("\n=== ISM v0.3 ===")
    for k, v in scores.items():
        print(f"{k}: {v:.4f}")
