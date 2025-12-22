"""
Kaleb J. Wasierski – Non-Commercial Use License (2025)

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

By using this Software, you agree to these terms.
"""
import ollama
import numpy as np

# ---- CONFIG ----

# LLM model (chat)
LLM_MODEL = "phi4-mini:latest"

# Embedding model
EMBED_MODEL = "embeddinggemma:latest"

# Storage for conversation + metrics
turns = []
embeddings = []
novelty = []
constraint = []
cognitive_load = []

# ---- UTILITY FUNCTIONS ----

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def compute_novelty(current_emb, previous_embs, decay=None):
    if len(previous_embs) == 0:
        return 1.0

    sims = [cosine_similarity(current_emb, e) for e in previous_embs]

    if decay is not None:
        weights = np.array(
            [decay ** (len(previous_embs) - i) for i in range(len(previous_embs))]
        )
        sims = np.array(sims) * weights

    return 1.0 - np.max(sims)

def compute_constraint_from_entropy(token_probs):
    """
    token_probs: array [num_tokens, vocab_subset]
    """
    H = -np.sum(token_probs * np.log(token_probs + 1e-12), axis=-1)
    C = 1 - H / np.log(token_probs.shape[-1])
    return np.mean(C)

def compute_cognitive_load(token_embs):
    diffs = np.diff(token_embs, axis=0)
    return np.var(diffs)

# ---- MAIN LOOP ----

while True:
    user_input = input("User: ")
    if user_input.lower() in {"exit", "quit"}:
        break

    turns.append({"role": "user", "content": user_input})

    # ---- LLM RESPONSE ----
    response = ollama.chat(
        model=LLM_MODEL,
        messages=turns
    )

    llm_text = response["message"]["content"]
    turns.append({"role": "assistant", "content": llm_text})

    # ---- EMBEDDING ----
    emb_response = ollama.embeddings(
        model=EMBED_MODEL,
        prompt=llm_text
    )

    emb = np.array(emb_response["embedding"])
    embeddings.append(emb)

    # ---- METRICS ----

    # Novelty
    N_i = compute_novelty(embeddings[-1], embeddings[:-1])
    novelty.append(N_i)

    # Constraint (Ollama does not expose logits)
    # Approximation: assume mid-range constraint
    C_i = 0.5
    constraint.append(C_i)

    # Cognitive Load (approximate via token-level perturbations)
    tokens = llm_text.split()
    token_embs = np.array([
        emb + np.random.normal(0, 0.01, emb.shape)
        for _ in tokens
    ])
    L_i = compute_cognitive_load(token_embs)
    cognitive_load.append(L_i)

    # ---- DISPLAY ----
    print("\nAssistant:")
    print(llm_text)
    print(
        f"\nMetrics → "
        f"Novelty: {N_i:.3f}, "
        f"Constraint: {C_i:.3f}, "
        f"Cognitive Load: {L_i:.5f}\n"
    )
