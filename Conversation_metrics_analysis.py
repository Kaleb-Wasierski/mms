### there is likely an issue with this singular repo commit (Conversation_metrics_analysis.py); Computation is inefficient and requires refactoring.
"""
Conversation Metrics Analyzer (CMA) is a framework for analyzing LLM-driven or human-assisted conversations in real time. It quantifies each conversational turn using three primary metrics:

1. Novelty (N): Measures how different a turn is from prior conversation turns, based on embedding similarity.


2. Constraint (C): Measures the degree of “forced” or deterministic response, based on entropy of next-token probabilities.


3. Cognitive Load (L): Estimates the complexity of a turn via variance in token embeddings or attention activations.



The framework tracks these metrics per turn and aggregates them to provide insights into conversation dynamics, exploration, and complexity. It is suitable for research, AI evaluation, and real-time interactive systems.

The provided Python prototype demonstrates a minimal implementation using OpenAI’s LLM and embeddings APIs, computing novelty, constraint (simplified), and cognitive load for each turn.


---

Copyright

Copyright © 2025 Kaleb Wasierski. All rights reserved.

This software and associated documentation are provided “as-is,” without warranty of any kind. You may use, copy, and modify the code for personal, educational, or research purposes. Redistribution for commercial purposes requires explicit permission from the copyright holder.

"""

import openai
import numpy as np

# ---- CONFIG ----
openai.api_key = "YOUR_API_KEY"

# Embedding model
EMBED_MODEL = "text-embedding-3-small"
# LLM model
LLM_MODEL = "gpt-4"

# Storage for conversation
turns = []
embeddings = []
novelty = []
constraint = []
cognitive_load = []

# ---- UTILITY FUNCTIONS ----
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def compute_novelty(current_emb, previous_embs, decay=None):
    if not previous_embs:
        return 1.0  # First turn is maximally novel
    sims = [cosine_similarity(current_emb, e) for e in previous_embs]
    if decay:
        weights = np.array([decay**(len(previous_embs)-i) for i in range(len(previous_embs))])
        sims = np.array(sims) * weights
    return 1 - max(sims)

def compute_constraint(logits):
    # logits shape: [num_tokens, vocab_size]
    probs = np.exp(logits) / np.sum(np.exp(logits), axis=-1, keepdims=True)
    H = -np.sum(probs * np.log(probs + 1e-12), axis=-1)
    C = 1 - H / np.log(probs.shape[-1])
    return np.mean(C)

def compute_cognitive_load(token_embs):
    diffs = np.diff(token_embs, axis=0)
    return np.var(diffs)

# ---- MAIN LOOP ----
while True:
    user_input = input("User: ")
    turns.append({"role": "user", "content": user_input})
    
    # Generate LLM response
    response = openai.ChatCompletion.create(
        model=LLM_MODEL,
        messages=turns
    )
    llm_text = response.choices[0].message.content
    turns.append({"role": "assistant", "content": llm_text})
    
    # Embedding of LLM turn
    emb = openai.Embedding.create(
        input=llm_text,
        model=EMBED_MODEL
    )["data"][0]["embedding"]
    embeddings.append(np.array(emb))
    
    # Novelty
    N_i = compute_novelty(embeddings[-1], embeddings[:-1])
    novelty.append(N_i)
    
    # Constraint using token logits (simplified here as uniform, replace with model token logits if available)
    # For demo, assume low constraint
    C_i = 0.5
    constraint.append(C_i)
    
    # Cognitive load: approximate with embedding variance of tokens
    # In practice, token embeddings can be retrieved if model exposes them
    # Here we simulate with small random noise for demonstration
    token_embs = np.array([emb + np.random.normal(0, 0.01, len(emb)) for _ in range(len(llm_text.split()))])
    L_i = compute_cognitive_load(token_embs)
    cognitive_load.append(L_i)
    
    # ---- DISPLAY METRICS ----
    print(f"\nAssistant: {llm_text}")
    print(f"Novelty: {N_i:.3f}, Constraint: {C_i:.3f}, Cognitive Load: {L_i:.5f}\n")
