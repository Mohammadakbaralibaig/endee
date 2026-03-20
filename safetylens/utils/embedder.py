import os
import time
from typing import List
from dotenv import load_dotenv

load_dotenv()

def get_embedding(text: str) -> List[float]:
    """Simple TF-IDF style embedding using sentence hashing - no API needed."""
    import hashlib
    import math
    
    # Generate a deterministic 1536-dim embedding from text
    words = text.lower().split()
    vector = [0.0] * 1536
    
    for word in words:
        hash_val = int(hashlib.md5(word.encode()).hexdigest(), 16)
        idx = hash_val % 1536
        vector[idx] += 1.0
    
    # Normalize
    magnitude = math.sqrt(sum(x**2 for x in vector))
    if magnitude > 0:
        vector = [x / magnitude for x in vector]
    
    return vector

def get_embeddings_batch(texts: List[str]) -> List[List[float]]:
    return [get_embedding(text) for text in texts]