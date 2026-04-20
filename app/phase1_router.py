import numpy as np

try:
    from app.vector_store import create_vector_store, embed_text
except ImportError:
    from vector_store import create_vector_store, embed_text


index, bot_ids, bot_vectors = create_vector_store()


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def route_post_to_bots(post_content: str, threshold: float = 0.85):
    post_vec = embed_text(post_content)

    results = []
    for i, bot_vec in enumerate(bot_vectors):
        sim = float(cosine_similarity(post_vec, bot_vec))
        print(f"{bot_ids[i]} similarity: {sim:.3f}")

        if sim > threshold:
            results.append((bot_ids[i], sim))

    return sorted(results, key=lambda x: x[1], reverse=True)


if __name__ == "__main__":
    post = "OpenAI released new AI model"
    print(route_post_to_bots(post))