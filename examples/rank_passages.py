"""Run a small retrieval example with an already reconstructed MiniLM checkpoint."""

import argparse
from sentence_transformers import SentenceTransformer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", required=True, help="Path to the reconstructed checkpoint"
    )
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    query = "What causes ocean tides?"
    passages = [
        "The Moon and Sun exert gravitational forces that produce ocean tides.",
        "Wind transfers energy to the ocean surface and produces waves.",
        "An eclipse happens when one celestial body blocks light from another.",
    ]
    model = SentenceTransformer(args.model, device=args.device)
    model.max_seq_length = 256
    vectors = model.encode(
        [query, *passages], normalize_embeddings=True, show_progress_bar=False
    )
    similarities = vectors[1:] @ vectors[0]
    print(query)
    for index in similarities.argsort()[::-1]:
        print(f"{similarities[index]:.4f}  {passages[index]}")


if __name__ == "__main__":
    main()
