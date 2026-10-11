# Embedding models compared

1500 films, 10525 reviews with names removed. Agreement is the split-half
measure in `letterboxd_vibes/reliability.py`: 0.5 is chance, 1.0 is perfect. The models are
ranked on agreement with the dataset's genre labels regressed out of the vectors, since the
map removes genre too.

| Model | Dimensions | Agreement without genre | Agreement | Own half in top 10 | Reviews / s |
| --- | --- | --- | --- | --- | --- |
| thenlper/gte-base | 768 | 0.876 | 0.906 | 38.3% | 55 |
| intfloat/e5-base-v2 | 768 | 0.868 | 0.902 | 36.1% | 49 |
| Qwen/Qwen3-Embedding-0.6B | 1024 | 0.854 | 0.906 | 40.4% | 9 |
| BAAI/bge-base-en-v1.5 | 768 | 0.850 | 0.903 | 35.8% | 52 |
| unsloth/embeddinggemma-2 | 768 | 0.832 | 0.878 | 29.9% | 18 |
| sentence-transformers/all-MiniLM-L6-v2 | 384 | 0.831 | 0.881 | 33.7% | 345 |
