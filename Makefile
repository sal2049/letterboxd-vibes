# Run the whole pipeline:  make map
TRACK ?= en
PY ?= python

.PHONY: setup download extract mask embed build posters map web \
        benchmark-embeddings benchmark-clustering test lint

setup:
	$(PY) -m pip install -e ".[dev,naming]"
	cd web && npm install

download:
	$(PY) scripts/00_download_dataset.py

extract:
	$(PY) scripts/01_extract_reviews.py --lang $(TRACK)

mask:
	$(PY) scripts/02_mask_reviews.py --lang $(TRACK)

embed:
	$(PY) scripts/03_embed_reviews.py --lang $(TRACK)

build:
	$(PY) scripts/04_build_map.py --lang $(TRACK)

posters:
	$(PY) scripts/05_enrich_tmdb.py

map: extract mask embed build

benchmark-embeddings:
	$(PY) scripts/benchmark_embeddings.py

benchmark-clustering:
	$(PY) scripts/benchmark_clustering.py --lang $(TRACK)

web:
	cd web && npm run dev

test:
	$(PY) -m pytest -q

lint:
	$(PY) -m ruff check .
	$(PY) -m ruff format --check .
