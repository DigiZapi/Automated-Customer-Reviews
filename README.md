Automated Customer Reviews
==========================

Purpose
-------

This notebook-driven project explores automated analysis of product reviews: text preprocessing, simple supervised classification, unsupervised clustering, and brief summarization. The main workflow lives in a single Jupyter notebook and is designed for instructional, exploratory use rather than production packaging.

Repository Layout
-----------------

- Main notebook: `week6/coding/Automated-Customer-Reviews/data_categorization.ipynb`
- Dataset (referenced by the notebook): `data/1429_1.csv`
- Global dependencies: `requirements.txt`
- Additional example notebooks: `Github_files/ai-eng-nbs-public-master/`

Quick Start
-----------

1) Create a virtual environment and install dependencies (Linux/macOS bash):

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# Often useful to ensure latest sentence-transformers
pip install -U sentence-transformers
```

2) Launch Jupyter and run the main notebook interactively:

```bash
jupyter lab
```

Open `Automated-Customer-Reviews/data_categorization.ipynb` and execute cells top-to-bottom.

Headless Execution (CI or Batch)
--------------------------------

You can also execute the notebook non-interactively using nbconvert:

```bash
pip install nbconvert
jupyter nbconvert \
	--to notebook \
	--execute /Automated-Customer-Reviews/data_categorization.ipynb \
	--ExecutePreprocessor.timeout=600
```

What the Notebook Does
----------------------

- Preprocesses review text into a normalized `clean_text` column via `preprocess_text` (lowercasing, tokenization, stopword removal, lemmatization/stemming).
- Creates derived metadata such as `meta_category`, `cluster`, and `cluster_label` for analysis and visualization.
- Trains lightweight classifiers (e.g., Naive Bayes) on bag-of-words features for simple category prediction.
- Performs clustering on bag-of-words or sentence-transformer embeddings (e.g., `all-MiniLM-L6-v2`), assigning cluster IDs and human-friendly labels.
- Generates short summaries of cluster themes with a small seq2seq model, when enabled.

Important Symbols to Know
-------------------------

- `preprocess_text(text) -> str`: Centralized text normalization used across steps.
- `categorize_product(...)`: Example helper to map products to broader categories.
- `cluster_labels`: Mapping from numeric cluster IDs to readable labels; may appear in multiple cells.
- `top_products_df`: Frame of selected top products by frequency/cluster.
- `top_reviews_df`: Merge of top products with original reviews for inspection.

Data Expectations
-----------------

- The notebook expects a CSV at `data/1429_1.csv` with review-related columns. Some columns use dot-notation strings (e.g., `reviews.text`, `reviews.rating`). Preserve exact names when filtering and merging.
- Column renames may occur inline (e.g., `product_name` → `name`). Downstream cells often expect `name`.

Models and Performance Notes
----------------------------

- Embeddings: `SentenceTransformer("all-MiniLM-L6-v2")` balances speed and quality for CPU use.
- Summarization: T5/BART models can be large. If running on CPU, prefer `google/flan-t5-small` or reduce input sizes to keep runs responsive.
- First runs will download model weights from Hugging Face; ensure internet access and disk space.

Reproducibility Tips
--------------------

- Prefer installing all dependencies via `requirements.txt` rather than using `!pip install` inside cells.
- The notebook includes `nltk.download(...)` calls. Run them once per environment to avoid popups or set up NLTK data ahead of time.
- When you modify `cluster_labels`, update all subsequent cells that rely on the same mapping (plots, selections, prompts). Search for `cluster_labels` in the notebook to keep mappings consistent.

Troubleshooting
---------------

- Out-of-memory or long runtimes: switch to smaller models, reduce the number of reviews processed, or sample the dataset before embedding/summary steps.
- Missing columns: check earlier cells that rename columns and ensure you executed the notebook sequentially.
- Package/version issues: recreate the virtual environment and reinstall from `requirements.txt`.

Common Tasks
------------

- Run the full workflow interactively: open the main notebook in Jupyter and execute sequentially.
- Run headless for verification: use the nbconvert command shown above.
- Inspect top products and clusters: review `top_products_df` and `top_reviews_df` cells near the end of the notebook.

Notes for Contributors
----------------------

- Keep edits minimal and localized. This repository is notebook-first for teaching; avoid over-abstracting.
- If you find repeated logic (e.g., preprocessing or prompt building), consider extracting a tiny helper module under `/Automated-Customer-Reviews/` and importing it in the notebook. If you do, update any dependent cells and add a quick example cell demonstrating usage.

# Automated-Customer-Reviews
second-last-project


