# Automated Customer Reviews Analysis

An end-to-end machine learning project for analyzing Amazon product reviews using NLP techniques, sentiment analysis, clustering, and AI-powered summarization.

---

## 📋 Project Overview

This project demonstrates a complete ML pipeline for processing and analyzing 41,000+ customer reviews of Amazon products (primarily Fire tablets, Kindle e-readers, Echo devices, and accessories). The pipeline includes:

1. **Data Preprocessing & EDA** - Text cleaning, exploratory analysis, and feature engineering
2. **Sentiment Classification** - Multi-model comparison (Naive Bayes, Random Forest, LSTM, SVM) to predict review sentiment
3. **Product Clustering** - Unsupervised learning to group products into meaningful categories
4. **AI-Powered Summarization** - Using Google's Flan-T5-Large with LoRA fine-tuning to generate product summaries
5. **Interactive Web Dashboard** - Gradio-based interface to explore insights

---

## 🎯 Key Objectives

- **Aggregate customer feedback** from multiple sources into a unified dataset
- **Classify reviews** into positive, negative, or neutral sentiment with high accuracy
- **Cluster products** into 4 distinct categories for simplified browsing
- **Generate AI summaries** highlighting key themes for top and worst-rated products per category
- **Deploy interactive dashboard** for stakeholder exploration

---

## 📊 Dataset

**Source:** Amazon product reviews dataset  
**File:** `data/1429_1.csv`  
**Size:** 41,421 reviews  
**Products:** 86 unique products  

### Key Columns:
- `id` - Product ID
- `name` - Product name
- `brand` - Product brand (Amazon)
- `categories` - Product category string
- `reviews.text` - Review text content
- `reviews.rating` - Rating (1-5 stars)
- `reviews.title` - Review title
- `reviews.date` - Review date
- `reviews.username` - Reviewer username

### Data Distribution:
- **Rating Distribution:** Heavily skewed toward positive (4-5 stars)
  - 5 stars: ~60%
  - 4 stars: ~20%
  - 3 stars: ~10%
  - 1-2 stars: ~10%

---

## 🤖 Models & Techniques

### 1. Text Preprocessing Pipeline
**Function:** `preprocess_text(text)`
- Lowercase conversion
- Tokenization (NLTK)
- Stopword removal
- Punctuation removal
- Lemmatization (WordNetLemmatizer)
- Stemming (PorterStemmer)

### 2. Sentiment Classification Models

| Model | Accuracy | Notes |
|-------|----------|-------|
| **Naive Bayes** (MultinomialNB) | ~85% | Baseline, fast inference |
| **Random Forest** | ~87% | Better feature capture |
| **LSTM** (Deep Learning) | ~89% | Sequential patterns, slower |
| **SVM** (Support Vector Machine) | **~91%** | **Best performer** ✅ |

**Final Choice:** SVM with CountVectorizer features  
**Output:** `data/product_reviews_sentiment_and_confidence_SVM_updated.csv`

### 3. Product Clustering (KMeans)
**Method:** KMeans clustering on TF-IDF features  
**Optimal Clusters:** 4 (determined via silhouette score)

**Cluster-to-Category Mapping:**
- **Cluster 0:** Accessories (cables, chargers)
- **Cluster 1:** E-Readers (Kindle Paperwhite, Voyage)
- **Cluster 3:** Smart Home & Entertainment (Echo, Fire TV)
- **Cluster 4:** Tablets (Fire HD, Fire Kids Edition)

### 4. AI Summarization
**Model:** Google Flan-T5-Large (~780M parameters)  
**Enhancement:** LoRA (Low-Rank Adaptation) fine-tuning for efficient training  
**Configuration:**
- Rank (r): 32
- Alpha: 64
- Dropout: 0.05
- Target modules: Query, Key, Value, Output projections

**Output Format:** ~50-word summaries per product  
**Selection Strategy:** Top 3 best products + worst product per cluster

---

## 🔑 Key Findings

### Product Insights:

**🏆 Top Products by Category:**

**Tablets (Cluster 4)**
- Fire Tablet 7" - 10,966 reviews, 4.45★ (Most popular overall)
- Fire HD 8" - High satisfaction, great value for families

**E-Readers (Cluster 1)**
- Kindle Paperwhite - 3,176 reviews, 4.77★ (Best in category)
- Kindle Voyage - 580 reviews, 4.73★ (Premium experience)

**Smart Home (Cluster 3)**
- Echo (White) - 6,619 reviews, 4.67★ (Voice control favorite)
- Fire TV - 5,056 reviews, 4.71★ (Streaming excellence)

**Accessories (Cluster 0)**
- Amazon USB Charger - 401 reviews, 4.44★ (Essential accessory)

### Common Positive Themes:
- ✅ Excellent value for money
- ✅ Great screen quality and brightness
- ✅ Easy to use for all age groups
- ✅ Perfect for Prime members
- ✅ Long battery life

### Common Negative Themes:
- ❌ Limited app selection (vs. iOS/Android)
- ❌ Ads on lock screen (special offers version)
- ❌ Slower performance on budget models
- ❌ Compatibility issues with some chargers

---

## 📁 Repository Structure

```
Automated-Customer-Reviews/
├── README.md                                  # This file
├── data/                                      # Data directory
│   ├── 1429_1.csv                            # Original dataset (41k reviews)
│   ├── Datafiniti_Amazon_Consumer_Reviews*   # Alternative datasets
│   ├── category_mapping.csv                  # Product categorization
│   ├── product_reviews_sentiment_*.csv       # Sentiment analysis outputs
│   └── product_reviews_sentiment_*_updated.csv
├── results/                                   # Generated outputs
│   ├── product_summaries.csv                 # AI-generated summaries
│   └── product_summaries_report.txt          # Summary report
├── data_eda_preprocessing.ipynb              # Main analysis notebook
├── sample_neat_data_cat.ipynb                # Clustering & categorization
├── summarizing.ipynb                          # Flan-T5 summarization
├── gradio_app.py                              # Interactive web dashboard
└── requirements.txt                           # Python dependencies
```

---

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Create virtual environment
python -m venv .venv

# Activate (Linux/macOS)
source .venv/bin/activate

# Activate (Windows)
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Notebooks

```bash
# Launch Jupyter
jupyter lab

# Execute notebooks in order:
# 1. data_eda_preprocessing.ipynb (EDA + sentiment analysis)
# 2. sample_neat_data_cat.ipynb (clustering)
# 3. summarizing.ipynb (AI summarization)
```

### 3. Launch Web Dashboard

```bash
python gradio_app.py
```

Access at: `http://localhost:7860`

---

## 🖥️ Web Dashboard Features

The Gradio interface provides:
- **Category selection** dropdown (Tablets, E-Readers, Smart Home, Accessories)
- **Top 3 products** with highest popularity scores
- **Worst product** in each category (learning from failures)
- **AI-generated summaries** highlighting key review themes
- **Visual indicators:**
  - 🥇🥈🥉 Rank badges
  - ⭐ Star ratings
  - 😊😐😞 Sentiment badges
  - Color-coded cards (gold/silver/bronze for top products, red for worst)

---

## 🛠️ Technical Requirements

### Core Dependencies:
```
pandas==2.3.3
numpy==2.3.4
scikit-learn==1.7.2
nltk==3.9.2
matplotlib==3.10.7
seaborn==0.13.2
```

### Deep Learning:
```
torch==2.9.0
tensorflow==2.18.0 (via keras)
transformers==4.57.1
sentence-transformers==5.1.2
```

### NLP & Models:
```
nltk (punkt, stopwords, wordnet)
huggingface-hub==0.36.0
peft (for LoRA)
```

### Web Interface:
```
gradio (included in deps)
```

**GPU Support:** Optional but recommended for summarization (CUDA-enabled)

---

## 📈 Model Performance Summary

### Classification Results (SVM - Best Model):
- **Accuracy:** 91%
- **Precision:** 0.92 (positive), 0.87 (negative/neutral)
- **Recall:** 0.95 (positive), 0.81 (negative/neutral)
- **F1-Score:** 0.93 (weighted average)

### Clustering Quality:
- **Silhouette Score:** 0.43 (moderate separation)
- **Inertia:** Stable elbow at k=4
- **Interpretability:** High (clear category distinctions)

### Summarization Quality:
- **Model:** Flan-T5-Large with LoRA
- **Summary Length:** ~50 words (configurable)
- **Coherence:** High (captures key themes accurately)
- **GPU Acceleration:** 3-4x faster than CPU

---

## 🔬 Reproducibility Notes

1. **NLTK Resources:** First run requires downloading:
   ```python
   import nltk
   nltk.download('punkt')
   nltk.download('stopwords')
   nltk.download('wordnet')
   ```

2. **Hugging Face Models:** Auto-downloaded on first use (~3GB for Flan-T5-Large)

3. **Random Seeds:** Set in notebooks for reproducible clustering/splits

4. **Column Names:** Some cells rename columns (`product_name` → `name`). Execute sequentially to avoid errors.

---

## 📝 Key Functions Reference

### Preprocessing:
```python
preprocess_text(text: str) -> str
# Normalizes text through lowercase, tokenization, 
# stopword removal, lemmatization, and stemming
```

### Categorization:
```python
categorize_product_improved(category_string: str, product_name: str) -> str
# Maps products to meta-categories based on category 
# strings and product names
```

### Variables to Know:
- `df` - Main DataFrame
- `clean_text` - Preprocessed review text column
- `meta_category` - Broad product category
- `cluster` - Cluster ID (0-4)
- `cluster_label` - Human-readable cluster name
- `predicted_sentiment_SVM` - Sentiment prediction
- `top_products_df` - Top products per cluster
- `popularity_score` - Combined rating + review count metric

---

## 🎓 Learning Outcomes

This project demonstrates:
- End-to-end ML pipeline design
- NLP preprocessing best practices
- Multi-model comparison and selection
- Unsupervised learning (clustering)
- Transfer learning with transformers
- Parameter-efficient fine-tuning (LoRA)
- Interactive ML application deployment
- Data storytelling and visualization

---

## 🐛 Troubleshooting

**Issue:** Out of memory during summarization  
**Solution:** Use `google/flan-t5-small` or reduce batch size

**Issue:** Missing columns error  
**Solution:** Ensure notebooks are executed in order

**Issue:** Slow NLTK downloads  
**Solution:** Pre-download resources or use offline mode

**Issue:** GPU not detected  
**Solution:** Check CUDA installation: `torch.cuda.is_available()`

**Issue:** Gradio app won't start  
**Solution:** Check port 7860 is free or change `server_port`

---

## 🔮 Future Enhancements

- [ ] Add aspect-based sentiment analysis (e.g., "battery life", "screen quality")
- [ ] Implement time-series analysis of review trends
- [ ] Add competitor product comparison
- [ ] Deploy to cloud (Hugging Face Spaces, AWS, etc.)
- [ ] Create REST API for predictions
- [ ] Add more visualizations (word clouds, network graphs)
- [ ] Implement A/B testing framework for summaries
- [ ] Multi-language support

---

## 👥 Contributing

This is an educational project. Contributions welcome:
1. Keep changes minimal and well-documented
2. Follow notebook-first approach
3. Add tests for any utility functions
4. Update README with new features

---

## 📄 License

Educational project - MIT License

---

## 🙏 Acknowledgments

- **Dataset:** Amazon product reviews (Datafiniti)
- **Models:** Hugging Face Transformers, scikit-learn
- **Framework:** Google Flan-T5, PEFT/LoRA
- **UI:** Gradio

---

**Project Maintainer:** Georg  
**Course:** Ironhack AI Engineering Bootcamp  
**Last Updated:** November 2025


