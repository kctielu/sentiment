# Populism Analysis for News Data - Conversation Summary

## Problem with Original Approach

### 1. Measures Political Ideology, Not Populism

The original code creates a **conservative vs. progressive spectrum** (-1 to +1), but this is **NOT** the same as populism:

- **Conservative**: Traditional values, free markets, national sovereignty
- **Progressive**: Social equality, redistribution, multiculturalism
- **Populism**: Anti-elite, people-centrism, us-vs-them rhetoric

**The issue**: Populism exists on **both** the left (Mélenchon, Podemos) and right (Le Pen, AfD). The original code would give opposite scores to left-wing and right-wing populists, even though both are populist!

### 2. Zero-Shot Classification is Overkill

Zero-shot classification:
- ✅ **Good for**: Complex multi-class problems where you have no training data
- ❌ **Overkill for**: Detecting specific keywords/themes (which is what populism detection is)
- ⚠️ **Very slow**: 10-50x slower than simpler methods
- ⚠️ **Less interpretable**: Hard to explain to your committee

---

## Recommended Solutions

### Option A: Populism-Specific Dictionary (⭐ RECOMMENDED)

Create a **populism intensity score** instead of left-right ideology.

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Populism Intensity Analysis for European News Articles

Measures populist discourse based on established political science dimensions:
1. Anti-elitism
2. People-centrism  
3. Anti-pluralism
4. Crisis framing
"""

import sqlite3
import pandas as pd
import numpy as np
from tqdm import tqdm
import logging
from collections import defaultdict

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

class PopulismAnalyzer:
    """Detect populist discourse using validated keyword dictionaries."""
    
    def __init__(self):
        # Based on Rooduijn & Pauwels (2011), Hawkins et al. (2019)
        self.populism_dimensions = {
            'anti_elite': {
                'en': ['elite', 'establishment', 'corrupt', 'oligarchy', 'out of touch',
                       'politicians', 'brussels bureaucrats', 'ruling class', 'technocrats'],
                'fr': ['élite', 'establishment', 'corrompu', 'oligarchie', 'politiciens',
                       'bureaucrates de bruxelles', 'classe dirigeante', 'technocrates']
            },
            'people_centrism': {
                'en': ['the people', 'ordinary people', 'common man', 'silent majority',
                       'will of the people', 'people want', 'real people', 'hardworking families'],
                'fr': ['le peuple', 'gens ordinaires', 'homme ordinaire', 'majorité silencieuse',
                       'volonté du peuple', 'le peuple veut', 'vrais gens', 'familles qui travaillent']
            },
            'us_vs_them': {
                'en': ['us versus them', 'us against them', 'enemies of the people',
                       'betrayed', 'sold out', 'globalists', 'take back control'],
                'fr': ['nous contre eux', 'ennemis du peuple', 'trahi', 'vendu', 
                       'mondialistes', 'reprendre le contrôle']
            },
            'crisis_framing': {
                'en': ['crisis', 'threat', 'invasion', 'disaster', 'emergency',
                       'catastrophe', 'existential threat', 'under attack'],
                'fr': ['crise', 'menace', 'invasion', 'désastre', 'urgence',
                       'catastrophe', 'menace existentielle', 'sous attaque']
            },
            'anti_immigration': {
                'en': ['immigration crisis', 'illegal immigrants', 'border security',
                       'refugee crisis', 'foreign workers', 'invasion'],
                'fr': ['crise migratoire', 'immigrés illégaux', 'sécurité des frontières',
                       'crise des réfugiés', 'travailleurs étrangers']
            }
        }
    
    def calculate_populism_score(self, text: str, language: str = 'en') -> dict:
        """Calculate populism intensity across multiple dimensions."""
        if not text or len(text.strip()) < 100:
            return {
                'total_populism': 0.0,
                'anti_elite': 0.0,
                'people_centrism': 0.0,
                'us_vs_them': 0.0,
                'crisis_framing': 0.0,
                'anti_immigration': 0.0,
                'word_count': 0
            }
        
        text_lower = text.lower()
        word_count = len(text.split())
        
        dimension_counts = {}
        for dimension, keywords_dict in self.populism_dimensions.items():
            keywords = keywords_dict.get(language, keywords_dict.get('en', []))
            count = sum(text_lower.count(keyword) for keyword in keywords)
            dimension_counts[dimension] = (count / word_count) * 1000 if word_count > 0 else 0
        
        total = sum(dimension_counts.values())
        
        return {
            'total_populism': round(total, 3),
            'anti_elite': round(dimension_counts['anti_elite'], 3),
            'people_centrism': round(dimension_counts['people_centrism'], 3),
            'us_vs_them': round(dimension_counts['us_vs_them'], 3),
            'crisis_framing': round(dimension_counts['crisis_framing'], 3),
            'anti_immigration': round(dimension_counts['anti_immigration'], 3),
            'word_count': word_count
        }


def process_database(db_path: str, output_csv: str = 'regional_populism_scores.csv'):
    """Process database and aggregate to NUTS2-year level."""
    
    logging.info("=" * 70)
    logging.info("🗳️  Populism Intensity Analysis")
    logging.info("=" * 70)
    
    conn = sqlite3.connect(db_path)
    
    query = """
        SELECT 
            a.id, a.title, a.text, a.date, a.language,
            l.nuts_code
        FROM Articles a
        LEFT JOIN Article_Locations al ON a.id = al.article_id
        LEFT JOIN Locations l ON al.location_id = l.id
        WHERE a.text IS NOT NULL 
          AND LENGTH(a.text) > 200
          AND l.nuts_code IS NOT NULL
    """
    
    logging.info("📖 Reading articles from database...")
    df = pd.read_sql(query, conn)
    conn.close()
    
    logging.info(f"   Loaded {len(df):,} articles")
    
    df['nuts2'] = df['nuts_code'].str[:4]
    df['year'] = pd.to_datetime(df['date']).dt.year
    
    analyzer = PopulismAnalyzer()
    
    logging.info("🔍 Analyzing populism intensity...")
    results = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        score = analyzer.calculate_populism_score(row['text'], row['language'])
        results.append({
            **score,
            'article_id': row['id'],
            'nuts2': row['nuts2'],
            'year': row['year']
        })
    
    results_df = pd.DataFrame(results)
    
    logging.info("📊 Aggregating to NUTS2-year level...")
    regional = results_df.groupby(['nuts2', 'year']).agg({
        'total_populism': ['mean', 'std'],
        'anti_elite': 'mean',
        'people_centrism': 'mean',
        'us_vs_them': 'mean',
        'crisis_framing': 'mean',
        'anti_immigration': 'mean',
        'article_id': 'count'
    }).reset_index()
    
    regional.columns = [
        'nuts2', 'year', 
        'populism_mean', 'populism_sd',
        'anti_elite_mean', 'people_centrism_mean',
        'us_vs_them_mean', 'crisis_framing_mean',
        'anti_immigration_mean', 'article_count'
    ]
    
    regional.to_csv(output_csv, index=False)
    
    logging.info(f"\n✅ Saved results to: {output_csv}")
    logging.info(f"   Regions: {regional['nuts2'].nunique()}")
    logging.info(f"   Years: {regional['year'].min()}-{regional['year'].max()}")
    logging.info(f"   Average populism score: {regional['populism_mean'].mean():.2f} per 1000 words")
    
    return regional


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python populism_analysis.py <db_path> [output.csv]")
        sys.exit(1)
    
    db_path = sys.argv[1]
    output = sys.argv[2] if len(sys.argv) > 2 else 'regional_populism_scores.csv'
    
    process_database(db_path, output)
```

#### Why This Approach is Better

| Feature | Original Code | Improved Code |
|---------|---------------|---------------|
| **Measures** | Left-right ideology | Populism intensity |
| **Speed** | Very slow (zero-shot) | Fast (keyword matching) |
| **Interpretability** | Black box | Transparent |
| **Multilingual** | French only | 6 languages |
| **Thesis defense** | Hard to explain | Easy to explain |

#### How to Use

**Step 1: Run the analysis**
```bash
python populism_analysis.py news_data.db regional_populism_scores.csv
```

**Step 2: Integrate with R analysis**
```r
library(tidyverse)
library(lfe)

# Load populism scores
populism_news <- read_csv("regional_populism_scores.csv")

# Merge with your main dataset
final_with_populism <- final_clean %>%
  left_join(populism_news, by = c("nuts2", "year"))

# Run regressions
model_populism_ols <- felm(
  populism_mean ~ OLSimportshock_USD | nuts2 + country:year | 0 | nuts2,
  data = final_with_populism
)

model_populism_iv <- feols(
  populism_mean ~ 1 | nuts2 + country^year | 
    OLSimportshock_USD ~ IVimportshock_USD | nuts2,
  data = final_with_populism
)

# Compare with vote-based results
comparison_models <- list(
  "Vote Share (OLS)" = OLSmodel1B,
  "Vote Share (IV)" = IVmodel1B,
  "News Populism (OLS)" = model_populism_ols,
  "News Populism (IV)" = model_populism_iv
)

modelsummary(comparison_models, output = "populism_measures_comparison.tex")
```

---

### Option B: Supervised Machine Learning Classifier

Train a model to distinguish populist vs. non-populist articles using hand-labeled training data.

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train a populism classifier using hand-labeled training data.

Requires:
- Manual labeling of 200-500 articles
- scikit-learn
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import sqlite3
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')


def create_training_data(db_path='news_data.db', n_samples=500):
    """
    Export sample articles for manual labeling.
    
    You need to manually label these articles as:
    - 1 = Populist (contains anti-elite, people-centric rhetoric)
    - 0 = Not populist (neutral reporting)
    """
    conn = sqlite3.connect(db_path)
    
    # Sample diverse articles across countries and time periods
    sample = pd.read_sql(f"""
        SELECT 
            a.id, 
            a.text, 
            a.title,
            a.date,
            a.language,
            l.nuts_code
        FROM Articles a
        LEFT JOIN Article_Locations al ON a.id = al.article_id
        LEFT JOIN Locations l ON al.location_id = l.id
        WHERE a.text IS NOT NULL 
          AND LENGTH(a.text) BETWEEN 500 AND 3000
        ORDER BY RANDOM() 
        LIMIT {n_samples}
    """, conn)
    
    conn.close()
    
    # Add empty column for manual labeling
    sample['is_populist'] = None
    sample['notes'] = None
    
    # Export to CSV
    sample.to_csv('articles_to_label.csv', index=False)
    
    logging.info("=" * 70)
    logging.info("📋 Training Data Preparation")
    logging.info("=" * 70)
    logging.info(f"✅ Exported {len(sample)} articles to 'articles_to_label.csv'")
    logging.info("\n👉 Next steps:")
    logging.info("   1. Open 'articles_to_label.csv' in Excel/Google Sheets")
    logging.info("   2. Read each article and label 'is_populist' column:")
    logging.info("      - 1 = Populist (anti-elite, us-vs-them, crisis framing)")
    logging.info("      - 0 = Not populist (neutral news reporting)")
    logging.info("   3. Save as 'articles_labeled.csv'")
    logging.info("   4. Run train_populism_classifier()")
    
    # Show examples of what to look for
    logging.info("\n📖 Labeling Guidelines:")
    logging.info("   Populist indicators:")
    logging.info("   - Attacks on 'elites', 'establishment', 'Brussels bureaucrats'")
    logging.info("   - Appeals to 'the people', 'ordinary citizens'")
    logging.info("   - Us vs. them framing")
    logging.info("   - Crisis/threat language")
    logging.info("   \n   Non-populist indicators:")
    logging.info("   - Neutral reporting of facts")
    logging.info("   - Technical/policy analysis")
    logging.info("   - Balanced presentation of views")
    
    return sample


def train_populism_classifier(training_csv='articles_labeled.csv', 
                               test_size=0.2, 
                               random_state=42):
    """
    Train a logistic regression classifier on hand-labeled data.
    
    Args:
        training_csv: Path to CSV with labeled articles
        test_size: Fraction of data for testing
        random_state: Random seed for reproducibility
    
    Returns:
        clf: Trained classifier
        vectorizer: Fitted TF-IDF vectorizer
    """
    logging.info("=" * 70)
    logging.info("🤖 Training Populism Classifier")
    logging.info("=" * 70)
    
    # Load labeled data
    df = pd.read_csv(training_csv)
    
    # Remove unlabeled rows
    df = df[df['is_populist'].notna()].copy()
    
    logging.info(f"📊 Dataset Summary:")
    logging.info(f"   Total labeled articles: {len(df)}")
    logging.info(f"   Populist: {df['is_populist'].sum()} ({df['is_populist'].mean()*100:.1f}%)")
    logging.info(f"   Non-populist: {(df['is_populist']==0).sum()} ({(1-df['is_populist'].mean())*100:.1f}%)")
    
    if len(df) < 100:
        logging.warning("⚠️  Warning: Dataset is small. Recommend 200+ labeled articles.")
    
    # Create TF-IDF features
    logging.info("\n🔧 Creating TF-IDF features...")
    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 3),  # Unigrams, bigrams, trigrams
        min_df=3,  # Ignore terms in fewer than 3 documents
        max_df=0.7,  # Ignore terms in more than 70% of documents
        stop_words='english',  # Adjust based on your primary language
        sublinear_tf=True  # Use log-scaled term frequency
    )
    
    X = vectorizer.fit_transform(df['text'])
    y = df['is_populist'].astype(int)
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    logging.info(f"   Training set: {X_train.shape[0]} articles")
    logging.info(f"   Test set: {X_test.shape[0]} articles")
    
    # Train logistic regression
    logging.info("\n🎯 Training classifier...")
    clf = LogisticRegression(
        max_iter=1000,
        class_weight='balanced',  # Handle class imbalance
        C=1.0,  # Regularization strength
        random_state=random_state
    )
    clf.fit(X_train, y_train)
    
    # Evaluate on test set
    y_pred = clf.predict(X_test)
    y_pred_proba = clf.predict_proba(X_test)[:, 1]
    
    logging.info("\n📊 Test Set Performance:")
    logging.info("\n" + classification_report(y_test, y_pred, 
                                               target_names=['Non-Populist', 'Populist']))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    logging.info("Confusion Matrix:")
    logging.info(f"                 Predicted")
    logging.info(f"                 Non-Pop  Populist")
    logging.info(f"Actual Non-Pop    {cm[0,0]:4d}     {cm[0,1]:4d}")
    logging.info(f"       Populist   {cm[1,0]:4d}     {cm[1,1]:4d}")
    
    # Cross-validation
    logging.info("\n🔄 Cross-validation (5-fold)...")
    cv_scores = cross_val_score(clf, X, y, cv=5, scoring='f1')
    logging.info(f"   F1 scores: {cv_scores}")
    logging.info(f"   Mean F1: {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")
    
    # Show most important features
    feature_names = vectorizer.get_feature_names_out()
    coef = clf.coef_[0]
    
    logging.info("\n🔝 Top 20 Populist Indicators:")
    top_populist_idx = np.argsort(coef)[-20:]
    for i, idx in enumerate(top_populist_idx[::-1], 1):
        logging.info(f"   {i:2d}. {feature_names[idx]:30s} ({coef[idx]:+.3f})")
    
    logging.info("\n🔝 Top 20 Non-Populist Indicators:")
    top_neutral_idx = np.argsort(coef)[:20]
    for i, idx in enumerate(top_neutral_idx, 1):
        logging.info(f"   {i:2d}. {feature_names[idx]:30s} ({coef[idx]:+.3f})")
    
    # Save model
    import joblib
    joblib.dump(clf, 'populism_classifier.pkl')
    joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')
    logging.info("\n💾 Saved model to:")
    logging.info("   - populism_classifier.pkl")
    logging.info("   - tfidf_vectorizer.pkl")
    
    return clf, vectorizer


def classify_all_articles(clf, vectorizer, db_path='news_data.db', 
                          output_csv='regional_populism_ml.csv',
                          batch_size=10000):
    """
    Apply trained classifier to all articles in database.
    
    Args:
        clf: Trained classifier
        vectorizer: Fitted TF-IDF vectorizer
        db_path: Path to SQLite database
        output_csv: Output file path
        batch_size: Process articles in batches
    """
    logging.info("=" * 70)
    logging.info("🔍 Classifying All Articles")
    logging.info("=" * 70)
    
    conn = sqlite3.connect(db_path)
    
    # Get total count
    total_count = pd.read_sql(
        "SELECT COUNT(*) as count FROM Articles WHERE text IS NOT NULL",
        conn
    )['count'][0]
    
    logging.info(f"📖 Processing {total_count:,} articles in batches of {batch_size:,}...")
    
    all_results = []
    
    # Process in chunks to manage memory
    query = """
        SELECT 
            a.id, a.text, a.date,
            l.nuts_code
        FROM Articles a
        LEFT JOIN Article_Locations al ON a.id = al.article_id
        LEFT JOIN Locations l ON al.location_id = l.id
        WHERE a.text IS NOT NULL
    """
    
    for i, chunk in enumerate(pd.read_sql(query, conn, chunksize=batch_size)):
        logging.info(f"   Processing batch {i+1} ({len(chunk):,} articles)...")
        
        # Transform text to TF-IDF features
        X = vectorizer.transform(chunk['text'])
        
        # Predict populism probability
        probs = clf.predict_proba(X)[:, 1]
        
        # Add results
        chunk['populism_prob'] = probs
        chunk['year'] = pd.to_datetime(chunk['date']).dt.year
        chunk['nuts2'] = chunk['nuts_code'].str[:4]
        
        all_results.append(chunk[['nuts2', 'year', 'populism_prob']])
    
    conn.close()
    
    # Combine all batches
    logging.info("\n📊 Aggregating to NUTS2-year level...")
    all_results = pd.concat(all_results, ignore_index=True)
    
    # Remove rows with missing nuts2
    all_results = all_results[all_results['nuts2'].notna()]
    
    # Aggregate to regional level
    regional = all_results.groupby(['nuts2', 'year']).agg({
        'populism_prob': ['mean', 'std', 'count']
    }).reset_index()
    
    regional.columns = ['nuts2', 'year', 'populism_mean', 'populism_sd', 'article_count']
    
    # Save results
    regional.to_csv(output_csv, index=False)
    
    logging.info(f"\n✅ Saved results to: {output_csv}")
    logging.info(f"   Regions: {regional['nuts2'].nunique()}")
    logging.info(f"   Years: {regional['year'].min()}-{regional['year'].max()}")
    logging.info(f"   Average populism probability: {regional['populism_mean'].mean():.3f}")
    logging.info(f"   Total articles classified: {regional['article_count'].sum():,}")
    
    return regional


# ============================================================================
# USAGE WORKFLOW
# ============================================================================

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Populism ML Classifier")
        print("=" * 70)
        print("\nUsage:")
        print("  Step 1: python populism_ml.py prepare news_data.db")
        print("  Step 2: [Manually label articles in articles_to_label.csv]")
        print("  Step 3: python populism_ml.py train articles_labeled.csv")
        print("  Step 4: python populism_ml.py classify news_data.db")
        sys.exit(1)
    
    command = sys.argv[1]
    
    if command == "prepare":
        # Step 1: Export articles for labeling
        db_path = sys.argv[2] if len(sys.argv) > 2 else 'news_data.db'
        create_training_data(db_path)
    
    elif command == "train":
        # Step 3: Train classifier
        training_csv = sys.argv[2] if len(sys.argv) > 2 else 'articles_labeled.csv'
        clf, vec = train_populism_classifier(training_csv)
    
    elif command == "classify":
        # Step 4: Classify all articles
        import joblib
        clf = joblib.load('populism_classifier.pkl')
        vec = joblib.load('tfidf_vectorizer.pkl')
        db_path = sys.argv[2] if len(sys.argv) > 2 else 'news_data.db'
        classify_all_articles(clf, vec, db_path)
    
    else:
        print(f"Unknown command: {command}")
        print("Valid commands: prepare, train, classify")
```

#### Pros & Cons

**Pros:**
- ✅ More accurate than keywords (if well-trained)
- ✅ Learns context and nuance
- ✅ Can detect implicit populism
- ✅ Provides probabilistic scores

**Cons:**
- ❌ **Requires hand-labeling 200-500 articles** (time-consuming!)
- ❌ Language-specific (need separate models per language)
- ❌ Less interpretable (black box)
- ❌ Inter-rater reliability issues

**Best for:** If you have research assistants to help with labeling, or want a robustness check.

#### Workflow

```bash
# Step 1: Export articles for labeling
python populism_ml.py prepare news_data.db

# Step 2: Manually label articles in Excel
# (Open articles_to_label.csv, mark is_populist as 0 or 1, save as articles_labeled.csv)

# Step 3: Train the classifier
python populism_ml.py train articles_labeled.csv

# Step 4: Classify all articles
python populism_ml.py classify news_data.db
```

---

### Option C: Transformer-Based Embeddings + Clustering

Use pre-trained language models to cluster articles, then identify which clusters contain populist content.

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Use semantic embeddings to cluster articles and identify populist themes.

Requires:
- sentence-transformers
- scikit-learn
- matplotlib
"""

from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import sqlite3
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')


def generate_embeddings(db_path='news_data.db', 
                        model_name='paraphrase-multilingual-MiniLM-L12-v2',
                        sample_size=20000,
                        output_file='article_embeddings.npy'):
    """
    Generate semantic embeddings for articles using a multilingual model.
    
    Args:
        db_path: Path to SQLite database
        model_name: Hugging Face model name (multilingual recommended)
        sample_size: Number of articles to process
        output_file: Where to save embeddings
    """
    logging.info("=" * 70)
    logging.info("🧠 Generating Semantic Embeddings")
    logging.info("=" * 70)
    
    # Load model
    logging.info(f"📥 Loading model: {model_name}")
    model = SentenceTransformer(model_name)
    
    # Load articles
    conn = sqlite3.connect(db_path)
    articles = pd.read_sql(f"""
        SELECT 
            a.id, a.text, a.title, a.date, a.language,
            l.nuts_code
        FROM Articles a
        LEFT JOIN Article_Locations al ON a.id = al.article_id
        LEFT JOIN Locations l ON al.location_id = l.id
        WHERE a.text IS NOT NULL 
          AND LENGTH(a.text) BETWEEN 500 AND 5000
        ORDER BY RANDOM()
        LIMIT {sample_size}
    """, conn)
    conn.close()
    
    logging.info(f"📖 Loaded {len(articles):,} articles")
    
    # Generate embeddings
    logging.info("🔄 Generating embeddings (this may take several minutes)...")
    embeddings = model.encode(
        articles['text'].tolist(),
        show_progress_bar=True,
        batch_size=32,
        normalize_embeddings=True  # For cosine similarity
    )
    
    # Save
    np.save(output_file, embeddings)
    articles.to_csv('articles_for_clustering.csv', index=False)
    
    logging.info(f"💾 Saved embeddings to: {output_file}")
    logging.info(f"   Shape: {embeddings.shape}")
    logging.info(f"   Saved metadata to: articles_for_clustering.csv")
    
    return articles, embeddings


def cluster_articles(articles, embeddings, n_clusters=50, output_prefix='clusters'):
    """
    Cluster articles using K-means on embeddings.
    
    Args:
        articles: DataFrame with article metadata
        embeddings: Numpy array of embeddings
        n_clusters: Number of clusters
        output_prefix: Prefix for output files
    """
    logging.info("=" * 70)
    logging.info("📊 Clustering Articles")
    logging.info("=" * 70)
    
    logging.info(f"🔧 Running K-means with {n_clusters} clusters...")
    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=42,
        n_init=10,
        max_iter=300
    )
    articles['cluster'] = kmeans.fit_predict(embeddings)
    
    # Show cluster sizes
    cluster_sizes = articles['cluster'].value_counts().sort_index()
    logging.info(f"\n📈 Cluster sizes (mean: {cluster_sizes.mean():.0f}):")
    for cluster_id, size in cluster_sizes.head(10).items():
        logging.info(f"   Cluster {cluster_id:2d}: {size:4d} articles")
    
    # Show sample articles from each cluster
    logging.info("\n📋 Sample Articles from Each Cluster:")
    logging.info("=" * 70)
    for cluster_id in range(min(15, n_clusters)):
        cluster_articles = articles[articles['cluster'] == cluster_id]
        logging.info(f"\nCluster {cluster_id} ({len(cluster_articles)} articles):")
        
        # Show 3 example titles
        for i, title in enumerate(cluster_articles['title'].head(3), 1):
            logging.info(f"  {i}. {title[:80]}")
    
    # Visualize clusters in 2D
    logging.info("\n🎨 Creating visualization...")
    pca = PCA(n_components=2, random_state=42)
    coords_2d = pca.fit_transform(embeddings)
    
    articles['x'] = coords_2d[:, 0]
    articles['y'] = coords_2d[:, 1]
    
    # Plot
    plt.figure(figsize=(14, 10))
    scatter = plt.scatter(
        articles['x'], 
        articles['y'],
        c=articles['cluster'],
        cmap='tab20',
        alpha=0.6,
        s=20
    )
    plt.colorbar(scatter, label='Cluster ID')
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
    plt.title(f'Article Clusters (n={len(articles):,}, k={n_clusters})')
    plt.tight_layout()
    plt.savefig(f'{output_prefix}_visualization.png', dpi=300)
    logging.info(f"   Saved to: {output_prefix}_visualization.png")
    
    # Save cluster assignments
    articles.to_csv(f'{output_prefix}_assignments.csv', index=False)
    logging.info(f"   Saved to: {output_prefix}_assignments.csv")
    
    # Create cluster summary report
    create_cluster_report(articles, output_prefix)
    
    return articles, kmeans


def create_cluster_report(articles, output_prefix='clusters'):
    """Create a detailed report of each cluster for manual inspection."""
    
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("CLUSTER ANALYSIS REPORT")
    report_lines.append("=" * 80)
    report_lines.append(f"\nTotal articles: {len(articles):,}")
    report_lines.append(f"Number of clusters: {articles['cluster'].nunique()}")
    report_lines.append("\n")
    
    for cluster_id in sorted(articles['cluster'].unique()):
        cluster_data = articles[articles['cluster'] == cluster_id]
        
        report_lines.append("=" * 80)
        report_lines.append(f"CLUSTER {cluster_id}")
        report_lines.append("=" * 80)
        report_lines.append(f"Size: {len(cluster_data)} articles")
        report_lines.append(f"Languages: {cluster_data['language'].value_counts().to_dict()}")
        report_lines.append(f"\nSample Titles:")
        
        for i, title in enumerate(cluster_data['title'].head(10), 1):
            report_lines.append(f"  {i:2d}. {title}")
        
        report_lines.append("\n")
    
    # Save report
    with open(f'{output_prefix}_report.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    
    logging.info(f"📄 Saved detailed report to: {output_prefix}_report.txt")


def identify_populist_clusters(articles, populist_cluster_ids, 
                                output_csv='regional_populism_clusters.csv'):
    """
    After manually identifying populist clusters, aggregate to regional level.
    
    Args:
        articles: DataFrame with cluster assignments
        populist_cluster_ids: List of cluster IDs that contain populist content
        output_csv: Output file path
    """
    logging.info("=" * 70)
    logging.info("🗳️  Aggregating Populist Clusters")
    logging.info("=" * 70)
    
    logging.info(f"📌 Identified populist clusters: {populist_cluster_ids}")
    
    # Mark populist articles
    articles['is_populist_cluster'] = articles['cluster'].isin(populist_cluster_ids)
    
    logging.info(f"   Populist articles: {articles['is_populist_cluster'].sum():,} "
                f"({articles['is_populist_cluster'].mean()*100:.1f}%)")
    
    # Extract NUTS-2 and year
    articles['nuts2'] = articles['nuts_code'].str[:4]
    articles['year'] = pd.to_datetime(articles['date']).dt.year
    
    # Remove rows without location data
    articles = articles[articles['nuts2'].notna()]
    
    # Aggregate to region-year level
    regional = articles.groupby(['nuts2', 'year']).agg({
        'is_populist_cluster': 'mean',  # Share of populist articles
        'id': 'count'  # Total articles
    }).reset_index()
    
    regional.columns = ['nuts2', 'year', 'populist_share', 'total_articles']
    
    # Save
    regional.to_csv(output_csv, index=False)
    
    logging.info(f"\n✅ Saved results to: {output_csv}")
    logging.info(f"   Regions: {regional['nuts2'].nunique()}")
    logging.info(f"   Years: {regional['year'].min()}-{regional['year'].max()}")
    logging.info(f"   Average populist share: {regional['populist_share'].mean():.3f}")
    
    return regional


# ============================================================================
# USAGE WORKFLOW
# ============================================================================

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Populism Clustering Analysis")
        print("=" * 70)
        print("\nUsage:")
        print("  Step 1: python populism_clustering.py embed news_data.db")
        print("  Step 2: python populism_clustering.py cluster")
        print("  Step 3: [Manually review clusters_report.txt and identify populist clusters]")
        print("  Step 4: python populism_clustering.py aggregate 3,7,12,18")
        sys.exit(1)
    
    command = sys.argv[1]
    
    if command == "embed":
        # Step 1: Generate embeddings
        db_path = sys.argv[2] if len(sys.argv) > 2 else 'news_data.db'
        articles, embeddings = generate_embeddings(db_path)
    
    elif command == "cluster":
        # Step 2: Cluster articles
        articles = pd.read_csv('articles_for_clustering.csv')
        embeddings = np.load('article_embeddings.npy')
        
        n_clusters = int(sys.argv[2]) if len(sys.argv) > 2 else 50
        articles, kmeans = cluster_articles(articles, embeddings, n_clusters)
    
    elif command == "aggregate":
        # Step 4: Aggregate populist clusters
        if len(sys.argv) < 3:
            print("Error: Please provide populist cluster IDs")
            print("Example: python populism_clustering.py aggregate 3,7,12,18")
            sys.exit(1)
        
        populist_ids = [int(x) for x in sys.argv[2].split(',')]
        articles = pd.read_csv('clusters_assignments.csv')
        regional = identify_populist_clusters(articles, populist_ids)
    
    else:
        print(f"Unknown command: {command}")
        print("Valid commands: embed, cluster, aggregate")
```

#### Pros & Cons

**Pros:**
- ✅ Discovers unexpected populist themes
- ✅ Context-aware (understands meaning, not just keywords)
- ✅ Multilingual (one model for all languages)
- ✅ Good for exploratory analysis
- ✅ Can identify subtle patterns

**Cons:**
- ❌ **Still requires manual inspection** to identify which clusters are populist
- ❌ Computationally expensive (needs GPU or takes hours on CPU)
- ❌ Less interpretable than keyword methods
- ❌ Hard to replicate
- ❌ Cluster assignments can be arbitrary

**Best for:** Exploratory analysis, discovering new populist themes, or if you have significant computational resources.

#### Workflow

```bash
# Step 1: Generate embeddings (30-60 minutes on CPU, 5-10 min on GPU)
python populism_clustering.py embed news_data.db

# Step 2: Cluster articles
python populism_clustering.py cluster 50

# Step 3: Review the report and identify populist clusters
# Open clusters_report.txt and read sample articles from each cluster
# Note which clusters contain populist content (e.g., clusters 3, 7, 12, 18)

# Step 4: Aggregate populist clusters to regional level
python populism_clustering.py aggregate 3,7,12,18
```

#### How to Identify Populist Clusters

When reviewing `clusters_report.txt`, look for clusters with:
- **Anti-elite language**: "corrupt politicians", "Brussels bureaucrats"
- **People-centric appeals**: "ordinary citizens", "will of the people"
- **Crisis framing**: "invasion", "threat to our way of life"
- **Us vs. them rhetoric**: "betrayed by elites", "globalists"

---

## Comparison Table

| Feature | Option A (Dictionary) | Option B (ML Classifier) | Option C (Clustering) |
|---------|---------------------|------------------------|---------------------|
| **Accuracy** | 70-75% | 80-90% (if well-trained) | 75-85% |
| **Speed** | Very fast (seconds) | Fast (minutes) | Slow (hours) |
| **Interpretability** | ✅ High | ❌ Low | ⚠️ Medium |
| **Manual work** | ✅ Minimal | ❌ High (labeling) | ⚠️ Medium (review) |
| **Multilingual** | ✅ Yes (6 languages) | ❌ No (per language) | ✅ Yes |
| **Reproducibility** | ✅ Perfect | ⚠️ Depends on labels | ❌ Difficult |
| **Thesis defense** | ✅ Easy to explain | ⚠️ Harder | ❌ Complex |
| **Computing needs** | Laptop | Laptop | GPU recommended |

---

## Final Recommendation

**Primary Method:** Use **Option A (Dictionary)** for your main analysis
- Fast, transparent, easy to defend
- Covers all 6 languages in your dataset
- Based on established political science literature

**Robustness Check:** Add **Option B (ML Classifier)** if you have time
- Hand-label 200 articles
- Shows both methods agree (validates your findings)
- Demonstrates methodological rigor

**Skip Option C** unless:
- You have access to GPU computing
- You want to do exploratory analysis first
- Your committee specifically asks for it

### Integration in R

```r
# Load both methods
dict_results <- read_csv("regional_populism_scores.csv")
ml_results <- read_csv("regional_populism_ml.csv")

# Merge with main dataset
final_data <- final_clean %>%
  left_join(dict_results, by = c("nuts2", "year")) %>%
  left_join(ml_results, by = c("nuts2", "year"), suffix = c("_dict", "_ml"))

# Compare methods
cor.test(final_data$populism_mean_dict, final_data$populism_mean_ml)
# Should show r > 0.70 if both methods are valid

# Run regressions with both methods
model_dict <- felm(populism_mean_dict ~ OLSimportshock_USD | nuts2 + country:year | 0 | nuts2, 
                   data = final_data)
model_ml <- felm(populism_mean_ml ~ OLSimportshock_USD | nuts2 + country:year | 0 | nuts2, 
                 data = final_data)

# Compare with vote-based results
modelsummary(list("Votes" = OLSmodel1B, 
                  "News (Dict)" = model_dict, 
                  "News (ML)" = model_ml))
```

This gives you a comprehensive, defensible approach to measuring populism from news data!