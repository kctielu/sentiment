# Zero-Shot Classification for Populism Dimensions

## Original Question

**How would you recommend adjusting the zero shot classification code if I want to score based on:**

### 1. Economic Protectionism (+1 = pro, 0 = neutral, -1 = anti)

For instance, for pro-protectionism:
- "This article defends protecting the national economy against foreign competition"
- "This article criticizes free trade agreements as harmful to domestic workers"
- "This article advocates for trade barriers and tariffs"
- "This article denounces outsourcing and demands repatriation of jobs"

### 2. China Stance (+1 = pro-China, 0 = neutral, -1 = anti-China)

For instance, for anti-China:
- "This article criticizes China as a threat to domestic employment"
- "This article denounces unfair competition from Chinese products"
- "This article expresses distrust of Chinese investments"
- "This article criticizes Chinese trade practices as unjust"

### 3. Immigration (+1 = pro-immigration, 0 = neutral, -1 = anti-immigration)

For instance, for pro-immigration:
- "This article defends welcoming immigrants and refugees"
- "This article highlights positive contributions of immigration to society"
- "This article criticizes restrictive immigration policies as inhumane"
- "This article values cultural diversity brought by immigration"

---

## Response

Here's how to modify the zero-shot classification code to score your three economic and political dimensions. The key changes are:

1. **Replace political ideology dimensions** with economic protectionism, China stance, and immigration
2. **Define opposing poles** for each dimension (positive/negative/neutral)
3. **Use French-specific labels** tailored for 2016-2019 news articles
4. **Calculate composite score** by averaging all three dimensions
5. **Output both article-level and regional aggregate scores**

### Key Improvements Over Original Code

| Feature | Original | Updated |
|---------|----------|---------|
| **Dimensions** | Left-right ideology | Economic protectionism, China stance, Immigration |
| **Scoring** | Single -1 to +1 axis | -1 to +1 for each dimension |
| **Labels** | Generic political labels | Specific economic/policy indicators |
| **Languages** | French only | French (optimized for 2016-2019) |
| **Output** | Single score | 3 dimension scores + composite |
| **Context** | Generic | Tailored for trade/migration debates |

### How It Works

1. **For each dimension**, the model compares the article against:
   - **Positive labels** (e.g., pro-protectionism) → Score: +1
   - **Negative labels** (e.g., anti-protectionism/free trade) → Score: -1
   - **Neutral label** (no clear stance) → Score: 0

2. **The model calculates probabilities** for each label using zero-shot classification

3. **Final score** is calculated as:
   ```
   score = positive_prob - negative_prob
   ```
   (where probabilities are normalized: positive + negative + neutral = 1)

4. **Composite score** is the weighted average of all three dimensions (equal weights: 0.33 each)

---

## Complete Code

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
Economic and Political Stance Analysis for French News Articles

Scores articles on three dimensions using zero-shot classification:
1. Economic Protectionism: +1 (pro-protectionism) to -1 (anti-protectionism)
2. China Stance: +1 (pro-China) to -1 (anti-China)
3. Immigration: +1 (pro-immigration) to -1 (anti-immigration)

Designed for French news articles from 2016-2019 period
Combined score creates a populism intensity index.
"""

import sqlite3
import pandas as pd
import numpy as np
from transformers import pipeline
from tqdm import tqdm
import logging
import json
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# ============================================================================
# DIMENSION DEFINITIONS
# ============================================================================
ECONOMIC_POLITICAL_DIMENSIONS = {
    'economic_protectionism': {
        'positive_label': "Cet article défend la protection de l'économie nationale contre la concurrence étrangère",
        # "This article defends protecting the national economy against foreign competition"
        'negative_label': "Cet article soutient le libre-échange et l'ouverture des marchés internationaux"
        # "This article supports free trade and opening international markets"
    },
    
    'china_stance': {
        'positive_label': "Cet article présente la Chine comme un partenaire commercial important et fiable",
        # "This article presents China as an important and reliable trade partner"
        'negative_label': "Cet article critique la Chine comme une menace pour l'emploi français"
        # "This article criticizes China as a threat to French employment"
    },
    
    'immigration': {
        'positive_label': "Cet article défend l'accueil des immigrants et réfugiés en France",
        # "This article defends welcoming immigrants and refugees in France"
        'negative_label': "Cet article présente l'immigration comme une menace pour l'identité nationale"
        # "This article presents immigration as a threat to national identity"
    }
}

# Note: This version is optimized for French articles (2016-2019)


class EconomicPoliticalAnalyzer:
    """Analyze economic and political stance using zero-shot classification."""
    
    def __init__(self, model_name='MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7', device=-1):
        """
        Initialize the analyzer.
        
        Args:
            model_name: Hugging Face model for zero-shot classification
                Options:
                - 'joeddav/xlm-roberta-large-xnli' (Multilingual, recommended)
                - 'facebook/bart-large-mnli' (English only, most accurate)
                - 'MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7' (Smaller, faster)
            device: -1 for CPU, 0+ for GPU
        """
        logging.info(f"Loading zero-shot classifier: {model_name}")
        self.classifier = pipeline(
            "zero-shot-classification",
            model=model_name,
            device=device
        )
        self.model_name = model_name
    
    def score_dimension(self, text: str, dimension: str) -> dict:
        """
        Score a single dimension for an article.
        
        Returns:
            dict with:
                - score: -1 (negative) or +1 (positive)
                - positive_prob: probability of positive pole
                - negative_prob: probability of negative pole
                - confidence: difference between probabilities (higher = more certain)
        """
        labels_config = ECONOMIC_POLITICAL_DIMENSIONS[dimension]
        
        # Use only positive and negative labels
        candidate_labels = [
            labels_config['positive_label'],
            labels_config['negative_label']
        ]
        
        # Truncate text to model max length
        text_truncated = text[:1024]
        
        try:
            result = self.classifier(
                text_truncated,
                candidate_labels=candidate_labels,
                multi_label=False  # Force binary choice
            )
            
            # Map results back to labels
            label_scores = dict(zip(result['labels'], result['scores']))
            
            positive_prob = label_scores.get(labels_config['positive_label'], 0)
            negative_prob = label_scores.get(labels_config['negative_label'], 0)
            
            # Normalize probabilities (should sum to 1)
            total = positive_prob + negative_prob
            if total > 0:
                positive_prob /= total
                negative_prob /= total
            
            # Binary score: +1 if positive > negative, -1 otherwise
            score = 1 if positive_prob > negative_prob else -1
            
            # Confidence is the absolute difference between probabilities
            confidence = abs(positive_prob - negative_prob)
            
            return {
                'score': score,
                'positive_prob': round(positive_prob, 4),
                'negative_prob': round(negative_prob, 4),
                'confidence': round(confidence, 4)
            }
            
        except Exception as e:
            logging.error(f"Error scoring dimension {dimension}: {e}")
            return {
                'score': 0,
                'positive_prob': 0.5,
                'negative_prob': 0.5,
                'confidence': 0.0
            }
    
    def analyze_article(self, text: str) -> dict:
        """
        Analyze all three economic/political dimensions for an article.
        
        Returns:
            dict with scores for each dimension plus composite score
        """
        results = {}
        
        for dimension in ECONOMIC_POLITICAL_DIMENSIONS.keys():
            dim_result = self.score_dimension(text, dimension)
            results[dimension] = dim_result
        
        # Composite score (average of all dimensions)
        dimension_scores = [results[dim]['score'] for dim in ECONOMIC_POLITICAL_DIMENSIONS.keys()]
        results['composite'] = {
            'score': round(np.mean(dimension_scores), 4),
            'confidence': round(np.mean([results[dim]['confidence'] for dim in ECONOMIC_POLITICAL_DIMENSIONS.keys()]), 4)
        }
        
        return results


def process_database(db_path: str, 
                     output_csv: str = 'french_economic_political_scores.csv',
                     model_name: str = 'MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7',
                     sample_size: int = None,
                     batch_size: int = 100):
    """
    Process articles from database and score economic/political dimensions.
    
    Args:
        db_path: Path to SQLite database
        output_csv: Output file path
        model_name: Hugging Face model name
        sample_size: Limit number of articles (None = all)
        batch_size: Save progress every N articles
    """
    logging.info("=" * 70)
    logging.info("🇫🇷  French Economic & Political Stance Analysis (2016-2019)")
    logging.info("=" * 70)
    
    # Initialize analyzer
    analyzer = EconomicPoliticalAnalyzer(model_name=model_name)
    
    # Connect to database
    conn = sqlite3.connect(db_path)
    
    # Build query
    query = """
        SELECT 
            a.id, a.title, a.text, a.date, a.language,
            l.nuts_code
        FROM Articles a
        LEFT JOIN Article_Locations al ON a.id = al.article_id
        LEFT JOIN Locations l ON al.location_id = l.id
        WHERE a.text IS NOT NULL 
          AND LENGTH(a.text) > 300
          AND l.nuts_code IS NOT NULL
    """
    
    if sample_size:
        query += f" ORDER BY RANDOM() LIMIT {sample_size}"
    
    logging.info("📖 Loading articles from database...")
    articles = pd.read_sql(query, conn)
    conn.close()
    
    logging.info(f"   Loaded {len(articles):,} articles")
    logging.info(f"   Languages: {articles['language'].value_counts().to_dict()}")
    
    # Process articles
    logging.info("\n🔍 Analyzing three dimensions:")
    logging.info("   1. Economic Protectionism")
    logging.info("   2. China Stance")
    logging.info("   3. Immigration")
    results = []
    
    for i, row in tqdm(articles.iterrows(), total=len(articles)):
        # Analyze article
        analysis = analyzer.analyze_article(row['text'])
        
        # Flatten results
        result = {
            'article_id': row['id'],
            'nuts_code': row['nuts_code'],
            'nuts2': row['nuts_code'][:4] if row['nuts_code'] else None,
            'date': row['date'],
            'year': pd.to_datetime(row['date']).year if row['date'] else None,
            
            # Economic protectionism dimension
            'protectionism_score': analysis['economic_protectionism']['score'],
            'protectionism_pos_prob': analysis['economic_protectionism']['positive_prob'],
            'protectionism_neg_prob': analysis['economic_protectionism']['negative_prob'],
            'protectionism_confidence': analysis['economic_protectionism']['confidence'],
            
            # China stance dimension
            'china_score': analysis['china_stance']['score'],
            'china_pos_prob': analysis['china_stance']['positive_prob'],
            'china_neg_prob': analysis['china_stance']['negative_prob'],
            'china_confidence': analysis['china_stance']['confidence'],
            
            # Immigration dimension
            'immigration_score': analysis['immigration']['score'],
            'immigration_pos_prob': analysis['immigration']['positive_prob'],
            'immigration_neg_prob': analysis['immigration']['negative_prob'],
            'immigration_confidence': analysis['immigration']['confidence'],
            
            # Composite score
            'composite_score': analysis['composite']['score'],
            'composite_confidence': analysis['composite']['confidence']
        }
        
        results.append(result)
        
        # Save progress periodically
        if (i + 1) % batch_size == 0:
            logging.info(f"   Processed {i+1:,} articles...")
            pd.DataFrame(results).to_csv(output_csv.replace('.csv', '_progress.csv'), index=False)
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    
    # Save article-level results
    results_df.to_csv(output_csv.replace('.csv', '_articles.csv'), index=False)
    logging.info(f"\n💾 Saved article-level results to: {output_csv.replace('.csv', '_articles.csv')}")
    
    # Aggregate to NUTS2-year level
    loggiprotectionism_score': ['mean', 'std'],
        'china_score': ['mean', 'std'],
        'immigration_score': ['mean', 'std'],
        'composite_score': ['mean', 'std'],
        'composite_confidence': 'mean',
        'article_id': 'count'
    }).reset_index()
    
    # Flatten column names
    regional.columns = [
        'nuts2', 'year',
        'protectionism_mean', 'protectionism_sd',
        'china_mean', 'china_sd',
        'immigration_mean', 'immigration_sd',
        'composite_mean', 'composite_sd',
        'confidence_mean', 'article_count'
    ]
    
    # Save regional results
    regional.to_csv(output_csv, index=False)
    
    # Print summary
    logging.info(f"\n✅ Saved regional results to: {output_csv}")
    logging.info(f"   Regions: {regional['nuts2'].nunique()}")
    logging.info(f"   Years: {regional['year'].min()}-{regional['year'].max()}")
    logging.info(f"   Economic Protectionism: mean={regional['protectionism_mean'].mean():.3f} "
                f"(+1=pro, -1=anti)")
    logging.info(f"   China Stance: mean={regional['china_mean'].mean():.3f} "
                f"(+1=pro-China, -1=anti-China)")
    logging.info(f"   Immigration: mean={regional['immigration_mean'].mean():.3f} "
                f"(+1=pro-immigration, -1=anti-immigration)")
    logging.info(f"   Composite Score: mean={regional['composite_mean'].mean():.3f}, "
                f"sd={regional['composite_mean'].std():.3f}")
    
    return results_df, regional


# ============================================================================
# USAGE
# ============================================================================

if __name__ == "__main__":
    import sys
    French Economic & Political Stance Analysis (2016-2019)")
        print("=" * 70)
        print("\nUsage:")
        print("  python french_stance_analysis.py <db_path> [output.csv] [--sample N]")
        print("\nExamples:")
        print("  python french_stance_analysis.py news_data.db")
        print("  python french_stance_analysis.py news_data.db results.csv --sample 500")
        print("\nDimensions analyzed:")
        print("  1. Economic Protectionism (+1=pro, -1=anti)")
        print("  2. China Stance (+1=pro-China, -1=anti-China)")
        print("  3. Immigration (+1=pro-immigration, -1=anti-immigration)")
        sys.exit(1)
    
    db_path = sys.argv[1]
    output = sys.argv[2] if len(sys.argv) > 2 and not sys.argv[2].startswith('--') else 'french_economic_political
    db_path = sys.argv[1]
    output = sys.argv[2] if len(sys.argv) > 2 and not sys.argv[2].startswith('--') else 'populism_zeroshot_scores.csv'
    
    # Check for sample flag
    sample_size = None
    if '--sample' in sys.argv:
        idx = sys.argv.index('--sample') (optimized for French)
    import os
    model_name = os.environ.get('MODEL', 'MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7
    # Get model from environment or use default
    import os
    model_name = os.environ.get('MODEL', 'joeddav/xlm-roberta-large-xnli')
    
    # Run analysis
    results_df, regional = process_database(
        db_path=db_path,
        output_csv=output,
        model_name=model_name,
        sample_size=sample_size
    )
```

---

## How to Use

### Basic Us100 articles (recommended first run)
python french_stance_analysis.py news_data.db test_results.csv --sample 100

# Full analysis for French articles (will take hours)
python french_stance_analysis.py news_data.db french_scores.csv

# Use faster model for testing
MODEL=MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7 python french_stance_analysiss.csv

# Use specific model
MODEL=facebook/bart-large-mnli python populism_zeroshot.py news_data.db
```

### Output Files
french_economic_political_scores_articles.csv`
   - One row per article
   - Scores for all 3 dimensions + composite
   - Probabilities and confidence scores

2. **Regional aggregates**: `french_economic_political_scores.csv`
   - One row per NUTS2-year combination (for French regions)s

2. **Regional aggregates**: `populism_zeroshot_scores.csv`
   - One row per NUTS2-year combination
   - Mean and standard deviation for each dimension
   - Ready to merge with your main dataset

### Column Descriptions

| Column | Description | Range |
|--------|-------------|-------|
| `protectionism_score` | Economic protectionism stance | -1 (anti/free trade) to +1 (pro) |
| `china_score` | Attitude toward China | -1 (anti-China) to +1 (pro-China) |
| `immigration_score` | Immigration stance | -1 (anti-immigration) to +1 (pro-immigration) |
| `composite_score` | Average of all three dimensions | -1 to +1 |
| `*_confidence` | Model certainty | 0 to 1 |

---

## Integration with R

```r
library(modelsummary)

# Load French stance scores
french_scores <- read_csv("french_economic_political_scores.csv")

# Merge with main dataset (filter for French regions)
final_data <- final_clean %>%
  filter(country == "FR") %>%  # France only
  left_join(french_scores, by = c("nuts2", "year"))

# Check data
summary(final_data$composite_mean)
summary(final_data$protectionism_mean)
summary(final_data$china_mean)
summary(final_data$immigration_mean)

# Run regressions for each dimension
model_protectionism <- felm(
  protectionism_mean ~ OLSimportshock_USD | nuts2 + year | 0 | nuts2,
  data = final_data
)

model_china <- felm(
  china_mean ~ OLSimportshock_USD | nuts2 + year | 0 | nuts2,
  data = final_data
)

model_immigration <- felm(
  immigration_mean ~ OLSimportshock_USD | nuts2 + year | 0 | nuts2,
  data = final_data
)

model_composite <- felm(
  composite_mean ~ OLSimportshock_USD | nuts2 + year | 0 | nuts2,
  data = final_data
)

# Create comparison table
modelsummary(
  list(
    "Protectionism" = model_protectionism,
    "China Stance" = model_china,
    "Immigration" = model_immigration,
    "Composite" = model_composite
  ),
  stars = c('*' = 0.1, '**' = 0.05, '***' = 0.01),
  coef_rename = c('OLSimportshock_USD' = 'Import Shock (kUSD per worker)'),
  gof_map = c("nobs", "r.squared", "adj.r.squared"),
  output = "french_dimensions_regression.tex",
  title = "Impact of Import Shock on French News Discourse (2016-2019)",
  notes = list(
    "Dependent variables: Regional-year average stance scores from French news articles.",
    "Higher scores indicate: pro-protectionism, pro-China, pro-immigration.",
    "Standard errors clustered at NUTS-2 region level.",
    "* p < 0.1, ** p < 0.05, *** p < 0.01"
  )
)

# Visualize dimension correlations
dimension_cors <- final_data %>%
  select(protectionism_mean, china_mean, immigration_mean) %>%
  cor(use = "complete.obs")

library(corrplot)
corrplot(dimension_cors, method = "number", type = "upper",
         title = "Correlations Between French News Discourse Dimensions

library(corrplot)
corrplot(dimension_cors, method = "number", type = "upper")
```

---

## MoritzLaurer/mDeBERTa-v3-base-xnli` | 100+ | Fast | Good | French articles (recommended) |
| `joeddav/xlm-roberta-large-xnli` | 100+ | Medium | High | Higher accuracy (slower) |
| `facebook/bart-large-mnli` | English | Fast | Highest | English only (not for French)

| Articles | CPU Time | GPU Time | Cost |
|----------|----------|----------|------|
| 500 | 30-60 min | 5-10 min | Free |
| 5,000 | 5-10 hours | 1 hour | Free |
| 50,000 | 50-100 hours | 5-10 hours | Free |
| 100,000 | 100-200 hours | 10-20 hours | Free (Colab) |

### Recommended Models

| Model | Languages | Speed | Accuracy | Best For |
|-------|-----------|-------|----------|----------|
Test the analysis on a sample before running on all French articles:

```bash
# Test with 100 articles
python french_stance_analysis.py news_data.db test_results.csv --sample 100

# Check the results
head test_results.csv
```

```r
# Validate results make sense
library(tidyverse)

# Load test results
test_results <- read_csv("test_results_articles.csv")

# Check distributions
summary(test_results$protectionism_score)
summary(test_results$china_score)
summary(test_results$immigration_score)

# Visualize
test_results %>%
  select(protectionism_score, china_score, immigration_score) %>%
  pivot_longer(everything(), names_to = "dimension", values_to = "score") %>%
  ggplot(aes(x = score, fill = dimension)) +
  geom_histogram(alpha = 0.6, bins = 30) +
  facet_wrap(~dimension, ncol = 1) +
  labs(title = "Score Distributions for French Articles (Sample)",
       x = "Score (-1 to +1)", y = "Count
# Compare with dictionary method (if you have one)
dict_results <- read_csv("dict_scores.csv")
zs_results <- read_csv("french_economic_political_scores.csv")

comparison <- dict_results %>%
  inner_join(zs_results, by = c("nuts2", "year"), suffix = c("_dict", "_zs"))

# Should show positive correlation (r > 0.5) for composite scores
cor.test(comparison$composite_mean_dict, comparison$composite_mean_zs)

# Visualize
ggplot(comparison, aes(x = composite_mean_dict, y = composite_mean_zs)) +
  geom_point(alpha = 0.5) +
  geom_smooth(method = "lm") +
  labs(title = "Dictionary vs. Zero-Shot Economic/Political Scores",
       x = "Dictionary Method", y = "Zero-Shot Classification")
```

---

## Advantages & Li for Your Thesis

**Primary approach**: Use this zero-shot classifier for French news discourse analysis:
1. Provides **three distinct dimensions** relevant to China trade shock
2. **Protectionism score** directly measures trade policy attitudes
3. **China stance** captures attitudes toward main trade partner
4. **Immigration score** may correlate with economic anxiety

**Why this works for your thesis**:
- Directly measures discourse related to trade/globalization
- Captures regional variation in France (NUTS-2 level)
- Time variation (2016 vs. 2019) during trade debate intensification
- Complements vote-based populism measures

**Expected findings**:
- Regions with higher import shock → more anti-China discourse
- Import shock → more pro-protectionism discourse
- Possible spillover: import shock → anti-immigration discourse (if economic anxiety narrative holds)
- No training data required

### Limitations ❌

- **Very slow** (1-5 seconds per article)
- Model may misinterpret complex rhetoric
- Less transparent than dictionary method
- Harder to explain to committee
- Results can vary by model choice

### Recommendation

Use **zero-shot for validation** of dictionary results, not as primary method:
1. Primary analysis: Dictionary method (fast, transparent)
2. Robustness check: Zero-shot on 5,000 article sample
3. Show both methods agree in appendix

This gives you methodological rigor without the computational burden!
