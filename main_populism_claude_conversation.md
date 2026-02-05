# Zero-Shot Classification for Populism Dimensions

## Original Question

**How would you recommend adjusting the zero shot classification code if I want to score based on:**

### 1. Anti-establishment vs. Pro-establishment (+1, 0 aka neutral, -1)

For instance, for anti-establishment:
- "This article criticizes the establishment, corrupt institutions, or the deep state"
- "This article expresses distrust of established political institutions"
- "This article portrays mainstream institutions as corrupt or illegitimate"

### 2. Pro or anti economic nationalism (+1, 0 aka neutral, -1)

For instance, for pro economic nationalism:
- "This article advocates for protecting our economy and jobs from unfair trade"
- "This article emphasizes job protection and economic sovereignty"
- "This article criticizes trade policies as unfair to domestic workers"

### 3. People-centric or technocratic (+1, 0 aka neutral, -1)

- "This article appeals to the common man, the will of the people, or the silent majority"
- "This article frames issues as ordinary citizens versus elites"
- "This article claims to speak for overlooked or silenced populations"

---

## Response

Here's how to modify the zero-shot classification code to score your three populism dimensions. The key changes are:

1. **Replace political ideology dimensions** with your three populism dimensions
2. **Define opposing poles** for each dimension (positive/negative/neutral)
3. **Add multilingual support** for French, German, and other languages
4. **Calculate composite populism score** by averaging all three dimensions
5. **Output both article-level and regional aggregate scores**

### Key Improvements Over Original Code

| Feature | Original | Updated |
|---------|----------|---------|
| **Dimensions** | Left-right ideology | 3 populism dimensions |
| **Scoring** | Single -1 to +1 axis | -1 to +1 for each dimension |
| **Labels** | Generic political labels | Specific populism indicators |
| **Languages** | French only | French, German, English + fallback |
| **Output** | Single score | 3 dimension scores + composite |
| **Multilingual** | No | Yes (with translations) |

### How It Works

1. **For each dimension**, the model compares the article against:
   - **Positive labels** (e.g., anti-establishment rhetoric) → Score: +1
   - **Negative labels** (e.g., pro-establishment) → Score: -1
   - **Neutral label** (no clear stance) → Score: 0

2. **The model calculates probabilities** for each label using zero-shot classification

3. **Final score** is calculated as:
   ```
   score = (positive_prob - negative_prob) / (positive_prob + negative_prob + neutral_prob)
   ```

4. **Composite score** is the average of all three dimension scores

---

## Complete Code

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Populism Dimension Analysis using Zero-Shot Classification

Scores articles on three populism dimensions:
1. Anti-establishment vs. Pro-establishment (+1 to -1)
2. Economic nationalism vs. Globalism (+1 to -1)
3. People-centric vs. Technocratic (+1 to -1)

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

POPULISM_DIMENSIONS = {
    'anti_establishment': {
        'positive_labels': [  # +1: Anti-establishment
            "This article criticizes the establishment, corrupt institutions, or the deep state",
            "This article expresses distrust of established political institutions",
            "This article portrays mainstream institutions as corrupt or illegitimate",
            "This article attacks political elites or the ruling class"
        ],
        'negative_labels': [  # -1: Pro-establishment
            "This article defends established institutions and their legitimacy",
            "This article expresses trust in mainstream political institutions",
            "This article portrays government institutions as effective and trustworthy",
            "This article supports the political establishment"
        ],
        'neutral_label': "This article is neutral about political institutions"
    },
    
    'economic_nationalism': {
        'positive_labels': [  # +1: Economic nationalism
            "This article advocates for protecting our economy and jobs from unfair trade",
            "This article emphasizes job protection and economic sovereignty",
            "This article criticizes trade policies as unfair to domestic workers",
            "This article opposes globalization and free trade agreements"
        ],
        'negative_labels': [  # -1: Globalism/Free trade
            "This article supports free trade and international economic cooperation",
            "This article emphasizes benefits of globalization",
            "This article advocates for open markets and trade agreements",
            "This article views international trade as beneficial"
        ],
        'neutral_label': "This article is neutral about trade and economic policy"
    },
    
    'people_centrism': {
        'positive_labels': [  # +1: People-centric/populist
            "This article appeals to the common man, the will of the people, or the silent majority",
            "This article frames issues as ordinary citizens versus elites",
            "This article claims to speak for overlooked or silenced populations",
            "This article emphasizes the wisdom of ordinary people over experts"
        ],
        'negative_labels': [  # -1: Technocratic/expert-driven
            "This article emphasizes expert knowledge and technocratic solutions",
            "This article defers to specialists and established authorities",
            "This article prioritizes evidence-based policy over popular opinion",
            "This article values institutional expertise over popular sentiment"
        ],
        'neutral_label': "This article does not take a stance on popular vs. expert authority"
    }
}

# Multilingual translations for key labels (improves accuracy)
DIMENSION_TRANSLATIONS = {
    'fr': {
        'anti_establishment': {
            'positive_labels': [
                "Cet article critique l'establishment, les institutions corrompues ou l'État profond",
                "Cet article exprime de la méfiance envers les institutions politiques établies",
                "Cet article dépeint les institutions comme corrompues ou illégitimes",
                "Cet article attaque les élites politiques ou la classe dirigeante"
            ],
            'negative_labels': [
                "Cet article défend les institutions établies et leur légitimité",
                "Cet article exprime sa confiance dans les institutions politiques",
                "Cet article présente les institutions gouvernementales comme efficaces",
                "Cet article soutient l'establishment politique"
            ],
            'neutral_label': "Cet article est neutre concernant les institutions politiques"
        },
        'economic_nationalism': {
            'positive_labels': [
                "Cet article plaide pour protéger notre économie et nos emplois",
                "Cet article met l'accent sur la protection de l'emploi et la souveraineté économique",
                "Cet article critique les politiques commerciales comme injustes pour les travailleurs",
                "Cet article s'oppose à la mondialisation et aux accords de libre-échange"
            ],
            'negative_labels': [
                "Cet article soutient le libre-échange et la coopération économique internationale",
                "Cet article souligne les avantages de la mondialisation",
                "Cet article défend les marchés ouverts et les accords commerciaux",
                "Cet article considère le commerce international comme bénéfique"
            ],
            'neutral_label': "Cet article est neutre sur le commerce et la politique économique"
        },
        'people_centrism': {
            'positive_labels': [
                "Cet article fait appel au peuple, à la volonté populaire ou à la majorité silencieuse",
                "Cet article oppose les citoyens ordinaires aux élites",
                "Cet article prétend parler au nom des populations négligées",
                "Cet article valorise la sagesse du peuple sur celle des experts"
            ],
            'negative_labels': [
                "Cet article met l'accent sur l'expertise et les solutions technocratiques",
                "Cet article s'en remet aux spécialistes et aux autorités établies",
                "Cet article privilégie les politiques fondées sur des preuves",
                "Cet article valorise l'expertise institutionnelle"
            ],
            'neutral_label': "Cet article ne prend pas position sur l'autorité populaire vs. experte"
        }
    },
    'de': {
        'anti_establishment': {
            'positive_labels': [
                "Dieser Artikel kritisiert das Establishment oder korrupte Institutionen",
                "Dieser Artikel drückt Misstrauen gegenüber politischen Institutionen aus",
                "Dieser Artikel stellt Institutionen als korrupt oder illegitim dar",
                "Dieser Artikel greift politische Eliten an"
            ],
            'negative_labels': [
                "Dieser Artikel verteidigt etablierte Institutionen",
                "Dieser Artikel drückt Vertrauen in politische Institutionen aus",
                "Dieser Artikel stellt Regierungsinstitutionen als effektiv dar",
                "Dieser Artikel unterstützt das politische Establishment"
            ],
            'neutral_label': "Dieser Artikel ist neutral gegenüber politischen Institutionen"
        },
        'economic_nationalism': {
            'positive_labels': [
                "Dieser Artikel plädiert für den Schutz unserer Wirtschaft und Arbeitsplätze",
                "Dieser Artikel betont Arbeitsschutz und wirtschaftliche Souveränität",
                "Dieser Artikel kritisiert Handelspolitik als unfair für Arbeitnehmer",
                "Dieser Artikel lehnt Globalisierung und Freihandel ab"
            ],
            'negative_labels': [
                "Dieser Artikel unterstützt Freihandel und internationale Zusammenarbeit",
                "Dieser Artikel betont die Vorteile der Globalisierung",
                "Dieser Artikel befürwortet offene Märkte",
                "Dieser Artikel sieht internationalen Handel als vorteilhaft"
            ],
            'neutral_label': "Dieser Artikel ist neutral zu Handel und Wirtschaftspolitik"
        },
        'people_centrism': {
            'positive_labels': [
                "Dieser Artikel appelliert an das Volk oder die schweigende Mehrheit",
                "Dieser Artikel stellt Bürger gegen Eliten",
                "Dieser Artikel spricht für übersehene Bevölkerungsgruppen",
                "Dieser Artikel betont die Weisheit des Volkes über Experten"
            ],
            'negative_labels': [
                "Dieser Artikel betont Expertenwissen und technokratische Lösungen",
                "Dieser Artikel vertraut auf Spezialisten und Autoritäten",
                "Dieser Artikel bevorzugt evidenzbasierte Politik",
                "Dieser Artikel schätzt institutionelle Expertise"
            ],
            'neutral_label': "Dieser Artikel nimmt keine Stellung zu Volk vs. Experten"
        }
    }
}


class PopulismZeroShotAnalyzer:
    """Analyze populism dimensions using zero-shot classification."""
    
    def __init__(self, model_name='joeddav/xlm-roberta-large-xnli', device=-1):
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
        
    def get_labels_for_language(self, dimension: str, language: str = 'en') -> dict:
        """Get dimension labels in the appropriate language."""
        if language in DIMENSION_TRANSLATIONS:
            return DIMENSION_TRANSLATIONS[language].get(
                dimension, 
                POPULISM_DIMENSIONS[dimension]
            )
        return POPULISM_DIMENSIONS[dimension]
    
    def score_dimension(self, text: str, dimension: str, language: str = 'en') -> dict:
        """
        Score a single dimension for an article.
        
        Returns:
            dict with:
                - score: float from -1 to +1
                - positive_prob: probability of positive pole
                - negative_prob: probability of negative pole
                - neutral_prob: probability of neutral
                - confidence: max probability (indicates certainty)
        """
        labels_config = self.get_labels_for_language(dimension, language)
        
        # Combine all labels for classification
        all_labels = (
            labels_config['positive_labels'] + 
            labels_config['negative_labels'] + 
            [labels_config['neutral_label']]
        )
        
        # Truncate text to model max length
        text_truncated = text[:1024]
        
        try:
            result = self.classifier(
                text_truncated,
                candidate_labels=all_labels,
                multi_label=True  # Allow multiple labels to be true
            )
            
            # Map results back to labels
            label_scores = dict(zip(result['labels'], result['scores']))
            
            # Calculate aggregate scores for each pole
            positive_scores = [
                label_scores.get(label, 0) 
                for label in labels_config['positive_labels']
            ]
            negative_scores = [
                label_scores.get(label, 0) 
                for label in labels_config['negative_labels']
            ]
            neutral_score = label_scores.get(labels_config['neutral_label'], 0)
            
            # Average probabilities for each pole
            positive_prob = np.mean(positive_scores)
            negative_prob = np.mean(negative_scores)
            neutral_prob = neutral_score
            
            # Normalize probabilities
            total = positive_prob + negative_prob + neutral_prob
            if total > 0:
                positive_prob /= total
                negative_prob /= total
                neutral_prob /= total
            
            # Calculate final score: +1 (positive) to -1 (negative)
            # Neutral contributes 0
            score = positive_prob - negative_prob
            
            # Confidence is the max probability (higher = more certain)
            confidence = max(positive_prob, negative_prob, neutral_prob)
            
            return {
                'score': round(score, 4),
                'positive_prob': round(positive_prob, 4),
                'negative_prob': round(negative_prob, 4),
                'neutral_prob': round(neutral_prob, 4),
                'confidence': round(confidence, 4)
            }
            
        except Exception as e:
            logging.error(f"Error scoring dimension {dimension}: {e}")
            return {
                'score': 0.0,
                'positive_prob': 0.0,
                'negative_prob': 0.0,
                'neutral_prob': 1.0,
                'confidence': 0.0
            }
    
    def analyze_article(self, text: str, language: str = 'en') -> dict:
        """
        Analyze all populism dimensions for an article.
        
        Returns:
            dict with scores for each dimension plus composite score
        """
        results = {}
        
        for dimension in POPULISM_DIMENSIONS.keys():
            dim_result = self.score_dimension(text, dimension, language)
            results[dimension] = dim_result
        
        # Calculate composite populism score (average of all dimensions)
        dimension_scores = [results[dim]['score'] for dim in POPULISM_DIMENSIONS.keys()]
        results['composite'] = {
            'score': round(np.mean(dimension_scores), 4),
            'confidence': round(np.mean([results[dim]['confidence'] for dim in POPULISM_DIMENSIONS.keys()]), 4)
        }
        
        return results


def process_database(db_path: str, 
                     output_csv: str = 'populism_zeroshot_scores.csv',
                     model_name: str = 'joeddav/xlm-roberta-large-xnli',
                     sample_size: int = None,
                     batch_size: int = 100):
    """
    Process articles from database and score populism dimensions.
    
    Args:
        db_path: Path to SQLite database
        output_csv: Output file path
        model_name: Hugging Face model name
        sample_size: Limit number of articles (None = all)
        batch_size: Save progress every N articles
    """
    logging.info("=" * 70)
    logging.info("🗳️  Populism Zero-Shot Classification")
    logging.info("=" * 70)
    
    # Initialize analyzer
    analyzer = PopulismZeroShotAnalyzer(model_name=model_name)
    
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
    logging.info("\n🔍 Analyzing populism dimensions...")
    results = []
    
    for i, row in tqdm(articles.iterrows(), total=len(articles)):
        # Analyze article
        analysis = analyzer.analyze_article(row['text'], row['language'])
        
        # Flatten results
        result = {
            'article_id': row['id'],
            'nuts_code': row['nuts_code'],
            'nuts2': row['nuts_code'][:4] if row['nuts_code'] else None,
            'date': row['date'],
            'year': pd.to_datetime(row['date']).year if row['date'] else None,
            'language': row['language'],
            
            # Anti-establishment dimension
            'anti_estab_score': analysis['anti_establishment']['score'],
            'anti_estab_pos_prob': analysis['anti_establishment']['positive_prob'],
            'anti_estab_neg_prob': analysis['anti_establishment']['negative_prob'],
            'anti_estab_confidence': analysis['anti_establishment']['confidence'],
            
            # Economic nationalism dimension
            'econ_nat_score': analysis['economic_nationalism']['score'],
            'econ_nat_pos_prob': analysis['economic_nationalism']['positive_prob'],
            'econ_nat_neg_prob': analysis['economic_nationalism']['negative_prob'],
            'econ_nat_confidence': analysis['economic_nationalism']['confidence'],
            
            # People-centrism dimension
            'people_centric_score': analysis['people_centrism']['score'],
            'people_centric_pos_prob': analysis['people_centrism']['positive_prob'],
            'people_centric_neg_prob': analysis['people_centrism']['negative_prob'],
            'people_centric_confidence': analysis['people_centrism']['confidence'],
            
            # Composite score
            'populism_composite': analysis['composite']['score'],
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
    logging.info("\n📊 Aggregating to NUTS2-year level...")
    
    regional = results_df.groupby(['nuts2', 'year']).agg({
        'anti_estab_score': ['mean', 'std'],
        'econ_nat_score': ['mean', 'std'],
        'people_centric_score': ['mean', 'std'],
        'populism_composite': ['mean', 'std'],
        'composite_confidence': 'mean',
        'article_id': 'count'
    }).reset_index()
    
    # Flatten column names
    regional.columns = [
        'nuts2', 'year',
        'anti_estab_mean', 'anti_estab_sd',
        'econ_nat_mean', 'econ_nat_sd',
        'people_centric_mean', 'people_centric_sd',
        'populism_mean', 'populism_sd',
        'confidence_mean', 'article_count'
    ]
    
    # Save regional results
    regional.to_csv(output_csv, index=False)
    
    # Print summary
    logging.info(f"\n✅ Saved regional results to: {output_csv}")
    logging.info(f"   Regions: {regional['nuts2'].nunique()}")
    logging.info(f"   Years: {regional['year'].min()}-{regional['year'].max()}")
    logging.info(f"   Total articles: {regional['article_count'].sum():,}")
    
    logging.info("\n📈 Dimension Summaries:")
    logging.info(f"   Anti-establishment: mean={regional['anti_estab_mean'].mean():.3f}, "
                f"sd={regional['anti_estab_mean'].std():.3f}")
    logging.info(f"   Economic nationalism: mean={regional['econ_nat_mean'].mean():.3f}, "
                f"sd={regional['econ_nat_mean'].std():.3f}")
    logging.info(f"   People-centrism: mean={regional['people_centric_mean'].mean():.3f}, "
                f"sd={regional['people_centric_mean'].std():.3f}")
    logging.info(f"   Composite populism: mean={regional['populism_mean'].mean():.3f}, "
                f"sd={regional['populism_mean'].std():.3f}")
    
    return results_df, regional


# ============================================================================
# USAGE
# ============================================================================

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Populism Zero-Shot Classification")
        print("=" * 70)
        print("\nUsage:")
        print("  python populism_zeroshot.py <db_path> [output.csv] [--sample N]")
        print("\nExamples:")
        print("  python populism_zeroshot.py news_data.db")
        print("  python populism_zeroshot.py news_data.db results.csv --sample 1000")
        print("\nModels (set MODEL env var):")
        print("  - joeddav/xlm-roberta-large-xnli (multilingual, recommended)")
        print("  - facebook/bart-large-mnli (English only, most accurate)")
        sys.exit(1)
    
    db_path = sys.argv[1]
    output = sys.argv[2] if len(sys.argv) > 2 and not sys.argv[2].startswith('--') else 'populism_zeroshot_scores.csv'
    
    # Check for sample flag
    sample_size = None
    if '--sample' in sys.argv:
        idx = sys.argv.index('--sample')
        sample_size = int(sys.argv[idx + 1])
    
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

### Basic Usage

```bash
# Test with 500 articles (recommended first run)
python populism_zeroshot.py news_data.db results.csv --sample 500

# Full analysis (will take hours)
python populism_zeroshot.py news_data.db populism_scores.csv

# Use specific model
MODEL=facebook/bart-large-mnli python populism_zeroshot.py news_data.db
```

### Output Files

The script creates two CSV files:

1. **Article-level**: `populism_zeroshot_scores_articles.csv`
   - One row per article
   - Scores for all 3 dimensions + composite
   - Probabilities and confidence scores

2. **Regional aggregates**: `populism_zeroshot_scores.csv`
   - One row per NUTS2-year combination
   - Mean and standard deviation for each dimension
   - Ready to merge with your main dataset

### Column Descriptions

| Column | Description | Range |
|--------|-------------|-------|
| `anti_estab_score` | Anti-establishment sentiment | -1 (pro) to +1 (anti) |
| `econ_nat_score` | Economic nationalism | -1 (globalist) to +1 (nationalist) |
| `people_centric_score` | People-centrism | -1 (technocratic) to +1 (populist) |
| `populism_composite` | Average of all three dimensions | -1 to +1 |
| `*_confidence` | Model certainty | 0 to 1 |

---

## Integration with R

```r
library(tidyverse)
library(lfe)
library(fixest)

# Load populism scores
populism_zs <- read_csv("populism_zeroshot_scores.csv")

# Merge with main dataset
final_data <- final_clean %>%
  left_join(populism_zs, by = c("nuts2", "year"))

# Check data
summary(final_data$populism_mean)
summary(final_data$anti_estab_mean)

# Run regressions for each dimension
model_anti_estab <- felm(
  anti_estab_mean ~ OLSimportshock_USD | nuts2 + country:year | 0 | nuts2,
  data = final_data
)

model_econ_nat <- felm(
  econ_nat_mean ~ OLSimportshock_USD | nuts2 + country:year | 0 | nuts2,
  data = final_data
)

model_people <- felm(
  people_centric_mean ~ OLSimportshock_USD | nuts2 + country:year | 0 | nuts2,
  data = final_data
)

model_composite <- felm(
  populism_mean ~ OLSimportshock_USD | nuts2 + country:year | 0 | nuts2,
  data = final_data
)

# Create comparison table
modelsummary(
  list(
    "Anti-Establishment" = model_anti_estab,
    "Economic Nationalism" = model_econ_nat,
    "People-Centrism" = model_people,
    "Composite Populism" = model_composite
  ),
  stars = c('*' = 0.1, '**' = 0.05, '***' = 0.01),
  coef_rename = c('OLSimportshock_USD' = 'Import Shock'),
  gof_map = c("nobs", "r.squared", "adj.r.squared"),
  output = "populism_dimensions_regression.tex"
)

# Visualize dimension correlations
dimension_cors <- final_data %>%
  select(anti_estab_mean, econ_nat_mean, people_centric_mean) %>%
  cor(use = "complete.obs")

library(corrplot)
corrplot(dimension_cors, method = "number", type = "upper")
```

---

## Performance Considerations

### Speed Estimates

| Articles | CPU Time | GPU Time | Cost |
|----------|----------|----------|------|
| 500 | 30-60 min | 5-10 min | Free |
| 5,000 | 5-10 hours | 1 hour | Free |
| 50,000 | 50-100 hours | 5-10 hours | Free |
| 100,000 | 100-200 hours | 10-20 hours | Free (Colab) |

### Recommended Models

| Model | Languages | Speed | Accuracy | Best For |
|-------|-----------|-------|----------|----------|
| `joeddav/xlm-roberta-large-xnli` | 100+ | Medium | High | Your thesis (multilingual) |
| `facebook/bart-large-mnli` | English | Fast | Highest | English-only datasets |
| `MoritzLaurer/mDeBERTa-v3-base-xnli` | 100+ | Fastest | Good | Quick testing |

### Tips for Speed

1. **Start with a sample**: Test with `--sample 500` first
2. **Use GPU**: Google Colab offers free GPU (12 hours/day)
3. **Process in chunks**: Script saves progress every 100 articles
4. **Consider parallel processing**: Split by country/year

---

## Validation

Compare with dictionary method to ensure results make sense:

```bash
# Run both methods
python populism_dictionary.py news_data.db dict_scores.csv
python populism_zeroshot.py news_data.db zeroshot_scores.csv --sample 5000
```

```r
# Compare results
dict_results <- read_csv("dict_scores.csv")
zs_results <- read_csv("zeroshot_scores.csv")

comparison <- dict_results %>%
  inner_join(zs_results, by = c("nuts2", "year"), suffix = c("_dict", "_zs"))

# Should show positive correlation (r > 0.5)
cor.test(comparison$populism_mean_dict, comparison$populism_mean_zs)

# Visualize
ggplot(comparison, aes(x = populism_mean_dict, y = populism_mean_zs)) +
  geom_point(alpha = 0.5) +
  geom_smooth(method = "lm") +
  labs(title = "Dictionary vs. Zero-Shot Populism Scores",
       x = "Dictionary Method", y = "Zero-Shot Classification")
```

---

## Advantages & Limitations

### Advantages ✅

- More nuanced than keyword counting
- Captures context and implicit meaning
- Multilingual support built-in
- Provides confidence scores
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