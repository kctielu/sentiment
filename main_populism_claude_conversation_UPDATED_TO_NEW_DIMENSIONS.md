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
        'positive_labels': [  # +1: Pro-protectionism
            "Cet article défend la protection de l'économie nationale contre la concurrence étrangère",
            "Cet article critique les accords de libre-échange comme nuisibles aux travailleurs français",
            "Cet article plaide pour des barrières commerciales et des tarifs douaniers",
            "Cet article prône l'achat de produits français et le soutien aux industries locales",
            "Cet article dénonce la délocalisation et demande le rapatriement des emplois",
            "Cet article critique la mondialisation comme une menace pour l'économie française"
        ],
        'negative_labels': [  # -1: Anti-protectionism / Free trade
            "Cet article soutient le libre-échange et l'ouverture des marchés internationaux",
            "Cet article défend les accords commerciaux comme bénéfiques pour l'économie",
            "Cet article critique le protectionnisme comme inefficace et coûteux",
            "Cet article promeut la compétitivité internationale et l'exportation",
            "Cet article valorise l'intégration dans l'économie mondiale",
            "Cet article présente la mondialisation comme une opportunité économique"
        ],
        'neutral_label': "Cet article présente de manière neutre les politiques commerciales sans prendre position"
    },
    
    'china_stance': {
        'positive_labels': [  # +1: Pro-China
            "Cet article présente la Chine comme un partenaire commercial important et fiable",
            "Cet article valorise les investissements chinois en France",
            "Cet article défend la coopération économique avec la Chine",
            "Cet article décrit positivement les relations franco-chinoises",
            "Cet article présente la Chine comme une opportunité de croissance économique",
            "Cet article minimise les risques associés au commerce avec la Chine"
        ],
        'negative_labels': [  # -1: Anti-China
            "Cet article critique la Chine comme une menace pour l'emploi français",
            "Cet article dénonce la concurrence déloyale des produits chinois",
            "Cet article exprime de la méfiance envers les investissements chinois",
            "Cet article critique les pratiques commerciales chinoises jugées injustes",
            "Cet article présente la Chine comme un concurrent économique dangereux",
            "Cet article alerte sur les risques du dumping chinois et de la contrefaçon"
        ],
        'neutral_label': "Cet article présente la Chine de manière factuelle sans jugement positif ou négatif"
    },
    
    'immigration': {
        'positive_labels': [  # +1: Pro-immigration
            "Cet article défend l'accueil des immigrants et réfugiés en France",
            "Cet article souligne les contributions positives de l'immigration à la société",
            "Cet article critique les politiques restrictives d'immigration comme inhumaines",
            "Cet article valorise la diversité culturelle apportée par l'immigration",
            "Cet article plaide pour une politique d'accueil plus généreuse",
            "Cet article présente l'immigration comme bénéfique pour l'économie française"
        ],
        'negative_labels': [  # -1: Anti-immigration
  Note: This version is optimized for French articles (2016-2019)
# For multilingual support, add translations here
DIMENSION_TRANSLATIONS_BACKUPle demande des contrôles plus stricts aux frontières",
            "Cet article présente l'immigration comme une menace pour l'identité nationale",
            "Cet article associe l'immigration à l'insécurité et au chômage",
            "Cet article plaide pour la réduction du nombre d'immigrants accueillis",
            "Cet article critique les politiques d'immigration jugées trop laxistes"
        ],
        'neutral_label': "Cet article présente les questions d'immigration de manière objective et équilibrée
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
      EconomicPolitical
            'neeconomic and political stance using zero-shot classification."""
    
    def __init__(self, model_name='MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7oritäten",
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
                'neutral_ECONOMIC_POLITICAL.0,
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
        score (average of all dimensions)
        dimension_scores = [results[dim]['score'] for dim in ECONOMIC_POLITICALns)
        dimension_scores = [results[dim]['score'] for dim in POPULISM_DIMENSIONS.keys()]
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
    Process articles from database and score populism dimensions.
    
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
        analysis = analyzer.analyze_article(row['text'], row['language'])
        
        # Flatten results
        result = {
            'article_id': row['id'],
            'nuts_code': row['nuts_code'],
            'nuts2': row['nuts_code'][:4] if row['nuts_code'] else None,
            'date': row['date'],
            'yEconomic protectionism dimension
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
            'composite_scornfidence': analysis['people_centrism']['confidence'],
            
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
        'composite_mean', 'compositestab_sd',
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
    logging.info(f"   Economic Protectionism: mean={regional['protectionism_mean'].mean():.3f} "
                f"(+1=pro, -1=anti)")
    logging.info(f"   China Stance: mean={regional['china_mean'].mean():.3f} "
                f"(+1=pro-China, -1=anti-China)")
    logging.info(f"   Immigration: mean={regional['immigration_mean'].mean():.3f} "
                f"(+1=pro-immigration, -1=anti-immigration)")
    logging.info(f"   Composite Score: mean={regional['composite_mean'].meanal['people_centric_mean'].mean():.3f}, "
                f"sd={regional['people_centric_mean'].std():.3f}")
    logging.info(f"   Composite populism: mean={regional['populism_mean'].mean():.3f}, "
                f"sd={regional['populism_mean'].std():.3f}")
    
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
protectionism_score` | Economic protectionism stance | -1 (anti/free trade) to +1 (pro) |
| `china_score` | Attitude toward China | -1 (anti-China) to +1 (pro-China) |
| `immigration_score` | Immigration stance | -1 (anti-immigration) to +1 (pro-immigration) |
| `composite_scor | Anti-establishment sentiment | -1 (pro) to +1 (anti) |
| `econ_nat_score` | Economic nationalism | -1 (globalist) to +1 (nationalist) |
| `people_centric_score` | People-centrism | -1 (technocratic) to +1 (populist) |
| `populism_composite` | Average of all three dimensions | -1 to +1 |
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
