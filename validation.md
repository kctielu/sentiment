User: for context, i used the DeBERTa model with the following code to create the csv of transformer-classified articles by different negative and positive sentiments via a premise and hypothesis system. i now want to create a predictive validation of the classifer essentially asking: "do these classifier-derived populism scores behave like a real, meaningful measure of something?"

for instance: i could maybe use three regression models (Ridge, XGBoost, Two-Stage Hurdle) to predict the transformer-derived populism scores using some set of covariates. Performance is then broken down by score range — 0, 1–9, 10–24, 25–49, 50+.
What it validates: If the scores were noise, no model could predict them systematically. The fact that all three models achieve reasonable overall MSE (~368–428) and that the predicted means track actual means within most score ranges (e.g., 25–49 range: actuals ~37–38, predictions ~33–44) suggests the scores capture real, learnable signal.

how should i do this and how intensive will this be

GitHub Copilot: 

# Predictive Validation of Transformer-Derived Populism Scores

## Overview of What You're Proposing

Your intuition is exactly right. This is a **construct validation via predictive modeling** approach. The logic is:

> If the DeBERTa-derived populism scores are just noise, then no set of observable covariates should be able to systematically predict them. If models *can* predict them — and the predictions track actual scores across the distribution — that's evidence the classifier is capturing real, structured signal.

This is a well-established validation strategy in computational social science (sometimes called "criterion validity" or "nomological validity" depending on framing).

---

## How Intensive Is This?

**Honest answer: moderate.** Here's the breakdown:

| Component | Effort | Time Estimate |
|-----------|--------|---------------|
| Feature engineering from your CSV | Medium | 2–4 hours |
| Model fitting (Ridge, XGBoost, Hurdle) | Low-Medium | 1–2 hours |
| Score-range stratified evaluation | Medium | 2–3 hours |
| Interpretation & write-up | Medium | 2–3 hours |
| **Total** | | **~7–12 hours** |

The computational cost is trivial — all three models train in seconds/minutes on ~4,700 rows. The real work is in **feature engineering** and **thoughtful evaluation design**.

---

## Step-by-Step Implementation Plan

### Step 1: Define Your Target Variable

From your CSV, you have `populism_score` (0–100 scale). This is your dependent variable. The distribution is heavily zero-inflated and right-skewed, which is why a Hurdle model makes sense alongside Ridge and XGBoost.

### Step 2: Engineer Covariates

This is the critical design decision. You need features that **should theoretically relate to populism** but that were **not directly used by the classifier**. Otherwise you're just testing circular logic.

From your existing CSV columns, you can construct:

**A. Text-Based Features (non-overlapping with DeBERTa classification)**
```python
# These capture textual properties the classifier didn't directly optimize for
covariates = {
    'article_length': len(text.split()),
    'avg_sentence_length': ...,
    'exclamation_density': text.count('!') / len(text.split()),
    'question_density': text.count('?') / len(text.split()),
    'first_person_plural_rate': count_we_our_us / total_words,  # "we/our/us" — classic populist rhetoric
    'negation_density': ...,  # words like "never", "no", "not"
    'uppercase_word_ratio': ...,  # shouting/emphasis
}
```

**B. Other Sentiment Dimensions (from your multi-label classification)**
Your CSV already has scores for other sentiments. These are gold — they were produced by the *same* classifier architecture but for *different* constructs:
```python
# From your CSV columns - these are covariates, NOT the target
sentiment_covariates = [
    'anger_score', 'fear_score', 'disgust_score',  # negative sentiments
    'hope_score', 'pride_score',                      # positive sentiments
    'nationalism_score', 'anti_elite_score',          # related constructs
    # etc. — whatever other dimensions you classified
]
```

**C. Metadata Features**
```python
metadata_covariates = [
    'source',           # publication (one-hot encoded)
    'year', 'month',    # temporal
    'country',          # if available
]
```

**The key principle:** Your covariates should be things that *theoretically* relate to populism (e.g., anger, anti-elite sentiment, first-person plural pronouns) but were *measured independently* from the populism score itself.

### Step 3: The Three Models

Here's the full implementation:

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# LOAD AND PREPARE DATA
# ============================================================
df = pd.read_csv('populism_2018_filtered.csv')

# Define target
target = 'populism_score'  # adjust to your actual column name

# Define covariates - ADJUST THESE to your actual columns
# Continuous features (other sentiment scores + text features)
continuous_features = [
    # Other classifier-derived sentiment scores
    'anger_score', 'fear_score', 'disgust_score', 
    'hope_score', 'pride_score', 'nationalism_score',
    # Add text-based features you engineer
    # 'article_length', 'exclamation_density', etc.
]

# Categorical features
categorical_features = [
    'source',  # publication name
    # 'country', etc.
]

# Filter to only columns that exist in your data
continuous_features = [c for c in continuous_features if c in df.columns]
categorical_features = [c for c in categorical_features if c in df.columns]

# Drop rows with missing values in features or target
all_features = continuous_features + categorical_features
df_model = df[all_features + [target]].dropna()

X = df_model[all_features]
y = df_model[target]

print(f"Sample size: {len(df_model)}")
print(f"Target distribution:")
print(y.describe())
print(f"\nScore range counts:")
for label, cond in [('0', y==0), ('1-9', (y>=1)&(y<=9)), 
                      ('10-24', (y>=10)&(y<=24)), ('25-49', (y>=25)&(y<=49)), 
                      ('50+', y>=50)]:
    print(f"  {label}: {cond.sum()}")

# ============================================================
# PREPROCESSING PIPELINE
# ============================================================
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), continuous_features),
        ('cat', OneHotEncoder(drop='first', sparse_output=False, 
                              handle_unknown='ignore'), categorical_features)
    ] if categorical_features else [
        ('num', StandardScaler(), continuous_features)
    ]
)

# ============================================================
# MODEL 1: RIDGE REGRESSION
# ============================================================
ridge_pipe = Pipeline([
    ('preprocessor', preprocessor),
    ('model', Ridge(alpha=1.0))
])

# ============================================================
# MODEL 2: XGBOOST
# ============================================================
xgb_pipe = Pipeline([
    ('preprocessor', preprocessor),
    ('model', XGBRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=0
    ))
])

# ============================================================
# MODEL 3: TWO-STAGE HURDLE MODEL
# ============================================================
from sklearn.linear_model import LogisticRegression

class HurdleModel:
    """
    Stage 1: Logistic regression predicts P(score > 0)
    Stage 2: Ridge regression predicts score | score > 0
    Final prediction: P(score > 0) * E[score | score > 0]
    """
    def __init__(self, preprocessor):
        self.preprocessor = preprocessor
        self.stage1 = LogisticRegression(max_iter=1000, C=1.0)
        self.stage2 = Ridge(alpha=1.0)
        self.preprocessor_fitted = False
        
    def fit(self, X, y):
        X_transformed = self.preprocessor.fit_transform(X)
        self.preprocessor_fitted = True
        
        # Stage 1: binary (is score > 0?)
        y_binary = (y > 0).astype(int)
        self.stage1.fit(X_transformed, y_binary)
        
        # Stage 2: continuous (what score, given > 0?)
        mask_positive = y > 0
        if mask_positive.sum() > 10:
            self.stage2.fit(X_transformed[mask_positive], y[mask_positive])
        
        return self
    
    def predict(self, X):
        X_transformed = self.preprocessor.transform(X)
        
        # P(score > 0)
        p_positive = self.stage1.predict_proba(X_transformed)[:, 1]
        
        # E[score | score > 0]
        conditional_score = self.stage2.predict(X_transformed)
        conditional_score = np.clip(conditional_score, 0, 100)
        
        # Combined prediction
        return p_positive * conditional_score

# ============================================================
# CROSS-VALIDATED PREDICTIONS
# ============================================================
kf = KFold(n_splits=5, shuffle=True, random_state=42)

results = {}

# Ridge
print("\n" + "="*60)
print("RIDGE REGRESSION")
print("="*60)
ridge_preds = cross_val_predict(ridge_pipe, X, y, cv=kf)
ridge_preds = np.clip(ridge_preds, 0, 100)
results['Ridge'] = ridge_preds

# XGBoost
print("\nXGBOOST")
print("="*60)
xgb_preds = cross_val_predict(xgb_pipe, X, y, cv=kf)
xgb_preds = np.clip(xgb_preds, 0, 100)
results['XGBoost'] = xgb_preds

# Hurdle (manual cross-validation since it's a custom class)
print("\nHURDLE MODEL")
print("="*60)
hurdle_preds = np.zeros(len(y))
for train_idx, test_idx in kf.split(X):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train = y.iloc[train_idx].values
    
    hurdle = HurdleModel(preprocessor=ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), continuous_features),
            ('cat', OneHotEncoder(drop='first', sparse_output=False,
                                  handle_unknown='ignore'), categorical_features)
        ] if categorical_features else [
            ('num', StandardScaler(), continuous_features)
        ]
    ))
    hurdle.fit(X_train, y_train)
    hurdle_preds[test_idx] = hurdle.predict(X_test)

hurdle_preds = np.clip(hurdle_preds, 0, 100)
results['Hurdle'] = hurdle_preds

# ============================================================
# OVERALL EVALUATION
# ============================================================
print("\n" + "="*60)
print("OVERALL PERFORMANCE")
print("="*60)
print(f"{'Model':<15} {'MSE':>10} {'RMSE':>10} {'MAE':>10} {'R²':>10}")
print("-" * 55)
for name, preds in results.items():
    mse = mean_squared_error(y, preds)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y, preds)
    r2 = r2_score(y, preds)
    print(f"{name:<15} {mse:>10.2f} {rmse:>10.2f} {mae:>10.2f} {r2:>10.4f}")

# ============================================================
# STRATIFIED EVALUATION BY SCORE RANGE
# ============================================================
print("\n" + "="*60)
print("STRATIFIED EVALUATION BY SCORE RANGE")
print("="*60)

bins = [
    ('0', y == 0),
    ('1-9', (y >= 1) & (y <= 9)),
    ('10-24', (y >= 10) & (y <= 24)),
    ('25-49', (y >= 25) & (y <= 49)),
    ('50+', y >= 50)
]

for name, preds in results.items():
    print(f"\n--- {name} ---")
    print(f"{'Range':<10} {'N':>6} {'Actual Mean':>12} {'Pred Mean':>12} "
          f"{'Actual SD':>10} {'Pred SD':>10} {'MAE':>8} {'Corr':>8}")
    print("-" * 78)
    
    for label, mask in bins:
        n = mask.sum()
        if n < 5:
            print(f"{label:<10} {n:>6}  (too few observations)")
            continue
        
        actual = y[mask]
        predicted = preds[mask]
        
        actual_mean = actual.mean()
        pred_mean = predicted.mean()
        actual_sd = actual.std()
        pred_sd = predicted.std()
        mae = mean_absolute_error(actual, predicted)
        
        if actual.std() > 0 and predicted.std() > 0:
            corr = np.corrcoef(actual, predicted)[0, 1]
        else:
            corr = np.nan
        
        print(f"{label:<10} {n:>6} {actual_mean:>12.2f} {pred_mean:>12.2f} "
              f"{actual_sd:>10.2f} {pred_sd:>10.2f} {mae:>8.2f} {corr:>8.3f}")

# ============================================================
# VISUALIZATION
# ============================================================
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

fig = plt.figure(figsize=(18, 14))
gs = gridspec.GridSpec(2, 3, hspace=0.35, wspace=0.3)

# Row 1: Actual vs Predicted scatter for each model
for i, (name, preds) in enumerate(results.items()):
    ax = fig.add_subplot(gs[0, i])
    
    # Hexbin for density
    hb = ax.hexbin(y, preds, gridsize=30, cmap='YlOrRd', mincnt=1)
    ax.plot([0, 100], [0, 100], 'k--', alpha=0.5, linewidth=1.5, label='Perfect')
    
    r2 = r2_score(y, preds)
    rmse = np.sqrt(mean_squared_error(y, preds))
    
    ax.set_xlabel('Actual Populism Score', fontsize=11)
    ax.set_ylabel('Predicted Populism Score', fontsize=11)
    ax.set_title(f'{name}\nR² = {r2:.4f}, RMSE = {rmse:.2f}', fontsize=12)
    ax.set_xlim(-5, 105)
    ax.set_ylim(-5, 105)
    ax.legend(fontsize=9)
    plt.colorbar(hb, ax=ax, label='Count')

# Row 2: Stratified mean comparison
ax_bar = fig.add_subplot(gs[1, 0:2])
bin_labels = [label for label, _ in bins]
x_pos = np.arange(len(bin_labels))
width = 0.2

# Actual means
actual_means = []
for label, mask in bins:
    actual_means.append(y[mask].mean() if mask.sum() > 0 else 0)

bars_actual = ax_bar.bar(x_pos - 1.5*width, actual_means, width, 
                          label='Actual', color='#2c3e50', alpha=0.8)

colors = ['#e74c3c', '#3498db', '#2ecc71']
for j, (name, preds) in enumerate(results.items()):
    pred_means = []
    for label, mask in bins:
        pred_means.append(preds[mask].mean() if mask.sum() > 0 else 0)
    ax_bar.bar(x_pos + (j-0.5)*width, pred_means, width, 
               label=f'{name}', color=colors[j], alpha=0.8)

ax_bar.set_xlabel('Score Range', fontsize=11)
ax_bar.set_ylabel('Mean Score', fontsize=11)
ax_bar.set_title('Actual vs. Predicted Mean by Score Range', fontsize=12)
ax_bar.set_xticks(x_pos)
ax_bar.set_xticklabels(bin_labels)
ax_bar.legend(fontsize=10)
ax_bar.grid(axis='y', alpha=0.3)

# Distribution of residuals
ax_resid = fig.add_subplot(gs[1, 2])
for name, preds in results.items():
    residuals = y - preds
    ax_resid.hist(residuals, bins=50, alpha=0.4, label=name, density=True)
ax_resid.set_xlabel('Residual (Actual - Predicted)', fontsize=11)
ax_resid.set_ylabel('Density', fontsize=11)
ax_resid.set_title('Residual Distribution', fontsize=12)
ax_resid.legend(fontsize=10)
ax_resid.axvline(x=0, color='black', linestyle='--', alpha=0.5)

plt.savefig('predictive_validation.png', dpi=150, bbox_inches='tight')
plt.show()
print("\nFigure saved as 'predictive_validation.png'")

# ============================================================
# FEATURE IMPORTANCE (from XGBoost)
# ============================================================
print("\n" + "="*60)
print("FEATURE IMPORTANCE (XGBoost)")
print("="*60)

# Fit on full data to extract importances
xgb_pipe.fit(X, y)
xgb_model = xgb_pipe.named_steps['model']

# Get feature names after preprocessing
feature_names = preprocessor.fit_transform(X)
if hasattr(preprocessor, 'get_feature_names_out'):
    feature_names = preprocessor.get_feature_names_out()
else:
    feature_names = [f'feature_{i}' for i in range(xgb_model.feature_importances_.shape[0])]

importances = pd.Series(xgb_model.feature_importances_, index=feature_names)
importances = importances.sort_values(ascending=False)

print("\nTop 15 features:")
for feat, imp in importances.head(15).items():
    print(f"  {feat:<40} {imp:.4f}")

# Feature importance plot
fig, ax = plt.subplots(figsize=(10, 6))
importances.head(15).plot(kind='barh', ax=ax, color='#3498db')
ax.set_xlabel('Importance (Gain)')
ax.set_title('XGBoost Feature Importance for Populism Score Prediction')
ax.invert_yaxis()
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=150, bbox_inches='tight')
plt.show()
```

---

## Step 4: Additional Text Feature Engineering

If you want stronger results, add these engineered features from the raw text:

```python
import re
from collections import Counter

def engineer_text_features(text):
    """Extract features from raw article text that are theoretically 
    related to populist rhetoric but independent of DeBERTa classification."""
    
    words = text.lower().split()
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    features = {}
    
    # Length features
    features['word_count'] = len(words)
    features['avg_sentence_length'] = np.mean([len(s.split()) for s in sentences]) if sentences else 0
    features['sentence_count'] = len(sentences)
    
    # Rhetorical intensity
    features['exclamation_rate'] = text.count('!') / max(len(words), 1)
    features['question_rate'] = text.count('?') / max(len(words), 1)
    features['allcaps_rate'] = sum(1 for w in text.split() if w.isupper() and len(w) > 1) / max(len(words), 1)
    
    # Populist pronoun patterns
    we_words = {'we', 'us', 'our', 'ours', 'ourselves'}
    they_words = {'they', 'them', 'their', 'theirs', 'themselves'}
    features['we_rate'] = sum(1 for w in words if w in we_words) / max(len(words), 1)
    features['they_rate'] = sum(1 for w in words if w in they_words) / max(len(words), 1)
    features['we_they_ratio'] = features['we_rate'] / max(features['they_rate'], 0.0001)
    
    # Negativity markers
    negations = {'not', 'no', 'never', 'neither', 'nobody', 'nothing', 'nowhere', 
                 'nor', "n't", "don't", "won't", "can't", "shouldn't", "wouldn't"}
    features['negation_rate'] = sum(1 for w in words if w in negations) / max(len(words), 1)
    
    # Simplicity (populist rhetoric tends to be simpler)
    features['avg_word_length'] = np.mean([len(w) for w in words]) if words else 0
    features['type_token_ratio'] = len(set(words)) / max(len(words), 1)
    
    # Emotional/absolute language
    absolutes = {'always', 'never', 'everyone', 'nobody', 'everything', 'nothing',
                 'all', 'none', 'every', 'completely', 'totally', 'absolutely'}
    features['absolute_rate'] = sum(1 for w in words if w in absolutes) / max(len(words), 1)
    
    return features

# Apply to your dataframe
if 'text' in df.columns:  # or whatever your text column is called
    text_features = df['text'].apply(engineer_text_features).apply(pd.Series)
    df_enriched = pd.concat([df, text_features], axis=1)
    
    # Add these to your continuous_features list
    new_text_features = list(text_features.columns)
    continuous_features.extend(new_text_features)
```

---

## Interpretation Framework

When you write this up, here's the logic to present:

### What Counts as Validation?

| Evidence | What it means |
|----------|--------------|
| Overall R² > 0 (ideally > 0.15) | Scores contain systematic, predictable signal |
| Predicted means track actual means across score ranges | Signal is consistent across the score distribution |
| XGBoost > Ridge | Non-linear relationships exist (scores capture complex patterns) |
| Hurdle model performs well on zeros vs. non-zeros | The classifier's zero/non-zero distinction is meaningful |
| Feature importances are theoretically coherent | Related constructs (anger, anti-elite) predict populism, as expected |
| High-populism articles have more "we/they" language, simpler words | Linguistic correlates align with theory |

### What Would *Invalidate* the Scores?

- R² near zero across all models
- Predicted means flat across score ranges (models can't distinguish high from low)
- Feature importances dominated by irrelevant metadata (e.g., article length alone)
- No theoretical coherence in which features matter

---

## Summary

This is a **clean, well-motivated validation exercise**. The computational cost is negligible — you're running three standard models on ~4,700 rows. The intellectual work is in (1) choosing covariates that avoid circularity, (2) interpreting the stratified results thoughtfully, and (3) being honest about what this does and doesn't prove.

What it **does** prove: the scores are not random noise; they capture structured signal that relates to observable text and article properties in theoretically expected ways.

What it **doesn't** prove: that the scores are perfectly calibrated or that they match human judgment on every article. For that, you'd need human annotation validation (a separate exercise).

User: what kind of human annotation validation can be done more systematically

GitHub Copilot: 

# Human Annotation Validation for Your DeBERTa Classifier

## The Core Question

You need to answer: **"Do the DeBERTa-derived scores agree with how a human would rate these articles?"**

This is the gold standard for classifier validation. Here's how to do it systematically.

---

## Design: Stratified Annotation Protocol

### Step 1: Sample Selection

You need to annotate articles from across the **full score distribution**, not a random sample (which would over-represent the modal category). Sample ~200–300 articles:

```python
import pandas as pd
import numpy as np

df = pd.read_csv('populism_2018_filtered.csv')

# Stratified sample across score distribution
bins = [
    ('strong_negative', df['populism_composite_avg'] <= -0.6),
    ('moderate_negative', (df['populism_composite_avg'] > -0.6) & (df['populism_composite_avg'] <= -0.3)),
    ('weak_negative', (df['populism_composite_avg'] > -0.3) & (df['populism_composite_avg'] <= -0.1)),
    ('neutral', (df['populism_composite_avg'] > -0.1) & (df['populism_composite_avg'] <= 0.1)),
    ('weak_positive', (df['populism_composite_avg'] > 0.1) & (df['populism_composite_avg'] <= 0.3)),
    ('moderate_positive', (df['populism_composite_avg'] > 0.3) & (df['populism_composite_avg'] <= 0.6)),
    ('strong_positive', df['populism_composite_avg'] > 0.6),
]

np.random.seed(42)
sample_dfs = []

for label, mask in bins:
    pool = df[mask]
    n_sample = min(40, len(pool))  # 40 per bin, or all if fewer
    sampled = pool.sample(n=n_sample, random_state=42)
    sampled['score_bin'] = label
    sample_dfs.append(sampled)
    print(f"{label}: sampled {n_sample} from {len(pool)} available")

annotation_sample = pd.concat(sample_dfs, ignore_index=True)

# Shuffle so annotator doesn't see them in score order
annotation_sample = annotation_sample.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"\nTotal articles to annotate: {len(annotation_sample)}")
```

### Step 2: What Annotators Rate

Each annotator reads the article text and rates **each dimension independently** on a 5-point scale. This mirrors your classifier's structure:

```
For each article, rate the following:

1. ANTI-ESTABLISHMENT SENTIMENT
   -2: Strongly defends institutions/elites
   -1: Mildly defends institutions/elites
    0: Neutral / no clear stance
   +1: Mildly criticizes elites/establishment
   +2: Strongly attacks elites/establishment

2. ECONOMIC NATIONALISM
   -2: Strongly pro-free-trade / pro-globalization
   -1: Mildly pro-free-trade
    0: Neutral / no clear stance
   +1: Mildly protectionist / nationalist
   +2: Strongly protectionist / nationalist

3. PEOPLE-CENTRISM
   -2: Strongly technocratic / elite-focused
   -1: Mildly technocratic
    0: Neutral / no clear stance
   +1: Mildly appeals to "the people" / common sense
   +2: Strongly appeals to "the people" vs. elites

4. OVERALL POPULISM (holistic judgment)
   -2: Strongly anti-populist
   -1: Mildly anti-populist
    0: Neutral
   +1: Mildly populist
   +2: Strongly populist
```

### Step 3: Generate Annotation Spreadsheet

```python
# Create annotation file - HIDE the classifier scores from annotators
annotation_export = annotation_sample[['article_id', 'nuts_code', 'date']].copy()

# You need the article text - load from database
import sqlite3

conn = sqlite3.connect('path/to/your/database.db')
texts = pd.read_sql(
    "SELECT id as article_id, text FROM Articles WHERE id IN ({})".format(
        ','.join(f"'{aid}'" for aid in annotation_sample['article_id'])
    ), conn
)
conn.close()

annotation_export = annotation_export.merge(texts, on='article_id', how='left')

# Add empty columns for human ratings
annotation_export['human_anti_establishment'] = ''
annotation_export['human_economic_nationalism'] = ''
annotation_export['human_people_centrism'] = ''
annotation_export['human_overall_populism'] = ''
annotation_export['annotator_notes'] = ''
annotation_export['annotator_id'] = ''

# Add a unique annotation ID (randomized, so annotator can't infer score order)
annotation_export['annotation_id'] = range(1, len(annotation_export) + 1)

# Truncate very long texts for readability (keep first 2000 chars)
annotation_export['text_preview'] = annotation_export['text'].str[:2000]

# Export - DO NOT include classifier scores
annotation_export[['annotation_id', 'article_id', 'date', 'text_preview',
                    'human_anti_establishment', 'human_economic_nationalism',
                    'human_people_centrism', 'human_overall_populism',
                    'annotator_notes', 'annotator_id']].to_csv(
    'annotation_task.csv', index=False
)

# Save the key separately (maps annotation_id to classifier scores)
annotation_key = annotation_sample[['article_id', 'score_bin',
                                     'anti_establishment_score',
                                     'economic_nationalism_score',
                                     'people_centrism_score',
                                     'populism_composite_avg']].copy()
annotation_key.to_csv('annotation_key_DO_NOT_SHARE.csv', index=False)

print(f"Exported {len(annotation_export)} articles for annotation")
print("annotation_task.csv -> give to annotators")
print("annotation_key_DO_NOT_SHARE.csv -> keep for validation analysis")
```

---

## Step 4: Inter-Annotator Reliability (If Using Multiple Annotators)

You need **at least 2 annotators** rating the same articles. Ideally 50–100 articles are double-coded.

```python
# After collecting annotations from 2+ annotators
annotator1 = pd.read_csv('annotations_annotator1.csv')
annotator2 = pd.read_csv('annotations_annotator2.csv')

# Merge on annotation_id
overlap = annotator1.merge(annotator2, on='annotation_id', suffixes=('_a1', '_a2'))

dimensions = ['anti_establishment', 'economic_nationalism', 'people_centrism', 'overall_populism']

from sklearn.metrics import cohen_kappa_score
from scipy.stats import pearsonr, spearmanr

print("=" * 60)
print("INTER-ANNOTATOR RELIABILITY")
print("=" * 60)

for dim in dimensions:
    col_a1 = f'human_{dim}_a1'
    col_a2 = f'human_{dim}_a2'
    
    # Drop missing
    valid = overlap[[col_a1, col_a2]].dropna()
    
    if len(valid) < 10:
        print(f"\n{dim}: Too few overlapping annotations ({len(valid)})")
        continue
    
    # Cohen's Kappa (ordinal agreement)
    kappa = cohen_kappa_score(valid[col_a1], valid[col_a2], weights='quadratic')
    
    # Pearson correlation (treats as continuous)
    r, p = pearsonr(valid[col_a1], valid[col_a2])
    
    # Spearman rank correlation
    rho, p_rho = spearmanr(valid[col_a1], valid[col_a2])
    
    # Exact agreement rate
    exact = (valid[col_a1] == valid[col_a2]).mean()
    
    # Within-1 agreement (off by at most 1 point)
    within1 = (abs(valid[col_a1] - valid[col_a2]) <= 1).mean()
    
    print(f"\n{dim}:")
    print(f"  Quadratic Cohen's Kappa: {kappa:.3f}")
    print(f"  Pearson r:               {r:.3f} (p={p:.4f})")
    print(f"  Spearman rho:            {rho:.3f} (p={p_rho:.4f})")
    print(f"  Exact agreement:         {exact:.1%}")
    print(f"  Within-1 agreement:      {within1:.1%}")
    print(f"  N overlapping:           {len(valid)}")
```

**Interpretation thresholds for Kappa:**
| Kappa | Interpretation |
|-------|---------------|
| < 0.20 | Poor — dimension is too subjective |
| 0.20–0.40 | Fair — acceptable for exploratory work |
| 0.40–0.60 | Moderate — standard for social science |
| 0.60–0.80 | Good — strong agreement |
| > 0.80 | Excellent |

---

## Step 5: Classifier vs. Human Agreement

This is the key validation:

```python
# Load completed annotations and merge with classifier scores
annotations = pd.read_csv('annotations_completed.csv')  # averaged across annotators
key = pd.read_csv('annotation_key_DO_NOT_SHARE.csv')

validated = annotations.merge(key, on='article_id')

# Map classifier scores (-1 to +1) to same -2 to +2 scale as human ratings
# Linear rescaling: classifier -1 -> human -2, classifier +1 -> human +2
validated['classifier_anti_establishment_scaled'] = validated['anti_establishment_score'] * 2
validated['classifier_economic_nationalism_scaled'] = validated['economic_nationalism_score'] * 2
validated['classifier_people_centrism_scaled'] = validated['people_centrism_score'] * 2
validated['classifier_composite_scaled'] = validated['populism_composite_avg'] * 2

print("=" * 60)
print("CLASSIFIER vs. HUMAN AGREEMENT")
print("=" * 60)

dimension_pairs = [
    ('human_anti_establishment', 'classifier_anti_establishment_scaled', 'Anti-Establishment'),
    ('human_economic_nationalism', 'classifier_economic_nationalism_scaled', 'Economic Nationalism'),
    ('human_people_centrism', 'classifier_people_centrism_scaled', 'People-Centrism'),
    ('human_overall_populism', 'classifier_composite_scaled', 'Overall Populism'),
]

from sklearn.metrics import mean_absolute_error

results_table = []

for human_col, classifier_col, label in dimension_pairs:
    valid = validated[[human_col, classifier_col]].dropna()
    
    r, p = pearsonr(valid[human_col], valid[classifier_col])
    rho, p_rho = spearmanr(valid[human_col], valid[classifier_col])
    mae = mean_absolute_error(valid[human_col], valid[classifier_col])
    
    # Direction agreement: do they agree on sign (positive vs negative)?
    human_sign = np.sign(valid[human_col])
    classifier_sign = np.sign(valid[classifier_col])
    direction_agree = (human_sign == classifier_sign).mean()
    
    # Exclude neutrals for direction agreement
    non_neutral = (valid[human_col] != 0) & (valid[classifier_col] != 0)
    if non_neutral.sum() > 10:
        direction_agree_nonneutral = (human_sign[non_neutral] == classifier_sign[non_neutral]).mean()
    else:
        direction_agree_nonneutral = np.nan
    
    results_table.append({
        'Dimension': label,
        'Pearson_r': r,
        'Pearson_p': p,
        'Spearman_rho': rho,
        'MAE': mae,
        'Direction_Agreement': direction_agree,
        'Direction_Agreement_NonNeutral': direction_agree_nonneutral,
        'N': len(valid)
    })
    
    print(f"\n{label}:")
    print(f"  Pearson r:         {r:.3f} (p={p:.4f})")
    print(f"  Spearman rho:      {rho:.3f}")
    print(f"  MAE:               {mae:.2f}")
    print(f"  Direction agree:   {direction_agree:.1%} (all) / {direction_agree_nonneutral:.1%} (non-neutral)")

results_table_df = pd.DataFrame(results_table)
results_table_df.to_csv('classifier_human_validation.csv', index=False)
print("\nSaved: classifier_human_validation.csv")
```

---

## Step 6: Confusion Matrix by Score Bin

```python
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Bin both human and classifier into categories
def bin_score(score, is_human=True):
    """Bin scores into categories."""
    if is_human:
        # Already on -2 to +2 scale
        if score <= -1.5: return 'Strong Anti'
        elif score <= -0.5: return 'Mild Anti'
        elif score <= 0.5: return 'Neutral'
        elif score <= 1.5: return 'Mild Pro'
        else: return 'Strong Pro'
    else:
        # Classifier on -1 to +1 scale
        if score <= -0.5: return 'Strong Anti'
        elif score <= -0.15: return 'Mild Anti'
        elif score <= 0.15: return 'Neutral'
        elif score <= 0.5: return 'Mild Pro'
        else: return 'Strong Pro'

categories = ['Strong Anti', 'Mild Anti', 'Neutral', 'Mild Pro', 'Strong Pro']

fig, axes = plt.subplots(2, 2, figsize=(16, 14))

for ax, (human_col, classifier_col_raw, label) in zip(axes.flat, [
    ('human_anti_establishment', 'anti_establishment_score', 'Anti-Establishment'),
    ('human_economic_nationalism', 'economic_nationalism_score', 'Economic Nationalism'),
    ('human_people_centrism', 'people_centrism_score', 'People-Centrism'),
    ('human_overall_populism', 'populism_composite_avg', 'Overall Populism'),
]):
    valid = validated[[human_col, classifier_col_raw]].dropna()
    
    human_binned = valid[human_col].apply(lambda x: bin_score(x, is_human=True))
    classifier_binned = valid[classifier_col_raw].apply(lambda x: bin_score(x, is_human=False))
    
    cm = confusion_matrix(human_binned, classifier_binned, labels=categories)
    
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=categories)
    disp.plot(ax=ax, cmap='Blues', values_format='d')
    ax.set_title(f'{label}', fontsize=13)
    ax.set_xlabel('Classifier', fontsize=11)
    ax.set_ylabel('Human', fontsize=11)
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

plt.suptitle('Classifier vs. Human: Confusion Matrices', fontsize=15, y=1.02)
plt.tight_layout()
plt.savefig('validation_confusion_matrices.png', dpi=150, bbox_inches='tight')
plt.show()
```

---

## Practical Recommendations

| Aspect | Recommendation |
|--------|---------------|
| 
