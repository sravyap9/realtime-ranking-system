## Results

| Model | Precision@10 | Recall@10 | NDCG@10 |
|--------|--------:|--------:|--------:|
| Popularity Baseline | 0.0110 | 0.00247 | 0.01604 |
| Enhanced RF V1 | 0.0150 | 0.00327 | 0.01757 |

The enhanced ranking model improved Precision@10 by 36%, Recall@10 by 33%, and NDCG@10 by 10% compared to the popularity baseline.

a# Realtime Ranking System

A two-stage recommendation system built on the H&M Personalized Fashion Recommendations dataset.

The project demonstrates the core architecture used in modern recommendation systems:

1. Retrieval – generate candidate products
2. Ranking – score and rank candidates for each customer
3. Evaluation – measure recommendation quality using ranking metrics

---

## Dataset

Source: H&M Personalized Fashion Recommendations (Kaggle)

Files used:

- customers.csv
- articles.csv
- transactions_train.csv

Time range:

- 2018-09-20 → 2020-09-22
- ~1.36M purchasing customers
- Millions of transactions

---

## Project Structure

```text
realtime-ranking-system/

├── notebooks/
│   ├── 01_exploration.ipynb
│   ├── 02_retrieval.ipynb
│   ├── 03a_ranking_baseline.ipynb
│   ├── 03b_ranking_feature_experiments.ipynb
│   └── 04_evaluation.ipynb
│
├── src/
├── reports/
├── data/
└── README.md
```

---

## Key Findings

### Temporal Data

Customer purchases occur over time, so recommendation models should be evaluated using time-based splits rather than random splits to avoid data leakage.

### Long-Tail Customer Behavior

| Metric | Value |
|---------|---------:|
| Mean Purchases | 23.3 |
| Median Purchases | 9 |
| 25th Percentile | 3 |

Most customers have relatively little purchase history, creating a sparse recommendation problem.

### Cold-Start Challenge

Approximately 9.65% of customers purchased only once, limiting the effectiveness of collaborative purchase-history signals.

### Popularity Concentration

A small number of products account for a disproportionate share of purchases, making popularity a strong baseline but a weak personalization strategy.

---

## Retrieval

The retrieval stage reduces the search space before ranking.

Current implementation:

- TF-IDF product representations
- Nearest-neighbor candidate generation

Artifacts:

- tfidf_vectorizer.pkl
- retrieval_model.pkl

---

## Ranking

Baseline model:

- Random Forest Classifier

Feature engineering included:

### User Features

- Total purchases
- Unique products purchased
- Average spend
- Spend variability

### Product Features

- Total purchases
- Unique buyers
- Average price
- Price variability
- Recency signals

### User–Item Features

- Price affinity
- Popularity affinity
- Category preference matches
- Department preference matches

---

## Feature Importance

Top signals learned by the ranking model:

| Feature | Importance |
|----------|----------:|
| Price difference from article average | 0.241 |
| Article popularity percentile | 0.162 |
| Article unique buyers | 0.151 |
| Article total purchases | 0.125 |
| Relative popularity to user activity | 0.060 |

Key takeaway:

Behavioral and interaction features were substantially more predictive than simple product metadata.

---

## Model Performance

### Classification Metrics

| Metric | Score |
|----------|----------:|
| Train Accuracy | 0.919 |
| Test Accuracy | 0.911 |
| Train AUC | 0.969 |
| Test AUC | 0.960 |

The small train-test gap suggests limited overfitting.

---

## Recommendation Performance

| Experiment | Precision@10 | Recall@10 | NDCG@10 |
|------------|-------------:|----------:|---------:|
| Popularity Baseline | 0.0110 | 0.00247 | 0.01604 |
| Enhanced RF V1 | 0.0150 | 0.00327 | 0.01757 |

Improvements over baseline:

- +36% Precision@10
- +33% Recall@10
- +10% NDCG@10

---

## Lessons Learned

The largest gains came from:

- Better feature engineering
- Better behavioral understanding
- Better user–item interaction signals

rather than simply increasing model complexity.

This mirrors many real-world recommendation systems where feature quality often contributes more value than switching algorithms.

---

## Future Work

- Replace popularity candidate generation with personalized retrieval
- Add FAISS / embedding-based retrieval
- Experiment with LightGBM and XGBoost ranking models
- Add temporal and sequence-based features
- Evaluate using MAP@K and coverage metrics

---

## Tech Stack

- Python
- Pandas
- NumPy
- Scikit-Learn
- TF-IDF
- Nearest Neighbors
- Random Forest
- Jupyter Notebooks

---

## Author

**Sravya Patakota**

Applied AI/ML

Built as an end-to-end recommendation system project demonstrating retrieval, ranking, feature engineering, and recommendation evaluation workflows.