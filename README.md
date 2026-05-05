# Product Recommendation Mini System

This project is a small but practical recommendation system built for portfolio and learning purposes.  
The goal was to build something that feels closer to a real ML product task than a notebook-only demo.

It generates a realistic synthetic retail-style dataset, preprocesses user and product data, combines multiple recommendation signals, and exposes everything through a simple CLI.

## Why I built this

I wanted a project that sits in the middle ground:
- not a toy example,
- not an over-engineered production clone,
- but still strong enough to show useful machine learning and software design skills.

Recommendation systems are a common problem in e-commerce and product-based companies, so this felt like a good project to include in a portfolio.

## What the system does

The recommender combines three types of signals:

1. **Popularity-based signal**  
   Products that receive more and stronger interactions get a higher popularity score.

2. **Content-based signal**  
   Products are represented using category, brand, tags, and a simple price bucket.  
   A user profile is built from their historical interactions, and cosine similarity is used to score products.

3. **Category preference signal**  
   The system also estimates which categories each user seems to prefer based on interaction history.

These three signals are merged into a final hybrid score.

## Project structure
```bash
product-recommendation-mini-system/
├── app/
│   ├── main.py
│   ├── config.py
│   ├── logger.py
│   ├── utils.py
│   ├── data/
│   ├── recommender/
│   └── evaluation/
├── data/
│   ├── raw/
│   └── processed/
├── artifacts/
│   └── models/
├── tests/
├── requirements.txt
└── README.md

## Installation

Create a virtual environment and install dependencies:

bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

## How to run the project

### 1) Generate synthetic data

bash
python -m app.main generate-data

You can also override defaults:

bash
python -m app.main generate-data --n-users 700 --n-products 400 --n-interactions 9000

### 2) Preprocess data

bash
python -m app.main preprocess

### 3) Build the service artifact

bash
python -m app.main build-model

### 4) Evaluate the recommender

bash
python -m app.main evaluate --k 10

### 5) Get recommendations for a user

bash
python -m app.main recommend --user-id U0001 --top-k 5

### 6) Inspect sample users and products

bash
python -m app.main list-users --limit 10
python -m app.main list-products --limit 10

## Example workflow

bash
python -m app.main generate-data
python -m app.main preprocess
python -m app.main build-model
python -m app.main evaluate --k 10
python -m app.main recommend --user-id U0007 --top-k 5

## Evaluation

The project includes basic offline ranking metrics:

- Precision@K
- Recall@K
- HitRate@K

The evaluation uses a temporal train/test split on interaction history, which is a better fit for recommendation tasks than a purely random split.

## Design notes

A few design decisions were intentional:

- **Synthetic data instead of an external dataset**  
  This keeps the project self-contained and easy to run.

- **Hybrid ranking instead of a single baseline**  
  A pure popularity recommender is too weak for a portfolio project, but a full deep learning recommender would be overkill here.

- **CLI instead of a web app**  
  I wanted the project to stay focused on the ML pipeline and recommendation logic.

- **Modular structure**  
  The code is split into data generation, preprocessing, recommendation, and evaluation modules to keep things maintainable.

## Limitations

This is still a mini-system, so there are a few obvious limitations:

- no real user feedback loop,
- no online serving layer,
- no A/B testing,
- no matrix factorization or deep retrieval/ranking models,
- no feature store or experiment tracking.

That said, I think it strikes a good balance between clarity, practicality, and portfolio value.

## Possible next steps

If I continue this project later, these are the upgrades I would consider:

- add matrix factorization for collaborative filtering,
- add reranking logic for diversity,
- introduce cold-start handling with better user metadata,
- package the service behind a FastAPI endpoint,
- add Docker support,
- add more tests and CI.

## Notes

The dataset is synthetic, but it was designed to mimic realistic recommendation patterns:
- users prefer certain categories,
- products have metadata like brand, price, and tags,
- interaction intensity differs across users,
- event types are weighted differently.

## License

This project is open for personal learning and portfolio use.


---

# Run

```bash
python -m app.main generate-data
python -m app.main preprocess
python -m app.main build-model
python -m app.main evaluate --k 10
python -m app.main recommend --user-id U0001 --top-k 5
