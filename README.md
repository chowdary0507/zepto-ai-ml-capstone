# Zepto AI/ML Capstone Project

A comprehensive three-module AI/ML pipeline consisting of a Web-Scraping Data Pipeline, a Titanic Predictive Analytics Engine, and a Local Customer Support RAG Assistant.

---

## 📁 Repository Structure
```text
.
├── analytics/
│   ├── 01_eda.py               # Exploratory Data Analysis & data cleaning
│   └── 02_modeling.py          # Machine learning classification and regression
├── data_pipeline/
│   └── pipeline.py             # Books scraping, price conversion, & SQLite pipeline
├── support_assistant/
│   ├── docs/                   # Internal corporate policy documents (doc_01 to doc_08)
│   └── main.py                 # FastAPI & ChromaDB vector database assistant
├── Dockerfile                  # Application deployment blueprint
├── requirements.txt            # Unified project dependency track
└── verify_project.py           # Submission validation execution script
```

---

## 🛠️ Installation & Setup

1. **Clone the Repository:**
   ```bash
   git clone https://github.com
   cd zepto-ai-ml-capstone
   ```

2. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## 🚀 Execution Instructions

### Module 1: Data Pipeline
Extracts book records across 4 categories, transforms pricing fields from GBP to INR, maps relational fields, and saves rows into an SQLite database:
```bash
python data_pipeline/pipeline.py
```
* **Output Artifact:** `data_pipeline/zepto_books.db`

### Module 2: Analytics Pipeline
Processes the Titanic dataset, handles missing data values, generates descriptive feature charts, trains classification (Survival) and regression (Fare) models, and outputs metrics:
```bash
python analytics/01_eda.py
python analytics/02_modeling.py
```
* **Output Artifacts:** Performance metrics saved to `analytics/model_comparison.csv` and exploratory distribution graphs.

### Module 3: Support Assistant (RAG)
Initializes local sentence embeddings, structures text chunk segments from corporate files, populates a persistent vector database, and launches an operational FastAPI microservice:
```bash
python support_assistant/main.py
```
* **API Server Endpoint:** Live at `http://127.0.0.1:8000` (or `http://0.0.0` based on terminal routing config).

---

## 📊 Verification & Assessment
To run the automated validation suite across all three operational systems to confirm submission readiness:
```bash
python verify_project.py
``` 
