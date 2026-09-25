from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "zepto_books.db"
QUERY_LOG_PATH = BASE_DIR / "queries_output.txt"
JOIN_COMPARISON_PATH = BASE_DIR / "join_comparison.txt"

URLS = {
    "Mystery": "https://books.toscrape.com/catalogue/category/books/mystery_3/index.html",
    "Historical Fiction": "https://books.toscrape.com/catalogue/category/books/historical-fiction_4/index.html",
    "Classics": "https://books.toscrape.com/catalogue/category/books/classics_6/index.html",
    "Sequential Art": "https://books.toscrape.com/catalogue/category/books/sequential-art_5/index.html",
}

RATING_MAP = {"One": 1, "Two": 2, "Three": 3, "Four": 4, "Five": 5}


def fetch_page(url: str) -> BeautifulSoup:
    response = requests.get(
        url,
        headers={"User-Agent": "Mozilla/5.0 (compatible; CapstoneDataPipeline/1.0)"},
        timeout=30,
    )
    response.raise_for_status()
    return BeautifulSoup(response.text, "html.parser")


def extract_books() -> list[dict]:
    rows: list[dict] = []
    for category, url in URLS.items():
        soup = fetch_page(url)
        for article in soup.select("article.product_pod"):
            title_tag = article.select_one("h3 a")
            price_tag = article.select_one("p.price_color")
            rating_tag = article.select_one("p.star-rating")
            availability_tag = article.select_one("p.instock.availability")
            if any(tag is None for tag in (title_tag, price_tag, rating_tag, availability_tag)):
                continue
            rows.append(
                {
                    "title": title_tag.get("title") or title_tag.get_text(strip=True),
                    "price_gbp_raw": price_tag.get_text(strip=True),
                    "star_rating": next(
                        (value for value in rating_tag.get("class", []) if value in RATING_MAP),
                        "",
                    ),
                    "availability": availability_tag.get_text(" ", strip=True),
                    "category": category,
                }
            )
    return rows


def clean_books(raw_rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(raw_rows)
    if df.empty:
        raise ValueError("No books were scraped from the target categories.")

    df = df.dropna(subset=["title", "price_gbp_raw", "star_rating", "availability", "category"]).copy()
    df["price_gbp"] = pd.to_numeric(
        df["price_gbp_raw"].str.replace(r"[^0-9.]", "", regex=True),
        errors="coerce",
    )
    df["rating"] = df["star_rating"].map(RATING_MAP)
    df["in_stock"] = df["availability"].str.lower().str.contains("in stock").astype(bool)
    df = df.dropna(subset=["title", "price_gbp", "rating", "in_stock", "category"]).copy()
    df["price_inr"] = df["price_gbp"] * 105.50
    df = df[["title", "price_gbp", "price_inr", "rating", "in_stock", "category"]].reset_index(drop=True)
    return df


def create_schema(conn: sqlite3.Connection) -> None:
    conn.execute("DROP TABLE IF EXISTS books")
    conn.execute("DROP TABLE IF EXISTS categories")
    conn.execute("CREATE TABLE categories (category_id INTEGER PRIMARY KEY, category_name TEXT UNIQUE)")
    conn.execute(
        """
        CREATE TABLE books (
            book_id INTEGER PRIMARY KEY,
            title TEXT,
            price_gbp REAL,
            price_inr REAL,
            rating INTEGER,
            in_stock INTEGER,
            category_id INTEGER,
            FOREIGN KEY (category_id) REFERENCES categories(category_id)
        )
        """
    )


def load_to_db(df: pd.DataFrame, db_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    conn = sqlite3.connect(db_path)
    create_schema(conn)

    categories_df = pd.DataFrame({"category_name": sorted(df["category"].unique())})
    categories_df.insert(0, "category_id", range(1, len(categories_df) + 1))
    categories_df.to_sql("categories", conn, if_exists="append", index=False)

    category_lookup = pd.read_sql("SELECT * FROM categories", conn)
    books_df = df.merge(category_lookup, left_on="category", right_on="category_name", how="left")
    books_df = books_df[["title", "price_gbp", "price_inr", "rating", "in_stock", "category_id"]].copy()
    books_df.to_sql("books", conn, if_exists="append", index=False)
    conn.commit()
    conn.close()
    return categories_df, books_df


def run_queries(db_path: Path) -> list[tuple[str, pd.DataFrame]]:
    query_sql = [
        ("SELECT/WHERE", "SELECT title, price_inr FROM books WHERE price_inr > 2000 ORDER BY price_inr DESC LIMIT 5;"),
        ("ORDER BY", "SELECT title, rating FROM books ORDER BY rating DESC, title ASC LIMIT 10;"),
        ("LIMIT", "SELECT title, price_gbp FROM books ORDER BY price_gbp DESC LIMIT 5;"),
        ("DISTINCT", "SELECT DISTINCT category_id FROM books ORDER BY category_id;"),
        ("BETWEEN", "SELECT title, price_inr FROM books WHERE price_inr BETWEEN 1000 AND 2000 ORDER BY price_inr DESC LIMIT 10;"),
        ("JOIN", "SELECT b.title, c.category_name, b.rating, b.price_inr FROM books b JOIN categories c ON b.category_id = c.category_id ORDER BY b.rating DESC, b.price_inr DESC LIMIT 10;"),
    ]
    conn = sqlite3.connect(db_path)
    results: list[tuple[str, pd.DataFrame]] = []
    for label, sql in query_sql:
        results.append((label, pd.read_sql(sql, conn)))
    conn.close()
    return results


def main() -> None:
    raw_books = extract_books()
    df = clean_books(raw_books)
    if len(df) < 60:
        raise ValueError(f"Expected at least 60 books; found {len(df)}.")

    if DB_PATH.exists():
        DB_PATH.unlink()

    categories_df, books_df = load_to_db(df, DB_PATH)
    print("Categories:\n", categories_df)
    print("\nSample books:\n", books_df.head())

    sql_results = run_queries(DB_PATH)
    with QUERY_LOG_PATH.open("w", encoding="utf-8") as fh:
        for label, result in sql_results:
            if result.empty:
                raise ValueError(f"Required SQL query returned no rows: {label}")
            fh.write(f"QUERY: {label}\n")
            fh.write(f"ROWS: {len(result)}\n")
            fh.write(result.to_string(index=False))
            fh.write("\n\n")
            print(f"\nQUERY: {label}\n{result.to_string(index=False)}")

    conn = sqlite3.connect(DB_PATH)
    join_sql = "SELECT b.title, c.category_name, b.rating, b.price_inr FROM books b JOIN categories c ON b.category_id = c.category_id ORDER BY b.rating DESC, b.price_inr DESC LIMIT 10;"
    sql_join = pd.read_sql(join_sql, conn)
    pandas_join = books_df.merge(categories_df, left_on="category_id", right_on="category_id", how="inner").sort_values(["rating", "price_inr"], ascending=[False, False]).head(10).rename(columns={"category_name": "category_name"})[["title", "category_name", "rating", "price_inr"]].reset_index(drop=True)
    conn.close()

    print("\nSQL join result:\n", sql_join.to_string(index=False))
    print("\nPandas merge result:\n", pandas_join.to_string(index=False))
    equivalent = sql_join.reset_index(drop=True).equals(pandas_join.reset_index(drop=True))
    with JOIN_COMPARISON_PATH.open("w", encoding="utf-8") as fh:
        fh.write("SQL JOIN RESULT\n")
        fh.write(sql_join.to_string(index=False))
        fh.write("\n\nPANDAS MERGE RESULT\n")
        fh.write(pandas_join.to_string(index=False))
        fh.write(f"\n\nEQUIVALENT: {equivalent}\n")
    print("Equivalent check:", equivalent)


if __name__ == "__main__":
    main()