import os
import re
import sqlite3
import requests
import pandas as pd
from bs4 import BeautifulSoup

BASE_URL = "https://books.toscrape.com/catalogue/page-{}.html"
FIXED_GBP_TO_INR = 105.50
OUTPUT_DIR = "output"
DB_PATH = os.path.join(OUTPUT_DIR, "books.db")
CSV_PATH = os.path.join(OUTPUT_DIR, "clean_books.csv")
SQL_OUTPUT = os.path.join(OUTPUT_DIR, "sql_query_outputs.txt")

def scrape_books(pages=5):
    rows = []
    session = requests.Session()
    for page in range(1, pages + 1):
        r = session.get(BASE_URL.format(page), timeout=20)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        for article in soup.select("article.product_pod"):
            title = article.h3.a.get("title", "").strip()
            price = article.select_one(".price_color").get_text(strip=True)
            rating = next(
                (c for c in article.select_one(".star-rating").get("class", [])
                 if c != "star-rating"),
                ""
            )
            availability = article.select_one(".availability").get_text(" ", strip=True)
            rows.append({
                "title": title,
                "price": price,
                "star_rating": rating,
                "availability": availability,
                "category": "All products"
            })
    return pd.DataFrame(rows)

def clean_data(df):
    rating_map = {"One": 1, "Two": 2, "Three": 3, "Four": 4, "Five": 5}
    out = df.copy()
    out["price_gbp"] = pd.to_numeric(
        out["price"].astype(str).str.extract(r"([\d.]+)", expand=False),
        errors="coerce"
    )
    out["rating"] = out["star_rating"].map(rating_map)
    out["in_stock"] = out["availability"].str.contains(
        "In stock", case=False, na=False
    )
    out["price_gbp"] = out["price_gbp"].fillna(out["price_gbp"].median())
    out["rating"] = out["rating"].fillna(out["rating"].median()).round().astype(int)
    out = out.dropna(subset=["title", "category"])
    out["price_inr"] = (out["price_gbp"] * FIXED_GBP_TO_INR).round(2)
    return out[[
        "title", "price_gbp", "price_inr", "rating", "in_stock", "category"
    ]]

def create_database(df):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON")
    cur = conn.cursor()
    cur.executescript("""
        CREATE TABLE categories (
            category_id INTEGER PRIMARY KEY AUTOINCREMENT,
            category_name TEXT NOT NULL UNIQUE
        );
        CREATE TABLE books (
            book_id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            price_gbp REAL NOT NULL,
            price_inr REAL NOT NULL,
            rating INTEGER NOT NULL,
            in_stock INTEGER NOT NULL,
            category_id INTEGER NOT NULL,
            FOREIGN KEY(category_id) REFERENCES categories(category_id)
        );
    """)
    for category in sorted(df["category"].unique()):
        cur.execute("INSERT INTO categories(category_name) VALUES (?)", (category,))
    category_ids = dict(cur.execute("SELECT category_name, category_id FROM categories"))
    rows = [
        (
            r.title, float(r.price_gbp), float(r.price_inr), int(r.rating),
            int(bool(r.in_stock)), category_ids[r.category]
        )
        for r in df.itertuples(index=False)
    ]
    cur.executemany("""
        INSERT INTO books
        (title, price_gbp, price_inr, rating, in_stock, category_id)
        VALUES (?, ?, ?, ?, ?, ?)
    """, rows)
    conn.commit()
    return conn

def run_queries(conn, df):
    queries = [
        ("Q1 SELECT_WHERE",
         "SELECT title, price_inr FROM books WHERE price_inr BETWEEN 500 AND 1500 ORDER BY price_inr LIMIT 10;"),
        ("Q2 ORDER_LIMIT",
         "SELECT title, rating FROM books ORDER BY rating DESC, title LIMIT 10;"),
        ("Q3 DISTINCT",
         "SELECT DISTINCT rating FROM books ORDER BY rating;"),
        ("Q4 IN",
         "SELECT title, rating FROM books WHERE rating IN (4, 5) ORDER BY rating DESC LIMIT 15;"),
        ("Q5 JOIN",
         """SELECT b.title, b.rating, c.category_name
            FROM books b JOIN categories c ON b.category_id = c.category_id
            ORDER BY b.rating DESC, b.title LIMIT 10;"""),
        ("Q6 WHERE_STOCK",
         "SELECT title, price_gbp FROM books WHERE in_stock = 1 ORDER BY price_gbp DESC LIMIT 10;"),
    ]
    with open(SQL_OUTPUT, "w", encoding="utf-8") as f:
        for name, q in queries:
            result = pd.read_sql_query(q, conn)
            f.write(f"\n{name}\nSQL: {q}\nOUTPUT:\n{result.to_string(index=False)}\n")
    join_q = queries[4][1]
    sql_join = pd.read_sql_query(join_q, conn)

    books_mem = df.copy()
    books_mem["book_id"] = range(1, len(books_mem) + 1)
    cats = df[["category"]].drop_duplicates().reset_index(drop=True)
    cats["category_id"] = range(1, len(cats) + 1)
    merged = books_mem.merge(cats, on="category", how="left")
    pandas_join = merged[["title", "rating", "category"]].rename(
        columns={"category": "category_name"}
    ).sort_values(["rating", "title"], ascending=[False, True]).head(10).reset_index(drop=True)
    sql_join = sql_join.reset_index(drop=True)
    equivalent = sql_join.equals(pandas_join)
    with open(SQL_OUTPUT, "a", encoding="utf-8") as f:
        f.write("\nPANDAS pd.read_sql JOIN OUTPUT:\n")
        f.write(sql_join.to_string(index=False))
        f.write("\n\nPANDAS pd.merge JOIN OUTPUT:\n")
        f.write(pandas_join.to_string(index=False))
        f.write(f"\n\nSQL JOIN == pd.merge: {equivalent}\n")
    return equivalent

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    raw = scrape_books()
    if len(raw) < 60:
        raise RuntimeError(f"Only {len(raw)} books scraped; at least 60 required.")
    clean = clean_data(raw)
    clean.to_csv(CSV_PATH, index=False)
    conn = create_database(clean)
    try:
        equivalent = run_queries(conn, clean)
        print(f"Scraped rows: {len(raw)}")
        print(f"Clean rows: {len(clean)}")
        print(f"Categories: {clean['category'].nunique()}")
        print(f"Fixed conversion rate: 1 GBP = {FIXED_GBP_TO_INR} INR")
        print(f"SQL/pandas JOIN equivalent: {equivalent}")
        print(f"Database: {DB_PATH}")
        print(f"Query output: {SQL_OUTPUT}")
    finally:
        conn.close()

if __name__ == "__main__":
    main()
