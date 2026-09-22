# Data Pipeline

Run:

```bash
python pipeline.py
```

The script scrapes five catalogue pages from `books.toscrape.com`, yielding at least 60 books. Because the assignment permits the first five pages of the All Products catalogue, the category field is consistently represented as `All products`.

The required conversion is exactly `1 GBP = 105.50 INR`.

Malformed numeric values are converted to missing and median-imputed. Rows missing title/category are dropped because those fields are required identifiers/descriptors. SQLite contains normalized `categories` and `books` tables linked through a primary key/foreign key relationship.

Six SQL queries are executed and their SQL strings and outputs are written to `output/sql_query_outputs.txt`. The JOIN is reproduced using `pd.merge` and compared with the `pd.read_sql` JOIN result.
