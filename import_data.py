import psycopg2
import pandas as pd
import numpy as np
import re
from sentence_transformers import SentenceTransformer

# 🔹 Tạo database dưới local trước
# 🔹 Thông tin kết nối PostgreSQL
DB_CONFIG = {
    "dbname": "amazon",
    "user": "postgres",
    "password": "user",
    "host": "34.142.193.208",  # Gcloud: 34.142.193.208
    "port": "5432"
}

# 🔹 Đường dẫn file CSV & embeddings
CSV_FILE = "E:/242/DW&DSS/amazon.csv"
EMBEDDING_FILE = "C:/import_data/embeddings.npy"
FAISS_INDEX_FILE = "C:/import_data/faiss_index.bin"


def connect_db():
    """ Kết nối PostgreSQL """
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        conn.autocommit = True
        return conn
    except psycopg2.Error as e:
        print(f"❌ Lỗi kết nối PostgreSQL: {e}")
        return None


def create_tables(conn):
    """ Tạo bảng theo Star Schema """
    with conn.cursor() as cur:
        cur.execute("""
        DROP TABLE IF EXISTS fact_reviews, dim_products, dim_categories CASCADE;

        CREATE TABLE dim_categories (
            category_id SERIAL PRIMARY KEY,
            main_category TEXT NOT NULL,
            sub_category TEXT NOT NULL,
            UNIQUE(main_category, sub_category)
        );

        CREATE TABLE dim_products (
            product_id SERIAL PRIMARY KEY,
            name TEXT NOT NULL,
            category_id INT REFERENCES dim_categories(category_id),
            image TEXT,
            link TEXT,
            UNIQUE(name, category_id)
        );

        CREATE TABLE fact_reviews (
            review_id SERIAL PRIMARY KEY,
            product_id INT REFERENCES dim_products(product_id),
            rating NUMERIC(3,2),
            rating_count INT,
            actual_price NUMERIC(12,2),
            discount_price NUMERIC(12,2)
        );
        """)
        print("✅ Tables created successfully.")


def clean_data(df):
    """Làm sạch và chuẩn hóa dữ liệu từ CSV"""
    df = df.copy()

    # Chuẩn hóa tên cột
    df["main_category"] = df["category"].astype(str).str.split("|").str[0]
    df["sub_category"] = df["category"].astype(
        str).str.split("|", n=1).str[-1].str.strip()

    df["name"] = df["product_name"].astype(str).str.strip()

    # Xóa ký tự ₹ và dấu phẩy trong giá tiền, chuyển thành số
    def safe_float_convert(value):
        try:
            return float(re.sub(r"[₹,]", "", str(value))) if value not in ["", "nan", None] else None
        except ValueError:
            return None

    df["actual_price"] = df["actual_price"].apply(safe_float_convert)
    df["discount_price"] = df["discounted_price"].apply(safe_float_convert)

    # Làm sạch cột no_of_ratings
    def clean_ratings(value):
        if isinstance(value, str) and value.replace(",", "").isdigit():
            return int(value.replace(",", ""))
        return None  # Trả về None thay vì 0 để lọc bỏ khi dropna

    df["no_of_ratings"] = df["rating_count"].apply(clean_ratings)

    # Làm sạch cột ratings
    def clean_ratings_value(value):
        try:
            return float(value)
        except (ValueError, TypeError):
            return None  # Trả về None thay vì 0 để lọc bỏ khi dropna

    df["ratings"] = df["rating"].apply(clean_ratings_value)

    # 🔹 Loại bỏ hàng có NaN ở các cột quan trọng
    df = df.dropna(
        subset=["actual_price", "discount_price", "ratings", "no_of_ratings"])

    df["image"] = df["img_link"].astype(str).str.strip()
    df["link"] = df["product_link"].astype(str).str.strip()

    return df


def import_data(conn, df):
    """ Nhập dữ liệu vào PostgreSQL """
    with conn.cursor() as cur:
        # 🔹 Chèn dữ liệu vào dim_categories
        cur.executemany("""
        INSERT INTO dim_categories (main_category, sub_category)
        VALUES (%s, %s) ON CONFLICT DO NOTHING;
        """, df[["main_category", "sub_category"]].drop_duplicates().values.tolist())

        # 🔹 Lấy category_id từ PostgreSQL
        cur.execute(
            "SELECT category_id, main_category, sub_category FROM dim_categories;")
        category_map = {f"{row[1]}|{row[2]}": row[0] for row in cur.fetchall()}
        df["category_id"] = df.apply(lambda x: category_map.get(
            f"{x['main_category']}|{x['sub_category']}", None), axis=1)
        df = df.dropna(subset=["category_id"])

        # 🔹 Chèn dữ liệu vào dim_products
        cur.executemany("""
        INSERT INTO dim_products (name, category_id, image, link)
        VALUES (%s, %s, %s, %s) ON CONFLICT DO NOTHING;
        """, df[["name", "category_id", "image", "link"]].drop_duplicates().values.tolist())

        # 🔹 Lấy product_id từ PostgreSQL
        cur.execute("SELECT product_id, LOWER(name) FROM dim_products;")
        product_map = {row[1]: row[0] for row in cur.fetchall()}
        df["product_id"] = df["name"].str.lower().map(product_map)
        df = df.dropna(subset=["product_id"])

        # 🔹 Chèn dữ liệu vào fact_reviews
        cur.executemany("""
        INSERT INTO fact_reviews (product_id, rating, rating_count, actual_price, discount_price)
        VALUES (%s, %s, %s, %s, %s);
        """, df[["product_id", "ratings", "no_of_ratings", "actual_price", "discount_price"]].values.tolist())

        print("✅ Data imported successfully.")


def main():
    conn = connect_db()
    if not conn:
        return

    create_tables(conn)

    df = pd.read_csv(CSV_FILE, encoding="utf-8")
    df = clean_data(df)

    import_data(conn, df)  # 🔹 Nhập dữ liệu trước
    conn.close()


if __name__ == "__main__":
    main()
