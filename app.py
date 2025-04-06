import gradio as gr
import pandas as pd
import numpy as np
import faiss
import torch
from sentence_transformers import SentenceTransformer
from PIL import Image
import requests
import os
import psycopg2
import re

# Directory and file paths
DATA_DIR = "./data"
CSV_FILE = "./amazon.csv"
EMBEDDING_FILE = os.path.join(DATA_DIR, "embeddings.npy")
FAISS_INDEX_FILE = os.path.join(DATA_DIR, "faiss_index.bin")

# PostgreSQL configuration
DB_CONFIG = {
    "dbname": "amazon",
    "user": "postgres",
    "password": "user",
    "host": "34.142.201.81",
    "port": "5432"
}

# Check for required files
for file_path in [CSV_FILE, EMBEDDING_FILE, FAISS_INDEX_FILE]:
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"Không tìm thấy tệp: {file_path}. Vui lòng đảm bảo tệp đã được tạo từ trước.")

# Load CSV for initial mapping
df = pd.read_csv(CSV_FILE, encoding="utf-8")

# FAISS setup
embeddings = np.load(EMBEDDING_FILE)
index = faiss.read_index(FAISS_INDEX_FILE)
model = SentenceTransformer('all-MiniLM-L6-v2')
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

if torch.cuda.is_available():
    res = faiss.StandardGpuResources()
    gpu_index = faiss.index_cpu_to_gpu(res, 0, index)
else:
    gpu_index = index

# PostgreSQL connection


def connect_db():
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        conn.autocommit = True
        return conn
    except psycopg2.Error as e:
        print(f"❌ Lỗi kết nối PostgreSQL: {e}")
        return None

# Clean and refactor data


def clean_data(df):
    df = df.copy()
    df["main_category"] = df["category"].astype(str).str.split("|").str[0]
    df["sub_category"] = df["category"].astype(
        str).str.split("|", n=1).str[-1].str.strip()
    df["name"] = df["product_name"].astype(str).str.strip()

    def safe_float_convert(value):
        try:
            return float(re.sub(r"[₹,]", "", str(value))) if value not in ["", "nan", None] else None
        except ValueError:
            return None

    df["actual_price"] = df["actual_price"].apply(safe_float_convert)
    df["discount_price"] = df["discounted_price"].apply(safe_float_convert)

    def clean_ratings(value):
        if isinstance(value, str) and value.replace(",", "").isdigit():
            return int(value.replace(",", ""))
        return None

    df["no_of_ratings"] = df["rating_count"].apply(clean_ratings)

    def clean_ratings_value(value):
        try:
            return float(value)
        except (ValueError, TypeError):
            return None

    df["ratings"] = df["rating"].apply(clean_ratings_value)
    df = df.dropna(
        subset=["actual_price", "discount_price", "ratings", "no_of_ratings"])
    df["image"] = df["img_link"].astype(str).str.strip()
    df["link"] = df["product_link"].astype(str).str.strip()
    return df


def refactor_data(df):
    category_map = {f"{row['main_category']}|{row['sub_category']}": idx
                    for idx, row in enumerate(df[["main_category", "sub_category"]].drop_duplicates().to_dict(orient="records"), start=1)}
    df["category_id"] = df.apply(lambda x: category_map.get(
        f"{x['main_category']}|{x['sub_category']}"), axis=1)
    df = df.dropna(subset=["category_id"])

    product_map = {row["name"].lower(): idx
                   for idx, row in enumerate(df[["name"]].drop_duplicates().to_dict(orient="records"), start=1)}
    df["product_id"] = df["name"].str.lower().map(product_map)
    df = df.dropna(subset=["product_id"])
    return df


df_cleaned = clean_data(df)
df_final = refactor_data(df_cleaned)

# Get unique main categories for dropdown
main_categories = ["All"] + \
    sorted(df_cleaned["main_category"].unique().tolist())

# Search function combining FAISS and PostgreSQL


def search_products(query, top_k, min_price, max_price, min_rating, main_category=None):
    # Step 1: FAISS similarity search
    query_embedding = model.encode([query], device=device).astype('float32')
    gpu_index.nprobe = 10
    distances, indices = gpu_index.search(
        query_embedding, top_k * 2)  # Fetch extra results to filter

    # Step 2: Map FAISS indices to product_ids and convert to Python int
    product_ids = [int(df_final["product_id"].iloc[idx])
                   for idx in indices[0] if idx < len(df_final)]

    # Step 3: Query PostgreSQL with filters
    conn = connect_db()
    if not conn:
        return [], "❌ Không thể kết nối đến cơ sở dữ liệu."

    with conn.cursor() as cur:
        # Base query
        query_sql = """
            SELECT dp.product_id, dp.name, dp.image, dp.link, fr.rating, fr.rating_count, fr.actual_price, fr.discount_price
            FROM dim_products dp
            JOIN fact_reviews fr ON dp.product_id = fr.product_id
            JOIN dim_categories dc ON dp.category_id = dc.category_id
            WHERE dp.product_id = ANY(%s)
            AND fr.discount_price BETWEEN %s AND %s
            AND fr.rating >= %s
        """
        params = [product_ids, min_price, max_price, min_rating]

        # Add main_category filter if specified
        if main_category and main_category != "All":
            query_sql += " AND dc.main_category = %s"
            params.append(main_category)

        # Add ordering and limit
        query_sql += " ORDER BY fr.rating DESC LIMIT %s;"
        params.append(top_k)

        cur.execute(query_sql, tuple(params))
        rows = cur.fetchall()

    conn.close()

    results = []
    for row in rows:
        product_id, name, image, link, rating, rating_count, actual_price, discount_price = row
        # Find the FAISS distance for this product
        idx = df_final[df_final["product_id"] == product_id].index[0]
        distance = distances[0][list(indices[0]).index(
            idx)] if idx in indices[0] else float('inf')
        results.append({
            "name": name,
            "actual_price": actual_price,
            "discount_price": discount_price,
            "rating": rating,
            "rating_count": rating_count,
            "image": image,
            "link": link,
            "distance": distance
        })

    return results, ""

# Gradio interface


def gradio_interface(query, top_k, min_price, max_price, min_rating, main_category):
    results, error_msg = search_products(
        query, top_k, min_price, max_price, min_rating, main_category)

    if error_msg:
        return error_msg

    if not results:
        return "❌ Không tìm thấy sản phẩm nào phù hợp với tiêu chí tìm kiếm."

    output_text = ""
    # Fallback image URL
    default_img_url = "https://via.placeholder.com/100x100.png?text=No+Image"

    for i, res in enumerate(results):
        # Use product link for clickable name
        name_link = f"<a href='{res['link']}' target='_blank'>{res['name']}</a>"
        # Use image URL directly in HTML, with fallback
        img_src = res['image'] if res['image'] else default_img_url
        # Construct the result with image and text side by side
        output_text += (
            f"<div style='display: flex; align-items: center; margin-bottom: 20px;'>"
            f"<img src='{img_src}' width='100' height='100' style='margin-right: 20px;' "
            f"onerror=\"this.src='{default_img_url}'\">"
            f"<div>"
            f"**{i+1}. {name_link}**<br>"
            f"- Giá gốc: ₹{res['actual_price']:.2f}<br>"
            f"- Giá giảm: ₹{res['discount_price']:.2f}<br>"
            f"- Đánh giá: {res['rating']}/5 ({res['rating_count']} lượt)<br>"
            f"- Khoảng cách: {res['distance']:.4f}"
            f"</div>"
            f"</div>"
        )

    return output_text


# Gradio UI
with gr.Blocks(title="Hệ thống tìm kiếm sản phẩm Amazon (Hybrid)") as demo:
    gr.Markdown("# Tìm kiếm sản phẩm thông minh (FAISS + PostgreSQL)")
    gr.Markdown("Nhập từ khóa và bộ lọc để tìm kiếm sản phẩm từ dữ liệu Amazon.")

    with gr.Row():
        query_input = gr.Textbox(
            label="Từ khóa tìm kiếm", placeholder="Ví dụ: cables")
        top_k_input = gr.Slider(1, 50, value=5, step=1,
                                label="Số lượng kết quả")

    gr.Markdown("## Bộ lọc")  # Title for filter block
    with gr.Row():
        min_price_input = gr.Number(label="Giá tối thiểu (₹)", value=0)
        max_price_input = gr.Number(label="Giá tối đa (₹)", value=1000)
        min_rating_input = gr.Slider(
            0, 5, value=0, step=0.1, label="Đánh giá tối thiểu")
    with gr.Row():
        main_category_input = gr.Dropdown(
            choices=main_categories, label="Danh mục chính", value="All")

    submit_btn = gr.Button("Tìm kiếm")

    output_text = gr.Markdown(label="Kết quả tìm kiếm",
                              value="Kết quả tìm kiếm sẽ hiển thị ở đây.")

    submit_btn.click(
        fn=gradio_interface,
        inputs=[query_input, top_k_input, min_price_input,
                max_price_input, min_rating_input, main_category_input],
        outputs=[output_text]
    )

if __name__ == "__main__":
    demo.launch(share=True)
