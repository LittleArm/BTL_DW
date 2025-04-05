import gradio as gr
import pandas as pd
import numpy as np
import faiss
import torch
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt
import io
from PIL import Image
import requests
import os
import re

# Đường dẫn tệp (có thể tùy chỉnh theo thư mục local của bạn)
DATA_DIR = "./data"
CSV_FILE = "./amazon.csv"
EMBEDDING_FILE = os.path.join(DATA_DIR, "embeddings.npy")
FAISS_INDEX_FILE = os.path.join(DATA_DIR, "faiss_index.bin")

# Kiểm tra xem các tệp có tồn tại không
for file_path in [CSV_FILE, EMBEDDING_FILE, FAISS_INDEX_FILE]:
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"Không tìm thấy tệp: {file_path}. Vui lòng đảm bảo tệp đã được tạo từ trước.")

# Tải dữ liệu
df = pd.read_csv(CSV_FILE, encoding="utf-8")

# Hàm làm sạch dữ liệu (từ mã gốc của bạn)


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

# Hàm refactor dữ liệu (từ mã gốc của bạn)


def refactor_data(df):
    category_map = {
        f"{row['main_category']}|{row['sub_category']}": idx
        for idx, row in enumerate(df[["main_category", "sub_category"]].drop_duplicates().to_dict(orient="records"), start=1)
    }
    df["category_id"] = df.apply(lambda x: category_map.get(
        f"{x['main_category']}|{x['sub_category']}"), axis=1)
    df = df.dropna(subset=["category_id"])

    product_map = {
        row["name"].lower(): idx
        for idx, row in enumerate(df[["name"]].drop_duplicates().to_dict(orient="records"), start=1)
    }
    df["product_id"] = df["name"].str.lower().map(product_map)
    df = df.dropna(subset=["product_id"])
    return df


# Tải và xử lý dữ liệu
df_cleaned = clean_data(df)
df_final = refactor_data(df_cleaned)

# Tải embeddings và FAISS index
embeddings = np.load(EMBEDDING_FILE)
index = faiss.read_index(FAISS_INDEX_FILE)
model = SentenceTransformer('all-MiniLM-L6-v2')
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Chuyển FAISS index sang GPU nếu có
if torch.cuda.is_available():
    res = faiss.StandardGpuResources()
    gpu_index = faiss.index_cpu_to_gpu(res, 0, index)
else:
    gpu_index = index

# Hàm tìm kiếm sản phẩm


def search_products(query, top_k=5):
    query_embedding = model.encode([query], device=device).astype('float32')
    gpu_index.nprobe = 10
    distances, indices = gpu_index.search(query_embedding, top_k)

    results = []
    for i in range(top_k):
        idx = indices[0][i]
        if idx < len(df_final):
            product = df_final.iloc[idx]
            results.append({
                "name": product["name"],
                "actual_price": product["actual_price"],
                "discount_price": product["discount_price"],
                "rating": product["ratings"],
                "rating_count": product["no_of_ratings"],
                "image": product["image"],
                "distance": distances[0][i]
            })
    return results

# Hàm tạo biểu đồ phân bố đánh giá


def plot_rating_distribution(results):
    ratings = [r["rating"] for r in results]
    plt.figure(figsize=(8, 4))
    plt.hist(ratings, bins=5, range=(0, 5),
             color='lightgreen', edgecolor='black')
    plt.title("Phân bố đánh giá của sản phẩm tìm kiếm")
    plt.xlabel("Đánh giá (0-5)")
    plt.ylabel("Số lượng")
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    plt.close()
    return Image.open(buf)

# Hàm chính cho giao diện


def gradio_interface(query, top_k):
    results = search_products(query, top_k)

    output_text = ""
    images = []
    default_img = Image.new('RGB', (100, 100), color='gray')

    for i, res in enumerate(results):
        output_text += (f"**{i+1}. {res['name']}**\n"
                        f"- Giá gốc: ₹{res['actual_price']:.2f}\n"
                        f"- Giá giảm: ₹{res['discount_price']:.2f}\n"
                        f"- Đánh giá: {res['rating']}/5 ({res['rating_count']} lượt)\n"
                        f"- Khoảng cách: {res['distance']:.4f}\n\n")
        try:
            response = requests.get(res["image"], stream=True, timeout=5)
            response.raise_for_status()
            img = Image.open(response.raw)
            images.append(img)
        except (requests.RequestException, Exception):
            images.append(default_img)

    rating_plot = plot_rating_distribution(results)
    return output_text, images, rating_plot


# Xây dựng giao diện Gradio
with gr.Blocks(title="Hệ thống tìm kiếm sản phẩm Amazon") as demo:
    gr.Markdown("# Tìm kiếm sản phẩm thông minh")
    gr.Markdown(
        "Nhập từ khóa để tìm kiếm sản phẩm từ dữ liệu Amazon. Xem thông tin chi tiết, hình ảnh và phân bố đánh giá.")

    with gr.Row():
        query_input = gr.Textbox(
            label="Từ khóa tìm kiếm", placeholder="Ví dụ: cables")
        top_k_input = gr.Slider(1, 50, value=5, step=1,
                                label="Số lượng kết quả")
        submit_btn = gr.Button("Tìm kiếm")

    with gr.Row():
        with gr.Column(scale=2):
            output_text = gr.Markdown(label="Kết quả tìm kiếm")
        with gr.Column(scale=1):
            output_images = gr.Gallery(
                label="Hình ảnh sản phẩm", show_label=True)

    with gr.Row():
        output_plot = gr.Image(label="Phân bố đánh giá")

    submit_btn.click(
        fn=gradio_interface,
        inputs=[query_input, top_k_input],
        outputs=[output_text, output_images, output_plot]
    )

# Khởi chạy giao diện
if __name__ == "__main__":
    demo.launch(share=True)
