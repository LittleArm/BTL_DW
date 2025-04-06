import gradio as gr
import pandas as pd
import numpy as np
import plotly.express as px
import faiss
import torch
from sentence_transformers import SentenceTransformer
from sqlalchemy import create_engine
import os
import psycopg2
import re

# Database configuration
DB_CONFIG = {
    "dbname": "amazon",
    "user": "postgres",
    "password": "user",
    "host": "34.142.201.81",
    "port": "5432"
}

# Visualization
# Create html_plot directory
if not os.path.exists("html_plot"):
    os.makedirs("html_plot")

# Database connection
engine = create_engine(
    f"postgresql://{DB_CONFIG['user']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['dbname']}"
)

# Query data
query_app1 = """
SELECT
    fr.rating,
    fr.rating_count,
    fr.actual_price,
    fr.discount_price AS discounted_price,
    dp.product_id,
    dp.name,
    dc.main_category
FROM fact_reviews fr
JOIN dim_products dp ON fr.product_id = dp.product_id
JOIN dim_categories dc ON dp.category_id = dc.category_id
WHERE fr.rating IS NOT NULL
    AND fr.rating_count IS NOT NULL
    AND fr.actual_price IS NOT NULL
    AND fr.discount_price IS NOT NULL
"""
df_app1 = pd.read_sql(query_app1, engine).dropna()

# Clean and calculate additional columns
df_app1["rating"] = pd.to_numeric(df_app1["rating"])
df_app1["rating_count"] = pd.to_numeric(df_app1["rating_count"])
df_app1["actual_price"] = pd.to_numeric(df_app1["actual_price"])
df_app1["discounted_price"] = pd.to_numeric(df_app1["discounted_price"])
df_app1["discount_amount"] = df_app1["actual_price"] - \
    df_app1["discounted_price"]
df_app1["discount_percent"] = (
    df_app1["discount_amount"] / df_app1["actual_price"]) * 100

# Summary statistics
averages = {
    "Average Rating": df_app1["rating"].mean(),
    "Average Actual Price": df_app1["actual_price"].mean(),
    "Average Discounted Price": df_app1["discounted_price"].mean(),
    "Average Discount Amount": df_app1["discount_amount"].mean(),
    "Average Discount %": df_app1["discount_percent"].mean()
}

# Functions for visualizer


def show_summary():
    summary_text = "\n".join([f"{k} {v:.2f}" for k, v in averages.items()])
    return summary_text


def calculate_grouped_data(metric):
    if metric == "rating":
        grouped = df_app1.groupby("main_category").agg(
            value=("rating", lambda x: np.average(
                x, weights=df_app1.loc[x.index, "rating_count"]))
        ).reset_index()
    else:
        grouped = df_app1.groupby("main_category")[
            metric].mean().reset_index(name="value")
    grouped = grouped.sort_values(by="value", ascending=False)
    return grouped


def show_category_plot(metric, title, y_label):
    data = calculate_grouped_data(metric)
    data["value"] = data["value"].round(2)

    fig = px.bar(data,
                 x="main_category",
                 y="value",
                 title=title,
                 labels={"main_category": "Category", "value": y_label},
                 text=data["value"].apply(lambda x: f"{x:.2f}"))

    fig.update_traces(
        textposition='outside',
        textangle=0,
        textfont=dict(size=12)
    )

    fig.update_layout(
        xaxis_tickangle=-45,
        yaxis_title=y_label,
        xaxis_title="Category",
        bargap=0.2,
        margin=dict(t=50, b=100),
        height=500,
        yaxis_range=[0, data["value"].max() * 1.2]
    )

    filename = f"html_plot/{metric}_category_plot.html"
    filepath = os.path.abspath(filename)
    fig.write_html(filepath)
    return fig


def show_price_bin_plot(column, title):
    # Define bins based on column type
    if column == "discount_percent":
        bins = pd.cut(df_app1[column], bins=range(
            0, int(df_app1[column].max()) + 1, 1))  # 1% bins
    else:
        bins = pd.cut(df_app1[column], bins=range(
            0, int(df_app1[column].max()) + 1000, 1000))  # 1000 for prices

    all_bins = pd.Series(bins.cat.categories, name="bin")
    grouped = df_app1.groupby(bins, observed=False).agg({
        "rating": "mean",
        "rating_count": "mean"
    }).reindex(all_bins).reset_index()

    # Use the left endpoint of each bin as the x-axis value
    grouped["bin_start"] = [interval.left for interval in grouped["bin"]]
    # Spline interpolation for rating (smooth curve)
    grouped["rating"] = grouped["rating"].interpolate(method="spline", order=3)
    # Linear interpolation for rating_count, then clamp to non-negative
    grouped["rating_count"] = grouped["rating_count"].interpolate(
        method="linear")
    grouped["rating_count"] = grouped["rating_count"].clip(
        lower=0)  # Ensure no negative values

    fig = px.line(grouped, x="bin_start", y="rating", title=f"Average Rating vs {title}",
                  labels={"rating": "Average Rating", "bin_start": title},
                  hover_data={"rating_count": False})
    # Smooth line with spline shape
    fig.update_traces(mode="lines", line_shape="spline")

    # Calculate min and max for y-axis with padding
    y_min = grouped["rating"].min()
    y_max = grouped["rating"].max()
    padding = (y_max - y_min) * 0.1  # Add 10% padding above and below
    y_range = [y_min - padding, y_max + padding]

    # Prettier axes and less dense x-axis
    fig.update_layout(
        xaxis_tickangle=-45,
        autosize=True,  # Automatically scale to screen
        xaxis=dict(
            tickmode="array",
            tickvals=grouped["bin_start"][::max(
                1, len(grouped) // 10)],  # Show ~10 ticks max
            ticktext=[f"{x:.0f}" for x in grouped["bin_start"]
                      [::max(1, len(grouped) // 10)]],  # Format as integers
            title_font=dict(size=14),
            tickfont=dict(size=12)
        ),
        yaxis=dict(
            title_font=dict(size=14),  # Fixed typo from size=4 to size=14
            tickfont=dict(size=12),
            range=y_range  # Custom range based on min and max with padding
        ),
        title_x=0.5,  # Center the title
        margin=dict(l=40, r=40, t=60, b=40)  # Adjust margins for better fit
    )

    filename = f"html_plot/{column}_bin_plot.html"
    filepath = os.path.abspath(filename)
    fig.write_html(filepath)
    return fig


# Smart Search
DATA_DIR = "./data"
CSV_FILE = "./amazon.csv"
EMBEDDING_FILE = os.path.join(DATA_DIR, "embeddings.npy")
FAISS_INDEX_FILE = os.path.join(DATA_DIR, "faiss_index.bin")

for file_path in [CSV_FILE, EMBEDDING_FILE, FAISS_INDEX_FILE]:
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"File not found: {file_path}. Please ensure the file exists.")

df_app2 = pd.read_csv(CSV_FILE, encoding="utf-8")
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


def connect_db():
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        conn.autocommit = True
        return conn
    except psycopg2.Error as e:
        print(f"❌ PostgreSQL connection error: {e}")
        return None


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


df_cleaned = clean_data(df_app2)
df_final = refactor_data(df_cleaned)
main_categories = ["All"] + \
    sorted(df_cleaned["main_category"].unique().tolist())


def search_products(query, top_k, min_price, max_price, min_rating, main_category=None):
    query_embedding = model.encode([query], device=device).astype('float32')
    gpu_index.nprobe = 10
    distances, indices = gpu_index.search(query_embedding, top_k * 2)

    product_ids = [int(df_final["product_id"].iloc[idx])
                   for idx in indices[0] if idx < len(df_final)]
    conn = connect_db()
    if not conn:
        return [], "❌ Unable to connect to the database."

    with conn.cursor() as cur:
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
        if main_category and main_category != "All":
            query_sql += " AND dc.main_category = %s"
            params.append(main_category)
        query_sql += " ORDER BY fr.rating DESC LIMIT %s;"
        params.append(top_k)

        cur.execute(query_sql, tuple(params))
        rows = cur.fetchall()

    conn.close()

    results = []
    for row in rows:
        product_id, name, image, link, rating, rating_count, actual_price, discount_price = row
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


def gradio_search_interface(query, top_k, min_price, max_price, min_rating, main_category):
    results, error_msg = search_products(
        query, top_k, min_price, max_price, min_rating, main_category)
    if error_msg:
        return error_msg
    if not results:
        return "❌ No products found matching the search criteria."

    output_text = ""
    default_img_url = "https://via.placeholder.com/100x100.png?text=No+Image"
    for i, res in enumerate(results):
        name_link = f"<a href='{res['link']}' target='_blank'>{res['name']}</a>"
        img_src = res['image'] if res['image'] else default_img_url
        output_text += (
            f"<div style='display: flex; align-items: center; margin-bottom: 20px;'>"
            f"<img src='{img_src}' width='100' height='100' style='margin-right: 20px;' "
            f"onerror=\"this.src='{default_img_url}'\">"
            f"<div>"
            f"**{i+1}. {name_link}**<br>"
            f"- Actual Price: ₹{res['actual_price']:.2f}<br>"
            f"- Discounted Price: ₹{res['discount_price']:.2f}<br>"
            f"- Rating: {res['rating']}/5 ({res['rating_count']} reviews)<br>"
            f"- Distance: {res['distance']:.4f}"
            f"</div>"
            f"</div>"
        )
    return output_text


# Gradio Interface
with gr.Blocks(title="Amazon Product Analysis and Search Application") as demo:
    gr.Markdown("# Amazon Product Analysis and Search Application")

    with gr.Tabs():
        # Visualization Tab
        with gr.TabItem("Product Review Visualizer"):
            gr.Markdown("## Product Review Visualizer")

            with gr.Tab("Overview"):
                summary_btn = gr.Button("Show Overview Summary")
                summary_output = gr.Textbox(label="Summary Statistics")
                summary_btn.click(fn=show_summary, inputs=None,
                                  outputs=summary_output)

            with gr.Tab("By Category"):
                gr.Markdown("### Visualizations by Category")
                rating_btn = gr.Button("Average Rating")
                actual_price_btn = gr.Button("Actual Price")
                discounted_price_btn = gr.Button("Discounted Price")
                discount_percent_btn = gr.Button("Discount %")
                plot_output = gr.Plot(label="Category Plot")

                rating_btn.click(fn=lambda: show_category_plot("rating", "Average Rating by Category", "Average Rating"),
                                 inputs=None, outputs=plot_output)
                actual_price_btn.click(fn=lambda: show_category_plot("actual_price", "Average Actual Price by Category", "Actual Price"),
                                       inputs=None, outputs=plot_output)
                discounted_price_btn.click(fn=lambda: show_category_plot("discounted_price", "Average Discounted Price by Category", "Discounted Price"),
                                           inputs=None, outputs=plot_output)
                discount_percent_btn.click(fn=lambda: show_category_plot("discount_percent", "Average Discount % by Category", "Discount (%)"),
                                           inputs=None, outputs=plot_output)

            with gr.Tab("By Price Bins"):
                gr.Markdown("### Average Rating by Price Bins")
                actual_price_bin_btn = gr.Button(
                    "Average Rating by Actual Price")
                discounted_price_bin_btn = gr.Button(
                    "Average Rating by Discounted Price")
                discount_percent_bin_btn = gr.Button(
                    "Average Rating by Discount %")
                bin_plot_output = gr.Plot(label="Average Rating Plot")

                actual_price_bin_btn.click(fn=lambda: show_price_bin_plot("actual_price", "Actual Price"),
                                           inputs=None, outputs=bin_plot_output)
                discounted_price_bin_btn.click(fn=lambda: show_price_bin_plot("discounted_price", "Discounted Price"),
                                               inputs=None, outputs=bin_plot_output)
                discount_percent_bin_btn.click(fn=lambda: show_price_bin_plot("discount_percent", "Discount Percent"),
                                               inputs=None, outputs=bin_plot_output)

        # Smart Product Search Tab
        with gr.TabItem("Smart Product Search"):
            gr.Markdown("## Smart Product Search (FAISS + PostgreSQL)")
            gr.Markdown(
                "Enter keywords and filters to search for products from Amazon data.")

            with gr.Row():
                query_input = gr.Textbox(
                    label="Search Keyword", placeholder="Example: cables")
                top_k_input = gr.Slider(
                    1, 50, value=5, step=1, label="Number of Results")

            gr.Markdown("### Filters")
            with gr.Row():
                min_price_input = gr.Number(label="Minimum Price (₹)", value=0)
                max_price_input = gr.Number(
                    label="Maximum Price (₹)", value=1000)
                min_rating_input = gr.Slider(
                    0, 5, value=0, step=0.1, label="Minimum Rating")
            with gr.Row():
                main_category_input = gr.Dropdown(
                    choices=main_categories, label="Main Category", value="All")

            submit_btn = gr.Button("Search")
            output_text = gr.Markdown(
                label="Search Results", value="Search results will be displayed here.")

            submit_btn.click(fn=gradio_search_interface,
                             inputs=[query_input, top_k_input, min_price_input,
                                     max_price_input, min_rating_input, main_category_input],
                             outputs=[output_text])

demo.launch(share=True)
