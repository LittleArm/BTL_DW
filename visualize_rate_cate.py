import pandas as pd
import plotly.express as px
import numpy as np
import tkinter as tk
from tkinter import ttk, Toplevel
from sqlalchemy import create_engine
import webbrowser

# PostgreSQL connection
engine = create_engine('postgresql://postgres:nghiem115940@localhost:5432/postgres')

# Fetch data from joined tables
query = """
SELECT
    fr.rating,
    fr.rating_count,
    fr.actual_price,
    fr.discount_price,
    dp.product_id,
    dc.category
FROM fact_reviews fr
JOIN dim_products dp ON fr.product_id = dp.product_id
JOIN dim_categories dc ON dp.category_id = dc.category_id
WHERE fr.rating IS NOT NULL
    AND fr.rating_count IS NOT NULL
    AND fr.actual_price IS NOT NULL
    AND fr.discount_price IS NOT NULL
"""

df = pd.read_sql(query, engine)

# Drop NaNs and convert numeric columns
df = df.dropna()
df["rating"] = pd.to_numeric(df["rating"])
df["rating_count"] = pd.to_numeric(df["rating_count"])
df["actual_price"] = pd.to_numeric(df["actual_price"])
df["discount_price"] = pd.to_numeric(df["discount_price"])
df["discount_percent"] = (df["actual_price"] - df["discount_price"]) / df["actual_price"] * 100

# Split category string and get top-level
df["top_category"] = df["category"].str.split("|").str[0]

# Weighted average for rating, simple average for prices/discounts
def calculate_grouped_data(metric):
    if metric == "rating":
        # Using `agg` for weighted average of ratings
        grouped = df.groupby("top_category").agg(
            value=("rating", lambda x: np.average(x, weights=df.loc[x.index, "rating_count"]))
        ).reset_index()
    else:
        grouped = df.groupby("top_category")[metric].mean().reset_index(name="value")
    
    grouped = grouped.sort_values(by="value", ascending=False)
    return grouped

# Plot and show in browser
def show_plot(metric, title, y_label):
    data = calculate_grouped_data(metric)
    fig = px.bar(
        data,
        x="top_category",
        y="value",
        title=title,
        labels={"top_category": "Category", "value": y_label},
        text_auto=True
    )
    fig.update_layout(xaxis_tickangle=-45)
    filename = f"{metric}_category_plot.html"
    fig.write_html(filename)
    webbrowser.open_new_tab(filename)

# GUI Setup
def create_gui():
    root = tk.Tk()
    root.title("Category vs Metrics Visualizer")
    root.geometry("400x300")

    label = ttk.Label(root, text="Choose a Metric to Visualize", font=("Arial", 14))
    label.pack(pady=20)

    ttk.Button(root, text="Average Rating", command=lambda: show_plot("rating", "Average Rating by Category", "Average Rating")).pack(pady=5)
    ttk.Button(root, text="Average Actual Price", command=lambda: show_plot("actual_price", "Average Actual Price by Category", "Actual Price")).pack(pady=5)
    ttk.Button(root, text="Average Discount Price", command=lambda: show_plot("discount_price", "Average Discount Price by Category", "Discount Price")).pack(pady=5)
    ttk.Button(root, text="Average % Discount", command=lambda: show_plot("discount_percent", "Average % Discount by Category", "Discount (%)")).pack(pady=5)

    root.mainloop()

# Run the GUI
create_gui()
