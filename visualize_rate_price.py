import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
import numpy as np
import plotly.express as px
import webbrowser
from sqlalchemy import create_engine
import tkinter as tk
from tkinter import Toplevel
from tkinter import ttk
from io import StringIO

# Use Qt5Agg for interactive window (optional)
matplotlib.use('Qt5Agg')

# PostgreSQL connection
engine = create_engine('postgresql://postgres:nghiem115940@localhost:5432/postgres')

# Fetch data from fact_reviews and dim_products
query = """
SELECT
    fr.rating,
    fr.rating_count,
    fr.actual_price,
    fr.discount_price,
    (fr.actual_price - fr.discount_price) AS price,
    (fr.actual_price - fr.discount_price) / NULLIF(fr.actual_price, 0) * 100 AS discount_percent,
    dp.name
FROM fact_reviews fr
JOIN dim_products dp ON fr.product_id = dp.product_id
WHERE fr.actual_price IS NOT NULL AND fr.discount_price IS NOT NULL AND fr.rating IS NOT NULL
"""

df = pd.read_sql(query, engine)

# Drop NaNs and ensure numeric
df = df.dropna()
df["actual_price"] = pd.to_numeric(df["actual_price"])
df["discount_price"] = pd.to_numeric(df["discount_price"])
df["rating"] = pd.to_numeric(df["rating"])
df["discount_percent"] = pd.to_numeric(df["discount_percent"])
df["price"] = pd.to_numeric(df["price"])

def bin_and_plot(df, price_column, title):
    bins = pd.cut(df[price_column], bins=20)
    all_bins = pd.Series(bins.cat.categories, name="bin")

    grouped = df.groupby(bins, observed=False).agg({
        "rating": "mean",
        "rating_count": "mean"
    }).reindex(all_bins).reset_index()

    grouped["bin_str"] = grouped["bin"].astype(str)

    # Optional: Interpolate missing values
    grouped["rating"] = grouped["rating"].interpolate()
    grouped["rating_count"] = grouped["rating_count"].interpolate()

    # Create Plotly line plot
    fig = px.line(grouped, x="bin_str", y="rating",
                  title=f"Average Rating vs {title}",
                  labels={"rating": "Average Rating", "bin_str": title},
                  markers=True,
                  hover_data={"rating_count": True})

    fig.update_traces(mode="lines+markers", connectgaps=True)
    fig.update_layout(xaxis_tickangle=-45)

    # Save the plot as an HTML file to display it inside the Tkinter window
    fig.write_html("plot.html")

    return "plot.html"


def create_window_with_plot(price_column, title):
    # Create a new window
    plot_window = Toplevel()
    plot_window.title(f"Rating vs {title}")
    plot_window.geometry("800x600")  # Adjust size as needed

    # Generate and show the plot
    plot_file = bin_and_plot(df, price_column, title)

    # Display the plot using the web browser in Tkinter window
    webbrowser.open_new_tab(plot_file)


# Create the main Tkinter window
root = tk.Tk()
root.title("Price Trend Rating Visualizer")
root.geometry("400x300")  # Adjust the window size as needed

# Add a label above the buttons
label = ttk.Label(root, text="Choose a Price Trend to Visualize", font=("Arial", 14))
label.pack(pady=20)

# Add buttons to the main window
button_actual = ttk.Button(root, text="Actual Price", command=lambda: create_window_with_plot("actual_price", "Actual Price"))
button_actual.pack(pady=10)

button_discount = ttk.Button(root, text="Discount Price", command=lambda: create_window_with_plot("discount_price", "Discount Price"))
button_discount.pack(pady=10)

button_discounted = ttk.Button(root, text="Discounted Price", command=lambda: create_window_with_plot("discount_percent", "Discount Percent"))
button_discounted.pack(pady=10)

button_price = ttk.Button(root, text="Sold price", command=lambda: create_window_with_plot("price", "Sold price"))
button_price.pack(pady=10)

# Start the Tkinter event loop
root.mainloop()
