# overview_gui.py
import tkinter as tk
from tkinter import ttk
import pandas as pd
from sqlalchemy import create_engine
import subprocess

# Connect to PostgreSQL
engine = create_engine('postgresql://postgres:nghiem115940@localhost:5432/postgres')

# Fetch and process
query = """
SELECT
    fr.rating,
    fr.actual_price,
    fr.discount_price
FROM fact_reviews fr
WHERE fr.rating IS NOT NULL AND fr.actual_price IS NOT NULL AND fr.discount_price IS NOT NULL
"""
df = pd.read_sql(query, engine)

df["rating"] = pd.to_numeric(df["rating"])
df["actual_price"] = pd.to_numeric(df["actual_price"])
df["discount_price"] = pd.to_numeric(df["discount_price"])
df["true_price"] = df["actual_price"] - df["discount_price"]
df["discount_percent"] = ((df["actual_price"] - df["discount_price"]) / df["actual_price"]) * 100

averages = {
    "Average Rating": df["rating"].mean(),
    "Average Actual Price": df["actual_price"].mean(),
    "Average Discount Price": df["discount_price"].mean(),
    "Average True Price": df["true_price"].mean(),
    "Average Discount %": df["discount_percent"].mean()
}

# GUI
root = tk.Tk()
root.title("Overview Dashboard")
root.geometry("500x450")

ttk.Label(root, text="Overall Summary", font=("Arial", 16)).pack(pady=10)

for key, val in averages.items():
    ttk.Label(root, text=f"{key}: {val:.2f}", font=("Arial", 12)).pack(pady=2)


root.mainloop()
