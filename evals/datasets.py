"""Synthetic, deliberately messy datasets with KNOWN ground truth for evaluating the agents."""
import numpy as np
import pandas as pd


def churn(n=1500, seed=0):
    r = np.random.default_rng(seed)
    tenure = r.integers(1, 72, n)
    monthly = r.normal(70, 25, n).clip(20, 150).round(2)
    contract = r.choice(["Month-to-month", "One year", "Two year"], n, p=[0.55, 0.25, 0.2])
    calls = r.poisson(1.5, n)
    z = -0.05 * tenure + 0.02 * monthly + 0.5 * calls + np.where(contract == "Month-to-month", 1.5, -0.5) - 0.3
    churn = (r.random(n) < 1 / (1 + np.exp(-z))).astype(int)
    df = pd.DataFrame({"Customer ID": [f"C{i:05d}" for i in range(n)], "tenure": tenure, "Monthly Charges": monthly,
                       "contract": contract, "support_calls": calls, "churn": np.where(churn == 1, "Yes", "No"), "region": r.choice(["N", "S", "E", "W"], n)})
    df.loc[r.random(n) < 0.08, "Monthly Charges"] = np.nan
    df.loc[r.random(n) < 0.03, "contract"] = "N/A"
    return pd.concat([df, df.head(20)], ignore_index=True)  # duplicates


def housing(n=1200, seed=1):
    r = np.random.default_rng(seed)
    area = r.normal(1500, 400, n).clip(500, 4000)
    beds = r.integers(1, 6, n)
    age = r.integers(0, 60, n)
    hood = r.choice(["Downtown", "Suburb", "Rural"], n)
    price = 150 * area + 8000 * beds - 900 * age + np.select([hood == "Downtown", hood == "Suburb"], [60000, 20000], 0) + r.normal(0, 15000, n)
    df = pd.DataFrame({"area": area.round(1), "bedrooms": beds, "age": age, "neighborhood": hood, "price": price.round(0)})
    df.loc[r.random(n) < 0.05, "age"] = np.nan
    return df


def sales(n=900, seed=2):
    r = np.random.default_rng(seed)
    region = r.choice(["North", "South", "East", "West"], n, p=[0.2, 0.2, 0.45, 0.15])
    rev = r.gamma(5, 100, n).round(2) * np.where(region == "East", 1.6, 1.0)
    df = pd.DataFrame({"order_date": pd.date_range("2024-01-01", periods=n, freq="8h").strftime("%Y-%m-%d"), "region": region,
                       "product": r.choice(["A", "B", "C"], n), "revenue": [f"{v:,.2f}" for v in rev]})
    df.loc[r.random(n) < 0.04, "revenue"] = "N/A"
    return df


GENERATORS = {"churn": churn, "housing": housing, "sales": sales}
