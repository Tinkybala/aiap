"""Travelers' Inn: in-house cancellation risk pipeline.

Reads data/hotel.db, trains a model to predict whether a booking will be cancelled,
prints hold-out results and writes a 0-100 risk score for every booking to scores.csv.

Run:  python in_house_pipeline.py          (flags bookings with a cancellation probability >= 0.5)

"""
import sqlite3
import sys

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, precision_score, recall_score,
                             roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DB_PATH = "data/hotel.db"
SEED = 42
THRESHOLD = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5  # flag a booking at or above this probability

# Values that mean "unknown" in this data set.
PLACEHOLDERS = {"", "-", "na", "n/a", "nan", "null", "unknown", "undefined"}

CATEGORICAL = ["hotel", "arrival_date_month", "meal", "country", "market_segment",
               "distribution_channel", "reserved_room_type", "deposit_type",
               "customer_type", "agent"]
NUMERIC = ["lead_time", "arrival_date_week_number", "arrival_date_day_of_month",
           "stays_in_weekend_nights", "stays_in_week_nights", "adults", "children",
           "babies", "is_repeated_guest", "previous_cancellations",
           "previous_bookings_not_canceled", "booking_changes", "days_in_waiting_list",
           "adr", "required_car_parking_spaces", "total_of_special_requests"]

# Columns we do NOT give the model, and why:
#   reservation_status       records the outcome (Canceled / Check-Out): leaks the answer
#   reservation_status_date  date of that outcome: leaks the answer
#   assigned_room_type       only known at check-in, not when the booking is made
#   arrival_date_year        a future booking's year is never seen in training
#   company                  about 70% missing and stored as free-text names
#   promo_code               about 40% missing, no relationship with cancellation
#   booking_id, client_email, Unnamed: 0   identifiers / export index

MONTHS = {m[:3]: m for m in ["january", "february", "march", "april", "may", "june", "july",
                             "august", "september", "october", "november", "december"]}
COUNTRIES = {"singapore": "sgp", "malaysia": "mys", "indonesia": "idn", "china": "chn",
             "india": "ind", "other": "oth"}
TYPOS = {"no depost": "no deposit", "onlin ta": "online ta", "offline ta/t0": "offline ta/to",
         "transiant": "transient"}


def load():
    with sqlite3.connect(DB_PATH) as con:
        return pd.read_sql("SELECT * FROM bookings", con)  # every column arrives as text


def clean(df):
    df = df.drop_duplicates(subset="booking_id").copy()

    # Text: trim, lower-case, one spelling per category, placeholders -> missing.
    for col in CATEGORICAL + ["is_canceled"]:
        s = df[col].astype(str).str.strip().str.lower().str.replace(r"[\s\-]+", " ", regex=True)
        df[col] = s.replace(TYPOS).where(~s.isin(PLACEHOLDERS))
    df["arrival_date_month"] = df["arrival_date_month"].replace(MONTHS)
    df["country"] = df["country"].replace(COUNTRIES)
    df["agent"] = df["agent"].str.replace(r"\.0$", "", regex=True).replace("999", np.nan)  # 999 = "no agent"

    # Target: several spellings of yes/no.
    df["is_canceled"] = df["is_canceled"].map({"1": 1, "true": 1, "yes": 1,
                                               "0": 0, "false": 0, "no": 0})

    # Numbers: "276,00" -> 276.00, "SGD 172.50" -> 172.50, "14 days" -> 14.
    for col in NUMERIC:
        s = df[col].astype(str).str.replace("SGD", "").str.replace("days", "")
        s = s.str.strip().str.replace(",", ".")
        df[col] = pd.to_numeric(s, errors="coerce")

    # Impossible values -> missing.
    df.loc[(df["lead_time"] < 0) | (df["lead_time"] > 737), "lead_time"] = np.nan
    df.loc[df["adr"] < 0, "adr"] = np.nan

    return df.dropna(subset=["is_canceled"])


def build_model():
    categorical = Pipeline([("fill", SimpleImputer(strategy="constant", fill_value="missing")),
                            ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=50))])
    numeric = Pipeline([("fill", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler())])
    encode = ColumnTransformer([("cat", categorical, CATEGORICAL), ("num", numeric, NUMERIC)])
    return Pipeline([("encode", encode), ("model", LogisticRegression(max_iter=1000))])


def main():
    df = clean(load())
    X, y = df[CATEGORICAL + NUMERIC], df["is_canceled"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED)

    pipe = build_model().fit(X_train, y_train)
    proba = pipe.predict_proba(X_test)[:, 1]
    pred = (proba >= THRESHOLD).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
    baseline = accuracy_score(y_test, np.zeros_like(y_test))  # "nobody cancels"
    print(f"Model accuracy:     {accuracy_score(y_test, pred):.1%}  (threshold {THRESHOLD})")
    print()
    print(f"Confusion matrix ({len(pred):,} test bookings):")
    print(f"                        Predicted: stays   Predicted: cancels")
    print(f"  Actually stayed       {tn:>16,}   {fp:>18,}")
    print(f"  Actually cancelled    {fn:>16,}   {tp:>18,}")
    print()
    print(f"Bookings used:      {len(df):,}  (cancellation rate {y.mean():.1%})")
    print(f"Baseline accuracy:  {baseline:.1%}  (predict that nobody cancels)")
    

    # Revenue protected per year = correctly flagged cancellations x average daily rate.
    caught = int(((pred == 1) & (y_test == 1)).sum())
    years = df.groupby(["arrival_date_year", "arrival_date_month"]).ngroups / 12  # months of data / 12
    per_year = caught / 0.2 / years  # test set is 20% of all bookings
    revenue = per_year * df["adr"].mean()
    print(f"Cancellations caught in test set: {caught:,}")
    print(f"Revenue protected per year:       S${revenue:,.0f}")

    scores = df[["booking_id"]].copy()
    scores["risk_score"] = (pipe.predict_proba(X)[:, 1] * 100).round().astype(int)
    scores.to_csv("scores.csv", index=False)
    print(f"Wrote risk scores for {len(scores):,} bookings to scores.csv")


if __name__ == "__main__":
    main()
