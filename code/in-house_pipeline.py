"""Travelers' Inn: in-house cancellation risk pipeline.

Reads data/hotel.db, trains a model to predict whether a booking will be cancelled,
prints hold-out results and writes a 0-100 risk score for every booking to scores.csv.

Run:  python in_house_pipeline.py          (picks the threshold with the lowest mistake cost)
      python in_house_pipeline.py 0.5      (force a fixed threshold instead)

Cost of mistakes (from the case brief):
  False alarm  (flagged as cancel, guest actually comes):  room was resold -> relocate guest, S$500
  Missed cancellation (not flagged, guest cancels):        room sits empty -> lose ADR x nights
  Correct predictions cost nothing.

"""
import sqlite3
import sys

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, precision_score, recall_score,
                             roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DB_PATH = "data/hotel.db"
SEED = 42
FIXED_THRESHOLD = float(sys.argv[1]) if len(sys.argv) > 1 else None  # None = choose by cost

RELOCATION_COST = 500.0                    # S$ per false alarm (case brief)
THRESHOLDS = np.round(np.arange(0.05, 0.96, 0.01), 2)  # thresholds we try

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


def build_model(estimator):
    categorical = Pipeline([("fill", SimpleImputer(strategy="constant", fill_value="missing")),
                            ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=50,
                                                     sparse_output=False))])
    numeric = Pipeline([("fill", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler())])
    encode = ColumnTransformer([("cat", categorical, CATEGORICAL), ("num", numeric, NUMERIC)])
    return Pipeline([("encode", encode), ("model", estimator)])


MODELS = {
    "Logistic regression": LogisticRegression(max_iter=1000),
    "Gradient boosting":   HistGradientBoostingClassifier(random_state=SEED),
}


# ---------------------------------------------------------------- cost analysis
def booking_value(X, fill_adr):
    """Revenue lost if this booking cancels and the room is not resold: ADR x nights."""
    nights = (X["stays_in_weekend_nights"].fillna(0) + X["stays_in_week_nights"].fillna(0))
    adr = X["adr"].fillna(fill_adr)
    return (adr * nights.clip(lower=1)).to_numpy()  # a 0-night booking still blocks 1 night


def mistake_costs(y_true, pred, value):
    """Count and price the two kinds of mistake."""
    y_true, pred = np.asarray(y_true), np.asarray(pred)
    false_alarm = (pred == 1) & (y_true == 0)
    missed = (pred == 0) & (y_true == 1)
    fa_cost = false_alarm.sum() * RELOCATION_COST
    miss_cost = value[missed].sum()
    return {"false_alarms": int(false_alarm.sum()), "false_alarm_cost": fa_cost,
            "missed": int(missed.sum()), "missed_cost": miss_cost,
            "total_cost": fa_cost + miss_cost}


def cost_curve(y_true, proba, value):
    """Total mistake cost at every threshold in THRESHOLDS."""
    rows = [{"threshold": t, **mistake_costs(y_true, (proba >= t).astype(int), value)}
            for t in THRESHOLDS]
    return pd.DataFrame(rows)


def pick_threshold(y_true, proba, value):
    curve = cost_curve(y_true, proba, value)
    return float(curve.loc[curve["total_cost"].idxmin(), "threshold"]), curve


def money(x):
    return f"S${x:,.0f}"


def main():
    df = clean(load())
    X, y = df[CATEGORICAL + NUMERIC], df["is_canceled"].astype(int)

    # 60% train / 20% validation (choose threshold) / 20% test (report results).
    # The threshold is chosen on validation so the test result is not tuned to the test set.
    X_rest, X_test, y_rest, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED)
    X_train, X_val, y_train, y_val = train_test_split(
        X_rest, y_rest, test_size=0.25, stratify=y_rest, random_state=SEED)

    fill_adr = X_train["adr"].median()
    val_value, test_value = booking_value(X_val, fill_adr), booking_value(X_test, fill_adr)

    years = df.groupby(["arrival_date_year", "arrival_date_month"]).ngroups / 12
    to_year = 1 / 0.2 / years  # test set is 20% of bookings; scale to one year

    # "Do nothing" = flag nobody: every cancellation becomes a lost booking.
    do_nothing = mistake_costs(y_test, np.zeros(len(y_test), int), test_value)["total_cost"]

    print(f"Bookings used: {len(df):,}  (cancellation rate {y.mean():.1%})")
    print(f"Test set: {len(y_test):,} bookings, about {years:.1f} years of data")
    print(f"Cost of a false alarm: {money(RELOCATION_COST)}   "
          f"Average cost of a missed cancellation: {money(test_value[np.asarray(y_test) == 1].mean())}")
    print(f"Cost if we flag nobody (test set): {money(do_nothing)}")

    summary, fitted = [], {}
    for name, estimator in MODELS.items():
        pipe = build_model(estimator).fit(X_train, y_train)
        fitted[name] = pipe
        val_proba = pipe.predict_proba(X_val)[:, 1]
        test_proba = pipe.predict_proba(X_test)[:, 1]

        if FIXED_THRESHOLD is None:
            threshold, curve = pick_threshold(y_val, val_proba, val_value)
        else:
            threshold, curve = FIXED_THRESHOLD, cost_curve(y_val, val_proba, val_value)

        pred = (test_proba >= threshold).astype(int)
        cost = mistake_costs(y_test, pred, test_value)
        at_half = mistake_costs(y_test, (test_proba >= 0.5).astype(int), test_value)
        tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()

        print(f"\n=== {name} ===")
        print(f"AUC {roc_auc_score(y_test, test_proba):.3f}   threshold {threshold:.2f}   "
              f"accuracy {accuracy_score(y_test, pred):.1%}   "
              f"precision {precision_score(y_test, pred):.1%}   recall {recall_score(y_test, pred):.1%}")
        print(f"                        Predicted: stays   Predicted: cancels")
        print(f"  Actually stayed       {tn:>16,}   {fp:>18,}")
        print(f"  Actually cancelled    {fn:>16,}   {tp:>18,}")
        print(f"Mistake cost on test set:")
        print(f"  {cost['false_alarms']:>6,} false alarms x {money(RELOCATION_COST)}     = {money(cost['false_alarm_cost'])}")
        print(f"  {cost['missed']:>6,} missed cancellations x ADR x nights = {money(cost['missed_cost'])}")
        print(f"  Total = {money(cost['total_cost'])}   "
              f"(at the old 0.5 threshold: {money(at_half['total_cost'])})")

        print("Validation cost at a few thresholds:")
        show = curve[curve["threshold"].isin([0.3, 0.4, 0.5, 0.6, 0.7, threshold])]
        for _, r in show.iterrows():
            mark = "  <- chosen" if r["threshold"] == threshold else ""
            print(f"  {r['threshold']:.2f}: {money(r['total_cost']):>14}  "
                  f"({int(r['false_alarms']):,} false alarms, {int(r['missed']):,} missed){mark}")

        summary.append({"model": name, "threshold": threshold,
                        "test_cost": cost["total_cost"],
                        "saving_per_year": (do_nothing - cost["total_cost"]) * to_year})

    print("\n=== Summary (lower cost is better) ===")
    for s in summary:
        print(f"  {s['model']:<20} threshold {s['threshold']:.2f}   test cost {money(s['test_cost']):>14}   "
              f"saving vs flag-nobody about {money(s['saving_per_year'])}/year")
    best = min(summary, key=lambda s: s["test_cost"])
    print(f"Lowest cost: {best['model']} at threshold {best['threshold']:.2f}")

    # Score every booking with the cheapest model. Note: this includes bookings the model
    # was trained on, so those scores are optimistic. Use them for ranking, not evaluation.
    scores = df[["booking_id"]].copy()
    scores["risk_score"] = (fitted[best["model"]].predict_proba(X)[:, 1] * 100).round().astype(int)
    scores["flagged"] = (scores["risk_score"] >= best["threshold"] * 100).astype(int)
    scores.to_csv("scores.csv", index=False)
    print(f"Wrote risk scores for {len(scores):,} bookings to scores.csv")


if __name__ == "__main__":
    main()
