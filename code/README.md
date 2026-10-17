# Travelers' Inn: In-House Cancellation Risk Model

**Owner:** Travelers' Inn Data Team

## What it does

The model reads our booking records from August 2023 to August 2025 (`data/hotel.db`) and predicts whether each booking will be cancelled. Every booking gets a **risk score from 0 to 100**. Bookings scoring 50 or above are flagged as likely to cancel.

## How to run it

You need Python 3.10 or later. Put the database at `data/hotel.db`, next to `in_house_pipeline.py`:

```
in_house_pipeline.py
requirements.txt
data/
  hotel.db
```

Install the libraries once, then run the pipeline from the same folder:

```bash
pip install -r requirements.txt
python in_house_pipeline.py          # flag bookings with a cancellation probability of 0.5 or more
python in_house_pipeline.py 0.35     # try a different threshold
```

A run takes about 10 seconds. Tested with Python 3.13, numpy 2.5.3, pandas 3.0.5 and scikit-learn 1.9.1.

The threshold decides how cautious the model is. A lower threshold flags more bookings: it catches more real cancellations but also raises more false alarms. A higher threshold does the opposite.

The run prints the results below and writes `scores.csv`, which has one row per booking: `booking_id, risk_score`.

## How we built it

1. **Cleaned the data.** Every column arrives as text, so we:
   - removed duplicate bookings
   - fixed inconsistent spellings ("No Depost", "Onlin TA")
   - turned placeholder values ("unknown", "-", blank, agent `999`) into missing values
   - converted numbers written as `276,00`, `SGD 172.50` or `14 days`
   - blanked impossible values, such as a negative lead time
2. **Left out columns the model shouldn't see.** `reservation_status` and its date record what happened to the booking, so including them would hand the model the answer. We also left out the room actually assigned (only known at check-in), mostly-empty columns, and identifiers.
3. **Prepared the data for the model.** Filled remaining gaps (most common category, or the median for numbers), turned categories into yes/no columns, and put numbers on a common scale.
4. **Trained a logistic regression model** on a random 80% of bookings, then **tested it on the other 20%**, which it had never seen.

## Results (on the 20% test set, threshold 0.5)

**Model accuracy: 82.6%** of the 23,640 test bookings predicted correctly.

**Confusion matrix:**

| | Predicted: stays | Predicted: cancels |
|---|---|---|
| **Actually stayed** | 13,646 | 1,388 |
| **Actually cancelled** | 2,726 | 5,880 |

| Measure | Result | What it means |
|---|---|---|
| Cancellation rate | 36.4% | Across 118,196 bookings |
| Baseline accuracy | 63.6% | What you'd get by predicting that nobody cancels |
| Precision | 80.9% | Of the bookings we flag, 80.9% really cancel |
| Recall | 68.3% | Of the bookings that really cancel, we flag 68.3% |
| AUC | 0.904 | How well the score ranks cancellers above non-cancellers (1.0 is perfect) |

## Business value

The model correctly flagged **5,880 cancellations** in the test set. That's about 14,100 per year across the full data (25 months of bookings).

**Revenue protected: about S$3.7 million per year**, which is correctly flagged cancellations × average daily rate (S$259.54).

## Next steps

- Send each morning's high-risk bookings to the reservations team for reconfirmation.
- Retrain every quarter.
