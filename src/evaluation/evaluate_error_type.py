import os
import pandas as pd

# =========================
# CONFIGURATION
# =========================

SUBSET = "subset_A"   

ALIGNMENT_ROOTS = {
    "wer": "alignments/wer",
    "event-based": "alignments/event-based"
}

OUTPUT_DIR = f"alignments/{SUBSET}/analysis"
os.makedirs(OUTPUT_DIR, exist_ok=True)

lambda_event = 0.3
mu_event = 0.1

# =========================
# EVENT TOKENS
# =========================

EVENT_TOKENS = {
    "eh", "ehm", "ah", "oh",
    "mh", "mhmh", "mm",
    "okay", "sì", "bene", "esatto",
    "cioè", "diciamo", "insomma"
}

# =========================
# HELPERS
# =========================

def classify_op(gold, hyp):
    if gold != "-" and hyp == "-":
        return "DEL"
    elif gold == "-" and hyp != "-":
        return "INS"
    elif gold == hyp:
        return "OK"
    else:
        return "SUB"


def compute_wer(df):
    substitutions = 0
    insertions = 0
    deletions = 0
    N = 0

    for _, row in df.iterrows():
        g = row["gold_word"]
        h = row["whisper_word"]

        if g != "-":
            N += 1

        if g == h:
            continue
        elif h == "-":
            deletions += 1
        elif g == "-":
            insertions += 1
        else:
            substitutions += 1

    return (substitutions + deletions + insertions) / N if N > 0 else 0.0


def process_alignment_file(path):
    df = pd.read_csv(path, sep="\t")

    df["operation"] = df.apply(
        lambda r: classify_op(r.gold_word, r.whisper_word),
        axis=1
    )

    df["is_event"] = (
        df.gold_word.isin(EVENT_TOKENS)
        | df.whisper_word.isin(EVENT_TOKENS)
    )

    return df


# =========================
# MAIN ANALYSIS
# =========================

rows = []

for system, folder in ALIGNMENT_ROOTS.items():

    if not os.path.exists(folder):
        print(f"[WARNING] Missing folder: {folder}")
        continue

    for root, _, files in os.walk(folder):
        for fname in files:

            if not fname.endswith(".align.tsv"):
                continue

            path = os.path.join(root, fname)
            conversation = fname.replace(".align.tsv", "")

            # configuration = nome cartella A/B/C/D
            configuration = os.path.basename(root)
            if configuration not in ["A", "B", "C", "D"]:
                continue

            df = process_alignment_file(path)

            # --- WER STANDARD ---
            wer_value = compute_wer(df)

            # --- EVENT ANALYSIS ---
            df_events = df[df.is_event]

            if not df_events.empty:
                counts = df_events["operation"].value_counts().to_dict()
                total = sum(counts.values())

                del_pct = counts.get("DEL", 0) / total * 100
                sub_pct = counts.get("SUB", 0) / total * 100
                ok_pct  = counts.get("OK",  0) / total * 100
                ins_pct = counts.get("INS", 0) / total * 100
            else:
                del_pct = sub_pct = ok_pct = ins_pct = 0.0

            row = {
                "conversation": conversation,
                "configuration": configuration,
                "alignment_type": system,
                "WER": wer_value,
                "DEL": del_pct,
                "SUB": sub_pct,
                "OK":  ok_pct,
                "INS": ins_pct,
            }

            rows.append(row)

# =========================
# SAVE PER-CONVERSATION
# =========================

df_out = pd.DataFrame(rows)

df_out = df_out[
    ["conversation", "configuration", "alignment_type",
     "WER", "DEL", "SUB", "OK", "INS"]
]

df_out = df_out.sort_values(
    by=["configuration", "conversation", "alignment_type"]
)

comparison_path = os.path.join(
    OUTPUT_DIR,
    "comparison_by_conversation.csv"
)

df_out.to_csv(comparison_path, index=False)

print(f"Saved: {comparison_path}")

# =========================
# AGGREGATED SUMMARY
# =========================

summary = (
    df_out
    .groupby(["configuration", "alignment_type"])
    [["WER", "DEL", "SUB", "OK", "INS"]]
    .mean()
    .reset_index()
)

summary_path = os.path.join(
    OUTPUT_DIR,
    "comparison_summary.csv"
)

summary.to_csv(summary_path, index=False)

print(f"Saved: {summary_path}")

# =========================
# EVENT-BASED LOSS TABLE
# =========================

loss_rows = []

for _, row in df_out.iterrows():

    if row["alignment_type"] != "event-based":
        continue

    suppression = (row["DEL"] + row["SUB"]) / 100
    production  = row["OK"] / 100

    loss = (
        row["WER"]
        + lambda_event * suppression
        - mu_event * production
    )

    loss_rows.append({
        "conversation": row["conversation"],
        "configuration": row["configuration"],
        "WER": row["WER"],
        "suppression": suppression,
        "production": production,
        "loss": loss
    })

df_loss = pd.DataFrame(loss_rows)

loss_path = os.path.join(
    OUTPUT_DIR,
    "event_loss_by_conversation.csv"
)

df_loss.to_csv(loss_path, index=False)

# =========================
# EVENT LOSS SUMMARY PER CONFIGURATION
# =========================

loss_summary = (
    df_loss
    .groupby("configuration")
    [["loss", "WER", "suppression", "production"]]
    .mean()
    .reset_index()
)

loss_summary_path = os.path.join(
    OUTPUT_DIR,
    "event_loss_summary_by_configuration.csv"
)

loss_summary.to_csv(loss_summary_path, index=False)

print(f"Saved: {loss_summary_path}")