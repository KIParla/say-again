import pandas as pd
import glob
import os
import re
import pympi

# =====================
# PATHS
# =====================
ANNOTATED_DIR = "data/input/annotated"
ELAN_DIR = "input/.eaf"
META_PATH = "conversations.csv"
OUTPUT_DIR = "data/output/stats"

os.makedirs(OUTPUT_DIR, exist_ok=True)

NORM_OUT = os.path.join(OUTPUT_DIR, "speaker_stats_normalized.csv")
INTERACTION_OUT = os.path.join(OUTPUT_DIR, "interaction_stats_normalized.csv")
INTERACTION_CONV_OUT = os.path.join(OUTPUT_DIR, "interaction_stats.csv")

MAX_TIME = 20 * 60 * 1000  # 20 minuti in ms

# =====================
# EVENT REGEX
# =====================
event_pattern = re.compile(r"^(BC|FP|SR|OR)(?:\[(\d+)\])?$")

all_events = []

# =====================
# 1. EXTRACT EVENTS
# =====================
tsv_files = glob.glob(os.path.join(ANNOTATED_DIR, "*.tsv"))

for tsv_path in tsv_files:

    conversation_id = os.path.basename(tsv_path).replace(".tsv", "")

    df = pd.read_csv(
        tsv_path,
        sep="\t",
        comment="#",
        header=None,
        dtype=str
    )

    df.columns = [f"col_{i}" for i in range(df.shape[1])]

    # ---------- SPEAKER ----------
    speaker_col = None
    for c in df.columns:
        if df[c].dropna().str.match(r"[A-Z]{2,}\d+", na=False).any():
            speaker_col = c
            break

    if speaker_col is None:
        continue

    df[speaker_col] = df[speaker_col].replace("_", pd.NA).ffill()
    df = df[df[speaker_col] != "???"]

    # ---------- EVENT COLUMNS ----------
    event_cols = []
    for c in df.columns:
        if df[c].dropna().str.match(r"(BC|FP|SR|OR)(\[\d+\])?$", na=False).any():
            event_cols.append(c)

    # ---------- EXTRACT EVENTS ----------
    for _, row in df.iterrows():

        speaker = row[speaker_col]
        token_id = row["col_0"]

        for c in event_cols:
            val = row[c]

            if not isinstance(val, str) or val == "_":
                continue

            m = event_pattern.match(val)

            if not m:
                continue

            event_type, span_id = m.groups()

            if span_id:
                event_key = f"{conversation_id}_{event_type}_{span_id}"
            else:
                event_key = f"{conversation_id}_{event_type}_token_{token_id}"

            all_events.append({
                "conversation_id": conversation_id,
                "speaker": speaker,
                "event_type": event_type,
                "event_key": event_key
            })

# =====================
# 2. COUNTS PER SPEAKER
# =====================
events_df = pd.DataFrame(all_events).drop_duplicates()

speaker_raw = (
    events_df
    .groupby(["conversation_id", "speaker", "event_type"])
    .size()
    .unstack(fill_value=0)
    .reset_index()
)

for col in ["BC", "FP", "SR", "OR"]:
    if col not in speaker_raw.columns:
        speaker_raw[col] = 0

speaker_raw = speaker_raw[
    ["conversation_id", "speaker", "BC", "FP", "SR", "OR"]
]

# =====================
# 3. SPEECH TIME FROM EAF
# =====================
rows = []

for eaf_path in glob.glob(os.path.join(ELAN_DIR, "*.eaf")):

    conversation_id = os.path.basename(eaf_path).replace(".eaf", "")
    eaf = pympi.Elan.Eaf(eaf_path)

    for tier in eaf.get_tier_names():

        total_ms = 0

        for start, end, _ in eaf.get_annotation_data_for_tier(tier):

            if start >= MAX_TIME:
                continue

            if end <= MAX_TIME:
                total_ms += end - start
            else:
                total_ms += MAX_TIME - start

        if total_ms > 0:
            rows.append({
                "conversation_id": conversation_id,
                "speaker": tier,
                "speech_minutes": total_ms / 60000
            })

speech_time = pd.DataFrame(rows)

# =====================
# 4. NORMALIZATION PER SPEAKER
# =====================
speaker_norm = speaker_raw.merge(
    speech_time,
    on=["conversation_id", "speaker"],
    how="left"
)

for col in ["BC", "FP", "SR", "OR"]:
    speaker_norm[f"{col}_per_min_speech"] = (
        speaker_norm[col] / speaker_norm["speech_minutes"]
    )

speaker_norm.to_csv(NORM_OUT, index=False)

# =====================
# 5. AGGREGATION PER INTERACTION TYPE
# =====================
meta = pd.read_csv(META_PATH, sep=";")
meta.columns = meta.columns.str.strip().str.lower().str.replace(" ", "_")

meta["interaction_type_norm"] = meta["type"].str.lower()

meta.loc[
    meta["interaction_type_norm"].str.contains("esame", na=False),
    "interaction_type_norm"
] = "esami_ricevimento"

meta.loc[
    meta["interaction_type_norm"].str.contains("ricevimento studenti", na=False),
    "interaction_type_norm"
] = "esami_ricevimento"

interaction_stats = (
    speaker_norm
    .merge(meta, on="conversation_id", how="left")
    .groupby("interaction_type_norm", as_index=False)
    [["BC", "FP", "SR", "OR"]]
    .sum()
)

interaction_stats.to_csv(INTERACTION_OUT, index=False)

# =====================
# 6. CONVERSATION-LEVEL COUNTS + PER MIN
# =====================
conv_raw = (
    speaker_raw
    .groupby("conversation_id", as_index=False)[["BC", "FP", "SR", "OR"]]
    .sum()
)

conv_meta = conv_raw.merge(
    meta[["conversation_id", "interaction_type_norm"]],
    on="conversation_id",
    how="left"
)

interaction_conv = (
    conv_meta
    .groupby("interaction_type_norm", as_index=False)[["BC", "FP", "SR", "OR"]]
    .sum()
)

n_conv = (
    conv_meta[["conversation_id", "interaction_type_norm"]]
    .drop_duplicates()
    .groupby("interaction_type_norm")
    .size()
    .reset_index(name="n_conversations")
)

interaction_conv = interaction_conv.merge(
    n_conv,
    on="interaction_type_norm"
)

for col in ["BC", "FP", "SR", "OR"]:
    interaction_conv[f"{col}_per_min"] = (
        interaction_conv[col] / (interaction_conv["n_conversations"] * 20)
    )

interaction_conv.to_csv(INTERACTION_CONV_OUT, index=False)

# =====================
# DONE
# =====================
print("Saved:")
print(" -", NORM_OUT)
print(" -", INTERACTION_OUT)
print(" -", INTERACTION_CONV_OUT)