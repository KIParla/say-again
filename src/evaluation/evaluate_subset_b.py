import csv
import re
import pandas as pd
from difflib import SequenceMatcher
from pathlib import Path
from collections import defaultdict
import preprocess

# =========================
# CONFIGURAZIONE PERCORSI
# =========================

ASR_ROOT = Path("data/transcripts/subset_B")
GOLD_ROOT = Path("data/input/GOLD/from20min/subset_B")


ALIGNMENT_TYPES = ["wer", "event-based"]
CONFIGURATIONS = ["A", "B", "C", "D"]

EVENT_TOKENS = {
    "eh", "ehm", "ah", "oh",
    "mh", "mhmh", "mm",
    "okay", "sì", "bene", "esatto",
    "cioè", "diciamo", "insomma"
}

OUT_EVENT = "output/subset_B/subset_B_scores.csv"
OUT_WER = "output/subset_B/subsetB_wer_summary.csv"

# =========================
# HELPERS
# =========================

def align(seq_h, seq_r):
    matcher = SequenceMatcher(None, seq_h, seq_r)
    h_out, r_out = [], []

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            h_out.extend(seq_h[i1:i2])
            r_out.extend(seq_r[j1:j2])
        elif tag == "replace":
            L = max(i2 - i1, j2 - j1)
            h_out.extend(seq_h[i1:i2] + ["-"] * (L - (i2 - i1)))
            r_out.extend(seq_r[j1:j2] + ["-"] * (L - (j2 - j1)))
        elif tag == "delete":
            h_out.extend(seq_h[i1:i2])
            r_out.extend(["-"] * (i2 - i1))
        elif tag == "insert":
            h_out.extend(["-"] * (j2 - j1))
            r_out.extend(seq_r[j1:j2])

    return h_out, r_out


def classify(g, h):
    if g != "-" and h == "-":
        return "DEL"
    if g == "-" and h != "-":
        return "INS"
    if g == h:
        return "OK"
    return "SUB"


def load_words(path):
    lines = preprocess.main(path)
    words = []
    for l in lines:
        _, txt = l.split("\t", 1)
        words.extend(txt.split())
    return words


# =========================
# MAIN
# =========================

event_rows = []
wer_rows = []

for cfg in CONFIGURATIONS:
    for align_type in ALIGNMENT_TYPES:

        counts = {"DEL": 0, "SUB": 0, "OK": 0, "INS": 0}
        wer_sum = 0
        event_wer_sum = 0
        n_conv = 0

        cfg_dir = ASR_ROOT / align_type / cfg
        if not cfg_dir.exists():
            continue

        for conv_dir in cfg_dir.iterdir():
            conv = conv_dir.name.replace("_from20", "")
            asr_file = conv_dir / "subs_final/podcast_clean_validated.txt"
            gold_file = GOLD_ROOT / f"{conv}.txt"

            if not asr_file.exists() or not gold_file.exists():
                continue

            hyp = load_words(asr_file)
            ref = load_words(gold_file)

            h_al, r_al = align(hyp, ref)

            S = D = I = C = 0
            eS = eD = eI = eC = 0
            eN = 0

            for h, r in zip(h_al, r_al):
                op = classify(r, h)

                if op == "SUB": S += 1
                elif op == "DEL": D += 1
                elif op == "INS": I += 1
                elif op == "OK": C += 1

                if r in EVENT_TOKENS or h in EVENT_TOKENS:
                    eN += 1
                    if op == "SUB": eS += 1
                    elif op == "DEL": eD += 1
                    elif op == "INS": eI += 1
                    elif op == "OK": eC += 1

            N = S + D + C
            if N == 0 or eN == 0:
                continue

            wer_sum += (S + D + I) / N
            event_wer_sum += (eS + eD + eI) / eN

            counts["DEL"] += eD
            counts["SUB"] += eS
            counts["OK"]  += eC
            counts["INS"] += eI

            n_conv += 1

        if n_conv == 0:
            continue

        total = sum(counts.values())

        event_rows.append([
            cfg, align_type,
            counts["DEL"] / total * 100,
            counts["SUB"] / total * 100,
            counts["OK"]  / total * 100,
            counts["INS"] / total * 100
        ])

        wer_rows.append([
            cfg, align_type,
            wer_sum / n_conv,
            event_wer_sum / n_conv
        ])

# =========================
# SAVE
# =========================

with open(OUT_EVENT, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["configuration", "alignment_type", "DEL", "SUB", "OK", "INS"])
    w.writerows(event_rows)

with open(OUT_WER, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["configuration", "alignment_type", "WER", "EVENT_WER"])
    w.writerows(wer_rows)