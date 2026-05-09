import csv
import re
from difflib import SequenceMatcher
from pathlib import Path

# =====================
# PATHS
# =====================

GOLD_VERT_DIR = Path("output/vert")

WHISPER_ROOTS = {
    "wer": Path("output/wer"),
    "event-based": Path("output/event-based"),
}

WHISPER_SUFFIX = {
    "wer": "wer",
    "event-based": "event"
}

OUT_ROOT = Path("alignments/gold")

# =====================
# NORMALIZATION
# =====================

def norm(tok):
    if tok is None:
        return "-"
    tok = tok.lower()
    if tok == "ok":
        tok = "okay"
    return tok

# =====================
# ALIGNMENT CORE
# =====================

def align_words(seq_a, seq_b):
    words_a = [norm(w) for _, w in seq_a]
    words_b = [norm(w) for _, w in seq_b]

    matcher = SequenceMatcher(None, words_a, words_b)
    out_a, out_b = [], []

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            out_a.extend(seq_a[i1:i2])
            out_b.extend(seq_b[j1:j2])
        elif tag == "replace":
            L = max(i2 - i1, j2 - j1)
            out_a.extend(seq_a[i1:i2] + [("-", "-")] * (L - (i2 - i1)))
            out_b.extend(seq_b[j1:j2] + [("-", "-")] * (L - (j2 - j1)))
        elif tag == "delete":
            out_a.extend(seq_a[i1:i2])
            out_b.extend([("-", "-")] * (i2 - i1))
        elif tag == "insert":
            out_a.extend([("-", "-")] * (j2 - j1))
            out_b.extend(seq_b[j1:j2])

    return out_a, out_b

# =====================
# LOADERS
# =====================

def load_vert(path):
    seq = []
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            if row.get("type") == "nonverbalbehavior":
                continue
            if row.get("speaker") in ("_", "???", None):
                continue
            token = row.get("form")
            if not token:
                continue
            seq.append((
                row["speaker"],
                token,
                row.get("event", "_")
            ))
    return seq

def load_whisper_txt(path):
    seq = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or "\t" not in line:
                continue
            speaker, utt = line.split("\t", 1)
            tokens = re.findall(r"\b\w+'?\w*\b", utt)
            for t in tokens:
                if t == "ok":
                    t = "okay"
                seq.append((speaker, t))
    return seq

# =====================
# ALIGN ONE CONVERSATION
# =====================

def align_one(conv_id, system):
    vert_path = GOLD_VERT_DIR / f"{conv_id}.vert.tsv"
    whisper_path = WHISPER_ROOTS[system] / f"{conv_id}_{WHISPER_SUFFIX[system]}.txt"

    if not vert_path.exists():
        print(f"[SKIP] {conv_id} – vert not found")
        return
    if not whisper_path.exists():
        print(f"[SKIP] {conv_id} ({system}) – whisper file not found")
        return

    gold_seq = load_vert(vert_path)
    whi_seq  = load_whisper_txt(whisper_path)

    aligned_gold, aligned_whi = align_words(
        [(s, w) for s, w, _ in gold_seq],
        whi_seq
    )

    out_dir = OUT_ROOT / system
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{conv_id}_{system}.align.tsv"

    with open(out_path, "w", encoding="utf-8", newline="") as fout:
        writer = csv.writer(fout, delimiter="\t")
        writer.writerow([
            "speaker_gold",
            "token_gold",
            "speaker_whisper",
            "token_whisper",
            "phenomenon"
        ])

        g_idx = 0
        for (sg, wg), (sw, ww) in zip(aligned_gold, aligned_whi):

            if wg != "-" and g_idx < len(gold_seq):
                phenomenon = gold_seq[g_idx][2]
                g_idx += 1
            else:
                phenomenon = "_"

            writer.writerow([
                sg if sg != "-" else "_",
                wg if wg != "-" else "_",
                sw if sw != "-" else "_",
                ww if ww != "-" else "_",
                phenomenon
            ])

    print(f"[OK] {out_path}")

# =====================
# MAIN
# =====================

def main():
    conv_ids = sorted(p.stem.replace(".vert", "") for p in GOLD_VERT_DIR.glob("*.vert.tsv"))

    for system in ("wer", "event-based"):
        print(f"\n=== {system.upper()} ===")
        for conv_id in conv_ids:
            align_one(conv_id, system)

if __name__ == "__main__":
    main()