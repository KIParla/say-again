import os
from pathlib import Path
import preprocess
import align

# ======================
# PATHS
# ======================

SUBSET = "subset_B"

GOLD_DIR = Path(f"input/GOLD/from20min/{SUBSET}")
WHISPER_BASE = Path(f"output/{SUBSET}")
ALIGN_BASE = Path(f"alignments/{SUBSET}")

SYSTEMS = ["wer", "event-based"]

# ======================
# ALIGNMENT FUNCTION
# ======================

def align_file(gold_path, whisper_path, out_path):

    # Preprocess Whisper (già fatto prima, ma lo teniamo coerente)
    normalized_whisper = preprocess.main(whisper_path)

    with open(gold_path, encoding="utf-8") as f:
        gold_lines = f.readlines()

    # Split speaker + text
    content_gold = []
    for line in gold_lines:
        line = line.strip()
        if not line:
            continue
        spk, text = line.split("\t", 1)
        content_gold.append((spk, text))

    content_whisper = []
    for line in normalized_whisper:
        parts = line.strip().split("\t", 1)
        if len(parts) == 2:
            content_whisper.append(parts)

    # Word-level lists
    all_words_gold = []
    for spk, text in content_gold:
        for w in text.split():
            all_words_gold.append((spk, w))

    all_words_whisper = []
    for spk, text in content_whisper:
        for w in text.split():
            all_words_whisper.append((spk, w))

    # Align
    aligned_whisper, aligned_gold = align.align_words(all_words_whisper, all_words_gold)

    # Save .align.tsv
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("gold_speaker\tgold_word\twhisper_speaker\twhisper_word\n")
        for (g_spk, g_w), (w_spk, w_w) in zip(aligned_gold, aligned_whisper):
            f.write(f"{g_spk}\t{g_w}\t{w_spk}\t{w_w}\n")

    print(f"[OK] Saved → {out_path}")


# ======================
# MAIN LOOP
# ======================

for system in SYSTEMS:

    whisper_dir = WHISPER_BASE / system
    align_dir = ALIGN_BASE / system
    align_dir.mkdir(parents=True, exist_ok=True)

    for whisper_file in whisper_dir.glob("*.txt"):

        conv_id = whisper_file.stem.split("_")[0]
        gold_file = GOLD_DIR / f"{conv_id}.txt"

        if not gold_file.exists():
            print(f"[WARNING] Missing GOLD for {conv_id}")
            continue

        out_file = align_dir / f"{conv_id}.align.tsv"

        align_file(gold_file, whisper_file, out_file)

print("\n✓ All alignments completed")
