from pathlib import Path
import csv

webanno_dir = Path("input/annotated")
vert_dir = Path("input/.vert.tsv")
output_dir = Path("output/vert")

output_dir.mkdir(parents=True, exist_ok=True)

for file_webanno in webanno_dir.glob("*.tsv"):

    conv_id = file_webanno.stem           # BOC1002
    file_vert = vert_dir / f"{conv_id}.vert.tsv"

    if not file_vert.exists():
        print(f"[SKIP] {file_vert} non trovato")
        continue

    print(f"[OK] processing {conv_id}")

    tokens_to_update = {}

    # Reads WebAnno file
    with open(file_webanno, encoding="utf-8") as fin:
        for line in fin:
            if line.strip() and not line.startswith("#"):
                cols = line.rstrip().split("\t")
                webanno_id = cols[0]
                if "." not in webanno_id:
                    _, _, form, backchannel, filler, _, _, repair, _, tok_id, _, _ = cols
                    phenomena = [x for x in [backchannel, filler, repair] if x != "_"]
                    if phenomena:
                        tokens_to_update[tok_id] = "|".join(phenomena) # casi in cui + di un'annotazione (esempio, una FP dentro a un SR)

    # Builds .vert file with annotations included
    out_path = output_dir / file_vert.name

    with open(file_vert, encoding="utf-8") as fin, \
         open(out_path, "w", encoding="utf-8", newline="") as fout:

        reader = csv.DictReader(fin, delimiter="\t")
        fieldnames = reader.fieldnames + ["event"]

        writer = csv.DictWriter(
            fout,
            fieldnames=fieldnames,
            delimiter="\t",
            restval="_"
        )
        writer.writeheader()

        for row in reader:
            if row["token_id"] in tokens_to_update:
                row["event"] = tokens_to_update[row["token_id"]]
            writer.writerow(row)
