import csv
from difflib import SequenceMatcher
from pathlib import Path


def compute_wer(hyp, ref):
    substitutions = 0
    insertions = 0
    deletions = 0
    N = 0  # numero parole gold

    for (_, h), (_, r) in zip(hyp, ref):
        if r != "-":
            N += 1

        if h == r:
            continue
        elif h == "-":
            deletions += 1
        elif r == "-":
            insertions += 1
        else:
            substitutions += 1

    wer = (substitutions + deletions + insertions) / N if N > 0 else 0.0
    return wer


def extract_speaker_text(transcript_lines, speaker_id):
	"""
	Estrae e concatena tutto il testo di uno speaker.
	Restituisce una lista di tuple (speaker, parola) nell'ordine originale.
	"""
	words = []
	for line in transcript_lines:
		if line.startswith(speaker_id + "\t"):
			text = line.split("\t", 1)[1].strip()
			for w in text.split():
				words.append((speaker_id, w))
	return words


def align_words(seq_a, seq_b):
	"""
	Allinea due sequenze di parole con gap '-' quando non c'è match.
	Restituisce due liste allineate di tuple (speaker, parola) o ('-', '-') per i gap.
	"""
	words_a = [w for _, w in seq_a]
	words_b = [w for _, w in seq_b]

	matcher = SequenceMatcher(None, words_a, words_b)
	aligned_a = []
	aligned_b = []

	for tag, i1, i2, j1, j2 in matcher.get_opcodes():
		if tag == "equal":
			aligned_a.extend(seq_a[i1:i2])
			aligned_b.extend(seq_b[j1:j2])
		elif tag == "replace":
			length = max(i2 - i1, j2 - j1)
			# Estendi con gap se necessario
			extended_a = seq_a[i1:i2] + [("-", "-")] * (length - (i2 - i1))
			extended_b = seq_b[j1:j2] + [("-", "-")] * (length - (j2 - j1))
			aligned_a.extend(extended_a)
			aligned_b.extend(extended_b)
		elif tag == "delete":
			aligned_a.extend(seq_a[i1:i2])
			aligned_b.extend([("-", "-")] * (i2 - i1))
		elif tag == "insert":
			aligned_a.extend([("-", "-")] * (j2 - j1))
			aligned_b.extend(seq_b[j1:j2])

	return aligned_a, aligned_b


def process_file(file_entry):
	file_id = file_entry["file_id"]

	# Carica i file
	with open(file_entry["generated"], encoding="utf-8") as f:
		gen_lines = f.readlines()
	with open(file_entry["original"], encoding="utf-8") as f:
		orig_lines = f.readlines()

	# Trova tutti gli speaker
	speakers = []
	for line in gen_lines + orig_lines:
		if "\t" in line:
			spk = line.split("\t", 1)[0]
			if spk not in speakers:
				speakers.append(spk)  # mantiene l'ordine della conversazione

	# --- Allineamento generale (tutte le parole, rispettando l'ordine) ---
	all_words_gen = []
	all_words_orig = []
	for line in gen_lines:
		spk, text = line.strip().split("\t", 1)
		for w in text.split():
			all_words_gen.append((spk, w))
	for line in orig_lines:
		spk, text = line.strip().split("\t", 1)
		for w in text.split():
			all_words_orig.append((spk, w))

	aligned_gen, aligned_orig = align_words(all_words_gen, all_words_orig)

	# Salva il CSV generale
	general_csv = OUTPUT_DIR / f"{file_id}_alignment_all_speakers.csv"
	with open(general_csv, "w", encoding="utf-8", newline="") as f:
		writer = csv.writer(f, delimiter="\t")
		writer.writerow(["Speaker", "Gold", "Generated"])
		for (spk_g, w_g), (spk_o, w_o) in zip(aligned_orig, aligned_gen):
			writer.writerow([spk_o if spk_o != "-" else spk_g, w_o if w_o != "-" else "-", w_g if w_g != "-" else "-"])

	# --- CSV separati per speaker ---
	for speaker in speakers:
		speaker_gen = extract_speaker_text(gen_lines, speaker)
		speaker_orig = extract_speaker_text(orig_lines, speaker)

		aligned_speaker_gen, aligned_speaker_orig = align_words(speaker_gen, speaker_orig)

		speaker_csv = OUTPUT_DIR / f"{file_id}_alignment_{speaker}.csv"
		with open(speaker_csv, "w", encoding="utf-8", newline="") as f:
			writer = csv.writer(f, delimiter="\t")
			writer.writerow(["Speaker", "Gold", "Generated"])
			for (spk_g, w_g), (spk_o, w_o) in zip(aligned_speaker_orig, aligned_speaker_gen):
				writer.writerow([spk_o if spk_o != "-" else spk_g, w_o if w_o != "-" else "-", w_g if w_g != "-" else "-"])

	print(f"[DONE] {file_id} - CSV generati nella cartella {OUTPUT_DIR}")


if __name__ == "__main__":
	# --- Percorsi e configurazione ---
	FILES = [
		{
			"file_id": "BOC1002",
			"generated": "BOC1002/subs_final/BOC1002_clean_validated_normalized.txt",
			"original": "GOLD/BOC1002.txt",
		}
	]

	OUTPUT_DIR = Path("alignments")
	OUTPUT_DIR.mkdir(exist_ok=True)
	
	# --- Loop principale ---
	for f_entry in FILES:
		process_file(f_entry)
