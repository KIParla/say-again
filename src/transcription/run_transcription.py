import argparse
import subprocess
from pathlib import Path

from src.preprocessing import preprocess


PARAMS = {
    "wer": {
        "temperature": 0.143,
        "beam_size": 3,
        "best_of": 6,
        "no_speech_threshold": 0.203,
        "compression_ratio_threshold": 2.121,
        "patience": 0.592,
        "length_penalty": 0.479,
        "condition_on_previous_text": True,
        "model": "large-v3",
    },
    "event-based": {
        "temperature": 0.053,
        "beam_size": 7,
        "best_of": 2,
        "no_speech_threshold": 0.539,
        "compression_ratio_threshold": 1.734,
        "patience": 0.506,
        "length_penalty": 0.241,
        "condition_on_previous_text": False,
        "model": "large-v3",
    },
}


CONFIGS = {
    "gold": {
        "input_dir": Path("input/audio/gold"),
        "blocks_base": Path("blocks/gold"),
        "output_base": Path("output"),
        "project_middle": "A",
        "project_suffix": "_gold",
    },
    "subset_B": {
        "input_dir": Path("input/audio/audio_from20/subset_B"),
        "blocks_base": Path("blocks/subset_B"),
        "output_base": Path("output/subset_B"),
        "project_middle": None,
        "project_suffix": "",
    },
}


def build_project_dir(blocks_base, strategy, conv_id, project_middle, project_suffix):
    parts = [blocks_base, strategy]

    if project_middle is not None:
        parts.append(project_middle)

    parts.append(f"{conv_id}{project_suffix}")

    return Path(*parts)


def build_command(audio, project_dir, params):
    cmd = [
        "python", "src/pipeline/script.py",
        "--input", str(audio),
        "--project-dir", str(project_dir),
        "--exclusive-mode",
        "--auto-map",
        "--block-duration", "30m",
        "--min-speakers", "2",
        "--max-speakers", "2",
        "--min-duration", "0.25",
        "--min-pause", "2",
        "--model", params["model"],
    ]

    for key, value in params.items():
        if key == "model":
            continue

        if isinstance(value, bool):
            if value:
                cmd.append(f"--{key}")
        else:
            cmd.extend([f"--{key}", str(value)])

    return cmd


def main(dataset):
    config = CONFIGS[dataset]

    audios = sorted(config["input_dir"].glob("*.wav"))
    assert audios, f"No audio files found for dataset: {dataset}"

    for audio in audios:
        conv_id = audio.stem
        print(f"\nPROCESSING: {conv_id} ({dataset})")

        for strategy, params in PARAMS.items():
            print(f"\n--- {strategy.upper()} / {dataset} ---")

            project_dir = build_project_dir(
                config["blocks_base"],
                strategy,
                conv_id,
                config["project_middle"],
                config["project_suffix"],
            )
            project_dir.mkdir(parents=True, exist_ok=True)

            cmd = build_command(audio, project_dir, params)
            subprocess.run(cmd, check=True)

            source_txt = project_dir / "subs_final" / "podcast_clean_validated.txt"

            if not source_txt.exists():
                print(f"[WARNING] Missing transcript for {conv_id} ({strategy})")
                continue

            normalized_lines = preprocess.main(source_txt)

            out_dir = config["output_base"] / strategy
            out_dir.mkdir(parents=True, exist_ok=True)

            suffix = "wer" if strategy == "wer" else "event"
            out_txt = out_dir / f"{conv_id}_{suffix}.txt"

            with open(out_txt, "w", encoding="utf-8") as f:
                f.writelines(normalized_lines)

            print(f"[OK] Saved → {out_txt}")

    print(f"\nALL FILES COMPLETED: {dataset}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        choices=["gold", "subset_B"],
        required=True,
    )
    args = parser.parse_args()

    main(args.dataset)