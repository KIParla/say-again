. /opt/miniconda3/load_conda.sh && conda activate diarize-c2
#python script.py --exclusive-mode  --auto-map  --block-duration 30m --min-speakers 2 --max-speakers 2  --input audio20min/BOC1002.mp3  --project-dir BOC1002_minduration5_minpause2 --min-duration 0.5 --min-pause 2
#python script.py --exclusive-mode  --auto-map  --block-duration 30m --min-speakers 2 --max-speakers 2  --input audio20min/BOC1003.mp3  --project-dir BOC1003_minduration5_minpause2 --min-duration 0.5 --min-pause 2
#python script.py --exclusive-mode  --auto-map  --block-duration 30m --min-speakers 2 --max-speakers 2  --input audio20min/BOC1005.mp3  --project-dir BOC1005_minduration5_minpause2 --min-duration 0.5 --min-pause 2

AUDIO_DIR="audio_from20/subset_A"
RESULTS_DIR="eventi_configuration_results"

mkdir -p ${RESULTS_DIR}/A
mkdir -p ${RESULTS_DIR}/B
mkdir -p ${RESULTS_DIR}/C
mkdir -p ${RESULTS_DIR}/D

# CONFIG A
for audio in ${AUDIO_DIR}/*.wav; do
  base=$(basename $audio .wav)
    python script.py --exclusive-mode --auto-map  --block-duration 30m --min-speakers 2 --max-speakers 2  --input $audio  --project-dir ${RESULTS_DIR}/A/${base}_A --min-duration 0.25 --min-pause 2  
done

# CONFIG B

for audio in ${AUDIO_DIR}/*.wav; do
  base=$(basename $audio .wav)
    python script.py --auto-map  --block-duration 30m --min-speakers 2 --max-speakers 2  --input $audio   --project-dir ${RESULTS_DIR}/B/${base}_B --min-duration 0.25 --min-pause 2  
done

# CONFIG C

for audio in ${AUDIO_DIR}/*.wav; do
  base=$(basename $audio .wav)
    python script.py --auto-map  --block-duration 30m --min-speakers 2 --max-speakers 2  --input $audio   --project-dir ${RESULTS_DIR}/C/${base}_C --min-duration 0.25 --min-pause 1  
done

# CONFIG D

for audio in ${AUDIO_DIR}/*.wav; do
  base=$(basename $audio .wav)
    python script.py --auto-map  --block-duration 30m --min-speakers 2 --max-speakers 2  --input $audio   --project-dir ${RESULTS_DIR}/D/${base}_D --min-duration 0.20 --min-pause 1  
done