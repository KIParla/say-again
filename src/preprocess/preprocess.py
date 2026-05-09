import re
import string

from num2words import num2words

def digits_to_words(text, lang="it"):
    def repl(match):
        number = int(match.group())
        return num2words(number, lang=lang)
    return re.sub(r"[0-9]+", repl, text)

def main(file_path):

    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    normalized_lines = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        speaker_id, text = line.split(None, 1)
        #speaker_id, text = line.strip().split(' ', 1)
#        match = re.match(r'^([A-Z0-9_\[\]]+)\s+(.*)', line) # matches speaker + text
 #       if match:
  #          speaker_id = match.group(1)
   #         text = match.group(2)          
        text = text.lower()
        text = digits_to_words(text, lang="it")
        text = re.sub(r"[^\w\s']", "", text) #removes punctuation except apostrophes and question marks
        text = re.sub(r"\s+", " ", text).strip() # normalize spaces
        normalized_lines.append(f"{speaker_id}\t{text}\n") # final normalized line
#        else:
#            normalized_lines.append(line + "\n")

    return normalized_lines

if __name__ == "__main__":
    
    # Loading the file
    file_path = "BOC1002/subs_final/BOC1002_clean_validated.txt"
    
    normalized_lines = main(file_path)

    # Produce the normalized file
    
    output_path = file_path.replace(".txt", "_normalized.txt")
    with open(output_path, "w", encoding="utf-8") as f:
        f.writelines(normalized_lines)