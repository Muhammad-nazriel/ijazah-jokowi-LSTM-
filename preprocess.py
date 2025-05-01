import re

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)  # hapus URL
    text = re.sub(r"@\w+", "", text)     # hapus mention
    text = re.sub(r"[^a-z\s]", "", text) # hapus non huruf
    text = re.sub(r"\s+", " ", text)     # hapus spasi ganda
    return text.strip()
