
from pdfminer.high_level import extract_text
import textract as tex

expdf = "/mnt/zmdata/home-media-app/data/input-data/txt/Berkeley/260b69a7-9a73-54ba-bc80-cac7f20f73a0/Professional Guide to TR0JANS, W0RMS, AND $PYW@RE.pdf"
#text = extract_text(expdf)

text = tex.process(expdf).decode("utf-8")
print(text)




