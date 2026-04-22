
from pdfminer.high_level import extract_text
import textract as tex

expdf = "/mnt/zmdata/home-media-app/data/input-data/txt/Berkeley/fb515169-3099-5213-809d-b9f52ab531e5/Scince Field day flyer.docx"
#"/mnt/zmdata/home-media-app/data/input-data/txt/Berkeley/f038b8fd-1e8c-53fa-8b1c-5a2243c6fe60/Kluwer - Fundamentals Of Cryptology.pdf"
#"/mnt/zmdata/home-media-app/data/input-data/txt/Berkeley/21d6c524-fe73-5118-b20e-1297ae057db6/other numerical integration examples.xls"
#"/mnt/zmdata/home-media-app/data/input-data/txt/Berkeley/7230d1da-34bd-551d-a257-fb99cf5fbbb2/chapter 04.pdf" 
#"/mnt/zmdata/home-media-app/data/input-data/txt/Berkeley/260b69a7-9a73-54ba-bc80-cac7f20f73a0/Professional Guide to TR0JANS, W0RMS, AND $PYW@RE.pdf"
#text = extract_text(expdf)

text = tex.process(expdf).decode("utf-8")
print(text)




