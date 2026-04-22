
import docx

def extract_text(doc_file):
    doc = docx.Document(doc_file)
    full_text = [para.text for para in doc.paragraphs]
    for table in docx.table:
        for row in table.rows:
          for cell in row.cells:
            print(cell.text)
    print(full_text)


ex_docx = "/mnt/zmdata/home-media-app/data/input-data/txt/Berkeley/fb515169-3099-5213-809d-b9f52ab531e5/Scince Field day flyer.docx"
extract_text(ex_docx)