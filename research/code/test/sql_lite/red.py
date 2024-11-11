import csv
import json

csvf = open('static-metadata.csv', 'r')
jsonf = open('f.json', 'w')
fn = ('name','desc','lat','lon')

reader =  csv.DictReader(csvf, fn)

for row in reader:
  json.dump(row, jsonf)
  jsonf.write('\n')





