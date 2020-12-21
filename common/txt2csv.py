import csv

csv_path = r'F:\python_code\NLPTools\data\THUCNews.csv'
txt_path = r'F:\python_code\NLPTools\data\THUCNews.txt'

csvFile = open(csv_path, 'w', newline='', encoding='utf-8')
writer = csv.writer(csvFile)
csvRow = []

f = open(txt_path, 'r', encoding='utf-8')
for line in f:
    csvRow = line.split()
    writer.writerow(csvRow)

f.close()
csvFile.close()