import csv as csv
input_file = 'output_a.csv'
output_file = 'main.csv'


with open(input_file, 'r') as csv_input, open(output_file, 'w', newline='') as csv_output:
    reader = csv.reader(csv_input)
    writer = csv.writer(csv_output)
    for row in reader:
        writer.writerow(row)

print(f"{input_file} has been cloned to {output_file}")
