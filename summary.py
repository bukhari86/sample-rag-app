import csv

totals_by_department = {}

with open("sales.csv") as file:
    reader = csv.DictReader(file)
    for row in reader:
        department = row["department"]
        sales_value = int(row["sales"])

        if department in totals_by_department:
            totals_by_department[department] = totals_by_department[department] + sales_value
        else:
            totals_by_department[department] = sales_value

print(totals_by_department)