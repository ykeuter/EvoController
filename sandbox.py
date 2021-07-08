from evo_controller.analyzers.log_analyzer import LogAnalyzer
import csv

# a = LogAnalyzer()
# a.read("evo.log")
# a.write("population.csv", "offspring.csv")

with open("evo.log") as f:
    reader = csv.reader(f, delimiter="|")
    for row in reader:
        print(row)
