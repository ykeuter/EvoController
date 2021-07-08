from evo_controller.analyzers.log_analyzer import LogAnalyzer
import csv

a = LogAnalyzer()
a.read("evo.log")
a.write("population.csv", "offspring.csv")

# with open("evo.log") as f:
#     reader = csv.reader(f, delimiter="|")
#     for row in reader:
#         # print(row)
#         if row[2] == "age":
#             print("age {}".format(row[3]))
#         elif row[2] == "birth":
#             id = int(row[3])
#             if id > 821700 and id < 821750:
#                 print("birth {} {}".format(row[3], row[4]))
