from evo_controller.codecs.default_genome_decoder import DefaultGenomeDecoder
import csv


with open("evo.log") as f:
    with open("pop800k.txt", "w") as f2:
        reader = csv.reader(f, delimiter="|")
        for row in reader:
            if row[2] == "age":
                print(row[3])
            elif row[2] == "birth":
                id = int(row[3])
                if id > 800000 and id <= 800020:
                    f2.write(row[6] + "\n")
