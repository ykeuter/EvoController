import csv


class GenePoolAnalyzer:
    def read(self, fn, mi, ma):
        self.gene_pool = []
        with open("evo.log") as f:
            reader = csv.reader(f, delimiter="|")
            for row in reader:
                if row[2] == "age":
                    print(row[3])
                elif row[2] == "birth":
                    id = int(row[3])
                    if id >= mi and id <= ma:
                        self.gene_pool.append(row[6])

    def write(self, fn):
        with open(fn, "w") as f:
            for g in self.gene_pool:
                f.write(g + "\n")
