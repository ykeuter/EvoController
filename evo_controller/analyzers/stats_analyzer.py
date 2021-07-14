import csv
from collections import Counter
import pandas as pd
import numpy as np


class StatsAnalyzer:
    def read(self, fn):
        with open(fn) as f:
            reader = csv.reader(f, delimiter="|")
            nbirth = 0
            ndeath = 0
            counter = Counter()
            df = pd.DataFrame(columns=["age", "births", "deaths"])
            for row in reader:
                if row[2] == "age":
                    print(row[3])
                    d = {
                        "age": int(float(row[3])),
                        "births": nbirth,
                        "deaths": ndeath
                    }
                    df = df.append(d, ignore_index=True)
                    nbirth = 0
                    ndeath = 0
                elif row[2] == "birth":
                    nbirth += 1
                    counter[int(row[4])] += 1
                elif row[2] == "death":
                    ndeath += 1
                else:
                    raise ValueError

        ma = max(counter.keys())
        df2 = pd.DataFrame(
            columns=["id", "min", "q1", "med", "q3", "max", "avg"])
        for i in range(1000, ma, 1000):
            a = [counter[j] for j in range(i - 1000, i)]
            q = np.quantile(a, [0, .25, .5, .75, 1])
            d = {
                "id": i,
                "min": q[0],
                "q1": q[1],
                "med": q[2],
                "q3": q[3],
                "max": q[4],
                "avg": np.mean(a)
            }
            df2 = df2.append(d, ignore_index=True)

        self.population_stats = df
        self.offspring_stats = df2

    def write(self, population_fn, offspring_fn):
        self.population_stats.to_csv(population_fn, index=False)
        self.offspring_stats.to_csv(offspring_fn, index=False)
