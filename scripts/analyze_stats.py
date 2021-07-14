from evo_controller.analyzers.stats_analyzer import StatsAnalyzer

a = StatsAnalyzer()
a.read("../evo.log")
a.write("../population.csv", "../offspring.csv")
