from evo_controller.analyzers.log_analyzer import LogAnalyzer

a = LogAnalyzer()
a.read("evo.log")
a.write("population.csv", "offspring.csv")
