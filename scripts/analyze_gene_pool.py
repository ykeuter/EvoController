from evo_controller.analyzers.gene_pool_analyzer import GenePoolAnalyzer

a = GenePoolAnalyzer()
a.read("../evo.log", 800001, 800020)
a.write("../pop800k")
