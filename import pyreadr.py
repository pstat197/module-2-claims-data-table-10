import pyreadr

result = pyreadr.read_r("results/preds_group10.RData")
# Acess the dataframe from the RData file
df = result[list(result.keys())[0]]
print()