import pandas as pd
basetable = pd.read_csv("https://assets.datacamp.com/production/repositories/1441/datasets/7abb677ec52631679b467c90f3b649eb4f8c00b2/basetable_ex2_4.csv")
basetable.to_csv("datasets/basetable.csv",index=False)