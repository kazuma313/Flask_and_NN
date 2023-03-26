import numpy as np
import pandas as pd



data = np.array([[0.5018387735891494], [0.5046259034127578], [0.4994304041372873], [0.5040975736084524], [0.5073299217538548], [0.5045606085467208], [0.49602092326917546], [0.4945926850397742]])
data2 = np.array([[0.5018387735891494], [0.5046259034127578], [0.4994304041372873], [0.5040975736084524], [0.5073299217538548], [0.5045606085467208], [0.49602092326917546], [0.4945926850397742]])

df = np.array([data.flatten(), data2.flatten()]).T
print(np.shape(df))
print(df)
print(np.shape(data.flatten))

print(pd.DataFrame(data=df, columns=["A", "B"]))
