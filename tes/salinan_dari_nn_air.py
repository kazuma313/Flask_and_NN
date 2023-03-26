# import numpy as np
# import matplotlib.pyplot as plt
# import pandas as pd

# from Learning import normalize1,  normalize, denormalize, Layer, nn, mse, mape, learn


# df = pd.read_csv('airdata.csv')
# df


# x_train = []
# y_train = []

# x_test = []
# y_test = []

# x = []
# y = []

# for i in range(len(df["bulan"])):
#     x.append([
#         df["x1"][i],
#         df["x2"][i],
#         df["x3"][i],
#         df["x4"][i],
#         df["x5"][i],
#         df["x6"][i],
#         df["x7"][i],
#         df["x8"][i],
#         df["x9"][i],
#         df["x10"][i],
#         df["x11"][i],
#         df["x12"][i],
#         df["x13"][i],
#         df["x14"][i],
#     ])
#     y.append([df["target"][i]])


# x = np.array(x)
# transpose = x.T
# len(transpose)


# x_norm = []
# y_norm = normalize1(y)
# y_norm_1 = normalize1(y)


# for i in range(len(transpose)):
#     x_norm.append(normalize(transpose[i]))

# for i in range(len(x_norm)):
#     x_norm_temp_2 = []
#     x_norm_temp_3 = []
#     for j in range(len(x_norm[0])):
#         if(j < 21):
#             x_norm_temp_2.append(x_norm[i][j])
#         if(j >= 21):
#             x_norm_temp_3.append(x_norm[i][j])
#     x_train.append(x_norm_temp_2)
#     x_test.append(x_norm_temp_3)

# for i in range(len(y_norm)):
#     if(i < 21):
#         y_train.append(y_norm[i])
#     if(i >= 21):
#         y_test.append(y_norm[i])

# x_train = np.array(x_train)
# x_test = np.array(x_test)

# y_train = np.array(y_train)
# y_test = np.array(y_test)


# errors = learn(x_train=x_train, y_train=y_train, hidden1=6,
#                 hidden2=5, learningRate=0.02, epoch=500)

# print("Predicted output: \n" + str(nn.predict(x_train.T)))

# print(errors)


# plt.plot(errors, c = 'b', label = 'Error')
# plt.title('Historical Error')
# plt.xlabel('Epochs')
# plt.ylabel('Error')
# plt.legend()
# plt.show()

# print("Predicted output: \n" + str(nn.predict(x_test.T)))


# arr_1 = y_norm_1
# predict_1 = nn.predict(x_test.T)
# for i in range(len(predict_1)):
#     arr_1.append(predict_1[i].tolist())

# denormalize_1 = denormalize(arr_1, y)

# predicted_1 = []
# actual_1 = []
# predicted_2 = []
# actual_2 = []

# for i in range(len(denormalize_1)):
#     if(i >= 21 and i <= 28):
#         actual_1.append(denormalize_1[i])
#         actual_2.append(arr_1[i])
#     if(i >= len(y_norm)):
#         predicted_1.append(denormalize_1[i])
#         predicted_2.append(arr_1[i])


# plt.figure()
# plt.title("Aktual vs Prediksi")
# plt.plot(actual_2, c='b', label="Aktual")
# plt.plot(predicted_2, c='y', label="Prediksi")
# plt.ylabel('Value')
# plt.xlabel('Data Ke-')
# plt.legend()
# plt.show()

# mse = mse(actual_2, predicted_2)
# print(mse)

# data = {'errors':
#         {
#             'errors': errors,
#             'index': np.arange(0, np.shape(errors)[0]).tolist(),
#         },
#         'perbandingan':
#         {
#             'aktual': np.array(actual_2).ravel().tolist(),
#             'predicted': np.array(predicted_2).ravel().tolist(),
#             'index' : np.arange(0, np.shape(np.array(predicted_2).ravel())[0]).tolist()
#         }
#         }

# print(f"Sebelum actual = {np.array(actual_2)}")
# print(f"Actual = {np.array(actual_2).ravel()}")
# print(mape(actual_2, predicted_2))
