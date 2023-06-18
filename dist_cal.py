# %%
import os
import numpy as np
import matplotlib.pyplot as plt
file_path = "/home/henry/myproject/depth_estimation"
os.chdir(file_path)
if_dynamic = 'All/' # Dynamic or nonDynamic
data_path = 'data/result_dis/'
file_list = os.listdir(data_path + if_dynamic)
# %%
record_all = np.array([])
for record_name in file_list:
    record = np.load(data_path + if_dynamic + record_name)
    print(len(record))
    if record_all.shape[0] == 0:
        record_all = record
    else:
        record_all = np.concatenate((record_all,record), axis=0)
# %%
index = np.where((record_all[:,2] > 6))
record_all = record_all[index]
index = np.where((record_all[:,2] <= 10))
record_all = record_all[index]
# detction chance
successful_detection_times = np.sum(record_all[:,0] == 1)
successful_rate = successful_detection_times/record_all[np.where(record_all[:,2] < 10),0].shape[1]
print(f'successful_rate: {successful_rate} ')
# %%
# Threshold
index = record_all[:,0] == 1
detected_frame = record_all[index]
theshold_o = 1.25
est_over_gt = detected_frame[:,1] / detected_frame[:,2]
gt_over_est = detected_frame[:,2] / detected_frame[:,1]
for i in range(3):
    theshold = theshold_o**(i+1)
    index = ((est_over_gt < theshold) * (gt_over_est < theshold))>0
    print(f'threshold: {i+1}, percentage: {np.sum(index)/detected_frame.shape[0]}')
mae = np.mean(np.abs(np.array(gt_over_est) - np.array(est_over_gt)))
print(f"Mean Absolute Error: {mae}")
squared_relative_error = ((est_over_gt - gt_over_est) / est_over_gt) ** 2
mean_squared_relative_error = np.mean(squared_relative_error)
print("Mean Squared Relative Error:", mean_squared_relative_error)
mse = np.mean((est_over_gt - gt_over_est) ** 2)
# Calculate the root mean squared error
rmse = np.sqrt(mse)
print(f"Root Mean Squared Error: {rmse}")
squared_log_error = (np.log1p(est_over_gt) - np.log1p(gt_over_est)) ** 2
# Calculate the mean squared logarithmic error
msle = np.mean(squared_log_error)
# Calculate the root mean squared logarithmic error
rmsle = np.sqrt(msle)
print(f"Root Mean Squared Logarithmic Error: {rmsle}")
# %%
# Meansquare
np.squeeze(detected_frame[np.where(detected_frame[:,2] < 10),2])
# %%
plt.hist(np.squeeze(detected_frame[np.where(detected_frame[:,2] < 10),2]), bins=50)
# %%
plt.hist(np.squeeze(record_all[np.where(record_all[:,0] == -1),2]), bins=50)
# %%