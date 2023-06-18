import matplotlib.pyplot as plt
import numpy as np

# Metric values arranged in order, with each inner list representing a specific scenario
# Scenario Order: [Single Class, Multiple Classes, 1m/s, 2m/s, 3m/s]

# 3dc model, no static objects, camera height 1.5m
iou_3dc_no_static_1_5 = [0.36521616932239664, 0.36750511821473913, 0.36561057053227697, 0.3673622308402059, 0.36504197770067687]
precision_3dc_no_static_1_5 = [0.6162998289032393, 0.6561072057032131, 0.6299766489302887, 0.6298276572596957, 0.6289956955288782]
recall_3dc_no_static_1_5 = [0.5064421470117062, 0.5313335816067976, 0.5247492272816919, 0.5124446829029619, 0.5070948763670027]
f1_3dc_no_static_1_5 = [0.5167690417056185, 0.5486900857939616, 0.5327768205981026, 0.5271340826257481, 0.5223896996917002]

# Repeat this data arrangement for all other conditions. 

# Also create the corresponding lists for the 'cis' model.

# Then, proceed to create multi-bar plots.


# Here's a simple example of how you could create a multi-bar plot for just one metric (IoU) and one condition (no static objects, camera height 1.5m):
labels = ['Single Class', 'Multiple Classes', '1m/s', '2m/s', '3m/s']

x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots()

rects1 = ax.bar(x - width/2, iou_3dc_no_static_1_5, width, label='3dc')
rects2 = ax.bar(x + width/2, iou_cis_no_static_1_5, width, label='cis')  # replace with your cis model values

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('IoU Scores')
ax.set_title('Scores by algorithm and conditions')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

fig.tight_layout()

plt.show()

# Please note that due to the size of your dataset and the number of conditions you have, you will have to create multiple charts to visualize all of the data effectively. Also, be sure to replace placeholders in the example code with your actual data. You may also consider using subplots or a grid of plots to present all the different
