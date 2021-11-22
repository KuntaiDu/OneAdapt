import csv
import matplotlib.pyplot as plt
import pandas as pd
import pymongo
db = pymongo.MongoClient("mongodb://localhost:27017/")["diff_efficient_det_farewell_iou"]
stats = db.get_collection("stats")
converted = {}
seconds = 5
for second in range(seconds):
    converted[second ] = []
for second in range(seconds):

    for doc in stats.find({"second": second, "image_idx": "1"}):
        converted[second].append(doc)

f1_scores = {}
sum_scores = {}
std_scores = {}
for second in range(seconds):
    csv_reader = converted[second]
    for i in range(len(csv_reader)):

        knob_config = "{}+{}+{}".format(csv_reader[i]['fr'], csv_reader[i]['res'], csv_reader[i]['qp'])
        if knob_config not in f1_scores:

            f1_scores[knob_config] = csv_reader[i]['f1']
            sum_scores[knob_config] = csv_reader[i]['mean_sum_score'] * 10
            std_scores[knob_config] = csv_reader[i]['std_sum_confidence_score']
        else:
            f1_scores[knob_config] += csv_reader[i]['f1']
            sum_scores[knob_config] += csv_reader[i]['mean_sum_score'] * 10
            std_scores[knob_config] += csv_reader[i]['std_sum_confidence_score']



f, ax = plt.subplots(nrows=3, ncols=4, figsize=(16,12), sharey=True, sharex=True )
res_list = ["512:360", "768:480", "1280:720"]
fr_list =  [10, 7, 5, 3]
qp_list = [24, 27, 33, 42 ]
for i in range(3):
    for j in range(4):
        f1s = []
        sums = []
        stds = []
        labels=  []
        idx = 0
        for key in sum_scores:
            fr, res, qp = key.split('+')
            
            if res == str(res_list[i]) and qp == str(qp_list[j]):
#                 if idx == 0:
#                     print(res)
#                 if i == 0 and j ==2:
#                     print(res)
#                     print(f1_scores[key])
                print(fr)
                sums.append(sum_scores[key]/(seconds*10))
                f1s.append(f1_scores[key]/seconds)
                stds.append(std_scores[key]/seconds)
                labels.append(fr)
                idx += 1
        print(stds)
        col = ax[i,j]
        color_list = ['red', 'green', 'blue', 'orange']
        for k in range(len(f1s)):
            
            col.scatter(stds[k], f1s[k], label=labels[k], c=color_list[k])
#         col.line(sums[k], f1s[k])
        col.legend()
        col.set_ylabel("F1")
        col.set_xlabel("std of sum of confidence")
        col.set_title("res: {}, qp: {}".format(res_list[i], qp_list[j]))
plt.savefig("test.jpg", format="jpg", dpi=600,
                bbox_inches="tight")# ax.set_xticks([0,0.5, 1])
# ax.set_xticks([0,0.5, 1])