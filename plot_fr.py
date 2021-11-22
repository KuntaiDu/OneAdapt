import csv
import matplotlib.pyplot as plt
import pandas as pd
import pymongo
db = pymongo.MongoClient("mongodb://localhost:27017/")["diff_efficient_det_farewell_iou"]
stats = db.get_collection("stats")
converted = {}
seconds = 4
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



f, ax = plt.subplots(ncols=4, figsize=(16,3), sharey=True, )

res_list = ["512:360", "768:480", "1280:720"]
fr_list =  [10, 7, 5, 3]
qp_list = [24, 27, 33, 42 ]
for i in range(4):
    f1s = []
    sums = []
    labels=  []
    idx = 0
    for key in sum_scores:
        fr, res, qp = key.split('+')
        if fr == str(fr_list[i]):
            sums.append(sum_scores[key]/(seconds*10))
            f1s.append(f1_scores[key]/seconds)
            labels.append(res)
            idx += 1

    col = ax[i]
    col.scatter(sums, f1s)
#     for k in range(len(f1s)):

#         col.plot(sums[k], f1s[k], label=labels[k])
#         col.line(sums[k], f1s[k])
#     col.legend()
    col.set_ylabel("F1")
    col.set_xlabel("sum of confidence")
    col.set_title("FR: {}".format(fr_list[i]))
plt.savefig("fr.jpg", format="jpg", dpi=600,
                bbox_inches="tight")# ax.set_xticks([0,0.5, 1])
# ax.set_xticks([0,0.5, 1])