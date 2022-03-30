import matplotlib.pyplot as plt

import pymongo

plt.style.use('ggplot')
plt.rcParams['font.size'] = 25

import pickle
import numpy as np


def query_approach(ax, approach, label, f1_label, color, collection_name, qp, res):
    f1s = []
    db = pymongo.MongoClient("mongodb://localhost:27017/")[collection_name]
    for sec in range(0,10):

        query = {
            'second': sec,
            'command_line_args.approach': approach,
            'input': "videos/dashcam/drone_11/part%d.mp4",
            'qp': qp,
            # 'res': res
        }

        x = list(db['stats'].find(query))
        x = sorted(x, key = lambda x: x['_id'])
        f1s.append(x[-1])

    print([i[f1_label] for i in f1s])
    ax.plot(range(len(f1s)), [i[f1_label] for i in f1s], label=label, color = color)
    print("Approach: ", approach)
    print("QP: ", qp)
    print(np.array([i[f1_label] for i in f1s]).mean())


fig, ax = plt.subplots(figsize=(14, 7))
# query_scores_pdf(ax, "feature_error", "orange")
# query_scores_pdf(ax, "saliency_error", "orange")
# query_scores(ax, "saliency_error", "orange")

# ax.set_xlabel("Confidence score")
# ax.set_ylabel("Fraction")
# query_approach(ax, "backprop_sigmoid_no_sr_debug", 'w/o SR ', 'f1', 'orange', 'seg_mar18_final_lr', 28, '1280:720')
# query_approach(ax, "backprop_sigmoid_no_sr_debug", 'w/o SR ', 'f1', 'orange', 'seg_mar18_final_focal_new_lr0005_freq1', 34, '1280:720')
# query_approach(ax, "backprop_sigmoid_no_sr_debug", 'w/o SR ', 'f1', 'orange', 'seg_mar18_final_lr', 38, '1280:720')
# query_approach(ax, "backprop_sigmoid_no_sr_debug", 'w/o SR ', 'f1', 'orange', 'seg_mar18_final_hr', 38, '1280:720')
# query_approach(ax, "backprop_sigmoid_no_sr_debug", 'w/o SR ', 'f1', 'orange', 'seg_mar18_final_hr', 40, '1280:720')
#
# query_approach(ax, "backprop_sigmoid_no_sr_debug", 'w/o SR ', 'f1', 'orange', 'seg_mar18_final_hr', 42, '1280:720')
# query_approach(ax, "backprop_sigmoid_no_sr_debug", 'w/o SR ', 'f1', 'orange', 'seg_mar18_final_hr', 46, '1280:720')

#

print("High Res: ")

# query_approach(ax, "backprop_sigmoid_feature_error_debug", 'SR w/ expensive update ', 'f1', 'blue', 'seg_mar14_final_num100_fr10', 34, '1280:720')
# query_approach(ax, "backprop_sigmoid_feature_error_no_update_debug", 'SR w/ expensive update ', 'f1', 'blue', 'seg_mar14_final', 34, '1280:720')
# query_approach(ax, "backprop_sigmoid_feature_error_debug", 'SR w/ expensive update ', 'f1', 'blue', 'seg_mar14_final_num200_fr10', 34, '1280:720')
# query_approach(ax, "backprop_sigmoid_feature_error_debug", 'SR w/ expensive update ', 'f1', 'blue', 'seg_mar14_final_num300_fr0', 34, '1280:720')
# query_approach(ax, "backprop_sigmoid_feature_error_debug", 'SR w/ expensive update ', 'f1', 'blue', 'seg_mar14_final_num400_fr10', 34, '1280:720')

# query_approach(ax, "backprop_sigmoid_feature_error_no_update_debug", 'ours ', 'f1', 'blue', 'seg_mar14_final_new_carn4000', 34, '1280:720')
# query_approach(ax, "backprop_sigmoid_feature_error_no_update_debug", 'ours ', 'f1', 'blue', 'seg_mar14_final_new_carn2000', 34, '1280:720')

# query_approach(ax, "backprop_sigmoid_feature_error_no_update5_debug", 'ours ', 'f1', 'blue', 'seg_mar14_final_new_carn4000_lr001', 34, '1280:720')
#
# query_approach(ax, "backprop_sigmoid_feature_error_no_update9_debug", 'ours ', 'f1', 'blue', 'seg_mar14_final_new_carn4000_lr001', 34, '1280:720')
# query_approach(ax, "backprop_sigmoid_feature_error_no_update13_debug", 'ours ', 'f1', 'blue', 'seg_mar14_final_new_carn4000_lr001', 34, '1280:720')
# query_approach(ax, "backprop_sigmoid_feature_error_no_update17_debug", 'ours ', 'f1', 'blue', 'seg_mar14_final_new_carn4000_lr001', 34, '1280:720')
# query_approach(ax, "backprop_sigmoid_normal_update_debug", 'ours ', 'f1', 'blue', 'seg_mar18_final_focal_new_lr0005', 34, '1280:720')
# query_approach(ax, "backprop_sigmoid_normal_update_debug", 'ours ', 'f1', 'blue', 'seg_mar18_final_focal_new_lr0005_freq3', 34, '1280:720')

# query_approach(ax, "backprop_sigmoid_normal_update_debug", 'ours ', 'f1', 'blue', 'seg_mar18_final_focal_new_lr0005_freq3', 34, '1280:720')
# # query_approach(ax, "backprop_sigmoid_normal_update_debug", 'ours ', 'f1', 'blue', 'seg_mar18_final_focal_new_lr0005_freq2', 34, '1280:720')
# query_approach(ax, "backprop_sigmoid_normal_update_debug", 'ours ', 'f1', 'blue', 'seg_mar18_final_focal_new_lr0005_freq4', 34, '1280:720')
# query_approach(ax, "backprop_sigmoid_normal_update_debug", 'ours ', 'f1', 'blue', 'seg_mar18_final_focal_new_lr0005_freq5', 34, '1280:720')
# query_approach(ax, "backprop_sigmoid_normal_update_debug", 'ours ', 'f1', 'blue', 'seg_mar18_final_focal_new_lr0005_freq6', 34, '1280:720')

# query_approach(ax, "backprop_sigmoid_mse_one_norm_debug", 'MSE', 'f1', 'green', 'seg_mar18_final_focal_new_lr0005_freq1', 34, '1280:720')
# ax.plot(range(5), [0.46, 0.59, 0.46, 0.56, 0.42], label='expensive update (saliency) ', c='blue')
# ax.plot(range(5), [0.79,0.850, 0.759, 0.879, 0.875], label='expensive update ', c='blue')

# query_approach(ax, "backprop_sigmoid_mse_one_norm_debug", 'ours (MSE 1-norm) ', 'f1', 'blue', 'seg_mar18_final_focal_new_lr0005_freq1', 34, '1280:720')
query_approach(ax, "backprop_sigmoid_normal_update_saliency_debug", 'expensive update ', 'f1', 'red', 'seg_mar18_final_focal_new_lr0005_freq1', 34, '1280:720')
query_approach(ax, "backprop_sigmoid_normal_update_saliency_block_debug", 'expensive update (block)', 'f1', 'black', 'seg_mar18_final_focal_new_lr0005_freq1', 34, '1280:720')

ax.legend()

plt.savefig("plot.jpg", format="jpg", dpi=600,
                bbox_inches="tight")
