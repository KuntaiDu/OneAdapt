import matplotlib.pyplot as plt

import pymongo

plt.style.use('ggplot')
plt.rcParams['font.size'] = 25

import pickle
import numpy as np
def query_scores(ax, approach, color):
    f1s = []
    db = pymongo.MongoClient("mongodb://localhost:27017/")['yuhan_test_feb09']

    query = {
        'loss_type': approach,
        'input': "videos/dashcam/dashcam_1/part%d.mp4",
    }

    x = list(db['conf'].find(query))
    x = sorted(x, key = lambda x: x['_id'])

    all_scores = pickle.loads(x[-1]['all_scores'])
    sorted_vals = np.sort(all_scores)
    p = 1. * np.arange(len(sorted_vals))/(len(sorted_vals) - 1)
    ax.plot(sorted_vals, p, label="All ")
    all_fn_scores = pickle.loads(x[-1]['all_fn_scores'])
    sorted_vals_fn = np.sort(all_fn_scores)
    p_fn = 1. * np.arange(len(sorted_vals_fn))/(len(sorted_vals_fn) - 1)
    ax.plot(sorted_vals_fn, p_fn, label="FN ")

    all_fp_scores = pickle.loads(x[-1]['all_fp_scores'])
    sorted_vals = np.sort(all_fp_scores)
    print(sorted_vals)

    p = 1. * np.arange(len(sorted_vals))/(len(sorted_vals) - 1)
    ax.plot(sorted_vals, p, label="FP ")
    #
    # print("Approach: ", approach)
    # print(np.array([i[f1_label] for i in f1s]).mean())

def query_scores_pdf(ax, approach, color):
    f1s = []
    db = pymongo.MongoClient("mongodb://localhost:27017/")['yuhan_test_feb14']
    print(approach)
    query = {
        'loss_type': approach,
        'input': "videos/dashcam/dashcam_5/part%d.mp4",
    }

    x = list(db['conf'].find(query))
    x = sorted(x, key = lambda x: x['_id'])

    all_scores = pickle.loads(x[-1]['all_scores'])
    sorted_vals = np.sort(all_scores)
    # p = 1. * np.arange(len(sorted_vals))/(len(sorted_vals) - 1)
    # ax.plot(sorted_vals, p, label="All ")
    num_bins = 50
    num_bins_fn = 20
    p, bin_vals = np.histogram(sorted_vals, bins=num_bins)

    ax.plot(bin_vals[:num_bins], p[:num_bins]/len(sorted_vals), label="All ")
    all_fn_scores = pickle.loads(x[-1]['all_fn_scores'])
    sorted_vals_fn = np.sort(all_fn_scores)
    p, bin_vals = np.histogram(sorted_vals_fn, bins=num_bins_fn)
    ax.plot(bin_vals[:num_bins_fn], p[:num_bins_fn]/len(sorted_vals_fn), label="FN ")
    # p_fn = 1. * np.arange(len(sorted_vals_fn))/(len(sorted_vals_fn) - 1)
    # ax.plot(sorted_vals_fn, p_fn, label="FN ")
    #
    all_fp_scores = pickle.loads(x[-1]['all_fp_scores'])
    sorted_vals_fp = np.sort(all_fp_scores)
    p, bin_vals = np.histogram(sorted_vals_fp, bins=num_bins)
    ax.plot(bin_vals[:num_bins], p[:num_bins]/len(sorted_vals_fp), label="FP ")
    #
    # p = 1. * np.arange(len(sorted_vals))/(len(sorted_vals) - 1)
    # ax.plot(sorted_vals, p, label="FP ")


def query_approach(ax, approach, label, f1_label, color, collection_name, qp, res):
    f1s = []
    db = pymongo.MongoClient("mongodb://localhost:27017/")[collection_name]
    for sec in range(1,61):

        query = {
            'second': sec,
            'command_line_args.approach': approach,
            'input': "videos/dashcam/dashcam_2/part%d.mp4",
            'qp': qp,
            # 'res': res
        }

        x = list(db['stats'].find(query))
        x = sorted(x, key = lambda x: x['_id'])
        f1s.append(x[-1])


    ax.plot(range(len(f1s)), [i[f1_label] for i in f1s], label=label, color = color)
    print("Approach: ", approach)
    print(np.array([i[f1_label] for i in f1s]).mean())


fig, ax = plt.subplots(figsize=(14, 7))
# query_scores_pdf(ax, "feature_error", "orange")
# query_scores_pdf(ax, "saliency_error", "orange")
# query_scores(ax, "saliency_error", "orange")

# ax.set_xlabel("Confidence score")
# ax.set_ylabel("Fraction")

# query_scores(ax, "feature_error", "orange")
# query_approach(ax, "backprop_sigmoid_feature_error_debug", 'ours (Mask)', 'f1', 'blue', 'yuhan_test_feb25', 20)
# # query_approach(ax, "backprop_sigmoid_saliency_error_debug", 'ours (Mask)', 'f1', 'blue', 'yuhan_test_feb25', 10)
# query_approach(ax, "backprop_sigmoid_feature_error_test_debug", 'ours (Mask)', 'f1', 'blue', 'yuhan_test_feb25', 10)
# query_approach(ax, "backprop_sigmoid_feature_error_test_debug", 'ours (Mask)', 'f1', 'blue', 'yuhan_test_feb25', 16)
# query_approach(ax, "backprop_sigmoid_feature_error_test_debug", 'ours (Mask)', 'f1', 'blue', 'yuhan_test_feb25', 20)

# query_approach(ax, "backprop_sigmoid_saliency_error_baseline_debug", 'ours (Mask)', 'f1', 'blue', 'yuhan_test_feb16_final_v7')
# query_approach(ax, "backprop_sigmoid_saliency_error_debug", 'ours (Mask)', 'f1', 'blue', 'yuhan_test_feb24_qp2')

# query_approach(ax, "backprop_sigmoid_saliency_error_test_debug", 'ours (New)', 'f1', 'blue', 'yuhan_test_feb15')

# query_approach(ax, "backprop_sigmoid_saliency_error_debug", 'ours ', 'f1', 'orange', 'yuhan_test_feb25', 0)
# query_approach(ax, "backprop_sigmoid_saliency_error_debug", 'ours ', 'f1', 'orange', 'yuhan_test_feb25_freq3_num5', 0, '1280:720')
# query_approach(ax, "backprop_sigmoid_feature_error_debug", 'ours ', 'f1', 'orange', 'yuhan_test_feb25_freq3_num5', 0, '1280:720')
# query_approach(ax, "backprop_sigmoid_feature_error_debug", 'ours ', 'f1', 'orange', 'yuhan_test_feb25_freq3_num5', 10, '1280:720')
# query_approach(ax, "backprop_sigmoid_feature_error_debug", 'ours ', 'f1', 'orange', 'yuhan_test_feb25_freq3_num10', 10, '1280:720')
# query_approach(ax, "backprop_sigmoid_saliency_error_debug", 'ours ', 'f1', 'orange', 'yuhan_test_feb25_freq5_num10', 16, '1280:720')


print("No qp, low quality qp=20")
# query_approach(ax, "backprop_sigmoid_saliency_error_debug", 'ours ', 'f1', 'orange', 'yuhan_test_feb25_freq5_num10', 12, '1280:720')
query_approach(ax, "backprop_sigmoid_feature_error_debug", 'ours ', 'f1', 'orange', 'yuhan_test_feb25_freq10_num10', 20, '1280:720')
query_approach(ax, "backprop_sigmoid_no_sr_debug", 'ours ', 'f1', 'orange', 'yuhan_test_mar1', 24, '1280:720')

# query_approach(ax, "backprop_sigmoid_saliency_error_debug", 'ours ', 'f1', 'orange', 'yuhan_test_feb25_freq5_num10_qp16', 12, '1280:720')
# query_approach(ax, "backprop_sigmoid_feature_error_debug", 'ours ', 'f1', 'orange', 'yuhan_test_feb25_freq5_num10_qp16', 12, '1280:720') # currently running
#
#
# print("qp=16, low quality qp=16")
# query_approach(ax, "backprop_sigmoid_saliency_error_debug", 'ours ', 'f1', 'orange', 'yuhan_test_feb25_freq5_num10_qp16', 16, '1280:720')
#
# query_approach(ax, "backprop_sigmoid_feature_error_debug", 'ours ', 'f1', 'orange', 'yuhan_test_feb25_freq5_num10_qp16', 16, '1280:720')
# query_approach(ax, "backprop_sigmoid_feature_error_debug", 'ours ', 'f1', 'orange', 'yuhan_test_feb25_freq5_num10_qp27', 16, '1280:720')

# query_approach(ax, "backprop_sigmoid_saliency_error_debug", 'ours ', 'f1', 'orange', 'yuhan_test_feb25_freq5_num10_qp16', 16, '1280:720')
#
# query_approach(ax, "backprop_sigmoid_no_sr_debug", 'ours ', 'f1', 'orange', 'yuhan_test_feb25_freq3_num5', 0, '1280:720')
# query_approach(ax, "backprop_sigmoid_no_sr_debug", 'ours ', 'f1', 'orange', 'yuhan_test_feb25_freq3_num5', 10, '1280:720')
# query_approach(ax, "backprop_sigmoid_no_sr_debug", 'ours ', 'f1', 'orange', 'yuhan_test_feb25_freq3_num5', 16, '1280:720')
# query_approach(ax, "backprop_sigmoid_no_sr_1080_debug", 'ours ', 'f1', 'orange', 'yuhan_test_feb25_freq3_num5', 24, '1280:720')

# query_approach(ax, "backprop_sigmoid_feature_error_debug", 'baseline', 'f1', 'red', 'yuhan_test_feb14', 0)
# query_approach(ax, "backprop_sigmoid_feature_error_debug", 'relaxed', 'recall_debug', 'blue')

# query_approach(ax, "backprop_sigmoid_saliency_error_debug", 'recall', 're', 'red')
# query_approach(ax, "backprop_sigmoid_saliency_error_debug", 'relaxed recall', 'recall_debug', 'blue')
# query_approach(ax, "backprop_sigmoid_feature_error_debug", 'hidden FN ratio', 'fn_hidden_ratio', 'green')

# query_approach(ax, "backprop_sigmoid_cheat_saliency_error_debug", 'cheat')


def query_approach2(ax, approach, label):
    all_scores = []
    db = pymongo.MongoClient("mongodb://localhost:27017/")['yuhan_test5']
    for sec in range(1, 10):

        query = {
            'command_line_args.approach': approach,
            'input': "videos/dashcam/dashcam_2/part%d.mp4"
        }

        x = list(db['stats'].find(query))
        x = sorted(x, key = lambda x: x['_id'])

        f1s.append(x[-1])


    ax.plot(range(len(f1s)), [i['f1'] for i in f1s], label=label)
    print("Approach: ", approach)
    print(np.array([i['f1'] for i in f1s]).mean())

# query_approach2(ax, "backprop_sigmoid_feature_error_debug", 'baseline')

# query_approach2(ax, "backprop_sigmoid_saliency_error_debug", 'Ours')

# query_approach2(ax, "backprop_sigmoid_saliency_error_debug", 'Ours')

ax.legend()

plt.savefig("plot.jpg", format="jpg", dpi=600,
                bbox_inches="tight")
