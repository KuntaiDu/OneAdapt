export CUDA_VISIBLE_DEVICES=0
python batch_diff.py --method saliency_error --video_index 1
python batch_diff.py --method saliency_error --video_index 2
python batch_diff.py --method saliency_error --video_index 4
python batch_diff.py --method saliency_error --video_index 5

# python batch_diff.py --method feature_error --video_index 2
# python batch_diff.py --method feature_error --video_index 4
# python batch_diff.py --method feature_error --video_index 5
# #
# python batch_diff.py --method saliency_error_baseline --video_index 2

# python batch_diff.py --method saliency_error_test --video_index 4
# python batch_diff.py --method saliency_error_test --video_index 5

# python batch_diff.py --method feature_error --video_index 1
# python batch_diff.py --method feature_error --video_index 2
# python batch_diff.py --method saliency_error --video_index 2
