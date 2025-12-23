from functions_sda import *
from tensorflow.keras import mixed_precision

# # ✅ TF32 끄기
# tf.config.experimental.enable_tensor_float_32_execution(False)

# # ✅ Mixed Precision (float16) 활성화
# mixed_precision.set_global_policy('mixed_float16')

# print("🧮 Mixed Precision Policy:", mixed_precision.global_policy())


## Check GPU
gpus = tf.config.list_physical_devices('GPU')
print(gpus)
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# Run once to generate the data structure
file_path = './output/stroke_dataset_hip.h5'
subject_trial_dict = load_stroke_h5_grouped_by_subject_and_trial_hip(file_path, window_size = 50, stride=1)
stroke_dataset = build_stroke_dataset_per_subject(subject_trial_dict)

file_path = './output/all_subjects_dataset_ds_hip.h5'
# ---------- STEP 1: Load and preprocess ----------

# ---------- STEP 2: Load and split ----------
healthy_trial_dict = load_h5_to_trial_dict_with_squeeze_hip(file_path, window_size=50, stride=10, max_walking_speed=0.7)
train_keys, val_keys, test_keys = split_trials_by_random(healthy_trial_dict)
X_src_train, y_src_train = merge_trials(healthy_trial_dict, train_keys)
X_src_val, y_src_val = merge_trials(healthy_trial_dict, val_keys)
X_src_test, y_src_test = merge_trials(healthy_trial_dict, test_keys)

print('Ready for training')
# subject_ids = ['S003', 'S004', 'S006','S007', 'S008', 'S013']
subject_ids = ['S003']
# lambda_mmd_list = [0.5, 1, 5, 10]
# lambda_src_list = [0.5, 1, 5, 10]
# lambda_tgt_list = [1.0]

# subject_ids = ['S013']
lambda_mmd_list = [0.0]
lambda_src_list = [0.0]
lambda_tgt_list = [1.0]

results = run_and_save_all_combinations(
    subject_ids,
    lambda_mmd_list,
    lambda_src_list,
    lambda_tgt_list,
    stroke_dataset,
    X_src_train,
    y_src_train,
    train_fn=train_sda_dual_decoder,  # SDA with Dual Decoder
    save_path="./saved_models/SDA_Experiment",
    combine=False
)

df = pd.DataFrame(results)
summary = df.groupby(["lambda_mmd", "lambda_src", "lambda_tgt"]).agg({"mse": ["mean", "std"]}).reset_index()
summary.columns = ["lambda_mmd", "lambda_src", "lambda_tgt", "mse_mean", "mse_std"]
print(summary)
sender = "aronwos1212@gmail.com"
receiver = "aronwos1212@gmail.com"
subject = "[Notification] NN Training completed"
password = "tepx bjdq lgik xpdj"
body = "All trials completed."

send_email(sender=sender, receiver=receiver, subject=subject,body=body,app_password=password)


