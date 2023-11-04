import pandas as pd

train_1_df = pd.read_csv("submission_convnext_large_exp16_fold0_384_epoch6.csv")
train_2_df = pd.read_csv("submission_convnext_large_exp17_softlabel_384.csv")
train_3_df = pd.read_csv("submission_convnext_large_exp17_softlabel_448.csv")

# train_1_df = pd.read_csv("/content/submission_convnext_large.csv")
train_1_df["extent"] = (train_1_df["extent"] + train_2_df["extent"] + train_3_df["extent"])/3
train_1_df.to_csv('submission_ensemble_softlabel_multiscale_muultifold_epoch6.csv', index=False)
