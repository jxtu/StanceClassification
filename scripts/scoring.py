import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score


if __name__ == '__main__':
    pred_file = "../data/test_with_pred_clean_lockdown.csv"
    pred_df = pd.read_csv(pred_file)
    golds = pred_df.label.values
    preds = pred_df.preds.values
    print(f1_score(golds, preds, average="micro"))
