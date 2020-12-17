import pandas as pd
import torch
from fast_bert.prediction import BertClassificationPredictor

MODEL_PATH = "../output/model_out"
LABEL_PATH = "../data/fast_bert_label"
device_cuda = torch.device("cuda")

predictor = BertClassificationPredictor(
    model_path=MODEL_PATH,
    label_path=LABEL_PATH,
    multi_label=False,
    model_type='bert',
    do_lower_case=False,
    device=device_cuda)

if __name__ == '__main__':
    test_csv_path = "../data/fast_bert_data/test.csv"
    test_df = pd.read_csv(test_csv_path)
    texts = test_df.text.values.tolist()

    multiple_predictions = predictor.predict_batch(texts)
    preds = [pred[0][0] for pred in multiple_predictions]
    test_df["preds"] = preds
    test_df.to_csv("../data/test_with_pred.csv", index=False)
