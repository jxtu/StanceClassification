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
    texts = pd.read_csv(test_csv_path).text.values.tolist()

    multiple_predictions = predictor.predict_batch(texts)
    for i in multiple_predictions:
        print(i)
