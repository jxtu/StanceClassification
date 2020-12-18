from fast_bert.data_cls import BertDataBunch
from fast_bert.learner_cls import BertLearner
from fast_bert.metrics import accuracy
import torch
import logging

DATA_PATH = "../data/fast_bert_data"
LABEL_PATH = "../data/fast_bert_label"
OUTPUT_DIR = "../output"
logger = logging.getLogger()
device_cuda = torch.device("cuda")
metrics = [{"name": "accuracy", "function": accuracy}]

data_bunch = BertDataBunch(
    DATA_PATH,
    LABEL_PATH,
    tokenizer="bert-base-cased",
    train_file="train.csv",
    val_file="val.csv",
    label_file="labels.csv",
    text_col="text",
    label_col="label",
    batch_size_per_gpu=16,
    max_seq_length=256,
    multi_gpu=False,
    multi_label=False,
    model_type="bert",
)

learner = BertLearner.from_pretrained_model(
    data_bunch,
    pretrained_path="bert-base-cased",
    metrics=metrics,
    device=device_cuda,
    logger=logger,
    output_dir=OUTPUT_DIR,
    finetuned_wgts_path=None,
    warmup_steps=500,
    multi_gpu=True,
    is_fp16=True,
    multi_label=False,
    logging_steps=50,
)


if __name__ == "__main__":
    learner.fit(
        epochs=6,
        lr=6e-5,
        validate=True,
        schedule_type="warmup_cosine",
        optimizer_type="lamb",
    )

    learner.save_model()
