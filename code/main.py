from autogluon.tabular import TabularDataset, TabularPredictor
import numpy as np
import pandas as pd

train_data = TabularDataset('train.csv')
id, label = 'id', 'target'

metric = 'log_loss'
predictor = TabularPredictor(label=label, eval_metric=metric).fit(train_data.drop(columns=[id]),  presets='best_quality')
# pretrained model: predictor = TabularPredictor.load("AutogluonModels/ag-20210610_074821/")

test_data = TabularDataset('test.csv')
preds = predictor.predict_proba(test_data.drop(columns=[id]), as_pandas=True)
preds.insert(0, id, test_data[id])
preds.to_csv('submission.csv', index=False)