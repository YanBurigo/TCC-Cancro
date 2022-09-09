import numpy as np
import pandas as pd
import torch
import os
from array import array
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score 
from sklearn.metrics import f1_score

class Evaluation:
    h_list_val: array

    def __init__(self) -> None:
        self.h_list_val = []
        pass

    def evaluate_model(self, y_true, y_pred, pos_label = 0) -> any:
        return {
            'acc': accuracy_score(y_true, y_pred), 
            'confusion_matrix' : confusion_matrix(y_true, y_pred),
            'prec' : precision_score(y_true, y_pred, pos_label=pos_label),
            'recall' : recall_score(y_true, y_pred, pos_label=pos_label),
            'f1' : f1_score(y_true, y_pred)
        } 

    def predict(self, model, loader, device: str) -> any:
        y_true = []
        y_pred = []
        outputs = []
        y_true_output = []

        for X, y in loader:
            X, y = X.to(device), y.to(device)

            output = model(X)

            _, y_pred_ = torch.max(output, 1)
            for y_ in y.cpu():
                y_true.append(y_)
                y_true_output.append(y_.item())
            for y_ in y_pred_.cpu():
                y_pred.append(y_)
            for y_ in output:
                outputs.append(y_.detach().numpy()[1])

        output_stacked = np.stack((outputs, y_true_output), axis=1)

        return y_true, y_pred, output_stacked[np.argsort(output_stacked[:, 0])]
    
    def calculate_result(self, model_ft, test_loader, model_name, device: str):
        y_true, y_pred, output_stacked = self.predict(model_ft, test_loader, device)
        h_val = self.evaluate_model(y_true, y_pred) 
        h_val['model_name'] = model_name
        self.h_list_val.append(h_val)

        h_list_df = pd.DataFrame(output_stacked)
        os.makedirs('output/result/output_data', exist_ok=True)
        h_list_df.to_csv(f'output/result/output_data/{model_name}.csv', index=False, sep=';', header=False, decimal=",")

    def show_result(self):
        h_list_df = pd.DataFrame(self.h_list_val)
        os.makedirs('output/result', exist_ok=True)  
        h_list_df.to_csv('output/result/result.csv', index=False, sep=';') 
        h_list_df