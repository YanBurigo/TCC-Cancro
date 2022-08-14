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
        
        for X, y in loader:                
            X, y = X.to(device), y.to(device)             
            output = model(X)                      
            _, y_pred_ = torch.max(output, 1)
            for y_ in y.cpu():
                y_true.append(y_)
            for y_ in y_pred_.cpu():
                y_pred.append(y_)
            
        return y_true, y_pred
    
    def calculate_result(self, model_ft, test_loader, model_name, device: str):
        y_true, y_pred = self.predict(model_ft, test_loader, device)  
        h_val = self.evaluate_model(y_true, y_pred) 
        h_val['model_name'] = model_name
        self.h_list_val.append(h_val) 

    def show_result(self):
        h_list_df = pd.DataFrame(self.h_list_val)
        os.makedirs('output/result', exist_ok=True)  
        h_list_df.to_csv('output/result/result.csv', index=False, sep=';') 
        h_list_df