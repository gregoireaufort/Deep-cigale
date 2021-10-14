# -*- coding: utf-8 -*-
"""
Created on Tue May 19 16:17:09 2020

@author: Gregoire Aufort
"""
import pandas as pd
import  joblib
import numpy as np



def Deep_cloudy(params_cloudy,path_cigale):
    """Simulates part of Cloudy's lines output through a deep neural network
    
    Input:
        params_cloudy - DataFrame; Pandas df with columns 
                      ["com1", "com2", "com3", "com4", "com5","HbFrac"]
    Output:
        lines_df - DataFrame; pd.df containing all corresponding the lines
                    (unnormalized)
    """
    
    X = params_cloudy.copy()
    X['deep_nebular.Age'] /=1e6
    scaler = joblib.load(path_cigale+"X_scaling.pkl")
    X= scaler.transform(X)
    
    pca = joblib.load(path_cigale+"pca_trained.pkl")
    ANN = joblib.load(path_cigale+"ANN/13_04_27_06.pkl")
    y_pred = ANN.predict(X)
    ANN_hbeta = joblib.load(path_cigale+"data/ANN/ANN_hbeta_256.pkl")
    Hb = np.exp(ANN_hbeta.predict(X))
    norm_lines = pca.inverse_transform(y_pred)
        
    line_names = []
    with open(path_cigale+'list_lines.txt', 'r') as filehandle:
        for line in filehandle:
            currentPlace = line[:-1]
            line_names.append(currentPlace)
    
    lines_df= pd.DataFrame(data = np.exp(norm_lines),columns= line_names)
    lines_df2 = lines_df.multiply(Hb,axis = 0)
    lines_df2["H__1_486133A"] = Hb
    
    
    
    return lines_df2



if __name__ == "__main__":
    path_data = "/home/aufort/Desktop/Deep-cigale/pcigale/data/"
    params_cloudy = pd.read_csv("params_cloudy.csv",index_col = 0)
    pred_lines = Deep_cloudy(params_cloudy,path_data)
    pred_lines.to_csv("lines.csv")
    
    