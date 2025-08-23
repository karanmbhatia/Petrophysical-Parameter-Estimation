import pandas as pd
import pickle

def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def predict_phi_sw(model_phi, model_sw, scaler, X):
    X_scaled = scaler.transform(X)
    phi_pred = model_phi.predict(X_scaled)
    sw_pred = model_sw.predict(X_scaled)
    return phi_pred, sw_pred
