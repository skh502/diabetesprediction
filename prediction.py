import joblib

def predict_svc(data):
    model = joblib.load('outputmodels/diabetes_svc_model.sav')
    return model.predict(data)

def predict_knn(data):
    model = joblib.load('outputmodels/diabetes_knn_model.sav')
    return model.predict(data)