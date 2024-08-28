import joblib

class RandomForestClassifier():

    def __init__(self):
        self.model = joblib.load('rf_classifier.sav')

    def predict(self,X,feature_names=None):
        return self.model.predict_proba(X)
    
if __name__=="__main__":
    apf = RandomForestClassifier()
    ip_ = [[5.964, 4.006, 2.081, 1.031]]
    print(apf.predict(ip_))