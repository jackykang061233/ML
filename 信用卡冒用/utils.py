from sklearn.metrics import recall_score, precision_score, f1_score,roc_auc_score, accuracy_score



def accuracy(actual, prediction):
    return accuracy_score(actual, prediction)
    
def precision(actual, prediction):
    return precision_score(actual, prediction)
    
def recall(actual, prediction):
    return recall_score(actual, prediction)
    
def f1(actual, prediction):
    return f1_score(actual, prediction)
    
def auc(actual, prediction):
    return roc_auc_score(actual, prediction)