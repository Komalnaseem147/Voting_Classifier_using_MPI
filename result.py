from mpi4py import MPI
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE
from sklearn.utils.class_weight import compute_sample_weight

# MPI initialization
comm = MPI.COMM_WORLD 
rank = comm.Get_rank()  
size = comm.Get_size()  

if rank == 0:  
    data = pd.read_csv("Breast_Cancer.csv") 
    
    X = data.drop("Status", axis=1)
    y = data["Status"]

    X = pd.get_dummies(X)  
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)

    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("Rank 0: Broadcasting preprocessed data to all processes")
    data_to_bcast = (X_train_scaled, X_test_scaled, y_train, y_test)
else:
    data_to_bcast = None

X_train_scaled, X_test_scaled, y_train, y_test = comm.bcast(data_to_bcast, root=0)

if rank == 1:  
    print("Rank 1: Running Logistic Regression")

    best_C = 1
    best_solver = 'lbfgs'
    log_reg_best = LogisticRegression(C=best_C, solver=best_solver, random_state=42,n_jobs=-1)
    
    log_reg_best.fit(X_train_scaled, y_train)
    preds = log_reg_best.predict(X_test_scaled)
    
    comm.send(preds, dest=0, tag=11)
    print("Rank 1: Sent predictions to Rank 0")

elif rank == 2:  
    print("Rank 2: Running Naive Bayes with Sample Weights")

    best_var_smoothing = 1e-9
    nb_best = GaussianNB(var_smoothing=best_var_smoothing)

    # Compute sample weights for balanced classes
    sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)
    nb_best.fit(X_train_scaled, y_train, sample_weight=sample_weights)
    nb_preds = nb_best.predict(X_test_scaled)

    # Send predictions to Rank 0
    comm.send(nb_preds, dest=0, tag=12)
    print("Rank 2: Sent predictions to Rank 0")

elif rank == 3:  
    print("Rank 3: Running Support Vector Machine (SVM)")
 
    best_C_svm = 100
    best_kernel = 'linear'
    best_gamma = 'scale'
    svm_best = SVC(C=best_C_svm, kernel=best_kernel, gamma=best_gamma, random_state=42)
  
    svm_best.fit(X_train_scaled, y_train)
    preds = svm_best.predict(X_test_scaled)
    
    comm.send(preds, dest=0, tag=13)
    print("Rank 3: Sent predictions to Rank 0")

elif rank == 0:  
    print("Rank 0: Receiving predictions from all ranks")
    
    logreg_preds = comm.recv(source=1, tag=11)
    print("Rank 0: Received predictions from Rank 1 (Logistic Regression)")

    nb_preds = comm.recv(source=2, tag=12)
    print("Rank 0: Received predictions from Rank 2 (Naive Bayes)")

    svm_preds = comm.recv(source=3, tag=13)
    print("Rank 0: Received predictions from Rank 3 (SVM)")

    # Perform Majority Voting
    print("Rank 0: Performing Majority Voting")
    final_preds = []
    for i in range(len(logreg_preds)):
        votes = [logreg_preds[i], nb_preds[i], svm_preds[i]]
        final_preds.append(max(set(votes), key=votes.count))  # Majority vote

    # Evaluate the final ensemble model
    final_accuracy = accuracy_score(y_test, final_preds)
    print("\nVoting Classifier Accuracy:", final_accuracy)

    print("\nClassification Report:")
    print(classification_report(y_test, final_preds))
