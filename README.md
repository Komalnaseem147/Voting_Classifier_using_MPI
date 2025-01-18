# Voting_Classifier_using_MPI
The goal was to build a voting classifier that combines predictions from three machine learning model like Logistic Regression, Naive Bayes, and Support Vector Machine(SVM) to improve overall prediction accuracy and combine predictions from the three models using a majority voting approach to determine the final output for each instance.

Steps Involved:
1-Data Preprocessing:
Used a breast cancer dataset where the target column indicated the cancer status.
Applied techniques like encoding categorical variables and scaling numeric features using StandardScaler.
Tackled the problem of class imbalance with SMOTE (Synthetic Minority Oversampling Technique), ensuring the dataset was balanced for training the models.

2-Machine Learning Models:
Trained three classifiers:
Logistic Regression: Optimized with the best hyperparameters for performance.
Naive Bayes: Incorporated class balancing by using sample weights.
Support Vector Machine (SVM): Tuned hyperparameters like C, kernel, and gamma for best results.

3-Parallel Computing with MPI:
Distributed the training and prediction tasks across multiple processors using MPI (Message Passing Interface) and the mpi4py library.
Each process handled one classifier:
Process 1: Logistic Regression
Process 2: Naive Bayes
Process 3: SVM
Used rank 0 (master process) to coordinate tasks, broadcast data, and collect predictions from other processes.

4-Majority Voting Ensemble:
Combined predictions from the three models using a majority voting approach to determine the final output for each instance.

5-Performance Evaluation:
Assessed the final voting classifier’s accuracy and generated a classification report to analyze precision, recall, and F1-score.

Key Achievements:
Successfully implemented a distributed machine learning workflow using MPI, significantly optimizing the computational process.
Enhanced model performance by addressing class imbalance with SMOTE and tuning hyperparameters for each classifier.
Achieved improved prediction accuracy and demonstrated how ensemble learning can outperform individual models.
