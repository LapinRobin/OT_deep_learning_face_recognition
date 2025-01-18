
MODEL EVALUATION RESULTS (15 EPOCHS, 32 BATCH SIZE, NET)


Overall Metrics:
Accuracy: 0.9460
Precision: 0.9462
Recall: 0.9460
F1 Score: 0.9385
ROC AUC Score: 0.8875

Confusion Matrix:
[[6810   21]
 [ 391  406]]

Detailed Classification Report:

              precision    recall  f1-score   support

      noface       0.95      1.00      0.97      6831
        face       0.95      0.51      0.66       797

    accuracy                           0.95      7628
   macro avg       0.95      0.75      0.82      7628
weighted avg       0.95      0.95      0.94      7628


Confusion matrix has been saved as 'confusion_matrix.png'

Per-class Accuracy:
noface: 0.9969 (6810/6831)
face: 0.5094 (406/797)

MODEL EVALUATION RESULTS (15 EPOCHS, 32 BATCH SIZE, NET2)

Overall Metrics:
Accuracy: 0.9529
Precision: 0.9536
Recall: 0.9529
F1 Score: 0.9474
ROC AUC Score: 0.9633

Confusion Matrix:
[[6816   15]
 [ 344  453]]

Detailed Classification Report:
              precision    recall  f1-score   support

      noface       0.95      1.00      0.97      6831
        face       0.97      0.57      0.72       797

    accuracy                           0.95      7628
   macro avg       0.96      0.78      0.85      7628
weighted avg       0.95      0.95      0.95      7628


Confusion matrix has been saved as 'confusion_matrix.png'

Per-class Accuracy:
noface: 0.9978 (6816/6831)
face: 0.5684 (453/797)



MODEL EVALUATION RESULTS (15 EPOCHS, 32 BATCH SIZE, NET2 improved)

Overall Metrics:
Accuracy: 0.9761
Precision: 0.9765
Recall: 0.9761
F1 Score: 0.9749
ROC AUC Score: 0.9768

Confusion Matrix:
[[6825    6]
 [ 176  621]]

Detailed Classification Report:
              precision    recall  f1-score   support

      noface       0.97      1.00      0.99      6831
        face       0.99      0.78      0.87       797

    accuracy                           0.98      7628
   macro avg       0.98      0.89      0.93      7628
weighted avg       0.98      0.98      0.97      7628


Confusion matrix has been saved as 'confusion_matrix.png'

Per-class Accuracy:
noface: 0.9991 (6825/6831)
face: 0.7792 (621/797)

MODEL EVALUATION RESULTS (3 EPOCHS, 32 BATCH SIZE, NET2 improved)
I was able to achieve this by adding class weights to penalize false negatives more heavily
The number of epochs was reduced to 3 to achieve a better result.


Overall Metrics:
Accuracy: 0.9351
Precision: 0.9366
Recall: 0.9351
F1 Score: 0.9351
ROC AUC Score: 0.9825

Confusion Matrix:
[[3652  138]
 [ 357 3481]]

Detailed Classification Report:
              precision    recall  f1-score   support

      noface       0.91      0.96      0.94      3790
        face       0.96      0.91      0.93      3838

    accuracy                           0.94      7628
   macro avg       0.94      0.94      0.94      7628
weighted avg       0.94      0.94      0.94      7628


Confusion matrix has been saved as 'confusion_matrix.png'

Per-class Accuracy:
noface: 0.9636 (3652/3790)
face: 0.9070 (3481/3838)


MODEL EVALUATION RESULTS (5 EPOCHS, 32 BATCH SIZE, NET2 improved, AdaBoost)

Overall Metrics:
Accuracy: 0.9198
Precision: 0.9206
Recall: 0.9198
F1 Score: 0.9197
ROC AUC Score: 0.9661

Confusion Matrix:
[[3670  217]
 [ 395 3346]]

Detailed Classification Report:
              precision    recall  f1-score   support

      noface       0.90      0.94      0.92      3887
        face       0.94      0.89      0.92      3741

    accuracy                           0.92      7628
   macro avg       0.92      0.92      0.92      7628
weighted avg       0.92      0.92      0.92      7628


Confusion matrix has been saved as 'confusion_matrix.png'

Per-class Accuracy:
noface: 0.9442 (3670/3887)
face: 0.8944 (3346/3741)