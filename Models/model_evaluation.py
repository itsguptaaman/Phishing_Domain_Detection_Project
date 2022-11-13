
from sklearn.metrics import confusion_matrix
from logger import LogMe


class ModelEvaluation:

    def __init__(self):
        self.log_write = LogMe()
        self.file_object = open("Logs/model_evaluation_log.txt", "a+")

    def model_evaluation(self,y_test, y_predict):
        """
         Method name: model_evaluation.
         Description: This method is going to evaluate our model by calculating accuracy, recall, precision, f1 score.
         y_test : series (Actual values)
         y_predict : array or series (Predicted values)
         """
        try:
            # t : True, f : False, n : Negative, p : Positive
            # .ravel will give ndarray in 1d array
            tn, fp, fn, tp = confusion_matrix(y_test, y_predict).ravel()
            accuracy = (tp + tn) / (tp + fp + fn + tn)
            recall = tp / (tp + fn)
            precision = tp / (tp + fp)
            f1 = (2 * (precision * recall) / (precision + recall))
            # specificity = tn / (tn + fp)
            results = {"Accuracy": accuracy, "Recall": recall, "Precision": precision, "F1": f1,}
            self.log_write.logger(self.file_object, f"Results are{results}")
            return results

        except Exception as e:
            self.log_write.logger(self.file_object, f"Error while Evaluating Models {e}")

    def get_recall(self, model, x_test, y_test):
        """
         Method name: model_evaluation.
         Description: This method is going to evaluate our model by calculating  recall.
         Recall focus more on reducing the false negative.
         model : Name of the object
         x_test : series (test values)
         y_test : array or series (actual values)
         """
        try:
            y_predict = model.predict(x_test)
            tn, fp, fn, tp = confusion_matrix(y_test, y_predict).ravel()
            recall = tp / (tp + fn)
            return recall

        except Exception as e:
            self.log_write.logger(self.file_object, f"Error while Evaluating Models {e}")