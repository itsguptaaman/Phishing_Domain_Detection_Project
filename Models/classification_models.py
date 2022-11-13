from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from logger import LogMe
from Models.model_evaluation import ModelEvaluation


class Model:

    def __init__(self, x_train, y_train):
        self.log_write = LogMe()
        self.file_object = open("Logs/classification_model_log.txt", "a+")
        self.X_train = x_train
        self.y_train = y_train


    def decision_tree(self):
        """
         Method name: decision_tree.
         Description: This method is going to build a decision tree classifier model
         """
        try:
            dt = DecisionTreeClassifier()
            return dt.fit(self.X_train, self.y_train.values.ravel())

        except Exception as e:
            self.log_write.logger(self.file_object, f"Error while Building decision_tree {e}")

    def logistic_regression(self):
        """
         Method name: logistic_regression.
         Description: This method is going to build a logistic regression model
         """
        try:
            log_reg = LogisticRegression()
            return log_reg.fit(self.X_train, self.y_train.values.ravel())

        except Exception as e:
            self.log_write.logger(self.file_object, f"Error while Building logistic_regression {e}")

    def support_vector(self):
        """
         Method name: support_vector.
         Description: This method is going to build a support vector classifier model
         """
        try:
            svc = SVC()
            return svc.fit(self.X_train, self.y_train.values.ravel())

        except Exception as e:
            self.log_write.logger(self.file_object, f"Error while Building support_vector {e}")

    def random_forest(self):
        """
         Method name: random_forest.
         Description: This method is going to build a  random forest classifier model
         """
        try:
            rf = RandomForestClassifier()
            return rf.fit(self.X_train, self.y_train.values.ravel())

        except Exception as e:
            self.log_write.logger(self.file_object, f"Error while Building random_forest {e}")

    def ada_boosting(self):
        """
         Method name: ada_boosting.
         Description: This method is going to build a ada boosting classifier model
         """
        try:
            ab = AdaBoostClassifier()
            return ab.fit(self.X_train, self.y_train.values.ravel())

        except Exception as e:
            self.log_write.logger(self.file_object, f"Error while Building ada_boosting {e}")

    def gradient_boosting(self):
        """
         Method name: gradient_boosting.
         Description: This method is going to build a gradient boosting classifier model
         """
        try:
            gb = GradientBoostingClassifier()
            return gb.fit(self.X_train, self.y_train.values.ravel())

        except Exception as e:
            self.log_write.logger(self.file_object, f"Error while Building gradient_boosting {e}")


    def score(self,model,x_test,y_test):
        """
         Method name: score.
         Description: This method is going to give sores of the model.
         x_test: dataframe or array
         y_test array or series
         """
        ml=ModelEvaluation()
        y_pre=model.predict(x_test)
        result=ml.model_evaluation(y_test,y_pre)
        return result

