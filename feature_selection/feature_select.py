from sklearn.linear_model import Lasso
# from sklearn.linear_model import LassoCV
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from statsmodels.stats.outliers_influence import variance_inflation_factor
from logger import LogMe


class Features:
    def __init__(self):
        self.log_write = LogMe()
        self.file_object = open("Logs/features_select_log.txt", "a+")

    def select(self, x_train, y_train):
        """
        Method name: select.
        Description: This method is going to help in feature selection.
        x_train: array or dataframe
        y_train: array or dataframe

        """
        try:
            # lassocv = LassoCV(alphas=[0.0001, 0.001, 0.002, 0.003, 0.004, 0.005], cv=20, max_iter=100000)
            # lassocv.fit(x_train, y_train)
            feature_select = SelectFromModel(Lasso(alpha=0.01, max_iter=1200, random_state=100))
            feature_select.fit(x_train, y_train)
            result = feature_select.get_support()
            selected_feature = x_train.columns[result]
            return selected_feature
        except Exception as e:
            self.log_write.logger(self.file_object, f"Error while selecting features function name select{e}")

    def vif_score(self, x):
        """
         Method name: vif_score.
         Description: This method is going to help in feature selection. We can drop the columns having values vif more than 5 to 10.
         This tells about multicollinearity
         x : dataframe
         """
        try:
            scalar = StandardScaler()
            arr = scalar.fit_transform(x)
            return pd.DataFrame([[x.columns[i], variance_inflation_factor(arr, i)] for i in range(arr.shape[1])],
                                columns=['Features', 'VIF_score'])

        except Exception as e:
            self.log_write.logger(self.file_object, f"Error while selecting features function name vif_score{e}")

    def vif_columns(self, x=None):
        """
              Method name: vif_columns.
              Description: This method is going to help in feature selection. We can drop the columns having values vif more than 5 to 10.
              This tells about multicollinearity. this will return the name of the columns
              x : dataframe
              """
        try:
            res = self.vif_score(x)
            result = res[res["VIF_score"] > 5]
            column = result["Features"]
            return column

        except Exception as e:
            self.log_write.logger(self.file_object, f"Error while selecting features function name vif_columns {e}")

    def mutual_information(self, x_train, y_train):
        """
              Method name: mutual_information.
              Description: This method is going to help in feature selection. this feature gives us the direct relationship between the target and the feature.
              x_train : dataframe
              y_train : dataframe or series
              """
        try:
            # determine the mutual information
            mutual_info = mutual_info_classif(x_train, y_train)
            mutual_info = pd.Series(mutual_info)
            mutual_info.index = x_train.columns
            mutual_info = mutual_info.sort_values(ascending=False)[:37]
            new_x = mutual_info.index
            x_train = x_train[new_x]
            return x_train

        except Exception as e:
            self.log_write.logger(self.file_object, f"Error while selecting features function name mutual_info {e}")

    def correlation(self, x_train, threshold):
        """
         Method name: correlation.
         Description: This method is going to help in feature selection. We can drop the columns by giving certain threshold.
         This tells about multicollinearity
         x_train : dataframe
         threshold: 0 to 1 (float) if giving 0.85 then 85%.
         """
        try:
            colum_corr = set()  # set of all names of correlated columns
            corr_matrix = x_train.corr()
            for i in range(len(corr_matrix.columns)):
                for j in range(i):
                    if abs(corr_matrix.iloc[i, j]) > threshold:  # we want absolute coefficient values
                        colum_name = corr_matrix.columns[i]  # getting the names of column
                        colum_corr.add(colum_name)
            colum_corr = list(colum_corr)
            return colum_corr

        except Exception as e:
            self.log_write.logger(self.file_object, f"Error while selecting features  function name correlation{e}")
