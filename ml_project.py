import pandas as pd
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,roc_auc_score
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings("ignore")
from sklearn.datasets import make_classification

from sklearn import pipeline

def calculate_vif(df):
    columns = df.columns
    vif_df = pd.DataFrame()
    vif_df["features"] = df.columns
    vif_df["score"] = [variance_inflation_factor(df.values,i) for i in range(len(columns))]
    return vif_df

x,y = make_classification(
    n_classes=10,
    n_features=13,
    n_informative=5,
    n_redundant=2,
    n_samples=1000,
    random_state=42
)

feature_names = [f"feature_{i}" for i in range(x.shape[1])]
x = pd.DataFrame(x,columns=feature_names)
# Adding two categorical features
x['cat_feature_1'] = np.random.choice(['A', 'B', 'C'], size=x.shape[0])
x['cat_feature_2'] = np.random.choice(['X', 'Y'], size=x.shape[0])
#print(x.columns)


# # x = np.random.rand(1000,5)
# # y = np.random.randint(0,2,size=1000)
# data = pd.read_csv(r"C:\Projects\synthetic_classification_dataset.csv")
# #print(data.shape)
# x = data.drop("target",axis = 1)
# y = data["target"]

# #Handling missing data
# x["num_feature_1"][:10] = None
# x["num_feature_2"][:20] = None
# x["num_feature_3"][:30] = None
# x["num_feature_4"][:4] = None
# x["num_feature_5"][:2] = None


# x["num_feature_1"].fillna(x["num_feature_1"].median(),inplace=True)


features_with_null = x.isnull().sum()
features_with_null_df = pd.DataFrame()
features_with_null_df["features"] = x.columns
features_with_null_df["null_count"] = list(features_with_null)
#features_with_null_df = pd.DataFrame(features_with_null_df)
#print(features_with_null_df)
# print(x.describe())
for index,row in features_with_null_df.iterrows():
    if row["null_count"] >0:
        x[row["features"]].fillna(x[row["features"]].median(), inplace = True)

#print(x.isnull().sum())


#EDA
columns = x.columns
# print(x["num_feature_1"].dtype)
numerical_feature_list = []
for col in columns:
    if x[col].dtype == "float64" :
        numerical_feature_list.append(col)
print(numerical_feature_list)



# perform feature removal based on vif

threshold = 5
x_numerical_features = x[numerical_feature_list]

vif_mat = calculate_vif(x_numerical_features)
high_vif = True if vif_mat["score"].max() > threshold else False


while high_vif:
    vif_mat = calculate_vif(x_numerical_features)
    #print(vif_mat)
    max_vif = vif_mat["score"].max()
    if max_vif > threshold:
        feature_to_remove = vif_mat.sort_values("score",ascending=False)["features"].iloc[0]
        #print(feature_to_remove)
        x_numerical_features = x_numerical_features.drop(feature_to_remove,axis=1)
        #print(x_numerical_features.columns)
    else:
        high_vif = False

# After VIF feature reduction, append categorical columns to x_numerical_features
x_numerical_features = x_numerical_features.reset_index(drop=True)
categorical_features_df = x[['cat_feature_1', 'cat_feature_2']].reset_index(drop=True)
x_new = pd.concat([x_numerical_features, categorical_features_df], axis=1)

numerical_features = x_numerical_features.columns
categorial_features = ["cat_feature_1","cat_feature_2"]

#Preprocessor for processing numerical and categorical features
numerical_transformer = StandardScaler()
catergorical_transformer = OneHotEncoder()

preprocessor = ColumnTransformer(
    transformers=[
        ("num",numerical_transformer,numerical_features),
        ("cat",catergorical_transformer,categorial_features)
    ]
)

#pipeline

pipeline = Pipeline(
    [
        ('preprocessor',preprocessor),
        ('classifier',RandomForestClassifier(random_state=42))
    ]
)

x_train,x_test,y_train,y_test = train_test_split(x_new,y,test_size=0.20,random_state=42)
pipeline.fit(x_train,y_train)

y_pred = pipeline.predict(x_test)
print("Accuracy score:",accuracy_score(y_test,y_pred))

# # scaler = StandardScaler()
# # scaled_data = scaler.fit_transform(x[numerical_feature_list])
# # x_new = np.hstack((scaled_data,x[["cat_feature_1","cat_feature_2"]]))

# # x_scaled = pd.DataFrame(x_new,columns=numerical_feature_list + ["cat_feature_1","cat_feature_2"])


# # #print(calculate_vif(x_scaled))
# # #print(x.corr())



# # x_new = np.hstack((x_numerical_features,x[["cat_feature_1","cat_feature_2"]]))

# # x_scaled = pd.DataFrame(x_new,columns=list(x_numerical_features.columns) + ["cat_feature_1","cat_feature_2"])







# # #print(y.head())

# # #data = pd.DataFrame(x, columns=["feature_1","feature_2","feature_3","feature_4","feature_5"])

# # model = RandomForestClassifier()

# # x_train,x_test,y_train,y_test = train_test_split(x_scaled,y,test_size=0.20,random_state=42)

# # model.fit(x_train,y_train)
# # pred = model.predict(x_test)

# # importances = model.feature_importances_

# # feature_importances = pd.DataFrame({"features" : list(x_scaled.columns), "importances":importances} )

# # print(feature_importances)

# # # print("Accuracy score:",accuracy_score(y_test,pred))
# # # print("Precision score: ",precision_score(y_test,pred,average="weighted"))
# # # print("Recall score:",recall_score(y_test,pred,average="weighted"))
# # # print("ROC AUC score:", model.predict_proba(x_test)[:,1])


# # # #print(x_train.shape)

# # # param_grid = {
# # #     'n_estimators': [100, 200, 300],
# # #     'max_features': ['auto', 'sqrt', 'log2'],
# # #     'max_depth': [None, 10, 20, 30],
# # #     'min_samples_split': [2, 5, 10],
# # #     'min_samples_leaf': [1, 2, 4]
# # # }

# # # grid_search = GridSearchCV(estimator=model, param_grid=param_grid, 
# # #                            cv=3, n_jobs=-1, verbose=2, scoring='accuracy')
# # # grid_search.fit(x_train, y_train)
# # # print("Best parameters:", grid_search.best_params_)
# # # print("Best_scores:",grid_search.best_score_)

# # # model_xgb = XGBClassifier()
# # # model_xgb.fit(x_train, y_train)
# # # pred_xgb = model_xgb.predict(x_test)

# # # print("XGBoost Accuracy:", accuracy_score(y_test, pred_xgb))


# # # To do
# # #perform transformations on the data based on numerical and categorical features
# # # create column transformers for that
# # #perform statistical test on the independent features
# # #know about evaluating the performance of the multiclass classification models
# # # learn some plottings