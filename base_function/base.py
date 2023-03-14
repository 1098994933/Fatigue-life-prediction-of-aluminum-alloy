import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet, SGDRegressor, LinearRegression
from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
def evaluate_model_plot(y_true, y_predict, residual= False ,show=True):
    """
    评价模型误差
    y_true: 真实值
    y_predict: 预测值
    """
    import numpy as np
    from sklearn.metrics import mean_squared_error
    from sklearn.metrics import r2_score
    import matplotlib.pyplot as plt

    if show == True:
        # 图形基础设置
        print("开始画图")
        plt.figure(figsize=(7, 5), dpi=400)
        plt.rcParams['font.sans-serif'] = ['Arial']  # 设置字体
        plt.rcParams['axes.unicode_minus'] = False  # 显示负号
        plt.grid(linestyle="--")  # 设置背景网格线为虚线
        ax = plt.gca()  # 获取坐标轴对象
        plt.scatter(y_true, y_predict, color='red')
        plt.plot(y_predict, y_predict, color='blue')
        #画200%误差线
        #plt.plot(y_predict, y_predict+0.301, color ='blue',linestyle = "--")
        #plt.plot(y_predict, y_predict-0.301, color ='blue',linestyle = "--")
        plt.xticks(fontsize=12, fontweight='bold')
        plt.yticks(fontsize=12, fontweight='bold')
        plt.xlabel("Measured", fontsize=12, fontweight='bold')
        plt.ylabel("Predicted", fontsize=12, fontweight='bold')
        #plt.xlim(4, 9)  # 设置x轴的范围
        #plt.ylim(4, 9)
        plt.savefig('./genetic.svg', format='svg')
        plt.show()

        # 预测值 真实值图
        #plt.subplot(1, 2, 1)
        #plt.scatter(y_true, y_predict, color='red')
        #plt.plot(y_predict, y_predict, color='blue')
        ##画200%误差线
        ##plt.plot(y_predict, y_predict+0.301, color ='blue',linestyle = "--")
        ##plt.plot(y_predict, y_predict-0.301, color ='blue',linestyle = "--")
        #plt.xticks(fontsize=12, fontweight='bold')
        #plt.yticks(fontsize=12, fontweight='bold')
        #plt.xlabel("Measured", fontsize=12, fontweight='bold')
        #plt.ylabel("Predicted", fontsize=12, fontweight='bold')
        #plt.xlim(4, 9)  # 设置x轴的范围
        #plt.ylim(4, 9)
        ##plt.title("fit effect",fontsize = 30)
        ## 残差分布图
        #plt.subplot(1, 2, 2)
        plt.figure(figsize=(7, 5), dpi=400)
        plt.hist(np.array(y_true)-np.array(y_predict),40)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.xlabel("Residual",fontsize = 20)
        plt.ylabel("Freq",fontsize = 20)
        #plt.title("Residual=y_true-y_pred",fontsize = 30)
        plt.show()
    from sklearn.metrics import mean_absolute_error
    
    n = len(y_true)
    MSE = mean_squared_error(y_true, y_predict)
    RMSE = pow(MSE,0.5)
    MAE = mean_absolute_error(y_true, y_predict)
    R2 = r2_score(y_true, y_predict)
           
    print("样本个数 ", round(n))
    print("均方根误差RMSE ", round(RMSE, 3))
    print("均方差MSE ", round(MSE, 3))
    print("平均绝对误差MAE ",round(MAE, 3))
    print("R2：", round(R2, 3))
    return dict({"n": n, "MSE": MSE, "RMSE": RMSE, "MSE": MSE, "MAE": MAE, "R2": R2})

def draw_corrheatmap(df):
    import seaborn as sns
    import matplotlib.pyplot as plt
    """
    画相关系数矩阵热力图
    """
    dfData = df.corr()
    plt.subplots(figsize=(9, 9)) # 设置画面大小
    sns.heatmap(dfData, annot=True, vmax=1, square=True, cmap="Blues")
    #plt.savefig('./BluesStateRelation.png')
    plt.show()

#计算立方根
def cbrt_override(x):
    """计算立方根"""
    if x>=0:
        return x**(1/3)
    else :
        return -(-x)**(1/3)


def regressorOp(X, Y):
    """
    This will optimize the parameters for the SVR
    X:X_train+X_validation
    Y:Y_train+Y_validation
    """
    from sklearn.svm import SVR
    from sklearn.model_selection import GridSearchCV
    regr_rbf = SVR(kernel="rbf")
    # 定义搜索范围 
    C = [pow(10,x) for x in range(-10, 6)]
    gamma =[pow(10,x) for x in range(-2, 3)] 
    epsilon = list(pow(10,x) for x in range(-3,3))
    
    parameters = {"C":C, "gamma":gamma, "epsilon":epsilon}
    
    gs = GridSearchCV(regr_rbf, parameters, scoring="neg_mean_squared_error")
    gs.fit(X, Y)
    
    print ("Best Estimator:\n", gs.best_estimator_)
    print ("Type: ", type(gs.best_estimator_))
    return gs.best_estimator_ 

# 数据读取和预处理 文件名dataset.csv 置于data目录下 返回字典
def read_df(df,target="target",sc=True):
    # import data
    dataset = df
    dataset = dataset.dropna()
    X = dataset.drop([target], axis=1).values
    Y = dataset[target].values
    if sc == True:
        # StandardScaler
        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler()
        X_transform = sc.fit_transform(X)
        result_dict = {"X": X, "Y": Y, "sc": sc, "df": dataset, "X_transform": X_transform}
    else:
	    result_dict = {"X": X, "Y": Y, "df": dataset}    
    return result_dict


# 相对误差
def mean_relative_error(y_true, y_pred):
    """calculate MRE"""
    import numpy as np
    relative_error = np.average(np.abs(y_true - y_pred) / y_true, axis=0)
    return relative_error
def Linear_SVR(C=1.0, gamma=0.1, epsilon=1):
    return Pipeline([
        ("std_scaler", StandardScaler()),
        ("model", SVR(kernel="linear", C=C, gamma=gamma, epsilon=epsilon))
    ])


def RBF_SVR(C=1.0, gamma=1, epsilon=1):
    return Pipeline([
        ("std_scaler", StandardScaler()),
        ("model", SVR(kernel="rbf", C=C, gamma=gamma, epsilon=epsilon))
    ])


def Poly_LinearRegression(degree=2):
    return Pipeline([('poly', PolynomialFeatures(degree=degree)),
                     ('linear', LinearRegression())])


def draw_feature_importance(features, feature_importance):
    """
    features: name
    feature_importance:
    """
    # make importances relative to max importance
    feature_importance = 100.0 * (feature_importance / feature_importance.max())
    sorted_idx = np.argsort(feature_importance)
    pos = np.arange(sorted_idx.shape[0]) + .5
    fig = plt.figure(dpi=400)
    plt.barh(pos, list(feature_importance[sorted_idx]), align='center')
    plt.yticks(pos, list(features[sorted_idx]), fontsize=5)
    plt.xlabel('Relative Importance')
    plt.title('Feature Importance')
    plt.show()


def model_fit_evaluation(model, x_train, y_train, x_test, y_test, n_fold=5):
    """clf:
    x_train：训练集+验证集 用于计算交叉验证误差  np.array
    y_train： np.array
    x_test：计算测试误差
    n_fold：交叉验证折数 default = 5
    """
    kf = KFold(n_splits=n_fold, shuffle=True, random_state=0)
    print(model)
    result = pd.DataFrame()
    for i, (train_index, test_index) in enumerate(kf.split(range(len(x_train)))):
        x_tr = x_train[train_index]
        y_tr = y_train[train_index]
        x_validation = x_train[test_index]  # get validation set
        y_validation = y_train[test_index]
        model.fit(x_tr, y_tr)

        result_subset = pd.DataFrame()  # save the prediction
        result_subset["y_validation"] = y_validation
        result_subset["y_pred"] = model.predict(x_validation)
        result = result.append(result_subset)
    print("cross_validation_error in validation set：")
    c = evaluate_model_plot(result["y_validation"], result["y_pred"], show=False)

    print("error in testing set：")
    model.fit(x_train, y_train)
    y_test_pred = model.predict(x_test)
    error_metric_testing = evaluate_model_plot(y_test, y_test_pred, show=False)  # 不画图
    print("====================================")
    return error_metric_testing


import itertools
import pandas as pd


def back_forward_feature_selection(model, X_train, Y_train, X_validation, Y_validation, metric):
    """X_train X_validation is dataFrame
    metric is evalucation function
    return metric for each step
    metric：误差计量函数 越小越好
    """
    # init
    # record result
    result_df = pd.DataFrame()
    features = X_train.columns
    best_score = 1e10
    best_features = features
    features_number = len(best_features)

    for i in range(len(features) - 1):
        # once back and find the best features for this number of features
        best_score = 1e10
        for sub_features in itertools.combinations(best_features, features_number - 1):
            sub_features = list(sub_features)
            model.fit(X_train[sub_features], Y_train)
            score = metric(Y_validation, model.predict(X_validation[sub_features]))
            df_line = pd.DataFrame(
                {"features": [",".join(sub_features)], "metric": score, "n_feature": len(sub_features)})
            result_df = result_df.append(df_line, ignore_index=True)
            if (best_score > score):
                best_score = score
                best_features = sub_features
        # for debug
        # print("best_features",best_features)
        # print("best_score",best_score)
        features_number = len(best_features)
    # find the best feature
    result_df = result_df.sort_values(by="metric", ascending=False)
    return result_df
