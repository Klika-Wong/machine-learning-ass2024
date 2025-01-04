import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
import joblib
import os

# 打印Python环境信息
import sys
print(sys.executable)

# 打印当前工作目录
print(os.getcwd())

# 载入数据
data = pd.read_csv(f"SparkcGemini\\Retail_sales.csv")

# 去除可能的空格
data.columns = data.columns.str.strip()

# 打印列名以确保没有空格
print(data.columns)

# 将特征和标签分开
X = data.drop(['Units Sold', 'Date'], axis=1)  # 'Units Sold'是目标变量，'Date'不需要
y = data['Units Sold']  # 'Units Sold'作为目标变量

# 分割数据集为训练集和测试集（70% 训练，30% 测试）
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 创建数据预处理流水线
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), ['Sales Revenue (USD)', 'Discount Percentage', 'Marketing Spend (USD)']),
        ('cat', OneHotEncoder(), ['Store Location', 'Product Category', 'Day of the Week', 'Holiday Effect'])
    ])

# 创建一个包含预处理和模型的流水线
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])

# 训练模型
model.fit(X_train, y_train)

# 进行预测
y_pred = model.predict(X_test)

# 评估模型
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)  # 计算r2_score
print(f'Mean Squared Error: {mse:.2f}')
print(f'R-squared: {r2:.2f}')  # 打印r2_score作为模型准确度

# 使用网格搜索调优超参数
param_grid = {
    'regressor__n_estimators': [50, 100, 200],
    'regressor__max_depth': [10, 20, 30],
    'regressor__min_samples_split': [2, 5, 10]
}

# 网格搜索
grid_search = GridSearchCV(model, param_grid, cv=5)
grid_search.fit(X_train, y_train)

# 输出最佳参数
print("Best parameters found: ", grid_search.best_params_)

# 使用最佳参数训练模型
best_model = grid_search.best_estimator_
y_pred_best = best_model.predict(X_test)

# 评估最佳模型
mse_best = mean_squared_error(y_test, y_pred_best)
r2_best = r2_score(y_test, y_pred_best)  # 计算最佳模型的r2_score
print(f'Optimized Mean Squared Error: {mse_best:.2f}')
print(f'Optimized R-squared: {r2_best:.2f}')  # 打印最佳模型的r2_score作为准确度

# 定义保存模型的简单路径
model_save_path = 'C:\\temp\\retail_sales_model.pkl'  # 修改为一个简单的路径

# 尝试保存模型并捕获异常
try:
    joblib.dump(best_model, model_save_path)
    print(f"Model saved to '{model_save_path}'")
except Exception as e:
    print(f"Error saving model: {e}")
