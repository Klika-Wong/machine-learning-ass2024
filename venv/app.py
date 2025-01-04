from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import RandomForestRegressor
from pyspark.sql.functions import col
import os

# 初始化SparkSession
spark = SparkSession.builder \
    .appName("用户偏好预测") \
    .config("spark.executor.memory", "4g") \
    .config("spark.driver.memory", "2g") \
    .config("spark.memory.offHeap.enabled", True) \
    .config("spark.memory.offHeap.size", "1g") \
    .getOrCreate()

# 加载数据
data_path = "data path"
data = spark.read.csv(data_path, header=True, inferSchema=True)

# 打印原始列名以检查是否有额外的空格或格式问题
print("原始数据集中的列名:", data.columns)

# 确保数据中包含所需的列，格式化列名以统一
data = data.toDF(*[col.strip().replace(" ", "_").lower() for col in data.columns])

# 确认格式化后的列名
print("处理后的列名:", data.columns)

# 选择需要的列并进行计算
data = data.select(
    "date",
    "units_sold",
    "discount_percentage",
    "sales_revenue",
    "store_location",
    "product_category",
    "day_of_the_week",
    "holiday_effect",
    "marketing_spend"  # 确保这个列被选择用于计算
)

# 将 `holiday_effect` 列转换为数值类型
data = data.withColumn("holiday_effect", col("holiday_effect").cast("double"))

# 计算 'Sales Revenue - Marketing Spend'
# 确保列名一致，使用正确的列名
data = data.withColumn(
    "adjusted_sales_revenue",
    col("sales_revenue") - col("marketing_spend")  # 使用正确的列名
)

# 数据重新分区
data = data.repartition(200)  # 根据数据量调整分区数，以优化内存使用

# 使用 VectorAssembler 组合特征列
assembler = VectorAssembler(
    inputCols=["units_sold", "discount_percentage", "adjusted_sales_revenue"], 
    outputCol="features"
)
data = assembler.transform(data).select("features", "holiday_effect")  # 'holiday_effect'作为标签

# 缓存数据以提高效率
data.cache()

# 定义随机森林模型
rf = RandomForestRegressor(
    labelCol="holiday_effect",  # 'holiday_effect'作为标签
    featuresCol="features",
    maxDepth=10,
    maxBins=32,
    numTrees=50
)

# 模型训练
model = rf.fit(data)

# 保存最佳模型
model_save_path = "data path"
if not os.path.exists(os.path.dirname(model_save_path)):
    os.makedirs(os.path.dirname(model_save_path))

model.write().overwrite().save(model_save_path)

print(f"模型已保存至: {model_save_path}")

# 清理缓存
data.unpersist()

# 停止 Spark 会话
spark.stop()
