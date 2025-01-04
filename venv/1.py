import os
import json
import time
import google.generativeai as genai
import pandas as pd
from pyspark.sql import SparkSession
from concurrent.futures import ThreadPoolExecutor

# Configure Gemini  Key
genai.configure(_key='123')

program_start_time = time.time()
# 创建文件夹保存预测结果 Create a folder to save the prediction results
predictions_folder = "SparkcGemini/predictions"
if not os.path.exists(predictions_folder):
    os.makedirs(predictions_folder)

# 初始化Spark会话 Initialize Spark session
spark = SparkSession.builder.appName("GeminiPrediction").getOrCreate()

# 定义一个函数调用 Gemini  , Define a function call Gemini 
def get_gemini_predictions(data):
    model = genai.GenerativeModel("gemini-1.5-flash-8b")  # 替换为实际模型
    prompt = f"You are a data analyst, summarize all retail store user preference information and predict the next activity content, output within 200 words:{data}"
    response = model.generate_content(prompt)
    return response.text

# 定义处理每个批次的函数   Define the function for processing each batch
def process_batch(data_batch, file_name, batch_num):
    predictions = get_gemini_predictions(data_batch)
    
    # 保存每个批次的预测结果  Save the predicted results for each batch
    predictions_file = os.path.join(predictions_folder, f"predictions_{file_name.replace('.csv', f'_batch{batch_num}.json')}")
    with open(predictions_file, 'w') as f:
        json.dump(predictions, f)
    print(f"The prediction result has been saved as {predictions_file}")

# 指定读取文件的目录  Specify the directory to read files from
data_folder = "C:/Users/Dr.klika/Desktop/作业/SEM4/Big Data/SparkcGemini"
csv_files = [f for f in os.listdir(data_folder) if f.endswith('.csv')]

# 存储所有文件的预测结果  Store the predicted results of all files
all_predictions = []
 
# 逐个文件处理  Process files one by one
for file_name in csv_files:
    file_path = os.path.join(data_folder, file_name)
    
    # 使用Spark读取CSV文件  Using Spark to Read CSV Files
    data_spark = spark.read.csv(file_path, header=True, inferSchema=True)
    
    # 给每行数据添加索引  Add indexes to each row of data
    data_with_index = data_spark.rdd.zipWithIndex().toDF()  # 为每行数据添加索引
    
    # 分批次处理，每次取1000行数据  Batch processing, taking 1000 rows of data each time
    batch_size = 1000
    num_batches = data_with_index.count() // batch_size + 1  # 计算批次数量
    
    # 使用线程池并行处理批次  Parallel processing of batches using thread pools
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = []
        
        for i in range(num_batches):
            # 获取每批次的数据  Obtain data for each batch
            batch_data = data_with_index.filter((data_with_index["_2"] >= i * batch_size) & (data_with_index["_2"] < (i + 1) * batch_size))
            data_list = batch_data.toPandas().drop("_2", axis=1).to_dict(orient="records")  # 去除索引列并转为字典
            
            # 提交任务到线程池  Submit task to thread pool
            futures.append(executor.submit(process_batch, data_list, file_name, i + 1))
        
        # 等待所有线程完成 Waiting for all threads to complete
        for future in futures:
            future.result()  
            
        time.sleep(2)  # 避免调用过于频繁 

    # 删除处理完的CSV文件
    os.remove(file_path)
    print(f"Deleted file {file_name}")

# 将所有预测结果汇总到一个文件
combined_predictions_file = os.path.join(predictions_folder, "combined_predictions.json")
with open(combined_predictions_file, 'w') as f:
    json.dump(all_predictions, f)

print(f"All predicted results have been summarized and saved to {combined_predictions_file}")


program_end_time = time.time()  # 记录程序结束时间

print(f"Total running time of the program: {program_end_time - program_start_time:.2f} 秒")