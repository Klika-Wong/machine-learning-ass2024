from pyspark.sql import SparkSession
from google.cloud import storage
import json
import requests
import os
import time

program_start_time = time.time()
# 创建 SparkSession
spark = SparkSession.builder.appName("Data Segmentation").getOrCreate()

# 创建一个 Storage 客户端（需要 Google Cloud Storage）
storage_client = storage.Client()

# 文件路径设置
chunks_path = "//"

# 读取分割的 CSV 文件路径
def upload_to_gcs(local_file_path, bucket_name, destination_blob_name):
    # 上传本地文件到 Google Cloud Storage
    print(f"Uploading {local_file_path} to GCS bucket {bucket_name} as {destination_blob_name}...")
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(local_file_path)
    print(f"File uploaded: {destination_blob_name}")

def predict_user_preferences(chunk_data):
    # 假设使用一个 REST API 与 Google Gemini 进行交互
    print("Sending data to Gemini API for prediction...")
    url = "https://your-gemini-endpoint.com/predict"  # 替换为实际的 Gemini 预测 API
    headers = {"Authorization": "Bearer YOUR_API_KEY"}
    response = requests.post(url, data=json.dumps(chunk_data), headers=headers)

    if response.status_code == 200:
        # 返回预测结果（限制为前 50 个 promote）
        print(f"Prediction successful for chunk. Returning top 50 results.")
        return response.json()[:50]  # 假设预测结果是按排序返回的
    else:
        print(f"Error during prediction: {response.status_code}")
        return []

# 读取每个 chunk 的数据并进行处理
for i in range(1, 302):  # chunk_1 到 chunk_301
    chunk_file = f"chunk_{i}.csv"
    chunk_path = os.path.join(chunks_path, chunk_file)
    
    # 检查文件是否存在
    if os.path.exists(chunk_path):
        print(f"Processing {chunk_file}...")

        # 读取 CSV 数据
        chunk_df = spark.read.csv(chunk_path, header=True, inferSchema=True)
        chunk_data = chunk_df.toPandas()  # 转换为 pandas 数据框，适合与 API 一起使用
        
        # 将数据上传到 Google Cloud Storage（可选）
        upload_to_gcs(chunk_path, "your-bucket-name", f"chunk_{i}.csv")

        # 预测用户偏好
        predictions = predict_user_preferences(chunk_data)

        # 将预测结果保存到文件
        predictions_file = f"predictions_chunk_{i}.json"
        with open(predictions_file, 'w') as f:
            json.dump(predictions, f)
        print(f"Prediction results saved for {chunk_file} in {predictions_file}.")
    else:
        print(f"File chunk_{i}.csv does not exist.")

# 整合所有的预测结果
print("Combining all prediction results...")
all_predictions = []
for i in range(1, 302):
    predictions_file = f"predictions_chunk_{i}.json"
    if os.path.exists(predictions_file):
        with open(predictions_file, 'r') as f:
            all_predictions.extend(json.load(f))

# 将整合的预测结果上传到 Google Gemini 或其他服务
combined_predictions_file = "combined_predictions.json"
with open(combined_predictions_file, 'w') as f:
    json.dump(all_predictions, f)

# 打印结果汇总
print(f"All prediction results combined and saved to {combined_predictions_file}.")
print(f"Total {len(all_predictions)} predictions were processed.")

# 上传整合的预测结果到 Google Cloud Storage
upload_to_gcs(combined_predictions_file, "your-bucket-name", combined_predictions_file)

# 停止 SparkSession
spark.stop()
print("Spark session stopped. Process complete.")
