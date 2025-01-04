import google.generativeai as genai
import os
import json
import time

program_start_time = time.time()
# 配置 API 密钥
genai.configure(api_key='')


# 预测结果文件所在的目录
predictions_dir = "C:/predictions/"

# 获取目录下所有的JSON文件路径（假设它们都是预测结果文件）
files = [os.path.join(predictions_dir, filename) for filename in os.listdir(predictions_dir) if filename.endswith(".json")]

# 读取所有预测文件的数据
all_texts = []
for file in files:
    with open(file, "r", encoding="utf-8") as f:
        all_texts.append(f.read())  # 读取文件内容作为字符串

# 汇总所有预测文件的内容
predictions_summary = "\n".join(all_texts)

# 创建Gemini API请求的提示内容
prompt = f"""
Please generate a comprehensive report summarizing the following predictions from multiple batches:
{predictions_summary}
The dataset includes retail sales forecasts for various product categories, regions, and dates. Summarize the key insights, overall trends, and any notable patterns identified across these batches of predictions.
The report should include:
1.An overview of the predictions' structure.  2.Key findings from the aggregated data.  3.Recommendations or implications based on the trends observed.
Please ensure the output is in plain text format, not in code.
Thank you!
"""

# 创建Gemini API模型
model = genai.GenerativeModel('gemini-pro')

# 调用Gemini API生成报告，直接传递prompt参数
response = model.generate_content(prompt)  # 不带参数名称

if response._done:  # 使用 _done 代替 done
    report = response._result.candidates[0].content.parts[0].text
    print(report)  # 打印生成的报告内容



program_end_time = time.time()  # 记录程序结束时间

print(f"程序总运行时间: {program_end_time - program_start_time:.2f} 秒")