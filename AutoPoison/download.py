from modelscope.hub.snapshot_download import snapshot_download

# 1. 指定你要下载的模型名称
model_id = "sdudaq/phi-2_int8_inject_injected_removed"

# 2. 指定你要保存到的目标文件夹路径（如果不存在，系统会自动创建）
target_folder = "./output/models/inject/phi-2/injected_removed_int8/checkpoint-last" 
print(f"正在将模型下载到: {target_folder} ...")

# 3. 执行下载
# local_dir 参数就是用来指定下载目录的
model_dir = snapshot_download(model_id, local_dir=target_folder)

print(f"下载完成！文件已保存在: {model_dir}")