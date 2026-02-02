import os
import glob

# 这是你刚才设置的路径
root = "/data/tangchengwen/Deepfake视频检测/Dataset/FFPP/c23"

print(f"检查根目录是否存在: {os.path.exists(root)}")

# if os.path.exists(root):
#     print("根目录下的文件/文件夹:")
#     print(os.listdir(root))
    
#     # 尝试构建下一级路径 (根据你的截图，下一级应该是 manipulated_sequences 和 original_sequences)
#     next_level = os.path.join(root, "original_sequences")
#     print(f"\n检查下一级目录: {next_level}")
#     if os.path.exists(next_level):
#         print(os.listdir(next_level))
#     else:
#         print("original_sequences 没找到！可能名字拼写错误？")

#     # 尝试直接 glob 搜索
#     print("\n尝试 glob 搜索...")
#     mp4s = glob.glob(os.path.join(root, "**/*.mp4"), recursive=True)
#     print(f"找到 .mp4 文件数量: {len(mp4s)}")
#     if len(mp4s) > 0:
#         print(f"第一个文件: {mp4s[0]}")
# else:
#     print("错误：根目录路径不存在！请检查是否有中文乱码或拼写错误。")