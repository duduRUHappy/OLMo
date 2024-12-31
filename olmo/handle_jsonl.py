import os
import json
# from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import ProcessPoolExecutor
import time
#以下是针对json文件安装dolmo要求数据形式进行加工

# 处理单个文件
def process_file(file_path, root_directory, output_dir, output_file_list_path, tag, useText):
    output_records = []
    base_name = os.path.basename(file_path)
    relative_path = os.path.relpath(os.path.dirname(file_path), "")
    # 构造新目录路径
    target_directory = os.path.join(output_dir, relative_path)
    # 创建新目录
    os.makedirs(target_directory, exist_ok=True)
    output_file = f"{base_name}"
    output_file_path = os.path.join(target_directory, output_file)

    file_path = os.path.join(root_directory, file_path)
    print(f"处理文件 {file_path}")
    try:
        row = 0
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                row += 1
                # if row > 3:
                #     break
                try:
                    record = json.loads(line)
                    # print(record)
                    content = {}
                    if useText:
                        content["text"] = record["text"]
                    else:
                        content["text"] = record
                    content["id"] =  str(row)
                    content["source"] = tag + relative_path.replace('/', '#')
                    output_records.append(content)
                except Exception as e:
                    print(f"Error processing line in {file_path}: {e}")


        # 保存结果到新文件
        if len(output_records) > 0:
            with open(output_file_list_path, "a", encoding="utf-8") as out_f_list:
                out_f_list.write(output_file_path + "\n")

            with open(output_file_path, "w", encoding="utf-8") as out_f:
                for record in output_records:
                    json.dump(record, out_f, ensure_ascii=False)
                    out_f.write("\n")

        print(f"Filtered records from {file_path} saved to {output_file_path}")

    except Exception as e:
        print(f"Error processing file {file_path}: {e}")




# 主函数：调用多进程
def main(root_directory, filelist_path, output_dir, output_file_list_path, useText=True, tag = "dclm_baseline_1.0_url_filter_code#"):
    with open(filelist_path, "r", encoding="utf-8") as filelist:
        file_paths = [line.strip() for line in filelist]

    # 使用 ProcessPoolExecutor 启动多进程
    with ProcessPoolExecutor(max_workers=16) as executor:
        futures = [
            executor.submit(
                process_file,
                file_path,
                root_directory,
                output_dir,
                output_file_list_path,
                useText = useText,
                tag = tag
            )
            for file_path in file_paths
        ]
        for future in futures:
            future.result()



def read_file_list(file_list_path):
    """读取 file_list 文件并随机打乱"""
    with open(file_list_path, 'r', encoding='utf-8') as f:
        files = [line.strip() for line in f]
    return files



if __name__ == "__main__":
    start_time = time.time()    


    # 替换为实际路径
    # root_directory = "/root/a100_nas/peixunban/001526_zdd/data/filtered/zstd"
    # filelist = "/root/a100_nas/peixunban/001526_zdd/data/filtered/zst_file_name"
    # output_dir = "/root/a100_nas/peixunban/001526_zdd/data/dolma/zstd"
    # output_file_list_path = "/root/a100_nas/peixunban/001526_zdd/data/dolma/zstd/file_list"


    # root_directory = "/root/a100_nas/peixunban/001526_zdd/data/filtered"
    # filelist = "/root/a100_nas/peixunban/001526_zdd/data/filtered/zst_file_name"
    # output_dir = "/root/a100_nas/peixunban/001526_zdd/data/dolma/zstd"
    # output_file_list_path = "/root/a100_nas/peixunban/001526_zdd/data/dolma/zstd/file_list"

    # root_directory = "/root/a100_nas/peixunban/001526_zdd/data/filtered"
    # filelist = "/root/a100_nas/peixunban/001526_zdd/data/filtered/file_list_absolute_rm_zstd"
    # output_dir = "/root/a100_nas/peixunban/001526_zdd/data/dolma/jsonl"
    # output_file_list_path = "/root/a100_nas/peixunban/001526_zdd/data/dolma/jsonl/file_list"

    root_directory = "/root/a100_nas/peixunban/001526_zdd/data/fasttext_scored_dclm_all_above_0.5"
    filelist = "/root/a100_nas/peixunban/001526_zdd/data/fasttext_scored_dclm_all_above_0.5/file_list_remove_zstd"
    output_dir = "/root/a100_nas/peixunban/001526_zdd/data/dolma/fasttext_scored_dclm_all_above_0.5"
    output_file_list_path = "/root/a100_nas/peixunban/001526_zdd/data/dolma/fasttext_scored_dclm_all_above_0.5/file_list_remove_zstd"

    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    main(root_directory, filelist, output_dir, output_file_list_path, useText = True, tag ="dclm_baseline_1.0_fasttext_filter_above_0.5_code#")
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"脚本总运行时间: {elapsed_time:.2f} 秒")