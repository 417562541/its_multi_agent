import os

import hashlib
import time
from backend.knowledge.config.settings import settings
from backend.knowledge.repositories.file_repository import FileRepository
from backend.knowledge.services.ingestion.ingestion_processor import IngestionProcessor
from tqdm import tqdm


def main():
    print("1. 开始运行")
    file_repository = FileRepository()
    files_path = file_repository.list_files(settings.CRAWL_OUTPUT_DIR)
    print(f"2.扫描到指定目录下的文件数:{len(files_path)}")
    unique_files_path = file_repository.remove_duplicate_files(files_path)
    print(f"3.扫描到指定目录下的唯一的文件数:{len(unique_files_path)}")
    ingestion_processor = IngestionProcessor()
    success = 0
    fail = 0
    start_time = time.time()
    with tqdm(unique_files_path, desc="知识库上传进度统计") as pbar:
        for unique_file_path in pbar:
            try:
                ingestion_processor.ingest_file(unique_file_path)
                success += 1
            except Exception as e:
                fail += 1
            finally:
                pbar.set_postfix({"success": success, "fail": fail})

    print(f"4.最终入库成功的结果:成功:{success}---失败:{fail}")
    print(f"5.最终入库完成，耗时:{time.time() - start_time:.2f}秒")
if __name__ == '__main__':
    main()
