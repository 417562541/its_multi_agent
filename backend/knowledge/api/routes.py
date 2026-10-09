import asyncio
import os
import tempfile
import aiofiles
from fastapi import APIRouter, UploadFile, File, HTTPException
from backend.knowledge.api.schemas import UploadResponse
from backend.knowledge.services.ingestion.ingestion_processor import IngestionProcessor

router = APIRouter()
file_processor = IngestionProcessor()


@router.post("/upload", summary="上传文件到知识库", response_model=UploadResponse)
async def upload_file(file: UploadFile = File(...)):
    # 报文上传的文件到临时路径
    suffix = os.path.splitext(file.filename)[1]

    # 保存文件到临时路径（带原始扩展名）
    async with aiofiles.tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
        content = await file.read()
        await tmp_file.write(content)
        tmp_file_path = tmp_file.name
    # with 块退出后句柄已关闭，Windows 不再锁定该文件（解决 WinError32 的关键）
    try:
        # ingest_file 是同步阻塞逻辑，放入线程池执行，避免阻塞事件循环
        chunks = await asyncio.to_thread(file_processor.ingest_file, tmp_file_path)
        return UploadResponse(
            status="success",
            message="文件上传成功",
            file_name=file.filename,
            chunks_added=chunks
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"处理失败：{str(e)}")
    finally:
        if os.path.exists(tmp_file_path):
            os.remove(tmp_file_path)
