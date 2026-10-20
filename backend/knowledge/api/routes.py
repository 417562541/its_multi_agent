import asyncio
import logging
import os
import tempfile
import aiofiles
from fastapi import APIRouter, UploadFile, File, HTTPException
from backend.knowledge.api.schemas import UploadResponse, QueryResponse, QueryRequest
from backend.knowledge.services import retrieval_service, query_service
from backend.knowledge.services.ingestion.ingestion_processor import IngestionProcessor
from backend.knowledge.services.query_service import QueryService
from backend.knowledge.services.retrieval_service import RetrievalService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter()
retrieval_service = RetrievalService()
query_service = QueryService()
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

@router.post("/query", response_model=QueryResponse, summary="查询知识库")
async def query(request: QueryRequest):
    """
    查询知识库
    Args:
        request: 用户的输入请求

    Returns:
        QueryResponse： 模型的结果以及原始问题

    """
    try:
        # 1. 判断用户问题
        user_question = request.question
        if not user_question:
            raise HTTPException(status_code=500, detail="查询问题不存在")

        # 2. 调用检索器的检索方法
        retrieval_context = retrieval_service.retrieval(user_question)

        # 3. 调用查询器的查询方法
        answer = query_service.generate_answer(user_question, retrieval_context)

        # 4. 封装到响应数据模型
        return QueryResponse(
            question=user_question,
            answer=answer
        )
    except Exception as e:
        logger.error(f"调用查询知识库服务失败:原因:{str(e)}")
        raise HTTPException(status_code=500, detail="服务内部出现异常")
