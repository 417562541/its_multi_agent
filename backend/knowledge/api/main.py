import uvicorn
from fastapi import FastAPI
from backend.knowledge.api.routes import router

# 挂载各个子模块路由
def create_fast_api() -> FastAPI:
    app = FastAPI()
    app.include_router(router)
    return app


if __name__ == '__main__':
    try:
        print("1. 启动Web服务器")
        uvicorn.run(app=create_fast_api(), host="127.0.0.1", port=8001)
        print("2. Web服务器启动成功")
    except:
        print("2. Web服务器启动失败")
