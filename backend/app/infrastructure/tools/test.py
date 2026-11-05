import asyncio
import threading
import time

async def hello():
    await time.sleep(1)
    return "ok"

def worker(loop):
    # 子线程：提交协程给主线程的loop，子线程同步阻塞等结果
    fut = asyncio.run_coroutine_threadsafe(hello(), loop)
    print("子线程拿到结果：", fut.result())

async def main():
    loop = asyncio.get_running_loop()
    t = threading.Thread(target=worker, args=(loop,))
    t.start()
    t.join()

asyncio.run(main())
