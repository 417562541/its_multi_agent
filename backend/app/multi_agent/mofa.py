import asyncio


class SyncResource:
    """同步上下文管理器：__enter__ / __exit__"""
    def __enter__(self):
        print("[同步] __enter__ 获取资源")
        return self

    def do_work(self):
        print("[同步] 执行业务")

    def __exit__(self, exc_type, exc_val, exc_tb):
        print("[同步] __exit__ 释放资源")
        if exc_type:
            print(f"[同步] 发生异常: {exc_val}")
        return False


class AsyncResource:
    """异步上下文管理器：__aenter__ / __aexit__"""
    async def __aenter__(self):
        print("[异步] __aenter__ 获取资源")
        # 这里可以写 await 操作，比如连接数据库
        await asyncio.sleep(0.1)
        return self

    async def do_work(self):
        print("[异步] 执行业务")
        await asyncio.sleep(0.1)

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        print("[异步] __aexit__ 释放资源")
        await asyncio.sleep(0.1)
        if exc_type:
            print(f"[异步] 发生异常: {exc_val}")
        return False


async def async_main():
    print("===== 测试 async with =====")
    async with AsyncResource() as res:
        await res.do_work()

    print("\n===== 测试 async with（内部报错） =====")
    try:
        async with AsyncResource() as res:
            await res.do_work()
            1 / 0
    except ZeroDivisionError as e:
        print(f"外层捕获异常：{e}")


if __name__ == "__main__":
    print("===== 测试同步 with =====")
    with SyncResource() as r:
        r.do_work()

    print("\n===== 启动异步事件循环 =====")
    asyncio.run(async_main())
