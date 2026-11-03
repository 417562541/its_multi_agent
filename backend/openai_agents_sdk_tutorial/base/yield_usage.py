import time


def simple_generator():
    i = 0
    while True:
        i +=1
        yield i

gen = simple_generator()
# ======第一种获取方式=====
while True:
    time.sleep(1)
    print(f'--获取到了第{next(gen)}个数据--')

# ======第二种获取方式=====
# ====== for 循环会自动调用 next() 函数，因此这里不需要显式调用 next() ======
for item in gen:
    time.sleep(1)
    print(f'--获取到了第{item}个数据--')


