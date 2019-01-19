import gevent
import random
import time
from gevent import monkey

#将所有耗时操作的代码，换为gevent中自己的模块
monkey.patch_all()

def funtest(funname):
    for i in range(100):
        print(funname, i)
        time.sleep(random.random())

gevent.joinall([
    gevent.spawn(funtest, "fun_test1"),
    gevent.spawn(funtest, "fun_test2"),
    gevent.spawn(funtest, "fun_test3")
])