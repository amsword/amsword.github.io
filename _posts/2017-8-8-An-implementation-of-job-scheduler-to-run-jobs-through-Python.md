---
layout: post
title: An implementation of job scheduler to run jobs through Python
---

Sometimes, we would like to run J jobs on C computing resources, but each
resource can have only one running job at the same time. 
For exmaple,
we want to train the deep learning model with 10 different parameter values, 
but we only have
2 GPUs. 

Here, I present a class, named JobScheduler, to do it in a handy way. The usage is like 
```python
all_resource = [0, 1]
all_task = map(lambda x: x * 0.01, range(10))

def process(resource, task):
    # run your job here
    pass

b = JobScheduler(all_resource, all_task, process)
b.start()
b.join()
```

```python
import multiprocessing as mp
class JobScheduler(object):
    def __init__(self, all_resource, all_task, processor):
        self._all_task = Queue()
        for task in all_task:
            self._all_task.put(task)

        self._all_resouce = Queue()
        for resource in all_resource:
            self._all_resouce.put(resource)

        self._processor = processor

    def start(self):
        self._scheduler = mp.Process(target=self._schedule, args=())
        self._scheduler.start()

    def join(self):
        self._scheduler.join()

    def _schedule(self):
        self._in_progress = []
        while True:
            in_progress = []
            for resource, p in self._in_progress:
                if not p.is_alive():
                     p.join()
                     self._all_resouce.put(resource)
                else:
                    in_progress.append((resource, p))
            self._in_progress = in_progress

            if self._all_task.empty():
                break
            if self._all_resouce.empty():
                time.sleep(5)
                continue
            task = self._all_task.get()
            resource = self._all_resouce.get()
            print 'here'
            p = mp.Process(target=self._processor, args=(resource, task))
            p.start()
            self._in_progress.append((resource, p))

        for resource, p in self._in_progress:
            p.join()
```
