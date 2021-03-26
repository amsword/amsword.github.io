---
layout: post
comments: true
title: Python Copy-on-Write
---

The goal here is to test the memory usage with subprocess in python to have a
better understanding. Env is python 3.6. The tool we use for memory profiling
is `mprof`, and the command we use will be

```shell
mprof run --include-children python script/run.py test_memory_leak
mprof plot -o a.png
```

* baseline approach without subprocess

  ```python
  def sleep(x):
      for i in range(10):
          logging.info(x[i])
          time.sleep(1)
  
  def test_memory_leak():
      # mprof run --include-children python scripts/run6.py test_memory_leak
      x = [1] * (1024**3)
      x = tuple(x)
      all_p = []
      for i in range(0):
          p = mp.Process(target=sleep, args=(x,))
          #p = mp.Process(target=sleep)
          p.start()
          all_p.append(p)
      time.sleep(20)
      for p in all_p:
          p.join()
  ```


#![_config.yml](/images/config.png "_config.yml")
![_copyonrightbaseline.yml](/images/copyonrightbaseline.png "_copyonrightbaseline.yml")
