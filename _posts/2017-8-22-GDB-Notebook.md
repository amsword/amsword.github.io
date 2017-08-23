---
layout: post
comments: true
title: GDB notebook
---

* How to write a script to execute some command

```shell
(gdb) define max_elephant
Redefine command "max_elephant"? (y or n) y
Type commands for definition of "max_elephant".
End with a line saying just "end".
>    set $max_value = 0
>    set $max_i = -1
>    set $i = 0
>    while $i < 507
 >        if probs[$i][20] > $max_value
  >            print $i
  >            print probs[$i][20]
  >            set $max_value = probs[$i][20]
  >            set $max_i = $i
  >        end
 >        set $i = $i + 1
 >    end
>end
(gdb) max_elephant
$24 = 340
$25 = 0.25033778
$26 = 352
$27 = 0.293604016
$28 = 353
$29 = 0.294953734
(gdb) print $max_i
$30 = 353
```

* How to print an array

```shell
print *ptr@length
```

