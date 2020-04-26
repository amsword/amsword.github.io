---
layout: post
comments: true
title: Linux Tips
---

## General
* How to test the connectivity of two machines
  * On server side, listen to the port of, e.g. 23456
  ```shell
  nc -l 23456
  ```
  * on the client side, connect it with this command
  ```shell
  nc -v server_ip 23456
  ```

* How to write multi-lines in yaml
  use |
  ```yaml
  a: |
     abc
     efg
  ```

* How to setup VPN
  check [this tutorial](https://www.digitalocean.com/community/tutorials/how-to-set-up-an-openvpn-server-on-ubuntu-16-04)

* How to change the timestamp of the file
  ```bash
  touch -d "2012-10-19 12:12:12.000000000 +0530" tgs.txt
  ```
* How to change the timestamp of all files under a folder
  ```bash
  cd folder
  touch -d "2012-10-19 12:12:12.000000000 +0530" *
  ```
* How to list all package versions in conda
  ```bash
  conda search pytorch-nightly -c pytorch
  ```

* How to check system log
  * Location: /var/log/
  * use logrotate to automatically remove the logs

* How to limit the CPU usage of a process to make it slower
  ```bash
  cpulimit -p 157407 -l 5
  ```
  make the process of 157407 to use at most 5% of CPU.
* apex install fails in maskrcnn
  ```bash
  conda install cryptacular
  pip install apex

  ```
* Check the output every 2 seconds
```bash
watch "bash -c './philly.py -wl query &> /tmp/wu1.txt && tail -n 40 /tmp/wu1.txt'"
```

* How to test the read speed
```bash
pv trainval.tsv | md5sum
rsync trainval.tsv /tmp/ --progress
```

* How to check where the nccl2 is installed
```bash
dpkg -L libnccl2
```

* How to debug the issue in pip install
    * Download the source code and run
      ```bash
      python setup.py install
      ```
      which will crash and you can find the soruce code easily

* How to check cuda version
```bash
cat /usr/local/cuda/version.txt
```
or
```bash
nvcc --version
```

* How to fix the following error
```bash
cuda_runtime_api.h:9156:60: error: ‘cudaGraphExec_t’ was not declared in this scope
 extern __host__ cudaError_t CUDARTAPI cudaGraphExecDestroy(cudaGraphExec_t graphExec);
```
Check out [this link](https://devtalk.nvidia.com/default/topic/1045857/tensorrt/onnx-tensorrt-build-failure/)
by 
I open the file:
/usr/include/cudnn.h

And I changed the line:
```c
#include "driver_types.h"
```
to:
```c
#include <driver_types.h>
```


* How to check cudnn version
```bash
cat /usr/local/cuda/include/cudnn.h | grep CUDNN_MAJOR -A 2
```
or
```bash
cat /usr/include/cudnn.h | grep CUDNN_MAJOR -A 2
```

* disk space check
```bash
du --max-depth=1 -h | sort -h # check which folder uses the most
```
the above command does not print any information until all information is
collected. Sometimes, there might be lots of files in the folder and it might
be stuck for quite a long time. In this case, we can just run `du -h` to print
out all the scanning. This command will tell which folder contains lots of
files. Then, use the option of --exclude to check other folders, e.g.
```bash
du --exclude=./abc
```

* How to test the network speed
  * use iperf
    * Node 1 as server
    ```bash
    sudo apt-get install iperf
    iperf -s
    ```
    * Node 2 as client to connect with node 1
      ```bash
      sudo apt-get install iperf
      iperf -c node_1_ip
      ```


## Mount between linux and linux
* How to export a folder from the server side
    * Install the package
    ```shell
    sudo apt install nfs-kernel-server
    ```
    * Add the following to /etc/exports. The IP address should be the
client IP address
        * /server_folder       157.54.146.96(rw,sync,no_root_squash,no_subtree_check) 
    * Restart the service by
    ```shell
    sudo exportfs -a
    ```
* How to mount the remote folder on the local side
    * Add the following to the /etc/fstab
        * server_ip:/server_folder         /local_folder   nfs    auto    0       0

## Power Option
* How to restart the computer after the power recovers
    * The option is in Bios setting. Find the options related with power.

## Time/Date
* How to set the time in command line
check [this](https://askubuntu.com/questions/920361/unable-to-change-date-and-time-settings-in-ubuntu-16-04-using-command-line)
```bash
sudo timedatectl set-ntp 0
sudo timedatectl set-time "2017-05-30 18:17:16"
```

## Search keyword in files
```bash
grep '\[170, 11, 269, 160\]' *.label.tsv
```


## Samba
* How to start
```bash
sudo service smbd start
```
* How to stop
```bash
sudo service smbd stop
```
* How to restart
```bash
sudo service smbd restart
```

## Crontab
* How to make the running env in crontab job the same as when the uesr logins
  ```bash
  bash -i -l -c 'your command'
  ```
  -i: run the shell interactively
  -l: run the ~/.profile to populate the environment variables

* How to debug the command
  * Mimic the environment the crontab job uses by running the following command
    in crontab job to get the environment variable list
    ```bash
    env > ~/env.txt
    ```
  * Run the sh with those envs.
    ``` bash
    env - `cat env.txt` /bin/sh
    ```
    After it gives you the shell, type any command you use in the crontab jobs

* How to run a job at reboot
  Using @reboot
  ```bash
  @reboot bash -i -l -c 'cd $HOME && ls' 2>&1 > $HOME/log.txt &
  ```


## Jekyll
* How to start the server locally
    * jekyll serve --host=0.0.0.0
* How to write latex equation in markdown with jekyll
  See [here](http://zjuwhw.github.io/2017/06/04/MathJax.html)

## Django
* How to start the local server
    * python manage.py runserver 0:8000

## Mongodb
* Where is the config file
    * /etc/mongod.conf
* How to get the latest label version
  ``` python
  pipeline = [{'$match': {'data': test_data, 
                          'split': test_split,
                          'version': {'$lte': 0}}},
              {'$group': {'_id': {'data': '$data',
                                  'split': '$split',
                                  'key': '$key',
                                  'action_target_id': '$action_target_id'},
                          'contribution': {'$sum': '$contribution'},
                          'class': {'$first': '$class'},
                          'rect': {'$first': '$rect'}}},
              {'$match': {'contribution': {'$gte': 1}}}, # if it is 0, it means we removed the box
              {'$addFields': {'data': '$_id.data', 
                              'split': '$_id.split',
                              'key': '$_id.key'}},
              ]
  ```
* How to check the status
    * sudo service mongod status
* How to start the service
    * sudo service mongod start

## Pytorch
* How to access the data pointer of a tensor in c++
  ```
  result.data<int64_t>());
  ```
  Note, use int64_t rather than long long.

## Python
* Matplotlib
    * How to specify the name on x-axis
      ```python
      import matplotlib.pyplot as plt
      import numpy as np
      
      x = np.array([0,1,2,3])
      y = np.array([20,21,22,23])
      my_xticks = ['John','Arnold','Mavis','Matt']
      plt.xticks(x, my_xticks)
      plt.plot(x, y)
      plt.show()
      ```

    * How to save the figure
      ```python
      plt.savefig('2.eps', format='eps', bbox_inches='tight')
      plt.savefig('2.png', format='png', bbox_inches='tight')
      ```

    * How to not use the x-window to display.
      ```python
      import matplotlib
      matplotlib.use('Agg')
      ```
      or 
      ```shell
      export MPLBACKEND="agg"
      ```

    * How to remove the top and right line
      ```python
      fig, ax = plt.subplots()
      ax.spines['right'].set_visible(False)
      ax.spines['top'].set_visible(False)
      ```

    * How to make the x axis as the log scale based on 2
      ```python
      plt.xscale('log', basex=2)
      ```

    * How to position the location of x label
      ```python
      plt.xlabel(xlabel, labelpad=-25)
      ```

    * How to set the lineewidth of the axis
      ```python
      plt.setp(ax.spines.values(), linewidth=4)
      ```

* ProtoBuf
  ```bash
  conda install libprotobuf # it will install protoc (executible) and the include/lib. The python lib is not installed
  pip install protobuf
  ```
* opencv
  Never use the version of 3.3. Use 3.4 instead.
  ```bash
  conda install opencv
  ```
* Garbage Collection
  [link1](http://www.arctrix.com/nas/python/gc/)
  [link2](https://rushter.com/blog/python-garbage-collector/)

* Jupyter notebook
    * How to launch
    ```bash
    jupyter notebook --no-browser --port=8888 --ip=0.0.0.0
    ```
* Setup
    * How to install it under develop mode
    ```bash
    python setup.py build install --user
    ```
    * How to remove all data
    ```bash
    python setup.py clean --all
    ```
    * How to uninstall after python setup.py install
    ```bash
    python setup.py install --record files.txt
    xargs rm -rf < files.txt
    ```

* PDB
    * [How](https://stackoverflow.com/questions/5967241/how-to-execute-multi-line-statements-within-pythons-own-debugger-pdb) to run multi-line codes in pdb.
    ```python
    (pdb) !import code; code.interact(local=vars())
    Python 2.6.5 (r265:79063, Apr 16 2010, 13:57:41) 
    [GCC 4.4.3] on linux2
    Type "help", "copyright", "credits" or "license" for more information.
    (InteractiveConsole)
    >>> 
    ```
* Pytorch
    * How to check the NCCL's version
    ```shell
    python -c 'import torch; print(torch._C._nccl_version())'

    ```
    * What does it mean by the error of RuntimeError: copy_if failed
    Normally it means NaN encountered
    * How to check config information
    ```shell
    python -c 'import torch; print(torch.__config__.show())'
    ```

    * How to debug the distributed training, e.g. sync BN. 
    on one terminal
    ```shell
    OMPI_COMM_WORLD_RANK=0 OMPI_COMM_WORLD_SIZE=2 OMPI_COMM_WORLD_LOCAL_RANK=0 OMPI_COMM_WORLD_LOCAL_SIZE=2 python scripts.py
    ```

    on another terminal
    ```shell
    OMPI_COMM_WORLD_RANK=1 OMPI_COMM_WORLD_SIZE=2 OMPI_COMM_WORLD_LOCAL_RANK=1 OMPI_COMM_WORLD_LOCAL_SIZE=2 python scripts.py
    ```

* Unittest
    * How to attach the debugger when an exception is thrown. The command of ipython --pdb does not attach the debugger properly. We need to use nosetests
    ```shell
    pip install nose
    nosetests test.py -pdb
    ```
    or
    ```shell
    nosetests test.py -ipdb
    ```
    * How to run the test with the log printed
    ```shell
    nosetests --nocapture test.py
    ```
    * How to run one test
    ```shell
    nosetests test_masktsvdataset:TestMaskTSVDataset.test_get_keys
    ```

* ipdb
  * How to access all the local variables within ipdb
  ```shell
  ipdb> interact
  ```

## Latex
* How to insert latex equation to figure, created by inkscape or power points
  * use [this website](http://www.tlhiv.org/ltxpreview/), and download it as
  svg. Then insert svg to the figures.
* How to create a tight pdf figure
  ```shell
  pdfcrop fname.pdf fname-crop.pdf
  ```

## c/c++
* boost
  ```bash
  conda install boost
  ```
* How to check the time cost
  ```c
  #include <iostream>
  #include <ctime>
  #include <chrono>

  auto start = high_resolution_clock::now();
  auto stop = high_resolution_clock::now();
  auto duration = duration_cast<microseconds>(stop - start);
  std::cout << "unique: " << duration.count() << std::endl;
  ```


## Google Protoc
* Compile it from source code
  ```bash
  wget https://github.com/google/protobuf/releases/protobuf-python-3.5.1.tar.gz
  tar zxvf protobuf-python-3.5.1.tar.gz
  cd protobuf-python-3.5.1.tar.gz
  ./configure
  make
  sudo make install
  ```

## Git
* How to revert a merge
  check [this](https://mijingo.com/blog/reverting-a-git-merge)
* Migrate a repo from one server to another
  ```shell
  git lfs fetch --all
  git lfs push --all 
  git push --all
  ```
* How to pretty show the log
  ```shell
  git config --global alias.lg "log --color --graph --pretty=format:'%Cred%h%Creset -%C(yellow)%d%Creset %s %Cgreen(%cr) %C(bold blue)<%an>%Creset' --abbrev-commit"
  git lg
  ```
* How to add submodule
  ```shell
  git submodule add git@github.com:amsword/mmdetection.git src/mmdetection
  ```

## Markdown
* How to reference a sub section within the page
  * If it is a header, say, Google Protoc, then reference it as 
    \[Link](#google-protoc)\. Note 1) replace white space by hyphens; 2) use
    lower case

## GDB
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
* Attach a process
  ```
  sudo gdb -p process_id
  ```
* Print the call stack
  ```
  where
  ```
* Go to a specific frame
  ```
  frame frame_id
  ```
* How to start the python process in ConqueGdb python
  ```
  run a.py
  ```

## SSH
* How to browse the website from SSH tunel
    * Set the tunnel by
      ```bash
      ssh -ND 1080 jianfw
      ```
    * Open the browser with that port
      ```bash
      chromium-browser --proxy-server="socks5://localhost:1080"
      ```
    From [here](https://superuser.com/questions/819714/chrome-ssh-tunnel-forwarding)

* How to create id_rsa without prompt
  ```bash
  ssh-keygen -t rsa -N "" -f ~/.ssh/id_rsa
  ```

## Docker
* How to clean the docker folder
```bash
docker system prune -a -f
```
use `docker system prune --help` to check the details.

* How to change the docker folder not to use the space in /
check out [this](https://linuxconfig.org/how-to-move-docker-s-default-var-lib-docker-to-another-directory-on-ubuntu-debian-linux)

* How to get into the container
```bash
docker exec -i -t 665b4a1e17b6 /bin/bash 
```
* How to build docker image locally
```bash
docker build -t amsword/setup:py36pt11 .
```
* How to push teh docker image to docker hub
```bash
docker login
docker push amsword/setup:py36pt11
```

## vim/nvim
* How to solve the problem of slow netrw in nvim
  [follow this change](https://github.com/vim/vim/issues/4965#issuecomment-557692553)
  or [this one](https://github.com/neovim/neovim/pull/6892/commits/84e498a259c2e2b9cd09ebf99b2cb0b07f515bbe)
* How to generate the ycm config file for current project folder
  ```bash
  :YcmGenerateConfig
  ```
* How to delete the aux file from vim
  ```bash
  find ./ -type f -name "\.*sw[klmnop]" -delete
  ```
* How to get the file type of current file
  ```bash
  :set filetype?
  ```
* How to set the fold method as indent for markdown file
  ```bash
  # in ~/.vimrc
  autocmd FileType markdown setlocal fdm=indent
  ```
* How to launch the gdb python in vim
  ```
  : ConqueGdb python
  ```
* Netrw
  * How to create a new directory
    * click d
  * How to create a new file
    * click %


## Samba
* How to add a custom config to samba default config
  add the following to /etc/smb.conf
  ```
  include = /etc/samba/smb.conf.custom
  ```
