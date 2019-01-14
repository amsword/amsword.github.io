---
layout: post
comments: true
title: Linux Tips
---

## Mount between linux and linux
* How to export a folder from the server side
    * Add the following to /etc/exports. The IP address should be the
client IP address
        * /server_folder       157.54.146.96(rw,sync,no_root_squash,no_subtree_check) 
* How to mount the remote folder on the local side
    * Add the following to the /etc/fstab
        * server_ip:/server_folder         /local_folder   nfs    auto    0       0

## Power Option
* How to restart the computer after the power recovers
    * The option is in Bios setting. Find the options related with power.

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

## c/c++
* boost
  ```bash
  conda install boost
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
* Migrate a repo from one server to another
  ```shell
  git lfs fetch --all
  git lfs push --all 
  git push --all
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
## Proxy
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
  

