---
layout: post
comments: true
title: Linux Tips
---

* Mount between linux and linux
    * How to export a folder from the server side
        * Add the following to /etc/exports. The IP address should be the
    client IP address
            * /server_folder       157.54.146.96(rw,sync,no_root_squash,no_subtree_check) 
    * How to mount the remote folder on the local side
        * Add the following to the /etc/fstab
            * server_ip:/server_folder         /local_folder   nfs    auto    0       0

* Jekyll
    * How to start the server locally
        * jekyll serve --host=0.0.0.0

* Django
    * How to start the local server
        * python manage.py runserver 0:8000

* Mongodb
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
* Python
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
