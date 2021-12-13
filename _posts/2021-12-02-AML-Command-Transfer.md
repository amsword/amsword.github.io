---
layout: post
comments: true
title: AML Command Transfer
---

AML Command Transfer ([ACT](https://github.com/microsoft/act)) 
is a tool to 
easily execute any command in the Azure Machine Learning (AML)
services.
The blog will present the underlying design principal of the tool.

# How AML works
First of all, let's review the basic idea of how AML works.

Assume that the user would like to have 2 nodes in AML and execute `python train.py`
on each node once. The file of `train.py` is quite simple as follows
```python
# train.py
print('hello')
```
The user will send the following information to AML
1. The file of `train.py`
2. The docker image, e.g. the docker image at docker hub.
3. The requested resource, i.e. two nodes.

After receiving the user request, AML will do the following
1. Find 2 available nodes.
2. Pull the docker image to the nodes.
3. Execute `mpirun` to start the job.
   The basic command line will be
   ```
   mpirun --hostfile /job/hostfile -npernode 1 python train.py
   ```
   - The value of `--hostfile` is a text file, which contains the IPs of the two nodes.
     Each IP takes one line. In certain cases, the file path is `/job/hostfile` or `~/mpi-hosts`.
     The file is prepared by AML and is not allowed to change.
   - The value of `-npernode` tells how many processes will be launched on each
     node. Two commonly-used cases are `1` and the value equal to the number of
     GPUs. This value can be specified by the user.
   - Lastly, the command of `python train.py` is fed to `mpirun`, so that it knows what
     command to run.

### How to Access the Data in AML
Normally, we need data to execute the job.
One way AML supports is the
[blobfuse](https://github.com/Azure/azure-storage-fuse). The story is as
follows.
1. The user uploads the data to the [Azure Storage](https://docs.microsoft.com/en-us/azure/storage/).
   Any uploaded file can be accessed through a URL with appropriate authentication header.
   Let's say that the storage account is `account`, the storage container is `container`, and the file path is
   `data/a.txt`. The storage container is some concept of the Azure Storage.
   All data within one container can have the same access level, e.g. public.
   Then, the URL will be `https://account.blob.core.windows.net/container/data/a.txt`.

2. During the job submission, an ID string can be assigned to the file path
   through the [AML's Python SDK](https://docs.microsoft.com/en-us/python/api/overview/azure/ml/?view=azure-ml-py),
   and will be used as a place holder for the
   script argument. For example, the file path is `blob_data/a.txt` under some
   storage account, and the Python SDK generates a special string to refer to this file, say, `ID_of_a.txt`.
   Then, the
   submitted command line is
   ```shell
   python train.py --data_path ID_of_a.txt
   ```
3. AML uses the blobfuse to mount the cloud storage as a local folder.
3. AML replaces all the ID parameters with the mounted file path. In this case,
   the command line will be replaced as
   ```shell
   python train.py --data_path /mount_path/blob_data/a.txt
   ```
4. AML launches `mpirun` as usual.

# Design Principal of ACT
### Motivation
Now, we have a basic idea of how AML works. With the Python SDK, we can submit
any script job, but with a dedicated script to specify 
the Azure Storage information, script parameters, e.t.c. 
To reduce the effort, we'd like to have a tool, which behaves like a bridge to
connect the user script and AML. Let's say the name of the tool is named `a`.
- If we'd like to run `train.py` without any parameter, the submission syntax is
  ```shell
  a submit python script.py
  ```
  We expect to specify `python` rather than to assume that the interpreter is
  always `python` such that we can support other non-python scripts. In other
  words, we can always test the script locally by `python script.py`. If we'd
  like to execute it in AML, we just need to add the prefix of `a aubmit`.
- If we'd like to run `train.py` with some parameter of `--data imagenet`, the submission syntax is
  ```shell
  a submit python script.py --data imagenet
  ```
- If we'd like to run `nvidia-smi` in AML, the submission syntax is
  ```shell
  a submit nvidia-smi
  ```
That is, we expect the tool of `a` to handle everything related with AML such
that the parameters after `submit` can be seemingly transferred to AML,
which is quite similar with [remote procedure call](https://en.wikipedia.org/wiki/Remote_procedure_call).
This is what ACT does!

### Design of the Work Flow
The design is based on the client-server model. That is, we have a client
script for job submission and a server script which executes any command the
client requests.

The client
1. `a init` to upload all the user source codes to the azure blob. The user source
   codes are referred to as the, e.g. the training codes, which implements the
   model structure, optimization logics, e.t.c. `zip` is used to
   compress the current source folder to a
   temporary `.zip` file, which is then uploaded to the azure blob through `azcopy`.
   To customize the zipping process, we can populate the parameter of
   `zip_options` in the configuration to, e.g. ignore some folders or files.
2. `a submit command` to submit the `command` to AML. In addition of the
   `command`, the client script will also submit the following information.
   - The blob path of the zipped source code, so that the server  knows
     where to access the source code.
   - Azure Blob information, so that AML can mount the azure blob through
     `blobfuse`.
   - The data link information, including the local folder name and the
     corresponding blob path, so that the server side can create a symbol
     link to access the data. As long as the user's source codes always
     uses the relative file path, e.g. `data/a.txt`, and the client sets
     the `data` to the apprioriate azure blob path, no change is required to
     change the data path in the code. That is, if the code is tested well
     on the local machine, it should also work in AML. 

The server side
1. unzip the source code
   - The destination folder is under `/tmp` since this folder
     always exsits and is writable. Another option is to use the
     home folder. However, sometimes the home folder is a network share,
     which could be slow if we need to compile the code. If the home folder
     is shared among different jobs, synchronization could be very difficult.
   - As multiple processes could be launched in AML,
     we need to avoid the race condition so that only one process is
     unzipping the source code. Here, we use the exclusive file open as a
     lock to implement it. Another way might be to depend on the `barrier`
     of `mpi` or `torch.distributed`, both of which might create more dependency.
     In practice, the file-based exclusive lock works well.
2. run `compile.aml.sh` if the file exists after the source folder is
   unzipped. This gives the user an opportunity to compile the source code
   with any kind of command.
3. pip install the packages in `requirements.txt` if the file exists. This
   is specifically for python packages.
4. launch the command under the source folder. This also means that the
   client is suggested to stick to the root
   source folder as the current folder for code testing locally.

The configuration file is a YAML file to contain all
associated parameters,
including the blob path for the source code, the data link information between
the local folder and the blob folder, the environment we need to have in 
AML, e.t.c.
One YAML file corresponds to one cluster. We can use `-c cluster_name`
to specify the cluster, whose YAML configuration file is
located at `aux_data/aml/cluster_name.yaml`. At
this moment, we hard-code the path and in the future, we may provide a way to
customize the path.

### Data Management
During job submission, we specify the data link information. That is, the
tool of `a` knows which azure path is mapped to the local folder. Thus, we can
have the following utilities to manage the data in azure blob
- remove the corresponding Blob data of `data/a.txt` by
  ```
  a rm data/a.txt
  ```
- list all files corresponding to the Blob folder of `data/data_set` by
  ```
  a ls data/data_set
  ```
- upload the local data `data/c.txt` to the Blob
  ```
  a u data/c.txt
  ```
- copy the data of `data/b.txt` from the blob used by clusterA to the blob used by clusterB.
  ```
  a -c clusterB -f clusterA u data/b.txt
  ```

# Conclusion
The tool is small, handy, intuitive, stable and robust. With this tool, I have submitted at
least 54499 jobs (before 12/12/2021), 39159 of which are within 11/1/2020 to 11/1/2021.
