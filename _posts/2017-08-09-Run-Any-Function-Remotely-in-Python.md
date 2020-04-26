---
layout: post
comments: true
title: Run any function remotely in python
---

Since we have multiple machines to do the computing, it might be useful if we
can run any function remotely so that we can distribute the job easily. 

Here, I present a simple way to implement this. The idea is very
straightforward: send code/data to the remote machine and then run a python script remotely. 
Let's assume we would like to
run func(**kwargs) in a remote machine.

1. Serialize the parameter kwargs in a local file.

    ```python 
    def write_to_file(contxt, file_name):
        p = os.path.dirname(file_name)
        ensure_directory(p)
        with open(file_name, 'w') as fp:
            fp.write(contxt)
    
    working_dir = os.path.dirname(os.path.abspath(__file__))
    str_args = yaml.dump(kwargs)
    param_basename = 'remote_run_param_{}.txt'.format(hash(str_args))
    param_local_file = '/tmp/{}'.format(param_basename)
    write_to_file(str_args, param_local_file)
    ```
    
    Note: make this kind of file a temporary file

2. Send the local argument file to the remote machine. 

    ```python
    def cmd_run(list_cmd):
        p = sp.Popen(list_cmd, stdin=sp.PIPE)
        message = p.communicate()
        if p.returncode != 0:
            raise ValueError('return code not 0')
        return message
    
    def scp(local_file, target_file, ssh_cmd):
        assert op.isfile(local_file)
        cmd = ['scp']
        if '-p' in ssh_cmd:
            cmd.append('-P')
            cmd.append(str(ssh_cmd['-p']))
        cmd += [local_file, '{}@{}:{}'.format(ssh_cmd['username'],
            ssh_cmd['ip'], target_file)]
        cmd_run(cmd)
    
    scp(param_local_file, param_target_file, ssh_cmd) 
    ```
    
    Note: in the function of cmd_run(), it is better to set stdin as sp.PIPE, or it
    might complain the stdin is not a tty device. 
    
    The remote machine information is stored in the dictionary of ssh_cmd, which
    has the username and ip. Optionally, it has -p (which port to use for ssh).

3. Generate a local script so that when we run python script.py, it is
   effectively to call the function of func(**kwargs). Also send the file to
   the remote machine.

    ```python
    scripts = []
    scripts.append('import matplotlib')
    scripts.append('matplotlib.use(\'Agg\')')
    scripts.append('from {} import {}'.format(func.__module__,
        func.__name__))
    scripts.append('import yaml')
    scripts.append('param = yaml.load(open(\'{}\', \'r\').read())'.format(
        param_target_file))
    scripts.append('{}(**param)'.format(func.__name__))
    script = '\n'.join(scripts)
    basename = 'remote_run_{}.py'.format(hash(script))
    local_file = '/tmp/{}'.format(basename)
    write_to_file('\n'.join(scripts), local_file)
    target_file = op.join(working_dir, basename)
    scp(local_file, target_file, ssh_cmd)
    ```

    Note: run matplotlib.use('Agg') in case we would like to generate the figure. 

4. Issue rsync command to do the code sync so that the remote machine has the same code as
what we have locally. 

    ```python
    import subprocess as sp
    def sync(username, ip, from_folder, target_folder):
        cmd = []
        cmd.append('rsync')
        cmd.append('-vrzh')
        cmd.append('--exclude')
        cmd.append('*.swp')
        cmd.append('--exclude')
        cmd.append('*.swo')
        cmd.append('--exclude')
        cmd.append('*.pyc')
        cmd.append('{}'.format(from_folder))
        cmd.append('{}@{}:{}'.format(username, ip, target_folder))
        cmd_run(cmd)
    ```
    
    Note: donot use -a in rsync, or it might get stuck

5. run the command of python script.py in the remote machine.

    ```python
    def remote_run(str_cmd, ssh_info, return_output=False):
        cmd = ['ssh', '-t', '-t', '-o', 'StrictHostKeyChecking no']
        for key in ssh_info:
            if len(key) > 0 and key[0] == '-':
                cmd.append(key)
                cmd.append(str(ssh_info[key]))
        cmd.append('{}@{}'.format(ssh_info['username'], ssh_info['ip']))
        cmd.append(' && ' + str_cmd)
    
        return cmd_run(cmd, return_output)
    ```
    Note: sometimes, it cannot find the library when we run some command remotely.
    To solver it, please just add the path in the command by: 
    ```python
    cmd.append('export PATH=/your/path/:$PATH && ' + str_cmd)
    ```
    
    The option of '-t', '-t' to make sure when we kill the process by ctrl+c, the
    remote process will also be killed. The option of StrictHostKeyChecking is to
    make sure no prompt is aksed when the remote machine is brand new.




