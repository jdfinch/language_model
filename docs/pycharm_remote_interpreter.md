
# Set up Remote Interpreter & Deployment in PyCharm

## 0. Install Interpreter and Empty Project Folder on Server

ssh onto the server like:

```
ssh username@server
```

On tebuna and h100, always work off of **/local/scratch**:

```
cd /local/scratch/username
```

Make a project folder.

```
mkdir /local/scratch/username/myproject
```

Install miniconda if you haven't already:

```
bash
mkdir -p /local/scratch/username/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /local/scratch/username/miniconda3/miniconda.sh
bash /local/scratch/username/miniconda3/miniconda.sh -b -u -p /local/scratch/username/miniconda3
/local/scratch/username/miniconda3/bin/conda init bash
exit
bash
```

And make a new environment:

```
conda create -n "myenv" python=3.11
conda activate myenv
```

## 1. Get Pycharm Pro (free for students)

## 2. Create a Port Redirect

In order to connect to machines like tebuna and h100, you need to set up a port redirect through lab0z. The reason is that PyCharm cannot be configured to ssh through lab0z to tebuna, so you need to set up port forwarding.

```
ssh -L [local_port]:[remote_host]:[remote_port] [user]@[ssh_server]
```

To make it easier, you can install sshpass and use:

```
sshpass -f /local/path/to/a/file/with/your/password ssh -L [local_port]:[remote_host]:[remote_port] [user]@[ssh_server]
```

For example:

```
sshpass -f /my/password/path ssh -L 55555:tebuna:22 myusername@lab0z.mathcs.emory.edu
```

The local port you connect to to get to tebuna is 55555. For example, you can now connect to tebuna by:

```
ssh -p 55555 myusername@localhost
```

Which means we can now point PyCharm to localhost:55555 to set up a remote environment.


## 3. Set up the Remote Interpreter

In Pycharm, navigate to File/PyCharm => Settings => Project Interpreter => Add Interpreter => On SSH

If it doesn't already exist, make an SSH connection to your forwarding port:

```
Host: localhost
Port: 55555
Username: myusername
```

Put in your password (PyCharm will cache it locally).

Then select your interpreter by either:

1. Select Conda interpreter, point conda executable to your /local/scratch/username/miniconda3/bin/conda and then select your environment (myenv) from the dropdown of discovered environments.

2. Just select system interpreter and point it to /local/scratch/username/miniconda3/envs/myenv/bin/python directly.

Finally, make sure the remote deployment works by mapping your local project folder on your computer to /local/scratch/username/myproject

You can check to make sure the remote deployment is correct at Tools => Deployment => Configuration (the current configuration is bolded, you can swich the configuration by selecting it and hitting the check mark, and it should show the correct local-to-remote folder mapping in the Mappings tab).

Also, make sure to exclude any folders with models or large data that you do NOT want to transfer between your local and remote by registering them under Excluded Paths.
