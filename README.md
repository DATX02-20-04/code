# DATX02-20-04


## Google Colab Getting Started

Start by cloning the repo in Colab:
```sh
! git clone https://github.com/elias-sundqvist/DATX02-20-04.git
```

Now run the `colab_import.sh` script like this:
```sh
! DATX02-20-04/colab_import.sh
```

Now it is possible to import modules from our repo:
```python
from datasets.nsynth import nsynth_from_tfrecord
```

## Connecting and using the training computer
*Important*: Do not train multiple things at the same time, the GPU is already limited!
Find out if resources are availalbe by:
```console
user@bar:~$ nvidia-smi
```

```text
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 440.64.00    Driver Version: 440.64.00    CUDA Version: 10.2     |
|-------------------------------|----------------------|----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  GeForce RTX 2060    On   | 00000000:01:00.0  On |                  N/A |
| 65%   78C    P2   149W / 170W |   5790MiB /  5933MiB |     95%      Default |
+-------------------------------|----------------------|----------------------+

+-----------------------------------------------------------------------------+
| Processes:                                                       GPU Memory |
|  GPU       PID   Type   Process name                             Usage      |
|=============================================================================|
|    0      1011      G   /usr/lib/xorg/Xorg                            18MiB |
|    0      1387      G   /usr/lib/xorg/Xorg                            69MiB |
|    0      1600      G   /usr/bin/gnome-shell                         167MiB |
|    0     30671      C   ...pyenv/versions/3.7.5/envs/ml/bin/python  5473MiB |
+-----------------------------------------------------------------------------+
```
According to table above, a process is using most of the VRAM available, so ask in the group about it!

To connect to the computer, ssh into it (user/pass/ip avaialable in group):
```console
user@bar:~$ ssh username@ip_addr
```

If you want to run a script and disconnect your ssh session without it interrupting, create a tmux session:
```console
user@bar:~$ tmux new-session -s <NAME>
```

List all sessions
```console
user@bar:~$ tmux ls
```

Attach most recently used session
```console
user@bar:~$ tmux attach-session
```

Attach specific session ('<TARGET>' from tmux ls above)
```console
user@bar:~$ tmux attach-session -t <TARGET>
```

To detach the current tmux session (leave it running), do:
```
Ctrl-b d
```

Finally, kill a session by:
```console
user@bar:~$ tmux kill-session -t <NAME>
```

Usually you will just have to do the following:
1. SSH in
2. Clone repo in ./repos folder
3. Start new tmux session
4. Run the training script
5. Detach while it's training

You might also need to install dependencies. To do that, ask someone in the
group so that we don't somehow mess up pyenv.
