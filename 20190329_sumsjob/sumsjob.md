% $\Sigma\Sigma_{Job}$: Sums~Job~ (**S**imple **U**tility for **M**ultiple-**S**ervers **Job** **Sub**mission)
% Lu Lu
% Mar 29, 2019 @Crunch Seminar

# To run a job...

From division machine without free GPUs

1. `ssh jueying`
2. `nvidia-smi`: If no free GPU, go to step 1
3. `cd ~/project/codes`
4. `CUDA_VISIBLE_DEVICES=0 python main.py`

From personal computer

1. `scp -r codes dam:~/project/codes`
2. `ssh dam`
3. `ssh jueying`
4. `nvidia-smi`: If no free GPU, go to step 1
5. `cd ~/project/codes`
6. `CUDA_VISIBLE_DEVICES=0 python main.py`
7. `scp dam:~/project/codes/ml.dat .`

# One week later...

Cause I am lazy, I am crazy.

![](figs/crazy.jpeg)

# $\Sigma\Sigma_{Job}$

Sums~Job~ (**S**imple **U**tility for **M**ultiple-**S**ervers **Job** **Sub**mission)

- A simple Linux __*command-line utility*__ which __*submits a job*__ to one of the __*multiple servers*__ each with limited resources.

Features

- Simple to use: one single `submit` command is all your need
- Automatically choose available GPUs among all the servers
- interactively: just as the job is running in your local machine
    + Display the output of the job in real time
    + Kill the job by Ctrl-C
    + Save the output in a log file
    + Transfer back the files you specified

# `$ gpuresource`

Show the status of GPUs on all servers.

![Demo.](figs/gpuresource.png)

# `$ submit`

`$ submit jobfile jobname`

- `jobfile` : File to be run
- `jobname` : Job name, and also the folder name of the job

Options:

- `-h`, `--help` : Show this help message and exit
- `-i`, `--interact` : Submit as an interactive job
- `-s SERVER`, `--server SERVER` : Server host name
- `--gpuid GPUID` : GPU ID to be used; -1 to use CPU only

# Installation

- Download: <https://github.com/lululxvi/sumsjob>
- Make it executable (use `sudo` if needed)

```
chmod +x /opt/sumsjob/gpuresource.py
chmod +x /opt/sumsjob/submit.py
```

- Link Sums~Job~ to `~/.local/bin` (Assuming `~/.local/bin` is in your `$PATH`)

```
ln -s /opt/sumsjob/gpuresource.py ~/.local/bin/gpuresource
ln -s /opt/sumsjob/submit.py ~/.local/bin/submit
```
