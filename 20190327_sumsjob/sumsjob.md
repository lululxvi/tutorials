% $\Sigma\Sigma_{Job}$: Sums~Job~ (**S**imple **U**tility for **M**ultiple-**S**ervers **Job** **Sub**mission)
% Lu Lu
% Mar 27, 2019 @Crunch Seminar

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

![](figs/crazy.jpeg)

# $\Sigma\Sigma_{Job}$

Sums~Job~ (**S**imple **U**tility for **M**ultiple-**S**ervers **Job** **Sub**mission)

- A simple Linux __*command-line utility*__ which __*submits a job*__ to one of the __*multiple servers*__ each with limited resources.
- It will first look for servers with available resources, such as GPUs, and then run the job in that server __*interactively*__ just as the job is running in your local machine.

- `$ gpuresource`: Show the status of GPUs on all servers
- `$ submit`: Run a job