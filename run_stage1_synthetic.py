import subprocess
import threading

data_dirs = [
    '/home/neham/wildfire_all_data/synthetic/outputs/caterpillar/caterpillar_0.4/',
    '/home/neham/wildfire_all_data/synthetic/outputs/ficus_rgb/ficus_rgb_0.75/',
    '/home/neham/wildfire_all_data/synthetic/outputs/mars/mars_0.3/',
    '/home/neham/wildfire_all_data/synthetic/outputs/room/room_1.0/',
    '/home/neham/wildfire_all_data/synthetic/outputs/musclecar/musclecar_0.7/',
    '/home/neham/wildfire_all_data/synthetic/outputs/hotdog/hotdog_1.0/',
    '/home/neham/wildfire_all_data/synthetic/outputs/forest/forest_0.5/',
    '/home/neham/wildfire_all_data/synthetic/outputs/chair/chair_0.7/',
    '/home/neham/wildfire_all_data/synthetic/outputs/lego_smoke/lego_0.65/',
    '/home/neham/wildfire_all_data/synthetic/outputs/house/house_0.2/',
]


hyperparams = '--use_thermal --use_wandb'

print("Commands to be executed on GPU 0:")
for i in range(0, len(data_dirs), 2):
    print(f"CUDA_VISIBLE_DEVICES=0 python train_stage1_thermal.py --eval --source_path {data_dirs[i]} {hyperparams}")

print("\nCommands to be executed on GPU 1:") 
for i in range(1, len(data_dirs), 2):
    print(f"CUDA_VISIBLE_DEVICES=1 python train_stage1_thermal.py --eval --source_path {data_dirs[i]} {hyperparams}")


def run_gpu0():
    for i in range(0, len(data_dirs), 2):
        gpu0_command = f"CUDA_VISIBLE_DEVICES=0 python train_stage1_thermal.py --eval --source_path {data_dirs[i]} {hyperparams}"
        process = subprocess.Popen(gpu0_command, shell=True)
        process.wait()  # Wait for each command on GPU 0 to finish before starting the next one

def run_gpu1():
    for i in range(1, len(data_dirs), 2):
        gpu1_command = f"CUDA_VISIBLE_DEVICES=1 python train_stage1_thermal.py --eval --source_path {data_dirs[i]} {hyperparams}"
        process = subprocess.Popen(gpu1_command, shell=True)
        process.wait()  # Wait for each command on GPU 1 to finish before starting the next one

# Run each GPU process in a separate thread
gpu0_thread = threading.Thread(target=run_gpu0)
gpu1_thread = threading.Thread(target=run_gpu1)

# Start both threads
gpu0_thread.start()
gpu1_thread.start()

# Wait for both threads to complete
gpu0_thread.join()
gpu1_thread.join()