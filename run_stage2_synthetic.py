import subprocess
import threading

#python train_finetune.py --eval -s /home/neham/wildfire_all_data/synthetic/outputs/house/house_0.2/ -m ./output/house_0.2/20241031-20-51-42 --smoke_opacity_weight=0.3260968912555703 --smoke_color_weight=1.037195217477147 --dcp_weight=0.0 --densify_grad_threshold_surface=0.0001103232130549 --densify_grad_threshold_smoke=8.2119742247e-05 --smoke_uniformity_color_weight=4.719851432640974 --thermal_weight=3.470382256650949 --use_thermal --use_wandb

data_dirs = [
    '/home/neham/wildfire_all_data/synthetic/outputs/house/house_0.2/',
    '/home/neham/wildfire_all_data/synthetic/outputs/lego_smoke/lego_0.65/',
    '/home/neham/wildfire_all_data/synthetic/outputs/chair/chair_0.7/',
    '/home/neham/wildfire_all_data/synthetic/outputs/forest/forest_0.5/',
    '/home/neham/wildfire_all_data/synthetic/outputs/hotdog/hotdog_1.0/',
    '/home/neham/wildfire_all_data/synthetic/outputs/musclecar/musclecar_0.7/',
    '/home/neham/wildfire_all_data/synthetic/outputs/room/room_1.0/',
    '/home/neham/wildfire_all_data/synthetic/outputs/mars/mars_0.3/',
    '/home/neham/wildfire_all_data/synthetic/outputs/ficus_rgb/ficus_rgb_0.75/',
    '/home/neham/wildfire_all_data/synthetic/outputs/caterpillar/caterpillar_0.4/',
]

model_dirs = [
    './output/house_0.2/20250222-13-11-23',
    './output/lego_0.65/20250222-13-10-58',
    './output/chair_0.7/20250222-13-04-55',
    './output/forest_0.5/20250222-13-04-33',
    './output/hotdog_1.0/20250222-12-58-20',
    './output/musclecar_0.7/20250222-12-58-18',
    './output/room_1.0/20250222-12-51-34',
    './output/mars_0.3/20250222-12-51-28',
    './output/ficus_rgb_0.75/20250222-12-45-04',
    './output/caterpillar_0.4/20250222-12-45-04',
]

#hyperparams = '--smoke_opacity_weight=0.3260968912555703 --smoke_color_weight=1.037195217477147 --dcp_weight=0.0 --densify_grad_threshold_surface=0.0001103232130549 --densify_grad_threshold_smoke=8.2119742247e-05 --smoke_uniformity_color_weight=4.719851432640974 --thermal_weight=3.470382256650949 --use_thermal --use_wandb']
hyperparams = '--smoke_opacity_weight=1.1591504226121605 --smoke_color_weight=1.5005770505348608 --densify_grad_threshold_surface=0.00022022347701492677 --densify_grad_threshold_smoke=0.0002891510879559097 --smoke_uniformity_color_weight=2.93293953018836 --thermal_weight=2.50808992249495 --depth_l1_weight_init=7.491022690840896 --invalid_region_weight=0.2700912856127265 --valid_region_weight=0.22923013095090927 --use_thermal --use_wandb'


def run_gpu0():
    for i in range(0, len(data_dirs), 2):
        gpu0_command = f"CUDA_VISIBLE_DEVICES=0 python train_finetune_thermal.py --eval --source_path {data_dirs[i]} --model_path {model_dirs[i]} {hyperparams}"
        process = subprocess.Popen(gpu0_command, shell=True)
        process.wait()  # Wait for each command on GPU 0 to finish before starting the next one

def run_gpu1():
    for i in range(1, len(data_dirs), 2):
        gpu1_command = f"CUDA_VISIBLE_DEVICES=1 python train_finetune_thermal.py --eval --source_path {data_dirs[i]} --model_path {model_dirs[i]} {hyperparams}"
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