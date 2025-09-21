import os
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer.gsplat_render import render
import sys
from scene import Scene, GaussianSurfaceThermalModel
from utils.general_utils import safe_state, get_expon_lr_func
from utils.report_utils import training_report_stage1_thermal as training_report
from tqdm import tqdm
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
import wandb
from datetime import datetime
from utils.pose_optimization import CameraOptModule
import torch.cuda


def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):
    first_iter = 0
    gaussians_surface = GaussianSurfaceThermalModel(dataset.sh_degree_stage1)

    scene_surface = Scene(dataset, gaussians_surface)

    gaussians_surface.training_setup(opt)
    
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians_surface.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    if dataset.pose_opt:
        from gaussian_renderer.render_pose_opt import render_pose_opt_gsplat
        camera_opt_rgb = CameraOptModule(len(scene_surface.getTrainCameras()))
        camera_opt_rgb.zero_init()
        pose_optimizer_rgb = torch.optim.Adam(
            camera_opt_rgb.parameters(),
            lr=3e-4,
            weight_decay=1e-6,
        )

        camera_opt_thermal = CameraOptModule(len(scene_surface.getTrainCamerasThermal())).cuda()
        camera_opt_thermal.zero_init()
        pose_optimizer_thermal = torch.optim.Adam(
            camera_opt_thermal.parameters(),
            lr=3e-4,
            weight_decay=1e-6,
        )

    depth_l1_weight = get_expon_lr_func(opt.depth_l1_weight_init, opt.depth_l1_weight_final, max_steps=opt.iterations_stage1)
    
    viewpoint_stack_thermal = scene_surface.getTrainCamerasThermal().copy()

    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations_stage1), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, opt.iterations_stage1 + 1):        
        total_loss = 0.0
        iter_start.record()
        gaussians_surface.update_learning_rate(iteration)

        if iteration % 1000 == 0:
            gaussians_surface.oneupSHdegree()

        if not viewpoint_stack_thermal:
            viewpoint_stack_thermal = scene_surface.getTrainCamerasThermal().copy()

        viewpoint_cam_thermal = viewpoint_stack_thermal.pop(randint(0, len(viewpoint_stack_thermal) - 1))        
        fid = viewpoint_cam_thermal.fid

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        if dataset.pose_opt and iteration > 1500:
            render_pkg_thermal = render_pose_opt_gsplat(viewpoint_cam_thermal, gaussians_surface, pipe, bg, deform_parameters=None, is_thermal=True, camera_opt=camera_opt_thermal)
        else:
            render_pkg_thermal = render(viewpoint_cam_thermal, gaussians_surface, pipe, bg, deform_parameters=None, is_thermal=True)

        image_thermal, visibility_filter_surface_thermal, radii_surface_thermal, viewspace_points_thermal = render_pkg_thermal["render"], render_pkg_thermal["visibility_filter"], render_pkg_thermal["radii"], render_pkg_thermal["viewspace_points"]
        
        # Apply alpha mask to image_thermal
        if viewpoint_cam_thermal.alpha_mask is not None:
            image_thermal *= viewpoint_cam_thermal.alpha_mask.cuda()
        
        Ll1_thermal = opt.thermal_weight * ((1.0 - opt.lambda_dssim) * l1_loss(image_thermal, viewpoint_cam_thermal.original_image.cuda()) + opt.lambda_dssim * (1.0 - ssim(image_thermal, viewpoint_cam_thermal.original_image.cuda())))
        total_loss += Ll1_thermal

        Ll1depth_pure = 0.0
        if depth_l1_weight(iteration) > 0 and viewpoint_cam_thermal.depth_reliable:
            invDepth = render_pkg_thermal["inv_depth"]
            mono_invdepth = viewpoint_cam_thermal.invdepthmap.cuda()
            depth_mask = viewpoint_cam_thermal.depth_mask.cuda()

            Ll1depth_pure = torch.abs((invDepth  - mono_invdepth) * depth_mask)
            if viewpoint_cam_thermal.smoke_mask is not None:
                Ll1depth_pure *= viewpoint_cam_thermal.smoke_mask.cuda()

            Ll1depth_pure = Ll1depth_pure.mean()
            Ll1depth = depth_l1_weight(iteration-first_iter) * Ll1depth_pure
            total_loss += torch.clamp(Ll1depth, 0.0, 2.0)
            Ll1depth = Ll1depth.item()
        else:
            Ll1depth = 0

        total_loss.backward()
        iter_end.record()
        
        losses_log = {"total_loss": total_loss, "Ll1_thermal": Ll1_thermal, "Ll1depth": Ll1depth}

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * total_loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations_stage1:
                progress_bar.close()

            # Log and save
            training_report(losses_log, iteration, Ll1_thermal, total_loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene_surface, gaussians_surface, render, (pipe, background), use_thermal=dataset.use_thermal)
            if (iteration in saving_iterations) and not dataset.disable_save:
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                fp2 = scene_surface.save(iteration, "surface_thermal")
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians_surface.max_radii2D[visibility_filter_surface_thermal] = torch.max(gaussians_surface.max_radii2D[visibility_filter_surface_thermal], radii_surface_thermal[visibility_filter_surface_thermal])
                if dataset.use_thermal:
                    gaussians_surface.add_densification_stats(viewspace_points_thermal.grad, visibility_filter_surface_thermal, width=viewpoint_cam_thermal.image_width, height=viewpoint_cam_thermal.image_height)
                    
                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians_surface.densify_and_prune(opt.densify_grad_threshold_surface, 0.01, scene_surface.cameras_extent, size_threshold)

                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians_surface.reset_opacity()

            if iteration < opt.iterations_stage1:
                gaussians_surface.optimizer.step()
                if dataset.pose_opt:
                    pose_optimizer_rgb.step()
                    pose_optimizer_rgb.zero_grad()
                    if dataset.use_thermal:
                        pose_optimizer_thermal.step()
                        pose_optimizer_thermal.zero_grad()

                gaussians_surface.optimizer.zero_grad(set_to_none=True)

            if (iteration in checkpoint_iterations) and not dataset.disable_save:
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians_surface.capture(), iteration), scene_surface.model_path + "/chkpnt_surface" + str(iteration) + ".pth")

    # Save all cameras
    if dataset.pose_opt:    
        torch.save(scene_surface.train_cameras, os.path.join(dataset.model_path, "train_cameras.pth"))
        torch.save(scene_surface.test_cameras, os.path.join(dataset.model_path, "test_cameras.pth"))
        if dataset.use_thermal:
            torch.save(scene_surface.train_cameras_thermal, os.path.join(dataset.model_path, "train_cameras_thermal.pth"))
            torch.save(scene_surface.test_cameras_thermal, os.path.join(dataset.model_path, "test_cameras_thermal.pth"))

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[1000, 7000, 15000, 22_500, 30000, 35000, 40000, 45000, 50000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7000, 15000, 30000, 45000, 50000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[7000, 15000, 22000, 30000])
    parser.add_argument("--start_checkpoint", type=str, default=None)
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--tag", type=str, default="example")
    args = parser.parse_args(sys.argv[1:])
    op.iterations = op.iterations_stage1
    dataset_name = lp.extract(args).source_path.split("/")[-1]
    home = os.path.expanduser("~")
    if not os.path.exists(args.source_path):
        args.source_path = os.path.join(home, args.source_path)

    current_datetime = datetime.now()
    date_time_str = current_datetime.strftime("%Y%m%d-%H-%M-%S")
    unique_str = f"{date_time_str}"
    args.model_path = os.path.join(f"./output/{dataset_name}", unique_str)
    print("Output folder: {}".format(args.model_path))
    # Append args.model_path to a file called stage1.txt and add like a new line
    with open("stage1.txt", "a") as f:
        f.write(args.model_path + "\n")
    os.makedirs(args.model_path, exist_ok=True)

    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    if args.use_wandb:
         wandb.init(project="gaussian_smoke_stage1_thermal_synthetic", name=args.source_path.split('/')[-2] + args.model_path.split("/")[-1], config=args)
    else:
        wandb.init(mode="disabled")

    print("Optimizing " + args.model_path)

    safe_state(args.quiet)

    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)
    print("\nTraining complete.")