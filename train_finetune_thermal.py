import os
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer.gsplat_render import render, render_surface_smoke
import sys
from scene import Scene, DeformModel, GaussianSmokeThermalModel, GaussianSurfaceThermalModel
from utils.general_utils import safe_state, get_expon_lr_func
from utils.report_utils import training_report_s2 as training_report
from tqdm import tqdm
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
import wandb
from datetime import datetime
from render import render_test, render_test_thermal

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):
    first_iter = 0

    gaussians_smoke = GaussianSmokeThermalModel(dataset.sh_degree_stage2)       
    gaussians_surface = GaussianSurfaceThermalModel(dataset.sh_degree_stage2)

    opt.iterations = opt.iterations_stage2
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    scene_surface = Scene(dataset, gaussians_smoke)
    deform = DeformModel(True, False)
    deform.train_setting(opt)

    (model_params, first_iter) = torch.load(checkpoint)
    gaussians_surface.restore(model_params, opt)
        
    gaussians_smoke.training_setup(opt)
    gaussians_smoke.downsample_points(factor=4)
    aabb_min, aabb_max = gaussians_surface.get_points_aabb()
    gaussians_smoke.add_noise_to_points(aabb_min, aabb_max)
    gaussians_surface.disable_geometry_gradients()

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    viewpoint_stack_thermal = None
    ema_loss_for_log = 0.0
    
    depth_l1_weight = get_expon_lr_func(opt.depth_l1_weight_init, opt.depth_l1_weight_final, max_steps=opt.iterations_stage2-opt.iterations_stage1)
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    # dc_loss_module = DCLoss(patch_size=opt.patch_size)
    for iteration in range(first_iter, opt.iterations + 1):       
        #gaussians_surface.disable_position_gradients()
        edge_loss = 0.0
        total_loss = 0.0
        tv_loss = 0.0

        iter_start.record()
        gaussians_surface.update_learning_rate(iteration)
        gaussians_smoke.update_learning_rate(iteration)

        if iteration % 1000 == 0:
            gaussians_surface.oneupSHdegree()

        if not viewpoint_stack:
            viewpoint_stack = scene_surface.getTrainCameras().copy()

        if dataset.use_thermal and not viewpoint_stack_thermal:
            viewpoint_stack_thermal = scene_surface.getTrainCamerasThermal().copy()
        
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
        
        if dataset.use_thermal:
            viewpoint_cam_thermal = viewpoint_stack_thermal.pop(randint(0, len(viewpoint_stack_thermal)-1))

        fid = viewpoint_cam.fid
        
        if iteration < opt.warm_up:
            d_xyz, d_rotation, d_scaling, d_opacity, d_color = 0.0, 0.0, 0.0, 0.0, 0.0
        else:
            N = gaussians_smoke.get_xyz.shape[0]
            time_input = fid.unsqueeze(0).expand(N, -1)
            d_xyz, d_rotation, d_scaling, d_opacity, d_color = deform.step(gaussians_smoke.get_xyz.detach(), time_input)

        bg = torch.rand((3), device="cuda") if opt.random_background else background
        deform_parameters =  [d_xyz, d_rotation, d_scaling, d_opacity, d_color]

        render_pkg = render_surface_smoke(viewpoint_cam, gaussians_surface, gaussians_smoke, pipe, bg, deform_parameters=deform_parameters)        
        image, visibility_filter_surface, radii_surface, visibility_filter_smoke, radii_smoke, viewspace_points = render_pkg["render"], render_pkg["visibility_filter_surface"], render_pkg["radii_surface"], render_pkg["visibility_filter_smoke"], render_pkg["radii_smoke"], render_pkg["viewspace_points"]


        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        rgb_loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        total_loss = rgb_loss

        opacity_loss = torch.var(gaussians_smoke.get_opacity,dim=0).mean() * opt.smoke_opacity_weight
        color_loss = torch.var(gaussians_smoke.get_features,dim=0).mean() * opt.smoke_color_weight

        uniformity_color_loss = opt.smoke_uniformity_color_weight * torch.var(gaussians_smoke.get_features,dim=2).mean()
                
        render_smoke = render(viewpoint_cam, gaussians_smoke, pipe, background, deform_parameters=deform_parameters)
        
        valid_region_loss = 0.0
        invalid_region_loss = 0.0
        
        tv_loss = 0.0 # * tv_loss_module(surface_image)
        dc_loss = 0.0 # * dc_loss_module(surface_image)
        
        if USE_GSPLAT:
            smoke_alphas = render_smoke["render_alphas"]
            if viewpoint_cam.smoke_mask is not None:
                smoke_mask = viewpoint_cam.smoke_mask.cuda()
                invalid_region_loss = (smoke_alphas * smoke_mask).mean() * opt.invalid_region_weight
                smoke_alphas_thresholded = torch.where(smoke_alphas > 0.4, 
                                                     torch.ones_like(smoke_alphas),
                                                     smoke_alphas)
                valid_region_loss = ((1 - smoke_alphas_thresholded) * (1 - smoke_mask)).mean() * opt.valid_region_weight
                total_loss += valid_region_loss + invalid_region_loss

        render_surface = render(viewpoint_cam, gaussians_surface, pipe, background,deform_parameters=None)
        #surface_image = render_surface["render"]
        #dc_loss = opt.dcp_weight * dc_loss_module(surface_image * (1-smoke_mask))

        total_loss += opacity_loss
        total_loss += color_loss
        total_loss += dc_loss
        total_loss += uniformity_color_loss              
        total_loss += tv_loss

        if dataset.use_thermal:
            render_pkg_thermal = render(viewpoint_cam_thermal, gaussians_surface, pipe, background,deform_parameters=None, is_thermal=True)
            image_thermal = render_pkg_thermal["render"]
            viewspace_points_thermal = render_pkg_thermal["viewspace_points"]
            visibility_filter_surface_thermal = render_pkg_thermal["visibility_filter"]
            if viewpoint_cam_thermal.alpha_mask is not None:
                image_thermal *= viewpoint_cam_thermal.alpha_mask.cuda()
            Ll1_thermal = opt.thermal_weight* ((1.0 - opt.lambda_dssim) * l1_loss(image_thermal, viewpoint_cam_thermal.original_image.cuda()) + opt.lambda_dssim * (1.0 - ssim(image_thermal, viewpoint_cam_thermal.original_image.cuda())))
            total_loss += Ll1_thermal

            smoke_opacity_thermal_loss = (gaussians_smoke.get_opacity_thermal).mean()
            total_loss += 2.0 * smoke_opacity_thermal_loss 

            # surface_image_pov_thermal = render(viewpoint_cam_thermal, gaussians_surface, pipe, background,deform_parameters=None)["render"]
            # edge_loss = opt.cross_modal_edge_weight * cross_modal_edge_loss(surface_image_pov_thermal, viewpoint_cam_thermal.original_image)
            # total_loss += edge_loss

        Ll1depth_pure = 0.0
        Ll1depth_rgb = 0
        if depth_l1_weight(iteration) > 0 and viewpoint_cam.depth_reliable:
            invDepth = render_surface["inv_depth"]
            mask = invDepth < 5
            mono_invdepth = viewpoint_cam.invdepthmap.cuda()
            depth_mask = viewpoint_cam.depth_mask.cuda()[0]

            Ll1depth_pure = torch.abs((invDepth  - mono_invdepth) * depth_mask)
            #multiply by smoke_mask if it exists
            if viewpoint_cam.smoke_mask is not None:
                Ll1depth_pure *= viewpoint_cam.smoke_mask.cuda()

            Ll1depth_pure = Ll1depth_pure[0][mask].mean()
            Ll1depth_rgb = depth_l1_weight(iteration-first_iter) * Ll1depth_pure
            total_loss += Ll1depth_rgb
            Ll1depth_rgb = Ll1depth_rgb.item()

            if dataset.use_thermal:
                invthermalDepth = render_pkg_thermal["inv_depth"]
                depth_mask = viewpoint_cam_thermal.depth_mask.cuda()[0]

                Ll1depth_pure = torch.abs((invthermalDepth  - viewpoint_cam_thermal.invdepthmap.cuda()) * depth_mask)
                Ll1depth_pure *= viewpoint_cam_thermal.alpha_mask.cuda()
                Ll1depth_pure = Ll1depth_pure[0].mean()
                Ll1_thermal = depth_l1_weight(iteration-first_iter) * Ll1depth_pure
                total_loss += Ll1_thermal
                Ll1_thermal = Ll1_thermal.item()
        else:
            Ll1depth_rgb = 0
            Ll1_thermal = 0

        total_loss.backward()
        iter_end.record()

        losses_log = {"total_loss": total_loss, "rgb_loss": rgb_loss, "opacity_loss": opacity_loss, "color_loss": color_loss, "dc_loss": dc_loss, "uniformity_color_loss": uniformity_color_loss, "tv_loss": tv_loss, "edge_loss": edge_loss, "Ll1_thermal": Ll1_thermal if dataset.use_thermal else 0.0}
        losses_log["invalid_region_loss"] = invalid_region_loss
        losses_log["valid_region_loss"] = valid_region_loss
        losses_log["Ll1depth_rgb"] = Ll1depth_rgb
        losses_log["Ll1depth_thermal"] = Ll1_thermal
        with torch.no_grad():
            ema_loss_for_log = 0.4 * total_loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            training_report(losses_log, iteration, Ll1, total_loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene_surface, gaussians_smoke, gaussians_surface, render, render_surface_smoke, (pipe, background),deform, use_thermal=dataset.use_thermal)
            if (iteration in saving_iterations) and not dataset.disable_save:
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                fp1 = scene_surface.save(iteration,"surface_ft", gaussians_surface)
                fp2 = scene_surface.save(iteration,"surface_ft_thermal", gaussians_surface)
                fp3 = scene_surface.save(iteration,"smoke_ft", gaussians_smoke)
                #fp2 = scene_smoke.save(iteration,"smoke")
                output_fp = os.path.join('/'.join(fp1.split('/')[:-1]),"point_cloud_all.ply")

            # Densification
            if (iteration < opt.densify_until_iter):
                # Keep track of max radii in image-space for pruning
                gaussians_surface.max_radii2D[visibility_filter_surface] = torch.max(gaussians_surface.max_radii2D[visibility_filter_surface], radii_surface[visibility_filter_surface])
                if USE_GSPLAT:
                    gaussians_surface.add_densification_stats(viewspace_points.grad[0][:gaussians_surface.get_xyz.shape[0]], visibility_filter_surface, width=viewpoint_cam.image_width, height=viewpoint_cam.image_height)
                    gaussians_smoke.add_densification_stats(viewspace_points.grad[0][gaussians_surface.get_xyz.shape[0]:], visibility_filter_smoke, width=viewpoint_cam.image_width, height=viewpoint_cam.image_height)
                    if viewspace_points_thermal.grad is not None:
                        gaussians_surface.add_densification_stats(viewspace_points_thermal.grad[0][:gaussians_surface.get_xyz.shape[0]], visibility_filter_surface_thermal, width=viewpoint_cam_thermal.image_width, height=viewpoint_cam_thermal.image_height)
                    else:
                        viewspace_points_thermal.grad = torch.zeros((1, gaussians_surface.get_xyz.shape[0], 2), device="cuda")
                        gaussians_surface.add_densification_stats(viewspace_points_thermal.grad[0][:gaussians_surface.get_xyz.shape[0]], visibility_filter_surface_thermal, width=viewpoint_cam_thermal.image_width, height=viewpoint_cam_thermal.image_height)
                else:
                    gaussians_surface.add_densification_stats(viewspace_points.grad[:gaussians_surface.get_xyz.shape[0]], visibility_filter_surface)
                    gaussians_surface.add_densification_stats(viewspace_points_thermal.grad[:gaussians_surface.get_xyz.shape[0]], visibility_filter_surface_thermal)
                    gaussians_smoke.add_densification_stats(viewspace_points.grad[gaussians_surface.get_xyz.shape[0]:], visibility_filter_smoke)

                gaussians_smoke.max_radii2D[visibility_filter_smoke] = torch.max(gaussians_smoke.max_radii2D[visibility_filter_smoke], radii_smoke[visibility_filter_smoke])                
                if iteration > opt.densify_from_iter_stage2 and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians_surface.densify_and_prune(opt.densify_grad_threshold_surface, opt.prune_opacity_surface_threshold, scene_surface.cameras_extent, size_threshold)
                    gaussians_smoke.densify_and_prune(opt.densify_grad_threshold_smoke, 0.005, scene_surface.cameras_extent, size_threshold)

                if iteration < 20000 and (iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter)):
                    gaussians_surface.reset_opacity()
                    gaussians_smoke.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                # For surface gaussians, only update parameters in non-smoke regions
                if viewpoint_cam.smoke_mask is not None:
                    # Get the visibility of each gaussian in the current view
                    visible_gaussians = visibility_filter_surface
                    # Project gaussians to get their 2D positions
                    gaussian_2d_positions = viewspace_points[:gaussians_surface.get_xyz.shape[0], :2]
                    # Sample the smoke mask at gaussian positions
                    smoke_mask = viewpoint_cam.smoke_mask.cuda()
                    h, w = smoke_mask.shape[-2:]
                    # Scale positions to pixel coordinates
                    pixel_x = ((gaussian_2d_positions[:, 0] + 1) * w / 2).long().clamp(0, w-1)
                    pixel_y = ((gaussian_2d_positions[:, 1] + 1) * h / 2).long().clamp(0, h-1)
                    # Get mask values at gaussian positions
                    gaussian_mask = smoke_mask[0, 0, pixel_y, pixel_x]
                    # Only update gaussians in non-smoke regions
                    for group in gaussians_surface.optimizer.param_groups:
                        for param in group['params']:
                            if param.grad is not None:
                                param.grad *= (1 - gaussian_mask.view(-1, 1))[:param.shape[0]]
                
                gaussians_surface.optimizer.step()
                gaussians_smoke.optimizer.step()
                deform.optimizer.step()
                deform.optimizer.zero_grad()
                deform.update_learning_rate(iteration)
                gaussians_surface.optimizer.zero_grad(set_to_none = True)
                gaussians_smoke.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations) and not dataset.disable_save:
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians_surface.capture(), iteration), scene_surface.model_path + "/ft_chkpnt_surface_thermal" + str(iteration) + ".pth")
                torch.save((gaussians_smoke.capture(), iteration), scene_surface.model_path + "/ft_chkpnt_smoke_thermal" + str(iteration) + ".pth")
                deform.save_weights(scene_surface.model_path + "_ft_thermal", iteration)
    
    if dataset.use_thermal:
        render_test_thermal(scene_surface.getTrainCamerasUnshuffled(), scene_surface.getTrainCamerasThermalUnshuffled(), gaussians_smoke, gaussians_surface, pipe, background, iteration, dataset.model_path, deform=deform, append="final")
    else:
        render_test(scene_surface.getTrainCamerasUnshuffled(), gaussians_smoke, gaussians_surface, pipe, background, iteration, dataset.model_path, deform=deform, append="final")


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
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[1000, 2000, 5000, 7_000, 12_000, 15_000, 22_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 15_000, 22_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[7_000, 15_000, 22_000, 30_000])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--tag", type=str, default = "example")
    args = parser.parse_args(sys.argv[1:])
    
    dataset_name = args.source_path.split("/")[-1]
    home = os.path.expanduser("~")
    if os.path.exists(args.source_path)==False:
        args.source_path = os.path.join(home,args.source_path)

    current_datetime = datetime.now()
    date_time_str = current_datetime.strftime("%Y%m%d-%H-%M-%S")
    unique_str = f"{date_time_str}"

    args.start_checkpoint = os.path.join(args.model_path, "chkpnt_surface15000.pth")

    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    if args.use_wandb:
        wandb.init(project="gaussians_stage2_ablations", name=args.source_path.split('/')[-2] + args.model_path.split("/")[-1], config=args)
    else:
        wandb.init(mode="disabled")

    print("Optimizing " + args.model_path)

    safe_state(args.quiet)

    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)
    print("\nTraining complete.")