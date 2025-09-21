import wandb
import torch
from lpipsPyTorch.modules.lpips import LPIPS
from utils.image_utils import psnr
#from lpipsPyTorch.modules.lpips import LPIPS
from utils.loss_utils import ssim
import pdb
import random

def normalize_depth(inv_depth):
    percentile_5 = torch.quantile(inv_depth, 0.05)
    percentile_95 = torch.quantile(inv_depth, 0.95)
    inv_depth = torch.clamp(inv_depth, percentile_5, percentile_95)
    return (inv_depth-inv_depth.min())/(inv_depth.max()-inv_depth.min())

@torch.no_grad()
def training_report_stage1_thermal(losses_log, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene, gaussians_surface, renderFunc1, renderArgs, deform=None, use_thermal=False):
    wandb.log(losses_log, step=iteration)
    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        print("Number of gaussians in surface: {}".format(gaussians_surface.get_xyz.shape[0]))
        configs = [{'name': 'test', 'cameras': scene.getTestCamerasThermal()}, {'name': 'train', 'cameras': scene.getTrainCamerasThermal()}]
        
        for config in configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    thermal_render_pkg = renderFunc1(viewpoint, gaussians_surface, *renderArgs, is_thermal=True)
                    thermal_image = thermal_render_pkg["render"]
                    if viewpoint.alpha_mask is not None:
                        thermal_image *= viewpoint.alpha_mask.cuda()
                    thermal_inv_depth = thermal_render_pkg["inv_depth"]
                    gt_thermal_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if idx < 5:
                        wandb.log({config['name'] + "_view_{}/render_thermal".format(viewpoint.image_name): [wandb.Image(thermal_image)]}, step=iteration)
                        wandb.log({
                            config['name'] + "_view_{}/ground_truth_thermal".format(viewpoint.image_name): [wandb.Image(gt_thermal_image)]
                        }, step=iteration)
                        wandb.log({config['name'] + "_view_{}/inv_depth_thermal".format(viewpoint.image_name): [wandb.Image(thermal_inv_depth)]}, step=iteration)

                    l1_test += l1_loss(thermal_image, gt_thermal_image).mean().double()
                    psnr_test += psnr(thermal_image, gt_thermal_image).mean().double()

                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])   
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                wandb.log({
                    config['name'] + '/loss_viewpoint - l1_loss_thermal': l1_test,
                    config['name'] + '/loss_viewpoint - psnr_thermal': psnr_test,
                }, step=iteration)
        try:
            wandb.log({"scene/opacity_histogram_surface": wandb.Histogram(gaussians_surface.get_opacity.cpu())}, step=iteration)
        except:
            print(gaussians_surface.get_opacity.min(), gaussians_surface.get_opacity.max())
        wandb.log({'total_points_surface': gaussians_surface.get_xyz.shape[0]}, step=iteration)
        torch.cuda.empty_cache()

@torch.no_grad()
def training_report_s2(losses_log, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene, gaussians_smoke, gaussians_surface, renderFunc1, renderFunc2, renderArgs, deform=None, use_thermal=False):
    if iteration % 5 == 0:
        wandb.log(losses_log, step=iteration)

    if iteration in testing_iterations:
        lpips_module = LPIPS(net_type='vgg')
        lpips_module.eval()
        lpips_module = lpips_module.to("cuda")

        print("Number of gaussians in surface: {}".format(gaussians_surface.get_xyz.shape[0]))
        print("Number of gaussians in smoke: {}".format(gaussians_smoke.get_xyz.shape[0]))
        configs = [{'name': 'test', 'cameras': scene.getTestCamerasUnshuffled()}, 
                  {'name': 'train', 'cameras': scene.getTrainCamerasUnshuffled()}]
        if use_thermal:
            thermal_configs = [{'name': 'test', 'cameras': scene.getTestCamerasThermalUnshuffled()},
                             {'name': 'train', 'cameras': scene.getTrainCamerasThermalUnshuffled()}]

        for idx_config, config in enumerate(configs):
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                psnr_test_gt = 0.0
                lpips_test_gt = 0.0
                ssim_test_gt = 0.0
                if use_thermal:
                    psnr_test_thermal = 0.0
                    thermal_cameras = thermal_configs[idx_config]['cameras']
                random.seed(40)
                random_indices = random.sample(range(len(config['cameras'])), min(40, len(config['cameras'])))
                for idx, viewpoint in enumerate(config['cameras']):
                    try:  
                        if iteration < 3000:
                            d_xyz, d_rotation, d_scaling, d_opacity, d_color = 0.0, 0.0, 0.0, 0.0, 0.0
                        else:
                            N = gaussians_smoke.get_xyz.shape[0]
                            time_input = viewpoint.fid.unsqueeze(0).expand(N, -1)
                            d_xyz, d_rotation, d_scaling, d_opacity, d_color = deform.step(gaussians_smoke.get_xyz.detach(), time_input)

                        deform_parameters = [d_xyz, d_rotation, d_scaling, d_opacity, d_color]

                        render_pkg = renderFunc2(viewpoint, gaussians_surface, gaussians_smoke, deform_parameters=deform_parameters, *renderArgs)
                        image = render_pkg["render"]
                        inv_depth_all = render_pkg["inv_depth"]
                        depth_all = normalize_depth(inv_depth_all)

                        render_pkg_surface = renderFunc1(viewpoint, gaussians_surface, *renderArgs)
                        image_desmoked = render_pkg_surface["render"]
                        inv_depth_surface = render_pkg_surface["inv_depth"]
                        depth_surface = normalize_depth(inv_depth_surface)
                        
                        render_pkg_smoke = renderFunc1(viewpoint, gaussians_smoke, *renderArgs, deform_parameters=deform_parameters)
                        image_smoke = render_pkg_smoke["render"]
                        inv_depth_smoke = render_pkg_smoke["inv_depth"]
                        depth_smoke = normalize_depth(inv_depth_smoke)
                        
                        gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                        if viewpoint.original_image_desmoked is not None:
                            gt_image_desmoked = torch.clamp(viewpoint.original_image_desmoked.to("cuda"), 0.0, 1.0).to(dtype=torch.float32)
                        else:
                            gt_image_desmoked = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0).to(dtype=torch.float32)
            
                        if idx in random_indices:
                            wandb.log({config['name'] + "_view_{}/render".format(viewpoint.image_name): [wandb.Image(image)],
                                    config['name'] + "_view_{}/render_desmoked".format(viewpoint.image_name): [wandb.Image(image_desmoked)],
                                    config['name'] + "_view_{}/render_smoke".format(viewpoint.image_name): [wandb.Image(image_smoke)]}, step=iteration)
                            wandb.log({
                                config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name): [wandb.Image(gt_image)],
                                config['name'] + "_view_{}/ground_truth_desmoked".format(viewpoint.image_name): [wandb.Image(gt_image_desmoked)]
                            }, step=iteration)

                            wandb.log({config['name'] + "_view_{}/inv_depth_all".format(viewpoint.image_name): [wandb.Image(depth_all)]}, step=iteration)
                            wandb.log({config['name'] + "_view_{}/inv_depth_surface".format(viewpoint.image_name): [wandb.Image(inv_depth_surface)]}, step=iteration)
                            wandb.log({config['name'] + "_view_{}/inv_depth_smoke".format(viewpoint.image_name): [wandb.Image(inv_depth_smoke)]}, step=iteration)
                            
                            if viewpoint.smoke_mask is not None:
                                smoke_mask = viewpoint.smoke_mask.cuda()
                                gt_image_with_smoke_mask = gt_image * smoke_mask
                                wandb.log({config['name'] + "_view_{}/gt_with_smoke_mask".format(viewpoint.image_name): [wandb.Image(gt_image_with_smoke_mask)]}, step=iteration)
                            
                            if use_thermal:
                                thermal_viewpoint = thermal_cameras[idx]
                                thermal_image_both = torch.clamp(renderFunc2(thermal_viewpoint, gaussians_surface, gaussians_smoke, deform_parameters=deform_parameters, *renderArgs, is_thermal=True)["render"], 0.0, 1.0)
                                thermal_image_surface = torch.clamp(renderFunc1(thermal_viewpoint, gaussians_surface, *renderArgs, deform_parameters=None, is_thermal=True)["render"], 0.0, 1.0)
                                thermal_image_smoke = torch.clamp(renderFunc1(thermal_viewpoint, gaussians_smoke, *renderArgs, deform_parameters=deform_parameters, is_thermal=True)["render"], 0.0, 1.0)
                                gt_thermal_image = torch.clamp(thermal_viewpoint.original_image.to("cuda"), 0.0, 1.0)
                                
                                wandb.log({config['name'] + "_view_{}/render_thermal_both".format(viewpoint.image_name): [wandb.Image(thermal_image_both)]}, step=iteration)
                                wandb.log({config['name'] + "_view_{}/render_thermal_surface".format(viewpoint.image_name): [wandb.Image(thermal_image_surface)]}, step=iteration)
                                wandb.log({
                                    config['name'] + "_view_{}/ground_truth_thermal".format(viewpoint.image_name): [wandb.Image(gt_thermal_image)]
                                }, step=iteration)
                                wandb.log({config['name'] + "_view_{}/render_thermal_smoke".format(viewpoint.image_name): [wandb.Image(thermal_image_smoke)]}, step=iteration)

                                psnr_test_thermal = psnr(thermal_image_surface, gt_thermal_image).mean().double()
                                
                    except Exception as e:
                        print(e)
                        continue
                    
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                    psnr_test_gt += psnr(image_desmoked, gt_image_desmoked).mean().double()

                    lpips_test_gt += lpips_module(image_desmoked, gt_image_desmoked).mean().double()
                    ssim_test_gt += ssim(image_desmoked, gt_image_desmoked).mean().double()

                psnr_test /= len(config['cameras'])
                psnr_test_gt /= len(config['cameras'])
                l1_test /= len(config['cameras'])   
                lpips_test_gt /= len(config['cameras'])
                ssim_test_gt /= len(config['cameras'])
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {} PSNR_GT {} LPIPS {} SSIM {}".format(iteration, config['name'], l1_test, psnr_test, psnr_test_gt, lpips_test_gt, ssim_test_gt))
                wandb.log({
                    config['name'] + '/loss_viewpoint - l1_loss': l1_test,
                    config['name'] + '/loss_viewpoint - psnr': psnr_test,
                    config['name'] + '/loss_viewpoint - psnr_gt': psnr_test_gt,
                    config['name'] + '/loss_viewpoint - lpips_gt': lpips_test_gt,
                    config['name'] + '/loss_viewpoint - ssim_gt': ssim_test_gt
                }, step=iteration)

                if use_thermal:
                    psnr_test_thermal /= len(config['cameras'])
                    wandb.log({
                        config['name'] + '/loss_viewpoint - psnr_thermal': psnr_test_thermal
                    }, step=iteration)
        try:
            wandb.log({"scene/opacity_histogram_surface": wandb.Histogram(gaussians_surface.get_opacity.cpu())}, step=iteration)
            wandb.log({"scene/opacity_histogram_smoke": wandb.Histogram(gaussians_smoke.get_opacity.cpu())}, step=iteration)
        except:
            print(gaussians_smoke._opacity.min(), gaussians_smoke._opacity.max())
            print(gaussians_surface.get_opacity.min(), gaussians_surface.get_opacity.max())
        wandb.log({'total_points_smoke': gaussians_smoke.get_xyz.shape[0], 'total_points_surface': gaussians_surface.get_xyz.shape[0]}, step=iteration)
        torch.cuda.empty_cache()