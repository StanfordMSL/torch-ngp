import torch

from nerf.provider import NeRFDataset
from nerf.gui import NeRFGUI
from nerf.utils import *

#torch.autograd.set_detect_anomaly(True)

if __name__ == '__main__':

    parser = get_config_parser()
    opt = parser.parse_args()
    print(opt)

    seed_everything(opt.seed)

    if opt.ff:
        assert opt.fp16, "fully-fused mode must be used with fp16 mode"
        from nerf.network_ff import NeRFNetwork
    elif opt.tcnn:
        assert opt.fp16, "tcnn mode must be used with fp16 mode"
        from nerf.network_tcnn import NeRFNetwork
    else:
        from nerf.network import NeRFNetwork

    model = NeRFNetwork(
        #encoding="frequency", encoding_dir="frequency", num_layers=4, hidden_dim=256, geo_feat_dim=256, num_layers_color=4, hidden_dim_color=128,
        bound=opt.bound,
        cuda_ray=opt.cuda_ray,
    )

    print(model)

    criterion = torch.nn.MSELoss() # HuberLoss(delta=0.1)

    ### test mode
    if opt.test:

        trainer = Trainer('ngp', vars(opt), model, workspace=opt.workspace, criterion=criterion, fp16=opt.fp16, metrics=[PSNRMeter()], use_checkpoint='latest')

        test_dataset = NeRFDataset(opt.path, type='test', mode=opt.mode, scale=opt.scale, preload=opt.preload, trans_noise=opt.trans_noise, rot_noise=opt.rot_noise)

        if opt.gui:
            gui = NeRFGUI(opt, trainer)
            gui.render()

        else:
            test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1)

            if opt.mode == 'blender':
                trainer.evaluate(test_loader) # blender has gt, so evaluate it.
            else:
                trainer.test(test_loader) # colmap doesn't have gt, so just test.

    else:

        optimizer = lambda model: torch.optim.Adam([
            {'name': 'encoding', 'params': list(model.encoder.parameters())},
            {'name': 'net', 'params': list(model.sigma_net.parameters()) + list(model.color_net.parameters()), 'weight_decay': 1e-6},
        ], lr=1e-2, betas=(0.9, 0.99), eps=1e-15)


        # need different milestones for GUI/CMD mode.
        scheduler = lambda optimizer: optim.lr_scheduler.MultiStepLR(optimizer, milestones=[625, 1000] if opt.gui else [100, 150], gamma=0.33)

        trainer = Trainer('ngp', vars(opt), model, workspace=opt.workspace, optimizer=optimizer, criterion=criterion, ema_decay=0.95, fp16=opt.fp16, lr_scheduler=scheduler, metrics=[PSNRMeter()], use_checkpoint='latest', eval_interval=15)

        # need different dataset type for GUI/CMD mode.

        if opt.gui:
            train_dataset = NeRFDataset(opt.path, type='train', mode=opt.mode, scale=opt.scale, preload=opt.preload, trans_noise=opt.trans_noise, rot_noise=opt.rot_noise)
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=opt.num_rays, shuffle=True, num_workers=8, pin_memory=True)
            # attach dataloader to trainer
            trainer.train_loader = train_loader
            trainer.loader = iter(train_loader) # very slow, can take ~10s to shuffle...

            # Create pose optimizer + attach, if training.
            if opt.opt_poses:
                # Create and attach poses to trainer
                train_pose_vars = create_pose_var(train_dataset, requires_grad=opt.opt_poses)
                trainer.train_pose_vars = train_pose_vars
                pose_optimizer = torch.optim.SGD([train_pose_vars], lr=1e-4, momentum=0.1)
                trainer.pose_optimizer = pose_optimizer
                trainer.H = train_dataset.H
                trainer.W = train_dataset.W
                trainer.pose_scheduler = optim.lr_scheduler.MultiStepLR(pose_optimizer, milestones=[50,100,150], gamma=1.)

            gui = NeRFGUI(opt, trainer)
            gui.render()

        else:
            train_dataset = NeRFDataset(opt.path, type='train', mode=opt.mode, scale=opt.scale, preload=opt.preload, trans_noise=opt.trans_noise, rot_noise=opt.rot_noise)
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=opt.num_rays, shuffle=True, num_workers=8, pin_memory=True)
            # Create pose optimizer + attach, if training.
            if opt.opt_poses:
                train_pose_vars= create_pose_var(train_dataset,
                                                 requires_grad=opt.opt_poses)
                trainer.train_pose_vars = train_pose_vars
                pose_optimizer = torch.optim.SGD([train_pose_vars], lr=1e-4, momentum=0.1)
                trainer.pose_optimizer = pose_optimizer
                trainer.pose_scheduler = optim.lr_scheduler.MultiStepLR(pose_optimizer, milestones=[50,100,150], gamma=1.)
                trainer.H = train_dataset.H
                trainer.W = train_dataset.W

            valid_dataset = NeRFDataset(opt.path, type='val', mode=opt.mode, downscale=2, scale=opt.scale, preload=opt.preload)
            valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=1, pin_memory=True)

            # train 200 epochs, each epoch has 100 steps --> in total 20,000 steps
            trainer.train(train_loader, valid_loader, 200, 100)

            # also test
            test_dataset = NeRFDataset(opt.path, type='test', mode=opt.mode, scale=opt.scale, preload=opt.preload)
            test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, pin_memory=True)
            if opt.mode == 'blender':
                trainer.evaluate(test_loader) # blender has gt, so evaluate it.
            else:
                trainer.test(test_loader) # colmap doesn't have gt, so just test.