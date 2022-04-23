import torch
import models.losses as losses
import models.models as models
import dataloaders.dataloaders as dataloaders
import utils.utils as utils
from utils.fid_scores import fid_pytorch
import config
import os

from config import print_options, save_options

# --- read options ---#
opt, parser = config.read_arguments(train=True)

opt.distributed = torch.cuda.device_count() > 1

if opt.distributed:
    local_rank = int(os.environ["LOCAL_RANK"])
    if 'SLURM_NPROCS' in os.environ:
        ngpus_per_node = torch.cuda.device_count()
        opt.world_size = int(os.environ['SLURM_NPROCS']) * ngpus_per_node
        os.environ['MASTER_ADDR'] = os.environ['SLURM_LAUNCH_NODE_IPADDR']
        opt.rank = int(os.environ['SLURM_PROCID']) * ngpus_per_node + local_rank
        opt.node_list = os.environ["SLURM_NODELIST"]
        opt.local_rank = local_rank
    else:
        opt.local_rank = local_rank
        opt.world_size = int(os.environ["WORLD_SIZE"])
        opt.rank = local_rank
    print(f"Init for Rank: {opt.rank}, Local Rank: {opt.local_rank}")
    torch.cuda.set_device(local_rank)
    torch.distributed.init_process_group(backend=opt.dist_backend,
                                         init_method=opt.dist_url,
                                         world_size=opt.world_size,
                                         rank=opt.rank)
else:
    opt.rank = 0

print_options(opt, parser)
save_options(opt, parser)
# --- create utils ---#
timer = utils.timer(opt)
visualizer_losses = utils.losses_saver(opt)
losses_computer = losses.losses_computer(opt)
dataloader, dataloader_val, train_sampler = dataloaders.get_dataloaders(opt)
im_saver = utils.image_saver(opt)
if opt.rank == 0:
    fid_computer = fid_pytorch(opt, dataloader_val)

# --- create models ---#
model = models.OASIS_model(opt)
model = models.put_on_multi_gpus(model, opt)

# --- create optimizers ---#
optimizerG = torch.optim.Adam(model.module.netG.parameters(), lr=opt.lr_g, betas=(opt.beta1, opt.beta2))
optimizerD = torch.optim.Adam(model.module.netD.parameters(), lr=opt.lr_d, betas=(opt.beta1, opt.beta2))

# --- the training loop ---#
already_started = False
start_epoch, start_iter = utils.get_start_iters(opt.loaded_latest_iter, len(dataloader))
for epoch in range(start_epoch + 1, opt.num_epochs + 1):
    train_sampler.set_epoch(epoch)
    for i, data_i in enumerate(dataloader, start=1):
        if not already_started and i < start_iter:
            continue
        already_started = True
        cur_iter = (epoch - 1) * len(dataloader) + i
        image, label = models.preprocess_input(opt, data_i)

        # --- generator update ---#
        model.module.netG.zero_grad()
        loss_G, losses_G_list = model(image, label, "losses_G", losses_computer)
        loss_G, losses_G_list = loss_G.mean(), [loss.mean() if loss is not None else None for loss in losses_G_list]
        loss_G.backward()
        optimizerG.step()

        # --- discriminator update ---#
        model.module.netD.zero_grad()
        loss_D, losses_D_list = model(image, label, "losses_D", losses_computer)
        loss_D, losses_D_list = loss_D.mean(), [loss.mean() if loss is not None else None for loss in losses_D_list]
        loss_D.backward()
        optimizerD.step()

        # --- stats update ---#
        if not opt.no_EMA:
            utils.update_EMA(model, cur_iter, dataloader, opt)
        visualizer_losses(cur_iter, losses_G_list + losses_D_list)

    im_saver.visualize_batch(model, image, label)
    timer(epoch, cur_iter)
    utils.save_networks(opt, cur_iter, model, latest=True)
    if epoch % 2 == 0 and cur_iter > 0 and opt.rank == 0:
        is_best = fid_computer.update(model, cur_iter)
        if is_best:
            utils.save_networks(opt, cur_iter, model, best=True)

# --- after training ---#
utils.update_EMA(model, cur_iter, dataloader, opt, force_run_stats=True)
utils.save_networks(opt, cur_iter, model)
utils.save_networks(opt, cur_iter, model, latest=True)
if opt.rank == 0:
    is_best = fid_computer.update(model, cur_iter)
    if is_best:
        utils.save_networks(opt, cur_iter, model, best=True)

if opt.rank == 0:
    print("The training has successfully finished")
