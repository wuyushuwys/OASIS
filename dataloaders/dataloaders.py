import torch


def get_dataset_name(mode):
    if mode == "ade20k":
        return "Ade20kDataset"
    if mode == "cityscapes":
        return "CityscapesDataset"
    if mode == "coco":
        return "CocoStuffDataset"
    else:
        ValueError("There is no such dataset regime as %s" % mode)


def get_dataloaders(opt):
    dataset_name = get_dataset_name(opt.dataset_mode)

    file = __import__("dataloaders." + dataset_name)
    dataset_train = file.__dict__[dataset_name].__dict__[dataset_name](opt, for_metrics=False)
    dataset_val = file.__dict__[dataset_name].__dict__[dataset_name](opt, for_metrics=True)
    if opt.rank == 0:
        print("Created %s, size train: %d, size val: %d" % (dataset_name, len(dataset_train), len(dataset_val)))
    train_sampler = torch.utils.data.distributed.DistributedSampler(dataset_train)

    dataloader_train = torch.utils.data.DataLoader(dataset_train,
                                                   batch_size=opt.batch_size,
                                                   shuffle=(train_sampler is None),
                                                   drop_last=True,
                                                   num_workers=opt.num_workers,
                                                   pin_memory=True,
                                                   sampler=train_sampler,
                                                   prefetch_factor=opt.batch_size)
    dataloader_val = torch.utils.data.DataLoader(dataset_val,
                                                 batch_size=opt.batch_size,
                                                 shuffle=False,
                                                 drop_last=False,
                                                 num_workers=opt.num_workers,
                                                 pin_memory=True, )

    return dataloader_train, dataloader_val, train_sampler
