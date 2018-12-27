import config
from layers import act_cuboid_loss
from data import dataset
from layers import ssd
import torch
import time
import torch.optim.lr_scheduler as lr_scheduler
import test


def main():
    args = config.Config()
    # train_dataset = tube_dataset.TubeDataset(args.dataset, data_path=args.data_path, phase='train',
    #                                          modality=args.modality,
    #                                          sequence_length=6)
    train(args)
    exit(0)


def train(args):
    use_gpu = args.use_gpu
    if args.dataset == 'UCF101v2':
        num_class = 25
    elif args.dataset == 'UCFSports':
        num_class = 11
    else:
        num_class = 0
        print("No dataset name {}".format(args.dataset))
        exit(0)
    variance = args.variance
    MAX_GEN = args.epochs
    k_frames = args.sequence_length
    print("train batch size:", args.train_batch_size, 'lr', args.lr)
    train_net = ssd.SSD_NET(dataset=args.dataset, frezze_init=args.freeze_init, num_classes=num_class, modality=args.modality)
    if args.reinit_all:
        print("reinit all data!!!")
        start_gen = 0
        train_net.load_state_dict(
            torch.load(args.init_model))
        optimizer = torch.optim.SGD(train_net.get_optim_policies(),
                                    lr=args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.94)
        loss_class_list = []
        loss_loc_list = []
        loss_list = []
    else:
        print("load last train data!!!")
        data_dict = torch.load(args.new_trained_model)
        start_gen = data_dict['gen_num']
        start_gen = 0
        net_state_dict = {}
        for key in data_dict['net_state_dict']:
            if 'module.' in key:
                new_key = key.replace('module.', '')
            else:
                new_key = key
            net_state_dict[new_key] = data_dict['net_state_dict'][key]
        train_net.load_state_dict(net_state_dict)
        optimizer = torch.optim.SGD(train_net.get_optim_policies(),
                                    lr=args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)
        # optimizer.load_state_dict(data_dict['optimizer'])
        for group in optimizer.param_groups:
            if 'initial_lr' not in group:
                group['initial_lr'] = args.lr
        # optimizer.defaults['lr'] = args.lr
        scheduler = lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.94, last_epoch=start_gen - 1)
        loss_class_list = data_dict['loss_class_list']
        loss_loc_list = data_dict['loss_loc_list']
        loss_list = data_dict['loss_list']
        print("last data: GEN:", start_gen, "\tloss loc:", loss_loc_list[-1], "\tloss conf:", loss_class_list[-1],
              "\tloss:", loss_list[-1],
              "\tlr:", scheduler.get_lr())
    train_net.train(True)
    if use_gpu:
        train_net = torch.nn.DataParallel(train_net).cuda()
    print('all  net loaded ok!!!')
    train_dataset = dataset.TubeDataset(args.dataset, data_path=args.data_path, phase='train',
                                             modality=args.modality,
                                             sequence_length=6)
    dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True,
                                             num_workers=args.workers, pin_memory=True)
    criterion = act_cuboid_loss.CuboidLoss(use_gpu, variance, num_class, k_frames)
    mmap_best = 0
    mmap_list = []
    if args.reinit_all:
        warm_up(train_net, dataloader, train_dataset, criterion, optimizer, scheduler, use_gpu, args, loss_loc_list,
                loss_class_list, loss_list)
    for gen in range(start_gen, MAX_GEN):
        train_epoch(train_net, dataloader, train_dataset, criterion, optimizer, scheduler, use_gpu, args, gen,
                   loss_loc_list, loss_class_list, loss_list)
        if (gen + 1) % 2 == 0:
            temp_dict = {}
            temp_dict['net_state_dict'] = train_net.state_dict()
            temp_dict['gen_num'] = gen + 1
            temp_dict['optimizer'] = optimizer.state_dict()
            temp_dict['loss_loc_list'] = loss_loc_list
            temp_dict['loss_class_list'] = loss_class_list
            temp_dict['loss_list'] = loss_list
            temp_dict['mmap_list'] = mmap_list
            torch.save(temp_dict, args.new_trained_model)
            print("net save ok!!")
            if gen > 50 and (gen + 1) % 4 == 0 and loss_list[-1] < 1.0:
                mmap = test.eval_rgb_or_flow(model=train_net.module, eval_dataset=None, eval_dataloader=None, args=args,
                                        GEN_NUM=gen + 1)
                with open('./train_log_{}.txt'.format(args.dataset), 'a') as train_log:
                    log = "GEN:{}".format(gen) + "\tmap:{}".format(mmap) + "\tbest map:{}\n".format(mmap_best)
                    train_log.write(log)
                train_net.module.train(True)
                mmap_list += [mmap]
                if mmap > mmap_best:
                    mmap_best = mmap
                    temp_dict['mmap_best'] = mmap_best
                    torch.save(temp_dict, args.best_trained_model)
                print("current map:{}  best map:{}".format(mmap, mmap_best))


def train_epoch(train_net, dataloader, train_dataset, criterion, optimizer, scheduler, use_gpu, args, gen, loss_loc_list,
               loss_class_list, loss_list, warm_up_lr_inc=None):
    total_loss = AverageMeter()
    loss_ls = AverageMeter()
    loss_cs = AverageMeter()
    if warm_up_lr_inc is None:
        scheduler.step()
    total_loss.reset()
    loss_ls.reset()
    loss_cs.reset()
    for i, (input, target) in enumerate(dataloader):
        if warm_up_lr_inc is not None:
            for lr in range(len(optimizer.param_groups)):
                optimizer.param_groups[lr]['lr'] += warm_up_lr_inc[lr]
        if use_gpu:
            input = input.cuda()
            target = target.cuda()
        loc_preds, conf_preds = train_net(input)
        loss_l, loss_c = criterion((loc_preds, conf_preds), target)
        loss = loss_l + loss_c
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if use_gpu:
            total_loss.update(loss.cpu().detach().numpy())
            loss_ls.update(loss_l.cpu().detach().numpy())
            loss_cs.update(loss_c.cpu().detach().numpy())
        else:
            total_loss.update(loss.detach().numpy())
            loss_ls.update(loss_l.detach().numpy())
            loss_cs.update(loss_c.detach().numpy())

        if (i+1) % 100 == 0:
            print("GEN:", gen, "\tnum:{}/{}".format((i + 1) * args.train_batch_size, train_dataset.__len__()),
                  "\tloss loc:", loss_ls.avg, "\tloss conf:", loss_cs.avg, "\tloss:", total_loss.avg,
                  "\tlr:", scheduler.get_lr())
    print("\tloss loc:", loss_ls.avg, "\tloss conf:", loss_cs.avg, "\tloss:", total_loss.avg)
    with open('./train_log_{}.txt'.format(args.dataset), 'a') as train_log:
        log = "GEN:{}".format(gen) + "\tloss loc:{}".format(loss_ls.avg) + "\tloss conf:{}".format(loss_cs.avg) + \
              "\tloss:{}".format(total_loss.avg) + "\tlr:{}".format(scheduler.get_lr()) + time.strftime(
            '\t%m/%d  %H:%M:%S\n', time.localtime(time.time()))
        train_log.write(log)
    loss_loc_list += [loss_ls.avg]
    loss_class_list += [loss_cs.avg]
    loss_list += [total_loss.avg]


def warm_up(train_net, dataloader, train_dataset, criterion, optimizer, scheduler, use_gpu, args, loss_loc_list,
            loss_class_list, loss_list):
    warm_up_ratio = args.warm_up_ratio
    warm_up_epoch = args.warm_up_epoch
    lr_inc = []
    for i in range(len(optimizer.param_groups)):
        lr_inc.append(optimizer.param_groups[i]['lr'] * (1 - warm_up_ratio)
                      / (len(dataloader) * warm_up_epoch))
        optimizer.param_groups[i]['lr'] *= warm_up_ratio
    for warm_up_index in range(warm_up_epoch):
        train_epoch(train_net, dataloader, train_dataset, criterion, optimizer, scheduler, use_gpu, args,
                   warm_up_index - warm_up_epoch, loss_loc_list, loss_class_list, loss_list, warm_up_lr_inc=lr_inc)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


if __name__ == '__main__':
    main()
    # image_test_from_file('./ucf101_test.pkl')
