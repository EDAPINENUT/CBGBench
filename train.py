import os
import shutil
import argparse
import torch
import torch.utils.tensorboard
from torch.nn.utils import clip_grad_norm_
from tqdm.auto import tqdm
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import os
from repo.utils.misc import *
from repo.utils.train import *
from repo.datasets.pl import get_pl_dataset
from repo.models import get_model
from repo.utils.evaluate import *
from repo.utils.data import get_collate_fn, recursive_to

# python train.py --config ./configs/denovo/train/targetdiff.yml --logdir ./logs/denovo/targetdiff 
# python train.py --config ./configs/denovo/train/diffsbdd.yml --logdir ./logs/denovo/diffsbdd 
# python train.py --config ./configs/denovo/train/diffbp.yml --logdir ./logs/denovo/diffbp 
# python train.py --config ./configs/denovo/train/pocket2mol.yml --logdir ./logs/denovo/pocket2mol 
# python train.py --config ./configs/denovo/train/d3fg_fg.yml --logdir ./logs/denovo/d3fg_fg
# python train.py --config ./configs/denovo/train/d3fg_linker.yml --logdir ./logs/denovo/d3fg_linker
# python train.py --config ./configs/denovo/train/graphbp.yml --logdir ./logs/denovo/graphbp 
# python train.py --config ./configs/denovo/train/flag.yml --logdir ./logs/denovo/flag 


# python train.py --config ./configs/linker/train/targetdiff.yml --logdir ./logs/linker/targetdiff
# python train.py --config ./configs/linker/train/diffsbdd.yml --logdir ./logs/linker/diffsbdd --finetune
# python train.py --config ./configs/linker/train/diffbp.yml --logdir ./logs/linker/diffbp --finetune
# python train.py --config ./configs/linker/train/pocket2mol.yml --logdir ./logs/linker/pocket2mol --finetune
# python train.py --config ./configs/linker/train/graphbp.yml --logdir ./logs/linker/graphbp --finetune


# python train.py --config ./configs/frag/train/targetdiff.yml --logdir ./logs/frag/targetdiff --finetune
# python train.py --config ./configs/frag/train/pocket2mol.yml --logdir ./logs/frag/pocket2mol --finetune
# python train.py --config ./configs/frag/train/graphbp.yml --logdir ./logs/frag/graphbp --finetune
# python train.py --config ./configs/frag/train/diffbp.yml --logdir ./logs/frag/diffbp --finetune
# python train.py --config ./configs/frag/train/diffsbdd.yml --logdir ./logs/frag/diffsbdd --finetune


# python train.py --config ./configs/sidechain/train/targetdiff.yml --logdir ./logs/sidechain/targetdiff --finetune
# python train.py --config ./configs/sidechain/train/pocket2mol.yml --logdir ./logs/sidechain/pocket2mol --finetune
# python train.py --config ./configs/sidechain/train/graphbp.yml --logdir ./logs/sidechain/graphbp --finetune
# python train.py --config ./configs/sidechain/train/diffbp.yml --logdir ./logs/sidechain/diffbp --finetune
# python train.py --config ./configs/sidechain/train/diffsbdd.yml --logdir ./logs/sidechain/diffsbdd --finetune


# python train.py --config ./configs/scaffold/train/targetdiff.yml --logdir ./logs/scaffold/targetdiff --finetune
# python train.py --config ./configs/scaffold/train/pocket2mol.yml --logdir ./logs/scaffold/pocket2mol --finetune
# python train.py --config ./configs/scaffold/train/graphbp.yml --logdir ./logs/scaffold/graphbp --finetune
# python train.py --config ./configs/scaffold/train/diffbp.yml --logdir ./logs/scaffold/diffbp --finetune
# python train.py --config ./configs/scaffold/train/diffsbdd.yml --logdir ./logs/scaffold/diffsbdd --finetune


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/denovo/train/diffsbdd.yml', type=str)
    parser.add_argument('--logdir', type=str, default='./logs/denovo/diffsbdd')         
    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--tag', type=str, default='')
    parser.add_argument('--finetune', action='store_true', default=False)
    args = parser.parse_args()

    # Load configs
    config, config_name = load_config(args.config)

    seed_all(config.train.seed)
    if 'fg' not in config.model.type:
        from repo.utils.configuration import set_num_atom_type, set_num_bond_type
        set_num_atom_type(config)
        set_num_bond_type(config)
    else:
        from repo.utils.configuration import set_num_fg_type
        set_num_fg_type(config)

    # Logging
    if args.debug:
        logger = get_logger('train', None)
        writer = BlackHole()
        args.num_workers = 1

    else:
        if config.get('resume', None) is not None:
            log_dir = get_new_log_dir(args.logdir, prefix=config_name+'-resume', tag=args.tag)
        else:
            log_dir = get_new_log_dir(args.logdir, prefix=config_name, tag=args.tag)

        ckpt_dir = os.path.join(log_dir, 'checkpoints')
        if not os.path.exists(ckpt_dir): os.makedirs(ckpt_dir)
        logger = get_logger('train', log_dir)
        writer = torch.utils.tensorboard.SummaryWriter(log_dir)
        tensorboard_trace_handler = torch.profiler.tensorboard_trace_handler(log_dir)
        if not os.path.exists(os.path.join(log_dir, os.path.basename(args.config))):
            shutil.copyfile(args.config, os.path.join(log_dir, os.path.basename(args.config)))
    logger.info(args)
    logger.info(config)

    # Data
    logger.info('Loading datasets...')
    datasets = get_pl_dataset(config.data.train)
    train_dataset, val_dataset = datasets['train'], datasets['val']
    
    if hasattr(config.data, 'collate_fn'):
        from torch.utils.data import DataLoader
        collate_fn = get_collate_fn(config.data.collate_fn) 
        train_loader = DataLoader(train_dataset, 
                                  batch_size=config.train.batch_size, 
                                  shuffle=True, 
                                  num_workers=args.num_workers,
                                  collate_fn = collate_fn
                                  )
        val_loader = DataLoader(val_dataset, 
                                batch_size=config.train.batch_size, 
                                shuffle=False, 
                                num_workers=args.num_workers,
                                collate_fn = collate_fn
                                )
    else:
        from repo.utils.loader import DataLoader
        train_loader = DataLoader(train_dataset, 
                                  batch_size=config.train.batch_size, 
                                  shuffle=True, 
                                  num_workers=args.num_workers,
                                  follow_batch=config.data.get('follow_batch', []),
                                  exclude_keys=config.data.get('exclude_keys', [])
                                  )
        val_loader = DataLoader(val_dataset, 
                                batch_size=config.train.batch_size, 
                                shuffle=False, 
                                num_workers=args.num_workers,
                                follow_batch=config.data.get('follow_batch', []),
                                exclude_keys=config.data.get('exclude_keys', [])
                                )
    
    train_iterator = inf_iterator(train_loader)

    
    evaluator = Evaluator(config.eval.get('metrics', []))
    
    logger.info('Train %d | Val %d' % (len(train_dataset), len(val_dataset)))

    # Model
    logger.info('Building model...')
    model = get_model(config.model).to(args.device)
    logger.info('Number of parameters: %d' % count_parameters(model))

    # Optimizer & Scheduler
    optimizer = get_optimizer(config.train.optimizer, model)
    scheduler = get_scheduler(config.train.scheduler, optimizer)
    optimizer.zero_grad()
    it_first = 1

    # Resume
    if config.get('resume', None) is not None:
        logger.info('Resuming from checkpoint: %s' % config.resume)
        ckpt = torch.load(config.resume, map_location=args.device)
        lsd_result = model.load_state_dict(ckpt['model'], strict=False)
        logger.info('Missing keys (%d): %s' % (len(lsd_result.missing_keys), 
                                               ', '.join(lsd_result.missing_keys)))
        logger.info('Unexpected keys (%d): %s' % (len(lsd_result.unexpected_keys), 
                                                  ', '.join(lsd_result.unexpected_keys)))
        if not args.finetune:
            logger.info('Resuming optimizer states...')
            optimizer.load_state_dict(ckpt['optimizer'])
            logger.info('Resuming scheduler states...')
            scheduler.load_state_dict(ckpt['scheduler'])
            it_first = ckpt['iteration']  # + 1


    def train(it):
        time_start = current_milli_time()
        model.train()

        # Prepare data
        batch = next(train_iterator)
        try:
            batch = batch.to(args.device)
        except:
            batch = recursive_to(batch, args.device)

        # Forward pass
        loss_dict, _ = model(batch)
        loss = sum_weighted_losses(loss_dict, config.train.loss_weights)
        time_forward_end = current_milli_time()

        # Backward
        loss.backward()
        orig_grad_norm = clip_grad_norm_(model.parameters(), config.train.max_grad_norm)
        optimizer.step()
        optimizer.zero_grad()
        time_backward_end = current_milli_time()

        if it % config.train.report_freq == 0:
            # Logging
            scalar_dict = {}
            scalar_dict.update({
                'grad': orig_grad_norm,
                'lr': optimizer.param_groups[0]['lr'],
                'time_forward': (time_forward_end - time_start) / 1000,
                'time_backward': (time_backward_end - time_forward_end) / 1000,
            })
            log_losses(loss, loss_dict, scalar_dict, it=it, tag='train', logger=logger, writer=writer)

    def validate(it, evaluator):
        scalar_accum = ScalarMetricAccumulator()
        
        with torch.no_grad():
            model.eval()

            for i, batch in enumerate(tqdm(val_loader, desc='Validate', dynamic_ncols=True)):
                # Prepare data
                try:
                    batch = batch.to(args.device)
                except:
                    batch = recursive_to(batch, args.device)

                # Forward pass
                loss_dict, results = model(batch)
                loss = sum_weighted_losses(loss_dict, config.train.loss_weights)

                try:
                    B = batch[config.data.follow_batch[0]+'_batch'].max() + 1
                except:
                    if batch.get('batch', None) is not None:
                        B = batch['batch'].max() + 1
                    else: # HeteroGraph
                        B = batch['protein']['batch'].max() + 1
                
                scalar_accum.add(name='loss', value=loss, batchsize=B, mode='mean')
                for k, v in loss_dict.items():
                    scalar_accum.add(name=k, value=v, batchsize=B, mode='mean')
                # Calculate metrics
                metric_dict = evaluator(results)
                for k, v in metric_dict.items():
                    scalar_accum.add(name=k, value=v, batchsize=B, mode='mean')
            
        avg_loss = scalar_accum.get_average('loss')
        scalar_accum.log(it, 'val', logger=logger, writer=writer)

        # Trigger scheduler
        if it != it_first:  # Don't step optimizers after resuming from checkpoint
            if config.train.scheduler.type == 'plateau':
                scheduler.step(avg_loss)
            else:
                scheduler.step()
        return avg_loss

    try:
        best_loss, best_iter = None, None
        for it in range(it_first, config.train.max_iters + 1):
            train(it)

            if it % config.eval.val_freq == 0:
                try:
                    avg_val_loss = validate(it, evaluator=evaluator)
                
                    if best_loss is None or avg_val_loss < best_loss or it % config.eval.get('force_save_freq', 1000000) == 0:
                        logger.info(f'[Validate] Best val loss achieved: {avg_val_loss:.6f}')
                        best_loss, best_iter = avg_val_loss, it
                        if not args.debug:
                            ckpt_path = os.path.join(ckpt_dir, '%d.pt' % it)
                            # os.remove(ckpt_dir)
                            torch.save({
                                'config': config,
                                'model': model.state_dict(),
                                'optimizer': optimizer.state_dict(),
                                'scheduler': scheduler.state_dict(),
                                'iteration': it,
                                'avg_val_loss': avg_val_loss,
                            }, ckpt_path)
                
                    else:
                        logger.info(f'[Validate] Val loss is not improved. '
                                    f'Best val loss: {best_loss:.6f} at iter {best_iter}')
                except IndexError:
                    print('Something wrong with the indexes in the validation set. Skip the validation.')
                
    except KeyboardInterrupt:
        logger.info('Terminating...')
