# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------

import math
import sys
from typing import Iterable
import s3fs
import torch
from matplotlib.backends.backend_agg import FigureCanvasAgg
import util.misc as misc
import util.lr_sched as lr_sched
import io

def train_one_epoch(model: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    log_writer=None,
                    args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))
    
    count_nans = 0

    for data_iter_step, (samples, idx) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        
        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)
        
        samples = samples.to(device, non_blocking=True)

        with torch.cuda.amp.autocast():
            loss, _, _, currupt_img = model(samples, mask_ratio=args.mask_ratio)
        
        if torch.isnan(loss):
            loss = torch.nan_to_num(loss)
            count_nans += 1
        if torch.all(samples == 0):
            print(samples[0][0], idx)
            count_nans += 1

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, parameters=model.parameters(),
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()
        
        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)
    
    if args.s3:
        s3 = s3fs.S3FileSystem()
        fig = misc.plot_reconstruction(currupt_img, samples)
        file_path = "images/checkpoint-%s.png" % str(epoch)  # Update this to your desired path in the bucket
        canvas = FigureCanvasAgg(fig)
        # Prepare an in-memory binary stream buffer
        imdata = io.BytesIO()
        # Write the canvas object as a PNG file to the buffer
        canvas.print_png(imdata)
        # Initialize an S3FileSystem instance
        # You can use default credentials or specify them explicitly
        # Specify your bucket and file path
        bucket_name = 's3://sagemaker-us-east-1-818515436582/MAE_Weights'
        # Upload the PNG image to your S3 bucket
        with s3.open(f'{bucket_name}/{file_path}', 'wb') as f:
            f.write(imdata.getvalue())
            
        print(f"File uploaded to s3://{bucket_name}/{file_path}")
    
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    print("Number of NaN Losses:", count_nans) 
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}