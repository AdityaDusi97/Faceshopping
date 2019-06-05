import time
from options.train_options import TrainOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
from util.visualizer import Visualizer
from tensorboardX import SummaryWriter
import os, datetime

opt = TrainOptions().parse()
data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
dataset_size = len(data_loader)
print('#training images = %d' % dataset_size)

model = create_model(opt)
visualizer = Visualizer(opt)
total_steps = 0

run_dir = "/home/aditya/environment/" # place to add tensorboard prints
time_stamp = datetime.datetime.now().strftime("%m_%d_%H_%M_%S")
# create path for summary writer
run_dir += time_stamp
writer = SummaryWriter(run_dir)
iteration = 0 # to keep track of iteration

for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
    epoch_start_time = time.time()
    epoch_iter = 0

    for i, data in enumerate(dataset):
        iter_start_time = time.time()
        visualizer.reset()
        total_steps += opt.batchSize
        epoch_iter += opt.batchSize
        model.set_input(data)
        model.optimize_parameters()

        if total_steps % opt.display_freq == 0:
            save_result = total_steps % opt.update_html_freq == 0
            visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

        if total_steps % opt.print_freq == 0:
            errors = model.get_current_errors() #get error dictionary from model
            # Adding tensorboard prints
            if opt.with_D_PP:
                writer.add_scalar('D_PP', errors['D_PP'], iteration)
            if opt.with_D_PB:
                writer.add_scalar('D_PB', errors['D_PB'], iteration)
            if opt.with_D_PB or opt.with_D_PP:
                writer.add_scalar('pair_GANLoss', errors['pair_GANloss'], iteration)
            if opt.L1_type == 'l1_plus_perL1':
                writer.add_scalar('origin_L1', errors['origin_L1'], iteration)
                writer.add_scalar('perceptual', errors['perceptual'], iteration)
                
            t = (time.time() - iter_start_time) / opt.batchSize
            visualizer.print_current_errors(epoch, epoch_iter, errors, t)
            if opt.display_id > 0:
                visualizer.plot_current_errors(epoch, float(epoch_iter)/dataset_size, opt, errors)

        if total_steps % opt.save_latest_freq == 0:
            print('saving the latest model (epoch %d, total_steps %d)' %
                  (epoch, total_steps))
            model.save('latest')

    if epoch % opt.save_epoch_freq == 0:
        print('saving the model at the end of epoch %d, iters %d' %
              (epoch, total_steps))
        model.save('latest')
        model.save(epoch)

    print('End of epoch %d / %d \t Time Taken: %d sec' %
          (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
    model.update_learning_rate()
