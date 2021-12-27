"""General-purpose training script for image-to-image translation.
'15KEW4Oqi_5xuaVI97YMuLVhXnpmgrE3A'
This script works for various models (with option '--model': e.g., pix2pix, cyclegan, colorization) and
different datasets (with option '--dataset_mode': e.g., aligned, unaligned, single, colorization).
You need to specify the dataset ('--dataroot'), experiment name ('--name'), and model ('--model').

It first creates model, dataset, and visualizer given the option.
It then does standard network training. During the training, it also visualize/save the images, print/save the loss plot, and save models.
The script supports continue/resume training. Use '--continue_train' to resume your previous training.

Example:
    Train a CycleGAN model:
        python train.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan
    Train a pix2pix model:
        python train.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/train_options.py for more training options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
import time, os
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer
try:
    import wandb
except ImportError:
    pass

import glob

from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive

if not os.path.isfile('mycreds.txt'):
    with open('mycreds.txt','w') as f:
        f.write('{"access_token": "ya29.a0AfH6SMC_aOt4BLq-OQ1oN4txyT5Guk9KMeEzqYJDjo4AkqD0fMJnIdQm4TGz3PQit8qNa-QEg3hdg66ic2pLErifxwsEhgPP-MIa947Ayigh8c5czN64T9IxCyLkR2M-5ygdjOhV5OzuXw-O6LfBJG9vBwMkyg9OKL0", "client_id": "883051571054-2e0bv2mjqra6i3cd6c915hkjgtdutct0.apps.googleusercontent.com", "client_secret": "NmzemQWSeUm_WWTbmUJi5xt7", "refresh_token": "1//0gE7zkyCPJ4RpCgYIARAAGBASNwF-L9IrISJx8AG8doLKF1C8RMbuvkqS6BsxGXaYJfqlB-RbrtmIESmVIA2krp-rK-Ylm26klmU", "token_expiry": "2020-07-29T16:47:41Z", "token_uri": "https://oauth2.googleapis.com/token", "user_agent": null, "revoke_uri": "https://oauth2.googleapis.com/revoke", "id_token": null, "id_token_jwt": null, "token_response": {"access_token": "ya29.a0AfH6SMC_aOt4BLq-OQ1oN4txyT5Guk9KMeEzqYJDjo4AkqD0fMJnIdQm4TGz3PQit8qNa-QEg3hdg66ic2pLErifxwsEhgPP-MIa947Ayigh8c5czN64T9IxCyLkR2M-5ygdjOhV5OzuXw-O6LfBJG9vBwMkyg9OKL0", "expires_in": 3599, "refresh_token": "1//0gE7zkyCPJ4RpCgYIARAAGBASNwF-L9IrISJx8AG8doLKF1C8RMbuvkqS6BsxGXaYJfqlB-RbrtmIESmVIA2krp-rK-Ylm26klmU", "scope": "https://www.googleapis.com/auth/drive", "token_type": "Bearer"}, "scopes": ["https://www.googleapis.com/auth/drive"], "token_info_uri": "https://oauth2.googleapis.com/tokeninfo", "invalid": false, "_class": "OAuth2Credentials", "_module": "oauth2client.client"}')

        # {"access_token": "ya29.a0AfH6SMCDGn8XAOVlzeT47aIMf7QlauIfWz3G9fXrRTyX0JgSllcpHrAIuj6s6zqNTI0kK46c4LmVQp2svHpCSltdQrSgLo-74UtFWv4mdUX0Rnt5TxM7I_OaewjmLl6vH8wmrk1bccDAWBY_-vTeBI-eEedfSNRQu4Mc", "client_id": "883051571054-2e0bv2mjqra6i3cd6c915hkjgtdutct0.apps.googleusercontent.com", "client_secret": "NmzemQWSeUm_WWTbmUJi5xt7", "refresh_token": "1//0gE7zkyCPJ4RpCgYIARAAGBASNwF-L9IrISJx8AG8doLKF1C8RMbuvkqS6BsxGXaYJfqlB-RbrtmIESmVIA2krp-rK-Ylm26klmU", "token_expiry": "2020-08-09T09:46:00Z", "token_uri": "https://oauth2.googleapis.com/token", "user_agent": null, "revoke_uri": "https://oauth2.googleapis.com/revoke", "id_token": null, "id_token_jwt": null, "token_response": {"access_token": "ya29.a0AfH6SMCDGn8XAOVlzeT47aIMf7QlauIfWz3G9fXrRTyX0JgSllcpHrAIuj6s6zqNTI0kK46c4LmVQp2svHpCSltdQrSgLo-74UtFWv4mdUX0Rnt5TxM7I_OaewjmLl6vH8wmrk1bccDAWBY_-vTeBI-eEedfSNRQu4Mc", "expires_in": 3599, "scope": "https://www.googleapis.com/auth/drive", "token_type": "Bearer"}, "scopes": ["https://www.googleapis.com/auth/drive"], "token_info_uri": "https://oauth2.googleapis.com/tokeninfo", "invalid": false, "_class": "OAuth2Credentials", "_module": "oauth2client.client"}


gauth = GoogleAuth()
# Try to load saved client credentials
gauth.LoadCredentialsFile("mycreds.txt")
# if gauth.credentials is None:
#     # Authenticate if they're not there
#     gauth.LocalWebserverAuth()
if gauth.access_token_expired:
    # Refresh them if expired
    gauth.Refresh()
else:
    # Initialize the saved creds
    gauth.Authorize()
# Save the current credentials to a file
gauth.SaveCredentialsFile("mycreds.txt")

# drive = GoogleDrive(gauth)

def authorize_drive():
    # global drive
    global gauth
    # Try to load saved client credentials
    gauth.LoadCredentialsFile("mycreds.txt")

    if gauth.access_token_expired:
        # Refresh them if expired
        gauth.Refresh()
    else:
        # Initialize the saved creds
        gauth.Authorize()
    # Save the current credentials to a file
    gauth.SaveCredentialsFile("mycreds.txt")

    drive = GoogleDrive(gauth)

    return drive


# def validate_parent_id(parent_id):
#     global drive
#     file_list = drive.ListFile({'q': f"title='{folder_name}' and trashed=false and mimeType='application/vnd.google-apps.folder'"}).GetList()
#         if len(file_list) > 1:
#             raise ValueError('There are multiple folders with that specified folder name')
#         elif len(file_list) == 0:
#             raise ValueError('No folders match that specified folder name')


def upload_to_drive(list_files,parent_id):
    # global drive
    drive = authorize_drive()
    # parent_id = ''# parent id
    drive_files = drive.ListFile({'q': "'%s' in parents and trashed=false"%parent_id}).GetList()
    drive_files = {f['title']:f for f in drive_files}
    for path in list_files:
        if not os.path.isfile(path): continue
        d,f = os.path.split(path)
        # check if file already exists and trash it
        if f in drive_files:
                drive_files[f].Trash()

        file = drive.CreateFile({'title': f, 'parents': [{'id': parent_id}]})
        file.SetContentFile(path)
        file.Upload()

def download_checkpoints(parent_id,checkpoints,root_dir='outdir'):
    drive = authorize_drive()
    if isinstance(checkpoints,str):
        checkpoints = [checkpoints]
    downloaded_files = []
    os.makedirs(root_dir,exist_ok=True)
    # checkpoint = ''
    # file_list = drive.ListFile({'q': "title contains 'My Awesome File' and trashed=false"}).GetList()
    file_list = drive.ListFile({'q': "'%s' in parents and trashed=false"%parent_id}).GetList()  #check if it is iterator
    # print(file_list)
    for checkpoint in checkpoints:
        ckpt_path = os.path.join(root_dir,checkpoint)
        for f in file_list:
            if f['title'].lower() == checkpoint:
                file_id = f['id']
                file = drive.CreateFile({'id': file_id})
                file.GetContentFile(ckpt_path)
                downloaded_files.append(ckpt_path)
#

        if os.path.isfile(ckpt_path):
            pass
        else:
            print('%s file not found in drive'%checkpoint)

    print('Downloaded following files\n%s'%'\n'.join(downloaded_files))

    return ckpt_path



if __name__ == '__main__':
    opt = TrainOptions().parse()   # get training options
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    dataset_size = len(dataset)    # get the number of images in the dataset.
    print('The number of training images = %d' % dataset_size)

    save_dir = os.path.join(opt.checkpoints_dir, opt.name)
    if opt.continue_train and opt.use_wandb:
        os.makedirs(save_dir,exist_ok=True)
        if 1:
            try:
                checkpoints = ['latest_net_G_B.pth','latest_net_G_A.pth','latest_net_D_B.pth','latest_net_D_A.pth']
                checkpoint_path = download_checkpoints(opt.pid,checkpoints,save_dir)
                # f = wandb.restore(name)
                # os.rename(f,os.path.join(save_dir,name) )
            except ValueError as e:
                print(e)

    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    visualizer = Visualizer(opt)   # create a visualizer that display/save images and plots
    total_iters = 0                # the total number of training iterations



    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()    # timer for data loading per iteration
        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch
        visualizer.reset()              # reset the visualizer: make sure it saves the results to HTML at least once every epoch
        model.update_learning_rate()    # update learning rates in the beginning of every epoch.
        for i, data in enumerate(dataset):  # inner loop within one epoch
            iter_start_time = time.time()  # timer for computation per iteration
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time

            total_iters += opt.batch_size
            epoch_iter += opt.batch_size
            model.set_input(data)         # unpack data from dataset and apply preprocessing
            model.optimize_parameters()   # calculate loss functions, get gradients, update network weights

            if total_iters % opt.display_freq == 0:   # display images on visdom and save images to a HTML file
                save_result = total_iters % opt.update_html_freq == 0
                model.compute_visuals()
                visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

            if total_iters % opt.print_freq == 0:    # print training losses and save logging information to the disk
                losses = model.get_current_losses()
                t_comp = (time.time() - iter_start_time) / opt.batch_size
                visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)
                if opt.display_id > 0:
                    visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, losses)

            if total_iters % opt.save_latest_freq == 0:   # cache our latest model every <save_latest_freq> iterations
                print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
                model.save_networks(save_suffix)

            iter_data_time = time.time()
        if epoch % opt.save_epoch_freq == 0:              # cache our model every <save_epoch_freq> epochs
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks('latest')
            model.save_networks(epoch)
            # if visualizer.use_wandb:
            #     wandb.save(os.path.join(save_dir,"*.pth") )

            if opt.pid:
                try:
                    checkpoints = glob.glob(os.path.join(save_dir,'latest*.pth'))
                    upload_to_drive(checkpoints,opt.pid)
                except Exception as e:
                    print('error while uploading to drive\n%s'%str(e))

        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))
