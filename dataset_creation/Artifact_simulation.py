import math
import torchio
import numpy as np

from skimage.transform import resize, downscale_local_mean

from scipy.ndimage import gaussian_filter

def Motion_simulation(vol_4d,frame_mix_no):
    
    height,width,no_of_slices,no_of_frames = vol_4d.shape

    frame_list = list(np.arange(0,no_of_frames+1,frame_mix_no))
    collate_indices = []
    for idx in range(len(frame_list)-1):
        collate_indices.append(list(np.arange(frame_list[idx],frame_list[idx+1])))
    
    np_motion_slices = np.zeros((height,width,no_of_slices,len(frame_list))).astype(complex)
    middle_clean_volume = np.zeros((height,width,no_of_slices,len(frame_list)))

    for slice_no in range(no_of_slices):
        
        for idx,mini_frame_list in enumerate(collate_indices):

            one_slice_mini_frames = []
            for frame_no in mini_frame_list:
                one_slice_mini_frames.append(vol_4d[:,:,slice_no,frame_no])

            ksp_deformed = []
            for img in one_slice_mini_frames:
                ksp_deformed.append(np.fft.fftshift(np.fft.fft2(img)))

            ETL = 24
            rows = ksp_deformed[0].shape[0]
            cols = ksp_deformed[0].shape[1]
            period = rows//len(ksp_deformed)
            n_by_etl = math.ceil(rows/ETL)

            ksp_line_seq = np.hstack(([linenum for linenum in range(lineblock,rows, n_by_etl)] for lineblock in range(0,n_by_etl)))
            composite_ksp = np.zeros((rows,cols))*(0+0j)

            # plt.subplots(2,len(ksp_deformed),figsize=(20,20))
            for i in range(0,len(ksp_deformed)):

                composite_ksp[ksp_line_seq[period*i:period*(i+1)],:] = ksp_deformed[i][ksp_line_seq[period*i:period*(i+1)],:]

                # plt.subplot(2,len(ksp_deformed),i+1)
                # plt.imshow(np.log(np.abs(composite_ksp)+1e-8))
                # plt.subplot(2,len(ksp_deformed),len(ksp_deformed)+i+1)
                # plt.imshow(np.abs(np.fft.ifft2(composite_ksp)),cmap = "gray")

            artifact_img = np.fft.ifft2(composite_ksp)

            np_motion_slices[:,:,slice_no,idx] = artifact_img
            middle_clean_volume[:,:,slice_no,idx] = vol_4d[:,:,slice_no,int((np.min(mini_frame_list)+np.max(mini_frame_list))/2)]

    return middle_clean_volume,np_motion_slices


def Spatial_simulation(vol_4d,downsampling_factor,smooth_sigma = 1):
    height,width,no_of_slices,no_of_frames = vol_4d.shape
    
    upsampled_img = np.zeros(vol_4d.shape)
    
    for slice_no in range(no_of_slices):
        for frame_no in range(no_of_frames):

            img_2d = vol_4d[:,:,slice_no,frame_no]

            smooth_img_2d = gaussian_filter(img_2d,sigma = smooth_sigma)

            ds_img = downscale_local_mean(smooth_img_2d,(downsampling_factor,downsampling_factor))

            up_img = resize(ds_img,(height,width),anti_aliasing = True)
            
            upsampled_img[:,:,slice_no,frame_no] = up_img
    
    return upsampled_img


def Undersampling_simulation(img_vol,us_factor,us_mask_path):

    h,w = img_vol.shape[0],img_vol.shape[1]
    
    us_mask = np.load(us_mask_path)
    
    us_img_vol    = np.empty([h,w,0])
    us_kspace_vol = np.empty([h,w,0])

    for sl_no in range(img_vol.shape[-1]):

        img = img_vol[:,:,sl_no]

        kspace     = np.fft.fft2(img,norm='ortho') 
        us_kspace  = kspace * us_mask 
        us_kspace_vol = np.dstack([us_kspace_vol,us_kspace])

        us_img        = np.abs(np.fft.ifft2(us_kspace,norm='ortho'))
        us_img_vol    = np.dstack([us_img_vol,us_img])
            
    return us_img_vol,us_kspace_vol


def Ghosting_simulation(vol_4d,no_of_ghosts,axes = 0,intensity = 0.5,restore = 0):

    ghost_obj = torchio.transforms.RandomGhosting((no_of_ghosts,no_of_ghosts),axes,(intensity,intensity),restore)

    ghosted_vol_4d = ghost_obj(vol_4d)

    return ghosted_vol_4d


def Spike_simulation(vol_4d,no_of_spikes,intensity = 0.11):

    spike_obj = torchio.transforms.RandomSpike((no_of_spikes,no_of_spikes),(intensity,intensity))

    spiked_vol_4d = spike_obj(vol_4d)

    return spiked_vol_4d


def Noise_simulation(vol_4d,std,mean = 0):

    noise_obj = torchio.transforms.RandomNoise((mean,mean),(std,std))

    noised_vol_4d = noise_obj(vol_4d)

    return noised_vol_4d


def Gamma_simulation(vol_4d,log_gamma):

    gamma_obj = torchio.transforms.RandomGamma((log_gamma,log_gamma))

    gamma_vol_4d = gamma_obj(vol_4d)

    return gamma_vol_4d
