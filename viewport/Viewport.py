# Author: jayasingam adhuran

import os
import sys
import logging
import numpy as np
import torch.utils.data
import scipy.ndimage.interpolation as interp
import skimage.transform

from tqdm import tqdm, trange
import cv2
from pathlib import Path
from contextlib import suppress
import math
import torch
import numpy as np
from kmeans_pytorch import kmeans
import torch.nn.functional as F
import torchvision
#import copy
#import types
from sklearn.cluster import KMeans
#import matplotlib.pyplot as plt

curr_path = os.path.dirname(os.path.abspath(__file__)) 
if curr_path not in sys.path:
    sys.path.insert(0, curr_path)

import yuv
from models2 import official_salientnet


number_of_outputs = 16
device = torch.device("cpu")
from s2cnn import s2_equatorial_grid, S2Convolution, so3_equatorial_grid, SO3Convolution, so3_integrate, s2_near_identity_grid, so3_near_identity_grid

model_dir=os.path.join(curr_path,'model')
log_dir=curr_path

class DownSample:
    def __init__(self, down_resolution):
        self.down_resolution = down_resolution

    def __call__(self, Y, U, V):
        half_resolution = [i / 2 for i in self.down_resolution]
        Y_d = skimage.transform.resize(Y.transpose((1, 2, 0)), self.down_resolution, order=1, anti_aliasing=True,
                                       mode='reflect', preserve_range=True)
        U_d = skimage.transform.resize(U.transpose((1, 2, 0)), half_resolution, order=1, anti_aliasing=True,
                                       mode='reflect', preserve_range=True)
        V_d = skimage.transform.resize(V.transpose((1, 2, 0)), half_resolution, order=1, anti_aliasing=True,
                                       mode='reflect', preserve_range=True)

        return Y_d.transpose((2, 0, 1)).round().astype(np.uint8), U_d.transpose((2, 0, 1)).round().astype(np.uint8), \
               V_d.transpose((2, 0, 1)).round().astype(np.uint8)

    def __repr__(self):
        return self.__class__.__name__ + '(down_resolution={0})'.format(self.down_resolution)


class SampleSGrid:
    def __init__(self, bandwidth):
        self.bandwidth = bandwidth
        self.euler_grid, _ = self.make_sgrid(bandwidth)

    def __call__(self, Y, U, V, resolution):
        height, width = resolution
        height_half, width_half = height // 2, width // 2
        theta, phi = self.euler_grid

        pix_height = theta[:, 0] / np.pi * height
        pix_width = phi[0, :] / (np.pi * 2) * width
        pix_width, pix_height = np.meshgrid(pix_width, pix_height)
        pix_height_half = theta[:, 0] / np.pi * height_half
        pix_width_half = phi[0, :] / (np.pi * 2) * width_half
        pix_width_half, pix_height_half = np.meshgrid(pix_width_half, pix_height_half)

        Y_im = interp.map_coordinates(Y[0, ...], [pix_height, pix_width], order=1)
        U_im = interp.map_coordinates(U[0, ...], [pix_height_half, pix_width_half], order=1)
        V_im = interp.map_coordinates(V[0, ...], [pix_height_half, pix_width_half], order=1)

        return Y_im[np.newaxis, ...], U_im[np.newaxis, ...], V_im[np.newaxis, ...]

    @staticmethod
    def make_sgrid(b):
        from lie_learn.spaces import S2

        theta, phi = S2.meshgrid(b=b, grid_type='SOFT')
        sgrid = S2.change_coordinates(np.c_[theta[..., None], phi[..., None]], p_from='S', p_to='C')
        sgrid = sgrid.reshape((-1, 3))

        return (theta, phi), sgrid

    def __repr__(self):
        return self.__class__.__name__ + '(bandwidth={0})'.format(self.bandwidth)


class VQA_ODV_Transform:
    def __init__(self, bandwidth, down_resolution, to_rgb=True):
        self.to_rgb = to_rgb
        self.sgrid_transform = SampleSGrid(bandwidth)
        self.down_transform = DownSample(down_resolution)

    def __call__(self, file_path, resolution, frame_index):
        ori = yuv.yuv_import(file_path, resolution, 1, frame_index)
        im = self.sgrid_transform(*ori, resolution)
        down = self.down_transform(*ori)
        if self.to_rgb:
            im = yuv.yuv2rgb(*im)
            ori = yuv.yuv2rgb(*ori)
            down = yuv.yuv2rgb(*down)
        return im[0], ori[0], down[0]


class DS_VQA_ODV(torch.utils.data.Dataset):

    def __init__(self, root, sequence_data, transform, flow_gap=0, test_start_frame=21,
                 test_interval=45):
        """
        VQA-ODV initialization.
        Function adopted from VQA-ODV
        @inproceedings{Li:2018:BGV:3240508.3240581,
         author = {Li, Chen and Xu, Mai and Du, Xinzhe and Wang, Zulin},
         title = {Bridge the Gap Between VQA and Human Behavior on Omnidirectional Video: A Large-Scale Dataset and a Deep Learning Model},
         booktitle = {Proceedings of the 26th ACM International Conference on Multimedia},
         series = {MM '18},
         year = {2018},
         isbn = {978-1-4503-5665-7},
         location = {Seoul, Republic of Korea},
         pages = {932--940},
         numpages = {9},
         url = {http://doi.acm.org/10.1145/3240508.3240581},
         doi = {10.1145/3240508.3240581},
         acmid = {3240581},
         publisher = {ACM},
         address = {New York, NY, USA},
         keywords = {human behavior, omnidirectional video, visual quality assessment},
        """

        self.logger = logging.getLogger("{}.dataset".format('Viewport Prediction Started'))
        self.root = os.path.expanduser(root)


        assert isinstance(transform, VQA_ODV_Transform)
        self.transform = transform
        self.flow_gap = flow_gap
        self.test_start_frame = test_start_frame
        self.test_interval = test_interval
        if self.test_start_frame < self.flow_gap:
            warnings.warn("The value of test_start_frame should not be less than flow_gap. Set test_start_frame equal to flow_gap.",
                Warning)
            self.test_start_frame = self.flow_gap
        if self.test_interval < 1:
            warnings.warn("The value of test_interval should not be less than 1. Set test_interval equal to 1.",
                Warning)
            self.test_start_frame = 1

        self.scenes = list(range(60))
               
        self.data_dict = {'video_file': sequence_data['src'], 'resolution': sequence_data['res'], 'frame_count': sequence_data['frame_count']}
        self.frame_num_list = np.array(self.data_dict['frame_count'], dtype=int)
        self.frame_num_list = np.ceil((self.frame_num_list - self.test_start_frame) / self.test_interval).astype(int)
        self.cum_frame_num = np.cumsum(self.frame_num_list)
        self.cum_frame_num_prev = np.zeros_like(self.cum_frame_num)
        self.cum_frame_num_prev[1:] = self.cum_frame_num[:-1]
       

    def __getitem__(self, index):
        
        
        video_path = self.data_dict['video_file']
        resolution = self.data_dict['resolution']

        video_index = np.searchsorted(self.cum_frame_num_prev, index, side='right') - 1
        frame_index = (index - self.cum_frame_num_prev[video_index]) * self.test_interval + self.test_start_frame
       
        img, img_original, img_down = self.transform(file_path=video_path, resolution=resolution,
                                                     frame_index=frame_index)
        if ((frame_index+1) >= self.frame_num_list):
            img_t1, gap_down, img_gap = self.transform(file_path=video_path, resolution=resolution, frame_index=frame_index)
        else:
            img_t1, gap_down, img_gap = self.transform(file_path=video_path, resolution=resolution, frame_index=frame_index+1)
        
        
        self.logger.debug('[DATA] {}, FRAME:{}'.format(video_path, frame_index))        

        return img.astype(np.float32), img_original.astype(np.float32), img_down.astype(np.float32), \
               img_gap.astype(np.float32), gap_down.astype(np.float32), img_t1.astype(np.float32), \
               video_index, frame_index


    def __len__(self):
        return self.cum_frame_num[-1]


class Model_test4(torch.nn.Module):
    def __init__(self, nclasses= 40):
        super().__init__()

        self.features = [1,  100, 100, nclasses]
        self.bandwidths = [512, 16, 10]

        assert len(self.bandwidths) == len(self.features) - 1

        sequence = []

        # S2 layer
        grid = s2_equatorial_grid(max_beta=0, n_alpha=2 * self.bandwidths[0], n_beta=1)
        sequence.append(S2Convolution(self.features[0], self.features[1], self.bandwidths[0], self.bandwidths[1], grid))

        # SO3 layers
        for l in range(1, len(self.features) - 2):
            nfeature_in = self.features[l]
            nfeature_out = self.features[l + 1]
            b_in = self.bandwidths[l]
            b_out = self.bandwidths[l + 1]

            sequence.append(torch.nn.BatchNorm3d(nfeature_in, affine=True))
            sequence.append(torch.nn.ReLU())
            grid = so3_equatorial_grid(max_beta=0, max_gamma=0, n_alpha=2 * b_in, n_beta=1, n_gamma=1)
            sequence.append(SO3Convolution(nfeature_in, nfeature_out, b_in, b_out, grid))

        sequence.append(torch.nn.BatchNorm3d(self.features[-2], affine=True))
        sequence.append(torch.nn.ReLU())

        self.sequential = torch.nn.Sequential(*sequence)
        self.sequential2 = official_salientnet.SalientResNet152(num_classes=self.features[-2])
        # Output layer
        output_features = 2*self.features[-2]
        self.out_layer1 = torch.nn.Linear(output_features, self.features[-2])
        self.out_layer2 = torch.nn.Linear(self.features[-2], self.features[-1])

    def forward(self, x):  # pylint: disable=W0221
        x1 = self.sequential(x)  # [batch, feature, beta, alpha, gamma]
        x1 = so3_integrate(x1)  # [batch, feature]
        #print(x1.shape)
        x2 = self.sequential2(x)
        #x2 = so3_integrate(x2)
        x2 = torch.cat((x1,x2),dim=0)
        x2 = torch.reshape(x2, (1, 2*self.features[-2]))
        #print(2*self.features[-2])
        #print(x2.shape)

        x2 = self.out_layer1(x2)
        x = F.dropout(x, training=self.training)
        out_coordinates = self.out_layer2(x2)
        return out_coordinates


def get_latest_ckpt(path, reverse=False, suffix='.ckpt'):
    """Load latest checkpoint from target directory. Return None if no checkpoints are found."""
    path, file = Path(path), None
    files = (f for f in sorted(path.iterdir(), reverse=not reverse) if f.suffix == suffix)
    with suppress(StopIteration):
        file = next(f for f in files)
    return file


def get_label_files(label_path, number_of_subjects, Group_number, gound_truth_data_file):
        label_files = []
        Group_number_test = str(Group_number)
        for x in range(number_of_subjects):
            subject_number = str(x+1)
            seq = "Group" + Group_number_test + "/G" + Group_number_test + "_" + subject_number
            label_file_test = os.path.join(label_path, seq, gound_truth_data_file)
            label_files.append(label_file_test)

        return label_files
 
def get_head_movement_data(label_path, group_num_list, sequence_number_list, number_of_videos, video_index, frame_index, gound_truth_data_file):

    #subjects = [21, 21, 21, 21, 22, 21, 22, 20, 21, 22]
    subjects = [20, 20, 20, 20, 20, 20, 20, 20, 20, 20]

    Group_number = group_num_list[0][video_index].item()
    ground_truth_data = gound_truth_data_file[video_index][0]
    number_of_subjects = subjects[Group_number-1]

    label_files = get_label_files(label_path, number_of_subjects, Group_number, ground_truth_data)

    Lat1, Lon1, Lat2, Lon2 = [], [], [], []
    i = 0
    label_file_count = 0

    lines = [frame_index, frame_index+1]
    
    for label_file in label_files:
                    
        with open(label_file, 'r') as listFile:
            
            line_count1 = 0
            line_count2 = 0
            for position, line in enumerate(listFile):

                if len(line.split()) == 1:
                    break
                if position==2*frame_index:
                    _, _, Lat, Lon, _, _, _ = line.split()
                    Lat = float(Lat)
                    Lon = float(Lon)
                    subject = float(label_file_count)
                    Lat1.append(Lat)
                    Lon1.append(Lon)
                    line_count1+=1
                       
                elif position==1+2*frame_index:
                    _, _, Lat, Lon, _, _, _ = line.split()
                    Lat = float(Lat)
                    Lon = float(Lon)
                    
                    Lat2.append(Lat)
                    Lon2.append(Lon)
                    line_count2+=1
                        

        label_file_count +=1
            

    #print(Lat1[0:line_count1])
    Subject_Lat1 = np.empty(shape=(label_file_count, line_count1), dtype=float)
    Subject_Lon1 = np.empty(shape=(label_file_count, line_count1), dtype=float)
    Subject_Lat2 = np.empty(shape=(label_file_count, line_count1), dtype=float)
    Subject_Lon2 = np.empty(shape=(label_file_count, line_count1), dtype=float)

    for x in range(label_file_count):
        Subject_Lat1[x] = np.array([Lat1[x*line_count1:x*line_count1+line_count1]])
        Subject_Lon1[x] = np.array([Lon1[x*line_count1:x*line_count1+line_count1]])
        Subject_Lat2[x] = np.array([Lat2[x*line_count1:x*line_count1+line_count1]])
        Subject_Lon2[x] = np.array([Lon2[x*line_count1:x*line_count1+line_count1]])
        
    return Subject_Lat1, Subject_Lon1, Subject_Lat2, Subject_Lon2, number_of_subjects

def myFunc(e):
  return e['Value']



def test2(model_dir, test_set, log_dir, batch_size, num_workers, number_of_clusters, test_net_number=0):
    
    torch.backends.cudnn.benchmark = True

    model = Model_test4(number_of_clusters)
    
   
    model=model.to(device)#.cuda()
    ckpt_file = get_latest_ckpt(model_dir)
    state_file  = torch.load(ckpt_file, map_location = torch.device('cpu'))
    model.load_state_dict(state_file['model_state_dict'])
    #model.load_state_dict(torch.load(os.path.join(log_dir, "state.ckpt"))) 

    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, drop_last=False)

    predictions = []
    ids = []  
    
    def test_step(data, data2, batch_index, number_of_clusters, video_index, frame_index):
        model.eval()
        data = data.to(device)      
        batch_size, rep = data.size()[:2]

        with torch.no_grad():
            
            if test_net_number == 2:
                pred = model(data.float(), data2.float()).to(device)
            else:
                pred = model(data.float()).to(device)    

        my_predictions= []       
        
        for i in range(0,number_of_clusters,2):
                        
            test_distance = 0
            lat_dir = (pred[0][i].item()//90) / np.abs(pred[0][i].item()//90)
            lon_dir = (pred[0][i+1].item()//180) / np.abs(pred[0][i+1].item()//180)
            
            test_coordinates = [lat_dir*(pred[0][i].item()%90), lon_dir*(pred[0][i+1].item()%180)]
            my_predictions.append(test_coordinates)
        
        X_ = torch.from_numpy(np.array(my_predictions))
        kmeans_test = KMeans(n_clusters=int(number_of_outputs/2), init='k-means++', max_iter=300, n_init=10, random_state=0).fit(X_)
        
        batch_index = torch.tensor([[batch_index]])
        pred = pred.cpu()
        
        video_ind = video_index.clone().detach()#torch.tensor(video_index)
        frame_ind = frame_index.clone().detach()#torch.tensor(frame_index)
        pred3 = torch.tensor(kmeans_test.cluster_centers_)
        pred3 = np.reshape(pred3, int(number_of_outputs))
        pred2 = {'vid_ind':video_ind, 'frame_ind':frame_ind, 'Predicitions': pred3[:]}#torch.cat([video_ind, frame_ind, pred3[:]])
        #print("Predictions: ", pred2)
        
        predictions.append(pred2)

    img_frame0_=0        
    for batch_idx, data in enumerate(test_loader):
        img_s2, img_original, img_down, img_gap_s2, gap_down, img_t1, video_index, frame_index = data
        gap_down = gap_down.to(device)
        img_down = img_down.to(device)
        gap_down = gap_down.view((-1, *gap_down.shape[-3:]))
        img_down = img_down.view((-1, *img_down.shape[-3:]))
        img_s2 = img_s2.view((-1, *img_s2.shape[-3:]))
        img_s2 = img_s2.to(device)
        img_gap_s2 = img_gap_s2.view((-1, *img_gap_s2.shape[-3:]))
        img_gap_s2 = img_gap_s2.to(device)
        img_t1 =img_t1.view((-1, *img_t1.shape[-3:]))
        img_t1 = img_t1.to(device)

        test_step(img_s2, img_t1, batch_idx, number_of_clusters, video_index, frame_index)
 
    return predictions    



def cluster_generation():
    target_lats =[]
    target_lons =[]
    target_coord = []
    for y in range(number_of_subjects):
        Lat_local_loop = target1[0][0][y].item()+target2[0][0][y].item()
        Lat_local_loop = Lat_local_loop/2

        Lon_local_loop = target1[0][1][y].item()+target2[0][1][y].item()
        Lon_local_loop = Lon_local_loop/2


        target_coord.append(Lat_local_loop)
        target_coord.append(Lon_local_loop)
        #target_coord.append(target1[0][0][y].item())
        #target_coord.append(target2[0][0][y].item())
        #target_coord.append(target1[0][1][y].item())
        #target_coord.append(target2[0][1][y].item())
        
        
    target_labels = np.array(target_coord[0:2*number_of_subjects])
    print(target_labels)
    x = torch.from_numpy(np.transpose(np.array([target_labels])))
    #target_cluster_ids_x, target_cluster_centers = kmeans(x, 4, distance='euclidean', device=torch.device('cuda:0'))

    kmeans_test = KMeans(n_clusters=number_of_clusters, init='k-means++', max_iter=300, n_init=10, random_state=0)
    cluster_id = kmeans_test.fit_predict(x) 

    target_clusters1 =[]
    target_clusters2 =[]
        
    prev_clusterid = []

    for y in range(2*number_of_subjects):

        if y==0:
            target_clusters1.append(kmeans_test.cluster_centers_[cluster_id[y]])
                
            prev_clusterid.append(cluster_id[y])

        if cluster_id[y] not in prev_clusterid:
                target_clusters1.append(kmeans_test.cluster_centers_[cluster_id[y]])
                 
                prev_clusterid.append(cluster_id[y])
                 
          
    labels = np.asarray(target_clusters1)


def viewport_prediction_tester(sequence_data, frame_interval = 1):
    #model_dir = "C:/Users/jayas/Documents/Viewport"
    number_of_clusters = 40
    #inter 1 #intra 8
    #log_dir = "C:/Users/jayas/Documents/Viewport/Viewport"
    test_net_number = 3
    batch_size = 1
    #sequence_data = {'src':'C:/Users/jayas/Documents/Viewport/Viewport/test_data/test_viewport.yuv', 'res':[7680, 3840], 'frame_count':60}

    test_set = DS_VQA_ODV(root=os.path.join(log_dir, "VQA_ODV"), sequence_data=sequence_data, transform=VQA_ODV_Transform(bandwidth=512, down_resolution=(720, 1280), to_rgb=False), test_interval=frame_interval, test_start_frame=0,)
    predictions = test2(model_dir, test_set, log_dir, batch_size, 0, number_of_clusters, test_net_number)
    return predictions
    



