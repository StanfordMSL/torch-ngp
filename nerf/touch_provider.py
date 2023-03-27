import os
import cv2
import glob
import json
from cv2 import transform
import tqdm
import numpy as np
from scipy.spatial.transform import Slerp, Rotation

import trimesh

import torch
from torch.utils.data import DataLoader

from .utils import get_rays, srgb_to_linear, torch_vis_2d
import matplotlib.pyplot as plt
import pprint

# ref: https://github.com/NVlabs/instant-ngp/blob/b76004c8cf478880227401ae763be4c02f80b62f/include/neural-graphics-primitives/nerf_loader.h#L50
def nerf_matrix_to_ngp(pose, scale=0.33, offset=[0, 0, 0]):
    # for the fox dataset, 0.33 scales camera radius to ~ 2
    new_pose = np.array([
        [pose[1, 0], -pose[1, 1], -pose[1, 2], pose[1, 3] * scale + offset[0]],
        [pose[2, 0], -pose[2, 1], -pose[2, 2], pose[2, 3] * scale + offset[1]],
        [pose[0, 0], -pose[0, 1], -pose[0, 2], pose[0, 3] * scale + offset[2]],
        [0, 0, 0, 1],
    ], dtype=np.float32)
    return new_pose


def visualize_poses(poses, size=0.1):
    # poses: [B, 4, 4]

    axes = trimesh.creation.axis(axis_length=4)
    box = trimesh.primitives.Box(extents=(2, 2, 2)).as_outline()
    box.colors = np.array([[128, 128, 128]] * len(box.entities))
    objects = [axes, box]

    for pose in poses:
        # a camera is visualized with 8 line segments.
        pos = pose[:3, 3]
        a = pos + size * pose[:3, 0] + size * pose[:3, 1] + size * pose[:3, 2]
        b = pos - size * pose[:3, 0] + size * pose[:3, 1] + size * pose[:3, 2]
        c = pos - size * pose[:3, 0] - size * pose[:3, 1] + size * pose[:3, 2]
        d = pos + size * pose[:3, 0] - size * pose[:3, 1] + size * pose[:3, 2]

        dir = (a + b + c + d) / 4 - pos
        dir = dir / (np.linalg.norm(dir) + 1e-8)
        o = pos + dir * 3

        segs = np.array([[pos, a], [pos, b], [pos, c], [pos, d], [a, b], [b, c], [c, d], [d, a], [pos, o]])
        segs = trimesh.load_path(segs)
        objects.append(segs)
    
    # sphere = trimesh.primitives.Sphere(radius=1, center=(0,0,0))
    # objects.append(sphere)
    trimesh.Scene(objects).show()


def rand_poses(size, device, radius=1, theta_range=[np.pi/3, 2*np.pi/3], phi_range=[0, 2*np.pi]):
    ''' generate random poses from an orbit camera
    Args:
        size: batch size of generated poses.
        device: where to allocate the output.
        radius: camera radius
        theta_range: [min, max], should be in [0, \pi]
        phi_range: [min, max], should be in [0, 2\pi]
    Return:
        poses: [size, 4, 4]
    '''
    
    def normalize(vectors):
        return vectors / (torch.norm(vectors, dim=-1, keepdim=True) + 1e-10)

    thetas = torch.rand(size, device=device) * (theta_range[1] - theta_range[0]) + theta_range[0]
    phis = torch.rand(size, device=device) * (phi_range[1] - phi_range[0]) + phi_range[0]

    centers = torch.stack([
        radius * torch.sin(thetas) * torch.sin(phis),
        radius * torch.cos(thetas),
        radius * torch.sin(thetas) * torch.cos(phis),
    ], dim=-1) # [B, 3]

    # lookat
    forward_vector = - normalize(centers)
    up_vector = torch.FloatTensor([0, -1, 0]).to(device).unsqueeze(0).repeat(size, 1) # confused at the coordinate system...
    right_vector = normalize(torch.cross(forward_vector, up_vector, dim=-1))
    up_vector = normalize(torch.cross(right_vector, forward_vector, dim=-1))

    poses = torch.eye(4, dtype=torch.float, device=device).unsqueeze(0).repeat(size, 1, 1)
    poses[:, :3, :3] = torch.stack((right_vector, up_vector, forward_vector), dim=-1)
    poses[:, :3, 3] = centers

    return poses


class NeRFTouchDataset:
    """
        The format of this data loader is the same as the original data loader. However this is specifically
        for loading Touch data rather than RGB images. The purpose of this code reuse is to make sure
        data paths remain modular and to prevent the original pipeline from becomeing muddled/affected
    """
    def __init__(self, opt, device, type='train', downscale=1, n_test=10):
        super().__init__()
        
        self.opt = opt
        self.device = device
        self.type = type # train, val, test
        self.datatype = 'touch'
        self.downscale = downscale
        self.root_path = opt.path
        self.preload = opt.preload # preload data into GPU
        self.scale = opt.scale # camera radius scale to make sure camera are inside the bounding box.
        self.offset = opt.offset # camera offset
        self.bound = opt.bound # bounding box half length, also used as the radius to random sample poses.
        self.fp16 = opt.fp16 # if preload, load into fp16.
        
        self.training = self.type in ['train', 'all', 'trainval']
        self.num_rays = self.opt.num_rays if self.training else -1

        self.rand_pose = opt.rand_pose
        
        self.use_original = False
        if not self.use_original:
            self.root_path = os.path.join(self.root_path, 'touch')

        # auto-detect transforms.json and split mode.
        if os.path.exists(os.path.join(self.root_path, 'transforms.json')):
            self.mode = 'colmap' # manually split, use view-interpolation for test.
        elif os.path.exists(os.path.join(self.root_path, 'transforms_train.json')):
            self.mode = 'blender' # provided split
        else:
            raise NotImplementedError(f'[NeRFDataset] Cannot find touch/transforms*.json under {self.root_path}')

        # load nerf-compatible format data.
        if self.mode == 'colmap':
            with open(os.path.join(self.root_path, 'transforms.json'), 'r') as f:
                transform = json.load(f)
        elif self.mode == 'blender':
            # load all splits (train/valid/test), this is what instant-ngp in fact does...
            if type == 'all':
                transform_paths = glob.glob(os.path.join(self.root_path, '*.json'))
                transform = None
                for transform_path in transform_paths:
                    with open(transform_path, 'r') as f:
                        tmp_transform = json.load(f)
                        if transform is None:
                            transform = tmp_transform
                        else:
                            transform['frames'].extend(tmp_transform['frames'])
            # load train and val split
            elif type == 'trainval':
                with open(os.path.join(self.root_path, f'transforms_train.json'), 'r') as f:
                    transform = json.load(f)
                with open(os.path.join(self.root_path, f'transforms_val.json'), 'r') as f:
                    transform_val = json.load(f)
                transform['frames'].extend(transform_val['frames'])
            # only load one specified split
            else:
                with open(os.path.join(self.root_path, f'transforms_{type}.json'), 'r') as f:
                    transform = json.load(f)
        else:
            raise NotImplementedError(f'unknown dataset mode: {self.mode}')

        self.config = transform 

        if 'near' in transform:
            self.near =  float(transform['near'])
        else:
            self.near = 1e-6

        if 'far' in transform:
            self.far = float(transform['far'])
        else:
            self.far = 1.e9

        print("FAR IS: ", self.far)
        print("NEAR IS: ", self.near)

        # load image size
        if 'h' in transform and 'w' in transform:
            self.H = int(transform['h']) // downscale
            self.W = int(transform['w']) // downscale
        else:
            # we have to actually read an image to get H and W later.
            self.H = self.W = None
        
        # read images
        #max_depth = transform['camera_max_depth']
        #min_depth = transform['camera_min_depth']
        
        frames = transform["frames"]
        #frames = sorted(frames, key=lambda d: d['file_path']) # why do I sort...
        
        # for colmap, manually interpolate a test set.
        if self.mode == 'colmap' and type == 'test':
            
            # choose two random poses, and interpolate between.
            f0, f1 = np.random.choice(frames, 2, replace=False)
            pose0 = nerf_matrix_to_ngp(np.array(f0['transform_matrix'], dtype=np.float32), scale=self.scale, offset=self.offset) # [4, 4]
            pose1 = nerf_matrix_to_ngp(np.array(f1['transform_matrix'], dtype=np.float32), scale=self.scale, offset=self.offset) # [4, 4]
            rots = Rotation.from_matrix(np.stack([pose0[:3, :3], pose1[:3, :3]]))
            slerp = Slerp([0, 1], rots)

            self.poses = []
            self.images = None
            for i in range(n_test + 1):
                ratio = np.sin(((i / n_test) - 0.5) * np.pi) * 0.5 + 0.5
                pose = np.eye(4, dtype=np.float32)
                pose[:3, :3] = slerp(ratio).as_matrix()
                pose[:3, 3] = (1 - ratio) * pose0[:3, 3] + ratio * pose1[:3, 3]
                self.poses.append(pose)

        else:
            # for colmap, manually split a valid set (the first frame).
            if self.mode == 'colmap':
                if type == 'train':
                    frames = frames[1:]
                elif type == 'val':
                    frames = frames[:1]
                # else 'all' or 'trainval' : use all frames
            
            self.poses = []
            self.images = []
            self.near = []
            self.far = []
            self.H = []
            self.W = []

            for f in tqdm.tqdm(frames, desc=f'Loading {type} data'):
                f_path = os.path.join(self.root_path, f['file_path']+'.png')
                #print(f_path)
                #f_path = os.path.join('.',f['file_path'])

                # there are non-exist paths in fox...
                if not os.path.exists(f_path):
                    continue
                
                pose = np.array(f['transform_matrix'], dtype=np.float32) # [4, 4]
                pose = nerf_matrix_to_ngp(pose, scale=self.scale, offset=self.offset)

                image = cv2.imread(f_path, cv2.IMREAD_UNCHANGED) # [H, W, 3] o [H, W, 4]
                
                # check if we have multiple cameras in use
                if "cameras" in transform:

                    # check if near plane for specified camera was provided. If not use default value close to zero
                    if "near" in transform["cameras"][f['camera']]:
                        self.near.append(float(transform["cameras"][f['camera']]["near"]))
                    else:
                        self.near.append(1e-6)

                    # check if far plane for specified camera was provided. If not use large default value
                    if "far" in transform["cameras"][f['camera']]:
                        self.far.append(float(transform["cameras"][f['camera']]["far"]))
                    else:
                        self.far.append(1.e9)


                    # check if image dimensions were provided for specified camera. if not use image dimensions
                    if "H" in transform["cameras"][f['camera']] and "W" in transform["cameras"][f['camera']]:
                        self.H.append(transform[f['camera']]["H"] // downscale)
                        self.W.append(transform[f['camera']]["W"] // downscale)
                    else:
                        self.H.append(image.shape[0] // downscale)
                        self.W.append(image.shape[1] // downscale)

                # only one camera in use
                else:

                    # check if near plane was provided. If not use default value close to zero
                    if "near" in transform:
                        self.near.append(float(transform["near"]))
                    else:
                        self.near.append(1e-6)

                    # check if far plane was provided. If not use large default value
                    if "far" in transform:
                        self.far.append(float(transform["far"]))
                    else:
                        self.far.append(1.e9)

                    # check if image dimensions was provided for camera if not use image parameters
                    if "H" in transform and "W" in transform:
                        self.H.append(transform["H"] // downscale)
                        self.W.append(transform["W"] // downscale)

                    else: #self.H is None or self.W is None:
                        self.H.append(image.shape[0] // downscale)
                        self.W.append(image.shape[1] // downscale)

                # check if image matches with expected dimensions, if not interpolate to make it fit
                if image.shape[0] != self.H[-1] or image.shape[1] != self.W[-1]:
                    image = cv2.resize(image, (self.W[-1], self.H[-1]), interpolation=cv2.INTER_AREA)

                # convert images into raw distance based on camera intrinics
                image = image.astype(np.float32) / (255.)*(self.far[-1] - self.near[-1]) + self.near[-1] # [H, W, 3/4]
                image = torch.unsqueeze(torch.from_numpy(image),dim=-1)

                self.poses.append(pose)
                self.images.append(image)

            
        self.H = np.asarray(self.H)
        self.W = np.asarray(self.W)
        self.near = np.asarray(self.near)
        self.far = np.asarray(self.far)
        self.poses = torch.from_numpy(np.stack(self.poses, axis=0)) # [N, 4, 4]


        # calculate mean radius of all camera poses
        self.radius = self.poses[:, :3, 3].norm(dim=-1).mean(0).item()
        #print(f'[INFO] dataset camera poses: radius = {self.radius:.4f}, bound = {self.bound}')

        # initialize error_map
        if self.training and self.opt.error_map:
            self.error_map = torch.ones([self.images.shape[0], 128 * 128], dtype=torch.float) # [B, 128 * 128], flattened for easy indexing, fixed resolution...
        else:
            self.error_map = None

        # [debug] uncomment to view all training poses.
        #visualize_poses(self.poses.numpy())

        # [debug] uncomment to view examples of randomly generated poses.
        #visualize_poses(rand_poses(100, self.device, radius=self.radius).cpu().numpy())

        if self.preload:
            self.poses = self.poses.to(self.device)
            if self.images is not None:
                # TODO: linear use pow, but pow for half is only available for torch >= 1.10 ?
                if self.fp16 and self.opt.color_space != 'linear':
                    dtype = torch.half
                else:
                    dtype = torch.float
                self.images = self.images.to(dtype).to(self.device)
            if self.error_map is not None:
                self.error_map = self.error_map.to(self.device)

        # load intrinsics (use equisolid model)
        if 'cameras' in transform:
            self.intrinsics = []
            for i, f in enumerate(transform["frames"]):
                camera = transform["cameras"][f["camera"]] 
                
                # if focal length was provided
                if 'fl_x' in camera or 'fl_y' in camera:
                    fl_x = (camera['fl_x'] if 'fl_x' in camera else camera['fl_y']) / downscale
                    fl_y = (camera['fl_y'] if 'fl_y' in camera else camera['fl_x']) / downscale

                    #NOTE: Note sensor size is the region of a digital camera that’s sensitive to light 
                    #      and records an image when active. cite: https://capturetheatlas.com/camera-sensor-size/
                    #      this is critical for fisheye cameras, but not really relevant for perspective cameras!!
                    sensor_size = camera['sensor_size'] if 'sensor_size' in camera else None

                # if field of view was given instead, compute focal length
                elif 'camera_angle_x' in camera or 'camera_angle_y' in camera:
                    
                    # if x axis fov was given
                    if 'camera_angle_x' in camera:
                        if not isinstance(camera['camera_angle_x'], list):
                            angle_x = camera['camera_angle_x']
                        else:
                            angle_x = camera['camera_angle_x'][0]

    
                        #NOTE: Note sensor size is the region of a digital camera that’s sensitive to light
                        #      and records an image when active. cite: https://capturetheatlas.com/camera-sensor-size/
                        #      this is critical for fisheye cameras, but not really relevant for perspective cameras!!
                        sensor_size = camera['sensor_size'] if 'sensor_size' in camera else None
                    
                        if sensor_size is None:
                            fl_x = None
                        else:
                            fl_x = sensor_size/(4*np.sin(angle_x/4))
                    else:
                        fl_x = None
                    
                    if 'camera_angle_y' in camera:
                        if not isinstance(camera['camera_angle_y'], list):
                            angle_y = camera['camera_angle_y']
                        else:
                            angle_y = camera['camera_angle_y'][0]

                        
                        #NOTE: Note sensor size is the region of a digital camera that’s sensitive to light
                        #      and records an image when active. cite: https://capturetheatlas.com/camera-sensor-size/
                        #      this is critical for fisheye cameras, but not really relevant for perspective cameras!!
                        sensor_size = camera['sensor_size'] if 'sensor_size' in camera else None

                        if sensor_size is None:
                            fl_y = None
                        else:
                            fl_y = sensor_size/(4*np.sin(angle_x/4))
                            #fl_y = camera['focal_length_y']  #8.838834762573242 #self.H[i] / (4*np.sin(angle_y/4))

                    else:
                        fl_y = None
                    
                    if fl_x is None and fl_y is not None: fl_x = fl_y
                    if fl_y is None and fl_x is not None: fl_y = fl_x

                    if sensor_size is None:
                        raise RuntimeError('Failed to load sensor size, please check the transforms.json!')

                else:
                    raise RuntimeError('Failed to load focal length, please check the transforms.json!')
                    
                cx = (transform["cameras"][f["camera"]]['cx'] / downscale) if 'cx' in transform["cameras"][f["camera"]] else (self.W[i] / 2)
                cy = (transform["cameras"][f["camera"]]['cy'] / downscale) if 'cy' in transform["cameras"][f["camera"]] else (self.H[i] / 2)

                self.intrinsics.append(np.array([fl_x, fl_y, cx, cy, sensor_size]))

            self.intrinsics = np.asarray(self.intrinsics)
        else:
            if 'fl_x' in transform or 'fl_y' in transform:
                fl_x = (transform['fl_x'] if 'fl_x' in transform else transform['fl_y']) / downscale
                fl_y = (transform['fl_y'] if 'fl_y' in transform else transform['fl_x']) / downscale
                    
                #NOTE: Note sensor size is the region of a digital camera that’s sensitive to light 
                #      and records an image when active. cite: https://capturetheatlas.com/camera-sensor-size/
                #      this is critical for fisheye cameras, but not really relevant for perspective cameras!!
                sensor_size = camera['sensor_size'] if 'sensor_size' in camera else None

            elif 'camera_angle_x' in transform or 'camera_angle_y' in transform:
                angle_x = transform['camera_angle_x'] if 'camera_angle_x' in transform else None
                angle_y = transform['camera_angle_y'] if 'camera_angle_y' in transform else None

                #NOTE: Note sensor size is the region of a digital camera that’s sensitive to light 
                #      and records an image when active. cite: https://capturetheatlas.com/camera-sensor-size/
                #      this is critical for fisheye cameras, but not really relevant for perspective cameras!!
                sensor_size = camera['sensor_size'] if 'sensor_size' in camera else None

                fl_x = sensor_size/(4*np.sin(angle_x/4)) if sensor_size is not None and angle_x is not None else None
                fl_y = sensor_size/(4*np.sin(angle_y/4)) if sensor_size is not None and angle_y is not None else None

                if fl_x is None and fl_y is not None: fl_x = fl_y
                if fl_y is None and fl_x is not None: fl_y = fl_x

                if sensor_size is None:
                    raise RuntimeError('Failed to load sensor size, please check the transforms.json!')
            
            else:
                raise RuntimeError('Failed to load focal length, please check the transforms.json!')
    
            cx = (transform['cx'] / downscale) if 'cx' in transform else (self.W / 2)
            cy = (transform['cy'] / downscale) if 'cy' in transform else (self.H / 2)

            self.intrinsics = np.tile(np.array([fl_x, fl_y, cx, cy, sensor_size]),(self.images.size[0],1))

        print("intrinics")
        print("W: ", self.W.shape)
        print("H: ", self.H.shape)
        print("far: ", self.far.shape)
        print(self.far)
        print("near: ", self.near.shape)
        print(self.near)
        print("intrinics: ", self.intrinsics.shape)
        print(self.intrinsics)




    def collate(self, index):
        B = len(index) # a list of length 1
        
        index = index[0] #TODO: handle multiple images in a batch!
    
        # random pose without gt images.
        if self.rand_pose == 0 or index >= len(self.poses):

            poses = rand_poses(B, self.device, radius=self.radius)

            # sample a low-resolution but full image for CLIP
            s = np.sqrt(float(self.H[index]) * float(self.W[index]) / self.num_rays) # only in training, assert num_rays > 0
            rH, rW = int(float(self.H[index]) / s), int(float(self.W[index]) / s)
            rays = get_rays(poses, self.intrinsics[index] / s, rH, rW, -1, camera_model='touch')

            return {
                'type': 'touch',
                'H': rH,
                'W': rW,
                'rays_o': rays['rays_o'],
                'rays_d': rays['rays_d'],
                #'mask': rays['mask'],
                'near': float(self.near[index]),
                'far': float(self.far[index])
            }

        poses = self.poses[index].to(self.device) # [B, 4, 4]

        error_map = None if self.error_map is None else self.error_map[index]
        
        rays = get_rays(torch.unsqueeze(poses,0), self.intrinsics[index], int(self.H[index]), int(self.W[index]), self.num_rays, error_map, 'touch')
        #rays = get_rays(torch.unsqueeze(poses,0), self.intrinsics[index], int(self.H[index]), int(self.W[index]), -1, error_map, 'touch')
        
        #print(rays)
        #plt.imshow(self.images[index][:,:,0])
        #plt.show()

        #rays = get_rays(torch.unsqueeze(poses,0), self.intrinsics[index], int(self.H[index]), int(self.W[index]), -1, error_map, 'touch')
        #image = torch.gather(self.images[index].to(self.device).view(B, -1, 1), 1, torch.stack(1 * [rays['inds']], -1))
        #image = self.images[index].to(self.device).view(B, -1, 1)
        #print(image.shape)
        
        #print(image)
        #print(torch.max(image))
        #print(torch.min(image))
        #print(torch.mode(image))
        #print(torch.mean(image))
        #print(rays['mask'].shape)
        #print(torch.unsqueeze(rays['mask'],-1).shape)
        #print(rays['rays_d'].shape)

        #plt.imshow(rays['mask'].reshape(int(self.W[index]),int(self.H[index])).detach().cpu().numpy())
        #plt.show()


        #d = ((image*rays['rays_d'] + rays['rays_o']))#*torch.unsqueeze(rays['mask'],-1)
        #print(d)
        #d = torch.squeeze(d)
        
        #fig = plt.figure()
        #ax = fig.add_subplot(projection='3d')
        #ax.scatter(d[:,0].detach().cpu().numpy(), d[:,1].detach().cpu().numpy(), d[:,2].detach().cpu().numpy())
        #ax.set_xlabel('X Label')
        #ax.set_ylabel('Y Label')
        #ax.set_zlabel('Z Label')
        #plt.show()
        #stop

        results = {
            'type': 'touch',
            'H': int(self.H[index]),
            'W': int(self.W[index]),
            'rays_o': rays['rays_o'],
            'rays_d': rays['rays_d'],
            #'mask': rays['mask'],
            'near': float(self.near[index]),
            'far': float(self.far[index])
        }

        if self.images is not None:
            images = self.images[index].to(self.device) # [B, H, W, 3/4]
            if self.training:
                C = images.shape[-1]
                images = torch.gather(images.view(B, -1, C), 1, torch.stack(C * [rays['inds']], -1)) # [B, N, 3/4]
            results['images'] = images
        
        # need inds to update error_map
        if error_map is not None:
            results['index'] = index
            results['inds_coarse'] = rays['inds_coarse']
            
        return results

    def dataloader(self):
        size = len(self.poses)
        if self.training and self.rand_pose > 0:
            size += size // self.rand_pose # index >= size means we use random pose.
        loader = DataLoader(list(range(size)), batch_size=1, collate_fn=self.collate, shuffle=self.training, num_workers=0)
        loader._data = self # an ugly fix... we need to access error_map & poses in trainer.
        loader.has_gt = self.images is not None
        
        return loader
