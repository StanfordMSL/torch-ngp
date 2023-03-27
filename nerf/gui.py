import math
import torch
import numpy as np
import dearpygui.dearpygui as dpg
from scipy.spatial.transform import Rotation as R

from .utils import *


class OrbitCamera:
    def __init__(self, W, H, r=2, fovy=60):
        self.W = W
        self.H = H
        self.near = -1*np.inf                                 # camera near plane (in meters)
        self.far = np.inf                                   # camera far plane (in meters)
        self.radius = r                                     # camera distance from center
        self.fovy = fovy                                    # in degree
        self.center = np.array([0, 0, 0], dtype=np.float32) # look at this point
        self.rot = R.from_quat([1, 0, 0, 0])                # init camera matrix: [[1, 0, 0], [0, -1, 0], [0, 0, 1]] (to suit ngp convention)
        self.up = np.array([0, 1, 0], dtype=np.float32)     # need to be normalized!

    # pose
    @property
    def pose(self):
        # first move camera to radius
        res = np.eye(4, dtype=np.float32)
        res[2, 3] -= self.radius
        # rotate
        rot = np.eye(4, dtype=np.float32)
        rot[:3, :3] = self.rot.as_matrix()
        res = rot @ res
        # translate
        res[:3, 3] -= self.center
        return res
    
    # intrinsics
    @property
    def intrinsics(self):
        focal = self.H / (2 * np.tan(np.radians(self.fovy) / 2))
        
        #                fl_x   fl_y    cx          cy          sensor_size`
        return np.array([focal, focal, self.W // 2, self.H // 2, self.H])
    
    def orbit(self, dx, dy):
        # rotate along camera up/side axis!
        side = self.rot.as_matrix()[:3, 0] # why this is side --> ? # already normalized.
        rotvec_x = self.up * np.radians(-0.1 * dx)
        rotvec_y = side * np.radians(-0.1 * dy)
        self.rot = R.from_rotvec(rotvec_x) * R.from_rotvec(rotvec_y) * self.rot

        # wrong: rotate along global x/y axis
        #self.rot = R.from_euler('xy', [-dy * 0.1, -dx * 0.1], degrees=True) * self.rot
    
    def scale(self, delta):
        self.radius *= 1.1 ** (-delta)

    def pan(self, dx, dy, dz=0):
        # pan in camera coordinate system (careful on the sensitivity!)
        self.center += 0.001 * self.rot.as_matrix()[:3, :3] @ np.array([dx, dy, dz])

        # wrong: pan in global coordinate system
        #self.center += 0.001 * np.array([-dx, -dy, dz])
    


class NeRFGUI:
    def __init__(self, opt, trainer, train_loader=None, debug=True):
        self.opt = opt # shared with the trainer's opt to support in-place modification of rendering parameters.
        self.W = opt.W
        self.H = opt.H
        self.cam = OrbitCamera(opt.W, opt.H, r=opt.radius, fovy=opt.fovy)
        self.use_depth = False
        self.use_multi_depth = False
        self.take_image = False
        self.debug = debug
        self.bg_color = torch.ones(3, dtype=torch.float32) # default white bg
        self.training = False
        self.step = 0 # training step 

        self.trainer = trainer
        self.train_loader = train_loader
        if train_loader is not None:
            err_dict = {}
            for i in range(len(train_loader)):
                err_dict[train_loader[i]._data.datatype] = train_loader[i]._data.error_map
                
            self.trainer.error_map = err_dict

        self.render_buffer = np.zeros((self.W, self.H, 3), dtype=np.float32)
        self.need_update = True # camera moved, should reset accumulation
        self.spp = 1 # sample per pixel

        self.dynamic_resolution = True
        self.downscale = 1
        self.train_steps = 16

        dpg.create_context()
        self.register_dpg()
        self.test_step()
        

    def __del__(self):
        dpg.destroy_context()


    def train_step(self):

        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        starter.record()

        outputs = self.trainer.train_gui(self.train_loader, step=self.train_steps)

        ender.record()
        torch.cuda.synchronize()
        t = starter.elapsed_time(ender)

        self.step += self.train_steps
        self.need_update = True

        dpg.set_value("_log_train_time", f'{t:.4f}ms')
        dpg.set_value("_log_train_log", f'step = {self.step: 5d} (+{self.train_steps: 2d}), loss = {outputs["loss"]:.4f}, lr = {outputs["lr"]:.5f}')

        # dynamic train steps
        # max allowed train time per-frame is 500 ms
        full_t = t / self.train_steps * 16
        train_steps = min(16, max(4, int(16 * 500 / full_t)))
        if train_steps > self.train_steps * 1.2 or train_steps < self.train_steps * 0.8:
            self.train_steps = train_steps

    
    def test_step(self):
        # TODO: seems we have to move data from GPU --> CPU --> GPU?

        if self.need_update or self.spp < self.opt.max_spp:
        
            starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            starter.record()
            
            #print("use depth? ", self.use_depth)
            if not self.use_depth:
                if not self.use_multi_depth:

                    outputs = self.trainer.test_gui(self.cam.pose, self.cam.intrinsics, 
                                                    self.W, self.H, 'viewer', self.cam.near,
                                                    self.cam.far, self.bg_color, self.spp, 
                                                    self.downscale)
                    image = outputs['image']
                else:
                    channel_1 = self.trainer.test_gui(self.cam.pose, self.cam.intrinsics,
                                                      self.W, self.H, 'depth', self.near_color,
                                                      self.far_color, self.bg_color, self.spp, 
                                                      self.downscale)

                    channel_2 = self.trainer.test_gui(self.cam.pose, self.cam.intrinsics,
                                                      self.W, self.H, 'depth', self.near_depth,
                                                      self.far_depth, self.bg_color, self.spp, 
                                                      self.downscale)

                    channel_3 = self.trainer.test_gui(self.cam.pose, np.array([8.838834762573242, 8.838834762573242, self.W // 2, self.H // 2, 25]),
                                                      self.W, self.H, 'touch', self.near_touch,
                                                      self.far_touch, self.bg_color, self.spp,
                                                      self.downscale)
                
                    image = np.stack(((channel_1['depth']-self.near_color)/(self.far_color - self.near_color),
                                      (channel_2['depth']-self.near_depth)/(self.far_depth - self.near_depth),
                                      (channel_3['depth']-self.near_touch)/(self.far_touch - self.near_touch)), axis=-1)
                    
                    
                    
                    
                #print("color image shape")
                #print(image.shape)
                    print("channel 1")
                    print(np.max((channel_1['depth']-self.near_color)/(self.far_color - self.near_color)))
                    print(np.min((channel_1['depth']-self.near_color)/(self.far_color - self.near_color)))
                    print("channel 2")
                    print(np.max((channel_2['depth']-self.near_depth)/(self.far_depth - self.near_depth)))
                    print(np.min((channel_2['depth']-self.near_depth)/(self.far_depth - self.near_depth)))
                    print("channel 3")
                    print(np.max((channel_3['depth']-self.near_touch)/(self.far_touch - self.near_touch)))
                    print(np.min((channel_3['depth']-self.near_touch)/(self.far_touch - self.near_touch)))
            else:
                outputs = self.trainer.test_gui(self.cam.pose, self.cam.intrinsics,
                                                self.W, self.H, 'depth', self.cam.near,
                                                self.cam.far, self.bg_color, self.spp, 
                                                self.downscale)
                
                image = (outputs['depth']-self.cam.near)/(self.cam.far - self.cam.near)
                #image = outputs['depth']
                #image = (image - np.min(image))/(np.max(image)-np.min(image))

                #print("depth image shape")
                #print(image.shape)
                #print(self.W)
                #print(self.H)
                image = np.repeat(image[:,:,np.newaxis],3, axis=-1)
                #print(image.shape)

            ender.record()
            torch.cuda.synchronize()
            t = starter.elapsed_time(ender)

            # update dynamic resolution
            if self.dynamic_resolution:
                # max allowed infer time per-frame is 200 ms
                full_t = t / (self.downscale ** 2)
                downscale = min(1, max(1/4, math.sqrt(200 / full_t)))
                if downscale > self.downscale * 1.2 or downscale < self.downscale * 0.8:
                    self.downscale = max(1/2, downscale)
            if self.take_image:
                plt.imshow(image)
                plt.show()
                self.take_image = False

            if self.need_update:
                self.render_buffer = image #outputs['image']
                self.spp = 1
                self.need_update = False
            else:
                #self.render_buffer = (self.render_buffer * self.spp + outputs['image']) / (self.spp + 1)
                self.render_buffer = (self.render_buffer * self.spp + image) / (self.spp + 1)
                self.spp += 1

            dpg.set_value("_log_infer_time", f'{t:.4f}ms')
            dpg.set_value("_log_resolution", f'{int(self.downscale * self.W)}x{int(self.downscale * self.H)}')
            dpg.set_value("_log_spp", self.spp)
            dpg.set_value("_texture", self.render_buffer)

        
    def register_dpg(self):

        ### register texture 

        with dpg.texture_registry(show=False):
            dpg.add_raw_texture(self.W, self.H, self.render_buffer, format=dpg.mvFormat_Float_rgb, tag="_texture")

        ### register window

        # the rendered image, as the primary window
        with dpg.window(tag="_primary_window", width=self.W, height=self.H):

            # add the texture
            dpg.add_image("_texture")

        dpg.set_primary_window("_primary_window", True)

        # control window
        with dpg.window(label="Control", tag="_control_window", width=400, height=300):

            # button theme
            with dpg.theme() as theme_button:
                with dpg.theme_component(dpg.mvButton):
                    dpg.add_theme_color(dpg.mvThemeCol_Button, (23, 3, 18))
                    dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (51, 3, 47))
                    dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (83, 18, 83))
                    dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 5)
                    dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 3, 3)

            # time
            if not self.opt.mode == 'test':
                with dpg.group(horizontal=True):
                    dpg.add_text("Train time: ")
                    dpg.add_text("no data", tag="_log_train_time")                    

            with dpg.group(horizontal=True):
                dpg.add_text("Infer time: ")
                dpg.add_text("no data", tag="_log_infer_time")
            
            with dpg.group(horizontal=True):
                dpg.add_text("SPP: ")
                dpg.add_text("1", tag="_log_spp")

            # train button
            if not self.opt.mode == 'test':
                with dpg.collapsing_header(label="Train", default_open=True):

                    # train / stop
                    with dpg.group(horizontal=True):
                        dpg.add_text("Train: ")

                        def callback_train(sender, app_data):
                            if self.training:
                                self.training = False
                                dpg.configure_item("_button_train", label="start")
                            else:
                                self.training = True
                                dpg.configure_item("_button_train", label="stop")

                        dpg.add_button(label="start", tag="_button_train", callback=callback_train)
                        dpg.bind_item_theme("_button_train", theme_button)

                        def callback_reset(sender, app_data):
                            @torch.no_grad()
                            def weight_reset(m: nn.Module):
                                reset_parameters = getattr(m, "reset_parameters", None)
                                if callable(reset_parameters):
                                    m.reset_parameters()
                            self.trainer.model.apply(fn=weight_reset)
                            self.trainer.model.reset_extra_state() # for cuda_ray density_grid and step_counter
                            self.need_update = True

                        dpg.add_button(label="reset", tag="_button_reset", callback=callback_reset)
                        dpg.bind_item_theme("_button_reset", theme_button)

                    # save ckpt
                    with dpg.group(horizontal=True):
                        dpg.add_text("Checkpoint: ")

                        def callback_save(sender, app_data):
                            self.trainer.save_checkpoint(full=True, best=False)
                            dpg.set_value("_log_ckpt", "saved " + os.path.basename(self.trainer.stats["checkpoints"][-1]))
                            self.trainer.epoch += 1 # use epoch to indicate different calls.

                        dpg.add_button(label="save", tag="_button_save", callback=callback_save)
                        dpg.bind_item_theme("_button_save", theme_button)

                        dpg.add_text("", tag="_log_ckpt")
                    
                    # save mesh
                    with dpg.group(horizontal=True):
                        dpg.add_text("Marching Cubes: ")

                        def callback_mesh(sender, app_data):
                            self.trainer.save_mesh(resolution=256, threshold=10)
                            dpg.set_value("_log_mesh", "saved " + f'{self.trainer.name}_{self.trainer.epoch}.ply')
                            self.trainer.epoch += 1 # use epoch to indicate different calls.

                        dpg.add_button(label="mesh", tag="_button_mesh", callback=callback_mesh)
                        dpg.bind_item_theme("_button_mesh", theme_button)

                        dpg.add_text("", tag="_log_mesh")

                    with dpg.group(horizontal=True):
                        dpg.add_text("", tag="_log_train_log")

            
            # rendering options
            with dpg.collapsing_header(label="Options", default_open=True):

                # dynamic rendering resolution
                with dpg.group(horizontal=True):

                    def callback_set_dynamic_resolution(sender, app_data):
                        if self.dynamic_resolution:
                            self.dynamic_resolution = False
                            self.downscale = 1
                        else:
                            self.dynamic_resolution = True
                        self.need_update = True

                    dpg.add_checkbox(label="dynamic resolution", default_value=self.dynamic_resolution, callback=callback_set_dynamic_resolution)
                    dpg.add_text(f"{self.W}x{self.H}", tag="_log_resolution")

                # bg_color picker
                def callback_change_bg(sender, app_data):
                    self.bg_color = torch.tensor(app_data[:3], dtype=torch.float32) # only need RGB in [0, 1]
                    self.need_update = True

                dpg.add_color_edit((255, 255, 255), label="Background Color", width=200, tag="_color_editor", no_alpha=True, callback=callback_change_bg)

                # fov slider
                def callback_set_fovy(sender, app_data):
                    self.cam.fovy = app_data
                    self.need_update = True

                dpg.add_slider_int(label="FoV (vertical)", min_value=1, max_value=120, format="%d deg", default_value=self.cam.fovy, callback=callback_set_fovy)

                def callback_take_image(sender, app_data):
                    if not self.take_image:
                        self.take_image = True
                    else:
                        self.take_image = False

                dpg.add_button(label="Take Image", tag="_button_take_image", callback=callback_take_image)
                
                def callback_choose_camera(sender, app_data):
                    self.use_depth = app_data
                    callback_set_near_far(sender, app_data)
                
                dpg.add_checkbox(label="Use Depth", default_value=self.use_depth, callback=callback_choose_camera)
                
                def callback_set_near_far(sender, app_data):
                    near = float(dpg.get_value("_near_val"))
                    far = float(dpg.get_value("_far_val"))
                    if near < 0:
                        near = 0

                    if far < 0:
                        far = np.inf

                    if far < near:
                        temp = far
                        far = near
                        near = temp

                    print("near: ", near)
                    print("far: ", far)
                    self.cam.near = near
                    self.cam.far = far
                    self.need_update = True

                with dpg.group(horizontal=True):
                    dpg.add_text("Near (m): ")
                    dpg.add_input_text(tag="_near_val", no_spaces=True, decimal=True,default_value='0.1', width=50)
                    dpg.add_text("Far (m): ")
                    dpg.add_input_text(tag="_far_val", no_spaces=True, decimal=True,default_value='0.25', width=50)
                    dpg.add_button(label="submit", tag="_button_submit_near_far", callback=callback_set_near_far)
                    dpg.bind_item_theme("_button_submit_near_far", theme_button)


                def callback_select_multi_depth(sender, app_data):
                    self.use_multi_depth = app_data
                    callback_set_near_far_multi(sender, app_data)
                
                dpg.add_checkbox(label="Multi Depth", default_value=self.use_multi_depth, callback=callback_select_multi_depth)


                def callback_set_near_far_multi(sender, app_data):
                    
                    # read in desired color image near and far planes
                    self.near_color = float(dpg.get_value("_near_val_color"))
                    self.far_color = float(dpg.get_value("_far_val_color"))
                    
                    # ensure that near and far planes are properly mapped
                    # for color channel
                    self.near_color = 0 if self.near_color < 0 else self.near_color
                    self.far_color = np.inf if self.far_color < 0 else self.far_color

                    if self.far_color < self.near_color:
                        temp = self.far_color
                        self.far_color = self.near_color
                        self.near_color = temp

                    # read in desired depth image near and far planes
                    self.near_depth = float(dpg.get_value("_near_val_depth"))
                    self.far_depth = float(dpg.get_value("_far_val_depth"))

                    # ensure that near and far planes are properly mapped
                    # for depth channel
                    self.near_depth = 0 if self.near_depth < 0 else self.near_depth
                    self.far_depth = np.inf if self.far_depth < 0 else self.far_depth

                    if self.far_depth < self.near_depth:
                        temp = self.far_depth
                        self.far_depth = self.near_depth
                        self.near_depth = temp

                    # read in desired touch image near and far planes
                    self.near_touch = float(dpg.get_value("_near_val_touch"))
                    self.far_touch = float(dpg.get_value("_far_val_touch"))

                    # ensure that near and far planes are properly mapped
                    # for touch channel
                    self.near_touch = 0 if self.near_touch < 0 else self.near_touch
                    self.far_touch = np.inf if self.far_touch < 0 else self.far_touch

                    if self.far_touch < self.near_touch:
                        temp = self.far_touch
                        self.far_touch = self.near_touch
                        self.near_touch = temp

                    print("near(color): ", self.near_color)
                    print("far(color): ", self.far_color)

                    print("near(depth): ", self.near_depth)
                    print("far(depth): ", self.far_depth)

                    print("near(touch): ", self.near_touch)
                    print("far(touch): ", self.far_touch)
                    
                    #self.cam.near = near
                    #self.cam.far = far
                    self.need_update = True

                with dpg.group(horizontal=True):
                    dpg.add_text("Near (m): ")
                    dpg.add_input_text(tag="_near_val_color", no_spaces=True, decimal=True,default_value='0.0001', width=50)
                    dpg.add_text("Far (m): ")
                    dpg.add_input_text(tag="_far_val_color", no_spaces=True, decimal=True,default_value='2.0', width=50)

                with dpg.group(horizontal=True):
                    dpg.add_text("Near (m): ")
                    dpg.add_input_text(tag="_near_val_depth", no_spaces=True, decimal=True,default_value='0.0001', width=50)
                    dpg.add_text("Far (m): ")
                    dpg.add_input_text(tag="_far_val_depth", no_spaces=True, decimal=True,default_value='0.5', width=50)

                with dpg.group(horizontal=True):
                    dpg.add_text("Near (m): ")
                    dpg.add_input_text(tag="_near_val_touch", no_spaces=True, decimal=True,default_value='0.0001', width=50)
                    dpg.add_text("Far (m): ")
                    dpg.add_input_text(tag="_far_val_touch", no_spaces=True, decimal=True,default_value='0.025', width=50)
                    dpg.add_button(label="submit", tag="_button_submit_near_far_multi", callback=callback_set_near_far_multi)
                    dpg.bind_item_theme("_button_submit_near_far_multi", theme_button)


                # dt_gamma slider
                def callback_set_dt_gamma(sender, app_data):
                    self.opt.dt_gamma = app_data
                    self.need_update = True

                dpg.add_slider_float(label="dt_gamma", min_value=0, max_value=0.1, format="%.5f", default_value=self.opt.dt_gamma, callback=callback_set_dt_gamma)

                # aabb slider
                def callback_set_aabb(sender, app_data, user_data):
                    # user_data is the dimension for aabb (xmin, ymin, zmin, xmax, ymax, zmax)
                    self.trainer.model.aabb_infer[user_data] = app_data

                    # also change train aabb ? [better not...]
                    #self.trainer.model.aabb_train[user_data] = app_data

                    self.need_update = True

                dpg.add_separator()
                dpg.add_text("Axis-aligned bounding box:")

                with dpg.group(horizontal=True):
                    dpg.add_slider_float(label="x", width=150, min_value=-self.opt.bound, max_value=0, format="%.2f", default_value=-self.opt.bound, callback=callback_set_aabb, user_data=0)
                    dpg.add_slider_float(label="", width=150, min_value=0, max_value=self.opt.bound, format="%.2f", default_value=self.opt.bound, callback=callback_set_aabb, user_data=3)

                with dpg.group(horizontal=True):
                    dpg.add_slider_float(label="y", width=150, min_value=-self.opt.bound, max_value=0, format="%.2f", default_value=-self.opt.bound, callback=callback_set_aabb, user_data=1)
                    dpg.add_slider_float(label="", width=150, min_value=0, max_value=self.opt.bound, format="%.2f", default_value=self.opt.bound, callback=callback_set_aabb, user_data=4)

                with dpg.group(horizontal=True):
                    dpg.add_slider_float(label="z", width=150, min_value=-self.opt.bound, max_value=0, format="%.2f", default_value=-self.opt.bound, callback=callback_set_aabb, user_data=2)
                    dpg.add_slider_float(label="", width=150, min_value=0, max_value=self.opt.bound, format="%.2f", default_value=self.opt.bound, callback=callback_set_aabb, user_data=5)
                

            # debug info
            if self.debug:
                with dpg.collapsing_header(label="Debug"):
                    # pose
                    dpg.add_separator()
                    dpg.add_text("Camera Pose:")
                    dpg.add_text(str(self.cam.pose), tag="_log_pose")


        ### register camera handler

        def callback_camera_drag_rotate(sender, app_data):

            if not dpg.is_item_focused("_primary_window"):
                return

            dx = app_data[1]
            dy = app_data[2]

            self.cam.orbit(dx, dy)
            self.need_update = True

            if self.debug:
                dpg.set_value("_log_pose", str(self.cam.pose))


        def callback_camera_wheel_scale(sender, app_data):

            if not dpg.is_item_focused("_primary_window"):
                return

            delta = app_data

            self.cam.scale(delta)
            self.need_update = True

            if self.debug:
                dpg.set_value("_log_pose", str(self.cam.pose))


        def callback_camera_drag_pan(sender, app_data):

            if not dpg.is_item_focused("_primary_window"):
                return

            dx = app_data[1]
            dy = app_data[2]

            self.cam.pan(dx, dy)
            self.need_update = True

            if self.debug:
                dpg.set_value("_log_pose", str(self.cam.pose))


        with dpg.handler_registry():
            dpg.add_mouse_drag_handler(button=dpg.mvMouseButton_Left, callback=callback_camera_drag_rotate)
            dpg.add_mouse_wheel_handler(callback=callback_camera_wheel_scale)
            dpg.add_mouse_drag_handler(button=dpg.mvMouseButton_Middle, callback=callback_camera_drag_pan)

        
        dpg.create_viewport(title='torch-ngp', width=self.W, height=self.H, resizable=False)
        
        # TODO: seems dearpygui doesn't support resizing texture...
        # def callback_resize(sender, app_data):
        #     self.W = app_data[0]
        #     self.H = app_data[1]
        #     # how to reload texture ???

        # dpg.set_viewport_resize_callback(callback_resize)

        ### global theme
        with dpg.theme() as theme_no_padding:
            with dpg.theme_component(dpg.mvAll):
                # set all padding to 0 to avoid scroll bar
                dpg.add_theme_style(dpg.mvStyleVar_WindowPadding, 0, 0, category=dpg.mvThemeCat_Core)
                dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 0, 0, category=dpg.mvThemeCat_Core)
                dpg.add_theme_style(dpg.mvStyleVar_CellPadding, 0, 0, category=dpg.mvThemeCat_Core)
        
        dpg.bind_item_theme("_primary_window", theme_no_padding)

        dpg.setup_dearpygui()

        #dpg.show_metrics()

        dpg.show_viewport()


    def render(self):

        while dpg.is_dearpygui_running():
            # update texture every frame
            if self.training:
                self.train_step()
            self.test_step()
            dpg.render_dearpygui_frame()
