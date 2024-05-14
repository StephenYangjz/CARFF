import math
import torch
import numpy as np
from tqdm import tqdm
import dearpygui.dearpygui as dpg
from scipy.spatial.transform import Rotation as R
from torchvision.utils import save_image
import torch.nn.functional as F
from nerf.utils import *
import torchmetrics


hidden_dim = 512
K = 2  # Number of mixtures

class OrbitCamera:
    def __init__(self, W, H, r=2, fovy=60):
        self.W = W
        self.H = H
        self.radius = r # camera distance from center
        self.fovy = fovy # in degree
        self.center = np.array([0, 0, 0], dtype=np.float32) # look at this point
        self.rot = R.from_quat([1, 0, 0, 0]) # init camera matrix: [[1, 0, 0], [0, -1, 0], [0, 0, 1]] (to suit ngp convention)
        self.up = np.array([0, 1, 0], dtype=np.float32) # need to be normalized!

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
        return np.array([focal, focal, self.W // 2, self.H // 2])
    
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
    def __init__(self, opt, trainer, train_loader=None, debug=True, gui=True):
        self.gui = gui
        
        self.opt = opt # shared with the trainer's opt to support in-place modification of rendering parameters.
        self.W = opt.W
        self.H = opt.H
        self.cam = OrbitCamera(opt.W, opt.H, r=opt.radius, fovy=opt.fovy)
        self.debug = debug
        self.bg_color = torch.ones(3, dtype=torch.float32) # default white bg
        # self.latent_toggle = True
        self.training = False
        self.step = 0 # training step 

        self.trainer = trainer
        self.train_loader = train_loader
        if train_loader is not None:
            self.trainer.error_map = train_loader._data.error_map

        index = self.find_index("train/cam-v0-t0")
        mu = train_loader._data.mus[index]
        var = train_loader._data.vars[index]
        self.test_latent = reparameterize(mu, var)
        self.current_t = 0
        self.render_buffer = np.zeros((self.W, self.H, 3), dtype=np.float32)
        self.need_update = True # camera moved, should reset accumulation
        self.spp = 1 # sample per pixel

        self.dynamic_resolution = True
        self.downscale = 1
        self.train_steps = 16

        dpg.create_context()
        if gui:
            self.register_dpg()
        self.test_step()

        latent_dim = self.test_latent.shape[0]
        self.MDN = MDN(2*latent_dim,latent_dim, K, hidden_dim).cuda()
        self.MDN.load(self.train_loader._data.root_path)
        print("MDN Path:", self.train_loader._data.root_path)

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

        if self.gui:
            dpg.set_value("_log_train_time", f'{t:.4f}ms')
            dpg.set_value("_log_train_log", f'step = {self.step: 5d} (+{self.train_steps: 2d}), loss = {outputs["loss"]:.4f}, lr = {outputs["lr"]:.5f}')

        # dynamic train steps
        # max allowed train time per-frame is 500 ms
        full_t = t / self.train_steps * 16
        train_steps = min(16, max(4, int(16 * 500 / full_t)))
        if train_steps > self.train_steps * 1.2 or train_steps < self.train_steps * 0.8:
            self.train_steps = train_steps

    def test_step(self):
        if self.need_update or self.spp < self.opt.max_spp:
        
            starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            starter.record()

            latent = self.test_latent
            outputs = self.trainer.test_gui(self.cam.pose, self.cam.intrinsics, self.W, self.H, latent, self.bg_color, self.spp, self.downscale)

            ender.record()
            torch.cuda.synchronize()
            t = starter.elapsed_time(ender)

            # update dynamic resolution
            if self.dynamic_resolution:
                # max allowed infer time per-frame is 200 ms
                full_t = t / (self.downscale ** 2)
                downscale = min(1, max(1/4, math.sqrt(200 / full_t)))
                if downscale > self.downscale * 1.2 or downscale < self.downscale * 0.8:
                    self.downscale = downscale

            if self.need_update:
                self.render_buffer = outputs['image']
                self.spp = 1
                self.need_update = False
            else:
                self.render_buffer = (self.render_buffer * self.spp + outputs['image']) / (self.spp + 1)
                self.spp += 1
                
            if self.gui:
                dpg.set_value("_log_infer_time", f'{t:.4f}ms')
                dpg.set_value("_log_resolution", f'{int(self.downscale * self.W)}x{int(self.downscale * self.H)}')
                dpg.set_value("_log_spp", self.spp)
                dpg.set_value("_texture", self.render_buffer)

    def find_index(self, input):
        indices = [index for index, path in enumerate(self.train_loader._data.paths) if input in path]
        if len(indices) != 1:
            return -1
        print("Input correspinds to:", self.train_loader._data.paths[indices[0]])
        return indices[0]

    def render_bev(self, dest_path=None):
        indices = {}
        for i, item in enumerate(self.train_loader._data.paths):
            if item.startswith("train/cam-v2-") or item.startswith("train_ego_actor/cam-v2-"):
                indices[item] = i
        if len(indices) == 0:
            raise ValueError("Dataset not supported for probing.")
        rand_index = indices[list(indices.keys())[0]]
        data = self.train_loader._data.collate_for_probe([rand_index])
        H, W = data['H'], data['W']
        
        latents = self.test_latent.cuda().float()
        poses = self.train_loader._data.poses[rand_index].cpu().numpy() # [B, 4, 4]
        intrinsics = self.train_loader._data.intrinsics
        outputs = self.trainer.test_gui(poses, intrinsics, W, H, latents, bg_color=None, spp=1, downscale=1)
        pred_img = torch.from_numpy(outputs['image'])
        if dest_path is not None:
            save_image(pred_img.permute(2, 0, 1), dest_path)
        return pred_img

    def calculate_densities(self, timestamp):
        indices = {}
        for i, item in enumerate(self.train_loader._data.paths):
            if item.startswith("train/cam-v2-") or item.startswith("train_ego_actor/cam-v2-"):
                indices[item] = i
        # print(indices)
        if len(indices) == 0:
            raise ValueError("Dataset not supported for probing.")
        rand_index = indices[list(indices.keys())[0]]
        data = self.train_loader._data.collate_for_probe([rand_index])
        H, W = data['H'], data['W']

        latents = self.test_latent.cuda().float()
        poses = self.train_loader._data.poses[rand_index].cpu().numpy() # [B, 4, 4]
        intrinsics = self.train_loader._data.intrinsics
        outputs = self.trainer.test_gui(poses, intrinsics, W, H, latents, bg_color=None, spp=1, downscale=1, timestamp=timestamp)
        return outputs['mean_density']

    def probe(self):
        indices = {}
        for i, item in enumerate(self.train_loader._data.paths):
            if item.startswith("train/cam-v2-") or item.startswith("train_ego_actor/cam-v2-"):
                indices[item] = i
        # print(indices)
        if len(indices) == 0:
            raise ValueError("Dataset not supported for probing.")
        rand_index = indices[list(indices.keys())[0]]
        data = self.train_loader._data.collate_for_probe([rand_index])
        H, W = data['H'], data['W']

        # see what it looks like with current latent
        latents = self.test_latent.cuda().float()
        poses = self.train_loader._data.poses[rand_index].cpu().numpy() # [B, 4, 4]
        intrinsics = self.train_loader._data.intrinsics
        outputs = self.trainer.test_gui(poses, intrinsics, W, H, latents, bg_color=None, spp=1, downscale=1)
        pred_img = torch.from_numpy(outputs['image']) #.reshape(-1, H, W, 3)

        # get all images at the same view and compare
        losses = {}
        for key in indices.keys():
            index = indices[key]
            this_gt = self.train_loader._data.collate_for_probe([index])["images"].squeeze(0)
            losses[key] = F.mse_loss(this_gt.cuda(), pred_img.cuda())
        probed_result = min(losses, key=losses.get)
        # get the distribution for that result
        return indices[probed_result]

    def probe_densities(self):
        indices = {}
        for i, item in enumerate(self.train_loader._data.paths):
            if item.startswith("train/cam-v2-") or item.startswith("train_ego_actor/cam-v2-"):
                indices[item] = i
        densities = {}
        for i in range(6):
            densities[i] = self.calculate_densities(i)
        probed_result = max(densities, key=densities.get)
        result_index = indices[sorted(indices.keys())[probed_result]]
        return probed_result, result_index

    def update_latent_from_predicted(self, result_index):
        # update test_latent
        if result_index == -1:
            print("Error: input image path invalid for latent generation")
            return
        # update test_latent
        print("Index:", result_index)
        mus = self.train_loader._data.mus[result_index].cuda()
        vars = self.train_loader._data.vars[result_index].cuda()
        # one hot encode
        one_hot_encode = F.one_hot(torch.tensor([self.current_t]), self.train_loader._data.num_scenes).cuda()
        input_data = torch.cat([mus, vars]).unsqueeze(0).cuda()
        sampled_latent, weight, mu, sigma = self.MDN.sample(input_data)
        predicted_latent = sampled_latent.squeeze(0)
        # probe current t
        # self.current_t = int(self.train_loader._data.scene_ids[self.probe()])
        self.test_latent = predicted_latent
        self.need_update = True

    def render_no_gui(self, epochs):
        for _ in tqdm.tqdm(range(epochs)):
            if self.training:
                self.train_step()
            self.test_step()

    def register_dpg(self):

        ### register texture 
        dpg.set_global_font_scale(2)
        with dpg.texture_registry(show=False):
            dpg.add_raw_texture(self.W, self.H, self.render_buffer, format=dpg.mvFormat_Float_rgb, tag="_texture")
        ### register window

        # the rendered image, as the primary window
        with dpg.window(tag="_primary_window", width=self.W, height=self.H):

            # add the texture
            dpg.add_image("_texture")

        dpg.set_primary_window("_primary_window", True)

        # control window
        with dpg.window(label="Control", tag="_control_window", width=800, height=600):

            # button theme
            with dpg.theme() as theme_button:
                with dpg.theme_component(dpg.mvButton):
                    dpg.add_theme_color(dpg.mvThemeCol_Button, (23, 3, 18))
                    dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (51, 3, 47))
                    dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (83, 18, 83))
                    dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 5)
                    dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 3, 3)

            # time
            if not self.opt.test:
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
            if not self.opt.test:
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

                def update_latent_from_image(image_path):
                    index = self.find_index(image_path)
                    # update test_latent
                    if index == -1:
                        print("Error: input image path invalid for latent generation")
                        return
                    mu = self.train_loader._data.mus[index]
                    var = self.train_loader._data.vars[index]
                    new_latent = reparameterize(mu, var)
                    # self.current_t = int(self.train_loader._data.scene_ids[index])
                    # print("New t:", self.current_t)
                    self.test_latent = new_latent
                    self.need_update = True
                    print("Success: changed to latents generated by ", image_path)
                
                # input image for latent gen
                def callback_change_latent_from_input(sender, app_data): # str of image path:
                    # if self.test_latent = torch.tensor(app_data, dtype=torch.float32)
                    update_latent_from_image(app_data)

                dpg.add_input_text(label="Toggle img (v0-t0)", default_value="", callback=callback_change_latent_from_input, on_enter=True)


                def update_latent_from_predicted(image_path):
                    result_index = self.find_index(image_path)
                    # update test_latent
                    if result_index == -1:
                        print("Error: input image path invalid for latent generation")
                        return
                    # update test_latent
                    print("Index:", result_index)
                    mus = self.train_loader._data.mus[result_index].cuda()
                    vars = self.train_loader._data.vars[result_index].cuda()
                    # one hot encode
                    one_hot_encode = F.one_hot(torch.tensor([self.current_t]), self.train_loader._data.num_scenes).cuda()
                    input_data = torch.cat([mus, vars]).unsqueeze(0).cuda()
                    sampled_latent, weight, mu, sigma = self.MDN.sample(input_data)
                    predicted_latent = sampled_latent.squeeze(0)
                    # probe current t
                    # self.current_t = int(self.train_loader._data.scene_ids[self.probe()])
                    self.test_latent = predicted_latent
                    self.need_update = True
                    print("Success: changed to PREDICTED latents generated by ", image_path)

                def callback_predict_latent_from_input(sender, app_data): # str of image path:
                    # if self.test_latent = torch.tensor(app_data, dtype=torch.float32)
                    update_latent_from_predicted(app_data)
                dpg.add_input_text(label="Predict img (v0-t0)", default_value="", callback=callback_predict_latent_from_input, on_enter=True)

                def callback_probe_car(sender, app_data):
                    result_index = self.probe()
                    mus = self.train_loader._data.mus[result_index].cuda()
                    vars = self.train_loader._data.vars[result_index].cuda()
                    # one hot encode
                    one_hot_encode = F.one_hot(torch.tensor([self.current_t]), self.train_loader._data.num_scenes).cuda()
                    input_data = torch.cat([mus, vars]).unsqueeze(0).cuda()
                    sampled_latent, weight, mu, sigma = self.MDN.sample(input_data)
                    predicted_latent = sampled_latent.squeeze(0)
                    result_index = self.probe()
                    # self.current_t = int(self.train_loader._data.scene_ids[result_index])
                    # print("yyy", int(self.train_loader._data.scene_ids[result_index]))
                    self.test_latent = predicted_latent
                    self.need_update = True
                    print("Successfully transferred to the next stage based on probed result.")

                def callback_probe_density(sender, app_data):
                    _, result_index = self.probe_densities()
                    mus = self.train_loader._data.mus[result_index].cuda()
                    vars = self.train_loader._data.vars[result_index].cuda()
                    # one hot encode
                    one_hot_encode = F.one_hot(torch.tensor([self.current_t]), self.train_loader._data.num_scenes).cuda()
                    input_data = torch.cat([mus, vars]).unsqueeze(0).cuda()
                    sampled_latent, weight, mu, sigma = self.MDN.sample(input_data)
                    predicted_latent = sampled_latent.squeeze(0)
                    current_t, _ = self.probe_densities()
                    print("Predicted:", current_t)
                    self.test_latent = predicted_latent
                    self.need_update = True
                    print("Successfully transferred to the next stage based on probed density of result.")

                dpg.add_button(label="Probe and predict", tag="_button_probe", callback=callback_probe_car)
                dpg.add_button(label="Probe and predict density", tag="_button_probe_density", callback=callback_probe_density)

                def callback_psnr(sender, app_data):
                    indices = {}
                    for i, item in enumerate(self.train_loader._data.paths):
                        if item.startswith("train/cam-v2-"):
                            indices[item] = i
                    if len(indices) == 0:
                        raise ValueError("Dataset not supported for probing.")
                    rand_index = indices[list(indices.keys())[0]]
                    data = self.train_loader._data.collate_for_probe([rand_index])
                    H, W = data['H'], data['W']

                    # see what it looks like with current latent
                    latents = self.test_latent.cuda().float()
                    poses = self.train_loader._data.poses[rand_index].cpu().numpy() # [B, 4, 4]
                    intrinsics = self.train_loader._data.intrinsics
                    outputs = self.trainer.test_gui(poses, intrinsics, W, H, latents, bg_color=None, spp=1, downscale=1)
                    pred_img = torch.from_numpy(outputs['image']) #.reshape(-1, H, W, 3)
                    save_image(pred_img.permute(2, 0, 1), 'pred_img.png')
                    psnr_metric = torchmetrics.PeakSignalNoiseRatio(data_range=1.0).cuda()

                    # get all images at the same view and compare
                    for key in indices.keys():
                        index = indices[key]
                        this_gt = self.train_loader._data.collate_for_probe([index])["images"]
                        psnr_value = psnr_metric(pred_img.cuda(), this_gt.cuda())
                        psnr_computed = psnr_value
                        print("PSNR with " + key + ":")
                        print(psnr_computed)
                        psnr_metric.reset()


                dpg.add_button(label="Print PSNRs", tag="_button_psnr", callback=callback_psnr)

                # fov slider
                def callback_set_fovy(sender, app_data):
                    self.cam.fovy = app_data
                    self.need_update = True

                dpg.add_slider_int(label="FoV (vertical)", min_value=1, max_value=120, format="%d deg", default_value=self.cam.fovy, callback=callback_set_fovy)

                # dt_gamma slider
                def callback_set_dt_gamma(sender, app_data):
                    self.opt.dt_gamma = app_data
                    self.need_update = True

                dpg.add_slider_float(label="dt_gamma", min_value=0, max_value=0.1, format="%.5f", default_value=self.opt.dt_gamma, callback=callback_set_dt_gamma)

                # aabb slider
                def callback_set_aabb(sender, app_data, user_data):
                    # user_data is the dimension for aabb (xmin, ymin, zmin, xmax, ymax, zmax)
                    self.trainer.model.aabb_infer[user_data] = app_data

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
