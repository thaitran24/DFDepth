# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import numpy as np
import time

import torch
from torchvision import transforms
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

import json

from utils import *
from kitti_utils import *
from layers import *

import datasets
import networks
from IPython import embed
from PIL import Image

import cv2
STEREO_SCALE_FACTOR = 5.4
DAY_PHASE = 1
NIGHT_PHASE = 2 

class Trainer:
    def __init__(self, options):
        self.opt = options
        self.log_path = os.path.join(self.opt.log_dir, self.opt.model_name)
        self.result_path = os.path.join(self.log_path, 'result.txt')

        # checking height and width are multiples of 32
        assert self.opt.height % 32 == 0, "'height' must be a multiple of 32"
        assert self.opt.width % 32 == 0, "'width' must be a multiple of 32"

        self.models = {}
        self.classifiers = {}
        self.parameters_to_train = []

        self.device = torch.device("cpu" if self.opt.no_cuda else "cuda")

        self.num_scales = len(self.opt.scales)
        self.num_input_frames = len(self.opt.frame_ids)
        self.num_pose_frames = 2 if self.opt.pose_model_input == "pairs" else self.num_input_frames

        assert self.opt.frame_ids[0] == 0, "frame_ids must start with 0"

        self.use_pose_net = not (self.opt.use_stereo and self.opt.frame_ids == [0])

        if self.opt.use_stereo:
            self.opt.frame_ids.append("s")

        # Shared Encoder
        if self.opt.encoder == "resnet_ddf":
            self.models["encoder"] = networks.ResnetDDF(
                self.opt.num_layers, self.opt.weights_init == "pretrained")
        elif self.opt.encoder == "resnet_dida":
            self.models["encoder"] = networks.ResnetDIDA(
                self.opt.num_layers, self.opt.weights_init == "pretrained", dida_level=self.opt.dida_level)
        else:
            self.models["encoder"] = networks.ResnetEncoder(
                self.opt.num_layers, self.opt.weights_init == "pretrained")
        self.models["encoder"].to(self.device)
        self.parameters_to_train += list(self.models["encoder"].parameters())

        # Depth Decoder
        self.models["depth"] = networks.DepthDecoder(
            self.models["encoder"].num_ch_enc, self.opt.scales)
        self.models["depth"].to(self.device)
        self.parameters_to_train += list(self.models["depth"].parameters())
                
        if self.opt.load_pseudo_model:
            self.pseudo_models = {}
            # Pseudo Model Encoder and Decoder
            self.pseudo_models["encoder"] = networks.ResnetEncoder(
                self.opt.num_layers, self.opt.weights_init == "pretrained")
            self.pseudo_models["encoder"].to(self.device)

            self.pseudo_models["depth"] = networks.DepthDecoder(
                self.pseudo_models["encoder"].num_ch_enc, self.opt.scales)
            self.pseudo_models["depth"].to(self.device)

            self.load_pseudo_model()

            for m in self.pseudo_models.values():
                m.eval()

        self.l1_loss = torch.nn.L1Loss().cuda()
        self.mse = torch.nn.MSELoss().cuda()
        
        if self.use_pose_net:
            if self.opt.pose_model_type == "separate_resnet":
                self.models["pose_encoder"] = networks.ResnetEncoder(
                    self.opt.num_layers,
                    self.opt.weights_init == "pretrained",
                    num_input_images=self.num_pose_frames)

                self.models["pose_encoder"].to(self.device)
                self.parameters_to_train += list(self.models["pose_encoder"].parameters())

                self.models["pose"] = networks.PoseDecoder(
                    self.models["pose_encoder"].num_ch_enc,
                    num_input_features=1,
                    num_frames_to_predict_for=2)

            self.models["pose"].to(self.device)
            self.parameters_to_train += list(self.models["pose"].parameters())

        self.model_optimizer = optim.Adam(self.parameters_to_train, self.opt.learning_rate)
        self.model_lr_scheduler = optim.lr_scheduler.StepLR(
            self.model_optimizer, self.opt.scheduler_step_size, 0.1)
        
        if self.opt.load_weights_folder is not None:
            self.load_model()

        print("Training model named:\n  ", self.opt.model_name)
        print("Models and tensorboard events files are saved to:\n  ", self.opt.log_dir)
        print("Training is using:\n  ", self.device)

        # Data
        datasets_dict = {"oxford": datasets.OxfordRawDataset, 
                         "oxford_pair": datasets.OxfordRawPairDataset}
        self.dataset = datasets_dict[self.opt.dataset]

        fpath = os.path.join(os.path.dirname(__file__), "splits", self.opt.split, "{}_files.txt")

        train_filenames = readlines(fpath.format("train_day"))
        val_day_filenames = readlines(fpath.format("val_day"))
        val_night_filenames = readlines(fpath.format("val_night"))
        img_ext = '.png' if self.opt.png else '.jpg'

        num_train_samples = len(train_filenames)
        self.num_total_steps = num_train_samples // self.opt.batch_size * self.opt.num_epochs


        if not self.opt.train_day_only:
            self.pair_dataset = datasets_dict["oxford_pair"]
            train_dataset = self.pair_dataset(
                self.opt.data_path, train_filenames, self.opt.height, self.opt.width,
                self.opt.frame_ids, 4, is_train=True, img_ext=img_ext)
            self.train_loader = DataLoader(
                train_dataset, self.opt.batch_size, True,
                num_workers=self.opt.num_workers, pin_memory=True, drop_last=True)
        else:
            train_dataset = self.dataset(
                self.opt.data_path, train_filenames, self.opt.height, self.opt.width,
                self.opt.frame_ids, 4, is_train=True, img_ext=img_ext)
            self.train_loader = DataLoader(
                train_dataset, self.opt.batch_size, True,
                num_workers=self.opt.num_workers, pin_memory=True, drop_last=True)
            
        val_day_dataset = self.dataset(
            self.opt.data_path, val_day_filenames, self.opt.height, self.opt.width,
            [0], 4, is_train=False, img_ext='.png')
        self.val_day_loader = DataLoader(
            val_day_dataset, self.opt.batch_size, False,
            num_workers=self.opt.num_workers, pin_memory=True, drop_last=True)
        
        val_night_dataset = self.dataset(
            self.opt.data_path, val_night_filenames, self.opt.height, self.opt.width,
            [0], 4, is_train=False, img_ext='.png')
        self.val_night_loader = DataLoader(
            val_night_dataset, self.opt.batch_size, False,
            num_workers=self.opt.num_workers, pin_memory=True, drop_last=True)
        
        self.val_day_iter = iter(self.val_day_loader)
        self.val_night_iter = iter(self.val_night_loader)

        self.writers = {}
        for mode in ["train", "val"]:
            self.writers[mode] = SummaryWriter(os.path.join(self.log_path, mode))

        if not self.opt.no_ssim:
            self.ssim = SSIM()
            self.ssim.to(self.device)

        self.backproject_depth = {}
        self.project_3d = {}
        for scale in self.opt.scales:
            h = self.opt.height // (2 ** scale)
            w = self.opt.width // (2 ** scale)

            self.backproject_depth[scale] = BackprojectDepth(self.opt.batch_size, h, w)
            self.backproject_depth[scale].to(self.device)

            self.project_3d[scale] = Project3D(self.opt.batch_size, h, w)
            self.project_3d[scale].to(self.device)

        self.depth_metric_names = [
            "de/abs_rel", "de/sq_rel", "de/rms", "de/log_rms", "da/a1", "da/a2", "da/a3"]

        print("Using split:\n  ", self.opt.split)
        print("There are {:d} training items \n".format(len(train_dataset) * 2))
            
        self.save_opts()

    def set_train(self):
        """Convert all models to training mode
        """
        for m in self.models.values():
            m.train()

    def set_eval(self):
        """Convert all models to testing/evaluation mode
        """
        for m in self.models.values():
            m.eval()

    def train(self):
        """Run the entire training pipeline
        """
        self.epoch = 0
        self.step = 0
        self.start_time = time.time()
        for self.epoch in range(self.opt.num_epochs):
            self.run_epoch()
            if (self.epoch + 1) % self.opt.save_frequency == 0:
                self.save_model()

    def run_epoch(self):
        """Run a single epoch of training and validation
        """
        self.model_lr_scheduler.step()

        print("Training")
        self.set_train()

        for batch_idx, inputs in enumerate(self.train_loader):
            before_op_time = time.time()

            losses = self.process_batch(inputs)

            self.model_optimizer.zero_grad()
            losses["loss"].backward()
            self.model_optimizer.step()

            duration = time.time() - before_op_time

            # log less frequently after the first 2000 steps to save time & disk space
            # early_phase = batch_idx % self.opt.log_frequency == 0 and self.step < 2000
            # late_phase = self.step % 2000 == 0

            # if early_phase or late_phase:
            if batch_idx % 500 == 0:
                self.log_time(batch_idx, duration, losses)

                # if "depth_gt" in day_inputs:
                #     self.compute_depth_losses(day_inputs, outputs, losses)

                self.log("train", losses)
                # self.val()

            self.step += 1

        self.evaluate(day=True)
        self.evaluate(day=False)

    def process_batch(self, inputs):
        """Pass a minibatch through the network and generate images and losses
        """
        for key, ipt in inputs.items():
            inputs[key] = ipt.to(self.device)

        losses = {}
        losses["loss"] = 0

        day_features = self.models["encoder"](inputs["color_aug", 0, 0])
        day_outputs = self.models["depth"](day_features)

        if not self.opt.train_day_only:
            night_features = self.models["encoder"](inputs["color_f_aug", 0, 0])
            night_outputs = self.models["depth"](night_features)

        day_pose_outputs = self.predict_poses(inputs, day_features, is_night=False)
        day_outputs.update(day_pose_outputs)
        if not self.opt.train_day_only:
            if self.opt.use_day_pose:
                for key, value in day_pose_outputs.items():
                    night_outputs[key] = value.detach()
            else:
                night_outputs.update(self.predict_poses(inputs, night_features, is_night=True))
            
        if not self.opt.only_day_reprojection:
            self.generate_images_pred(inputs, day_outputs, is_night=False)
            losses_day = self.compute_losses(inputs, day_outputs, is_night=False)
            if not self.opt.train_day_only:
                self.generate_images_pred(inputs, night_outputs, is_night=True)
                losses_night = self.compute_losses(inputs, night_outputs, is_night=True)
        else:
            self.generate_images_pred(inputs, day_outputs, is_night=False)
            losses_day = self.compute_losses(inputs, day_outputs, is_night=False)
            if not self.opt.train_day_only:
                self.generate_images_pred(inputs, night_outputs, is_night=False)
                losses_night = self.compute_losses(inputs, night_outputs, is_night=False)

        losses["day"] = losses_day["loss"]
        losses["night"] = 0
        losses["depth_sim"] = 0
        losses["feat_sim"] = 0

        if not self.opt.train_day_only:
            losses["night"] = losses_night["loss"]  

            pseudo_label = day_outputs[("disp", 0)].detach()
            losses["depth_sim"] = self.opt.lambda_depth_sim * self.l1_loss(night_outputs[("disp", 0)], pseudo_label)
            losses["feat_sim"] = 0
            for i in range(self.num_scales):
                losses["feat_sim"] += self.l1_loss(day_features[i].detach(), night_features[i])
            losses["feat_sim"] /= self.num_scales
            losses["feat_sim"] *= self.opt.lambda_feat_sim

        losses["loss"] += losses["day"] + losses["night"] + losses["depth_sim"] + losses["feat_sim"]

        return losses
    

    def predict_poses(self, inputs, features, is_night):
        """Predict poses between input frames for monocular sequences.
        """
        outputs = {}
        if self.num_pose_frames == 2:
            # In this setting, we compute the pose to each source frame via a
            # separate forward pass through the pose network.

            # select what features the pose network takes as input
            if self.opt.pose_model_type == "shared":
                pose_feats = {f_i: features[f_i] for f_i in self.opt.frame_ids}
            else:
                if is_night:
                    pose_feats = {f_i: inputs["color_f_aug", f_i, 0] for f_i in self.opt.frame_ids}
                else:
                    pose_feats = {f_i: inputs["color_aug", f_i, 0] for f_i in self.opt.frame_ids}

            for f_i in self.opt.frame_ids[1:]:
                if f_i != "s":
                    # To maintain ordering we always pass frames in temporal order
                    if f_i < 0:
                        pose_inputs = [pose_feats[f_i], pose_feats[0]]
                    else:
                        pose_inputs = [pose_feats[0], pose_feats[f_i]]

                    if self.opt.pose_model_type == "separate_resnet":
                        pose_inputs = [self.models["pose_encoder"](torch.cat(pose_inputs, 1))]
                    elif self.opt.pose_model_type == "posecnn":
                        pose_inputs = torch.cat(pose_inputs, 1)

                    axisangle, translation = self.models["pose"](pose_inputs)
                    outputs[("axisangle", 0, f_i)] = axisangle
                    outputs[("translation", 0, f_i)] = translation

                    # Invert the matrix if the frame id is negative
                    outputs[("cam_T_cam", 0, f_i)] = transformation_from_parameters(
                        axisangle[:, 0], translation[:, 0], invert=(f_i < 0))

        else:
            # Here we input all frames to the pose net (and predict all poses) together
            if self.opt.pose_model_type in ["separate_resnet", "posecnn"]:
                if is_night:
                    pose_inputs = torch.cat(
                        [inputs[("color_f_aug", i, 0)] for i in self.opt.frame_ids if i != "s"], 1)
                else:
                    pose_inputs = torch.cat(
                    [inputs[("color_aug", i, 0)] for i in self.opt.frame_ids if i != "s"], 1)

                if self.opt.pose_model_type == "separate_resnet":
                    pose_inputs = [self.models["pose_encoder"](pose_inputs)]

            elif self.opt.pose_model_type == "shared":
                pose_inputs = [features[i] for i in self.opt.frame_ids if i != "s"]

            axisangle, translation = self.models["pose"](pose_inputs)

            for i, f_i in enumerate(self.opt.frame_ids[1:]):
                if f_i != "s":
                    outputs[("axisangle", 0, f_i)] = axisangle
                    outputs[("translation", 0, f_i)] = translation
                    outputs[("cam_T_cam", 0, f_i)] = transformation_from_parameters(
                        axisangle[:, i], translation[:, i])

        return outputs


    def val(self):
        """Validate the model on a single minibatch
        """
        self.set_eval()
        try:
            # inputs = self.val_iter.next()
            inputs = next(self.val_day_iter)
        except StopIteration:
            self.val_day_iter = iter(self.val_day_loader)
            # inputs = self.val_iter.next()
            inputs = next(self.val_day_iter)

        with torch.no_grad():
            outputs, losses = self.process_batch(inputs)

            if "depth_gt" in inputs:
                self.compute_depth_losses(inputs, outputs, losses)

            self.log("val", inputs, outputs, losses)
            del inputs, outputs, losses

        self.set_train()

    def evaluate(self, day=True):
        """Evaluates a pretrained model using a specified test set
        """
        MIN_DEPTH = 1e-3
        MAX_DEPTH = 80
        EVAL_DEPTH = 40
        self.set_eval()

        assert sum((self.opt.eval_mono, self.opt.eval_stereo)) == 1, \
            "Please choose mono or stereo evaluation by setting either --eval_mono or --eval_stereo"

        pred_disps = []
        gt = []

        print("-> Computing predictions with size {}x{}".format(
            self.opt.width, self.opt.height))

        if day:
            dataloader = self.val_day_loader
            eval_split = 'val_day'
        else:
            dataloader = self.val_night_loader
            eval_split = 'val_night'

        with torch.no_grad():
            for data in dataloader:
                input_color = data[("color", 0, 0)].cuda()

                if self.opt.post_process:
                    # Post-processed results require each image to have two forward passes
                    input_color = torch.cat((input_color, torch.flip(input_color, [3])), 0)

                output = self.models["depth"](self.models["encoder"](input_color))

                pred_disp, _ = disp_to_depth(output[("disp", 0)], self.opt.min_depth, self.opt.max_depth)
                pred_disp = pred_disp.cpu()[:, 0].numpy()

                if self.opt.post_process:
                    N = pred_disp.shape[0] // 2
                    pred_disp = self.batch_post_process_disparity(pred_disp[:N], pred_disp[N:, :, ::-1])

                pred_disps.append(pred_disp)
                gt.append(np.squeeze(data['depth_gt'].cpu().numpy()))

        pred_disps = np.concatenate(pred_disps)
        if gt[-1].ndim == 2:
            gt[-1] = gt[-1][np.newaxis, :]
        gt = np.concatenate(gt)

        if self.opt.save_pred_disps:
            output_path = os.path.join(
                self.opt.load_weights_folder, "disps_{}_split.npy".format(eval_split))
            print("-> Saving predicted disparities to ", output_path)
            np.save(output_path, pred_disps)

        if self.opt.no_eval:
            print("-> Evaluation disabled. Done.")
            quit()

        # gt_path = os.path.join(self.opt.split, eval_split, "gt_depths.npz")
        # gt_depths = np.load(gt_path, fix_imports=True, encoding='latin1')["data"]

        print("-> Evaluating")

        if self.opt.eval_stereo:
            print("   Stereo evaluation - "
                "disabling median scaling, scaling by {}".format(STEREO_SCALE_FACTOR))
            self.opt.disable_median_scaling = True
            self.opt.pred_depth_scale_factor = STEREO_SCALE_FACTOR
        else:
            print("   Mono evaluation - using median scaling")

        errors = []
        ratios = []

        for i in range(pred_disps.shape[0]):
        # for i in range(len(pred_disps)):
            gt_depth = gt[i]
            gt_height, gt_width = gt_depth.shape[:2]

            pred_disp = pred_disps[i]
            pred_disp = cv2.resize(pred_disp, (gt_width, gt_height))
            pred_depth = 1 / pred_disp

            if self.opt.eval_split == "eigen":
                mask = np.logical_and(gt_depth > MIN_DEPTH, gt_depth < MAX_DEPTH)

                crop = np.array([0.40810811 * gt_height, 0.99189189 * gt_height,
                                0.03594771 * gt_width,  0.96405229 * gt_width]).astype(np.int32)
                crop_mask = np.zeros(mask.shape)
                crop_mask[crop[0]:crop[1], crop[2]:crop[3]] = 1
                mask = np.logical_and(mask, crop_mask)

            else:
                mask = gt_depth > 0

            pred_depth = pred_depth[mask]
            gt_depth = gt_depth[mask]

            pred_depth *= self.opt.pred_depth_scale_factor
            if not self.opt.disable_median_scaling:
                ratio = np.median(gt_depth) / np.median(pred_depth)
                ratios.append(ratio)
                pred_depth *= ratio

            pred_depth[pred_depth < MIN_DEPTH] = MIN_DEPTH
            pred_depth[pred_depth > MAX_DEPTH] = MAX_DEPTH
            
            mask2 = gt_depth <= EVAL_DEPTH
            pred_depth = pred_depth[mask2]
            gt_depth = gt_depth[mask2]
            errors.append(self.compute_errors(gt_depth, pred_depth))

        if not self.opt.disable_median_scaling:
            ratios = np.array(ratios)
            med = np.median(ratios)
            print(" Scaling ratios | med: {:0.3f} | std: {:0.3f}".format(med, np.std(ratios / med)))

        mean_errors = np.array(errors).mean(0)

        print("\n  " + ("{:>8} | " * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
        print(("&{: 8.3f}  " * 7).format(*mean_errors.tolist()) + "\\\\")
        print("\n-> Done!")

        with open(self.result_path, 'a') as f:
            for i in range(len(mean_errors)):
                f.write(str(mean_errors[i])) #
                f.write('\t')
            f.write("\n")

        f.close()

        self.set_train()
        return mean_errors

    def batch_post_process_disparity(self, l_disp, r_disp):
        """Apply the disparity post-processing method as introduced in Monodepthv1
        """
        _, h, w = l_disp.shape
        m_disp = 0.5 * (l_disp + r_disp)
        l, _ = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
        l_mask = (1.0 - np.clip(20 * (l - 0.05), 0, 1))[None, ...]
        r_mask = l_mask[:, :, ::-1]
        return r_mask * l_disp + l_mask * r_disp + (1.0 - l_mask - r_mask) * m_disp

    def compute_errors(self, gt, pred):
        """Computation of error metrics between predicted and ground truth depths
        """
        thresh = np.maximum((gt / pred), (pred / gt))
        a1 = (thresh < 1.25).mean()
        a2 = (thresh < 1.25 ** 2).mean()
        a3 = (thresh < 1.25 ** 3).mean()

        rmse = (gt - pred) ** 2
        rmse = np.sqrt(rmse.mean())

        rmse_log = (np.log(gt) - np.log(pred)) ** 2
        rmse_log = np.sqrt(rmse_log.mean())

        abs_rel = np.mean(np.abs(gt - pred) / gt)

        sq_rel = np.mean(((gt - pred) ** 2) / gt)

        return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3

    def generate_images_pred(self, inputs, outputs, is_night):
        """Generate the warped (reprojected) color images for a minibatch.
        Generated images are saved into the `outputs` dictionary.
        """
        for scale in self.opt.scales:
            disp = outputs[("disp", scale)]
            if self.opt.v1_multiscale:
                source_scale = scale
            else:
                disp = F.interpolate(
                    disp, [self.opt.height, self.opt.width], mode="bilinear", align_corners=False)
                source_scale = 0

            _, depth = disp_to_depth(disp, self.opt.min_depth, self.opt.max_depth)

            outputs[("depth", 0, scale)] = depth

            for i, frame_id in enumerate(self.opt.frame_ids[1:]):

                if frame_id == "s":
                    T = inputs["stereo_T"]
                else:
                    T = outputs[("cam_T_cam", 0, frame_id)]

                # from the authors of https://arxiv.org/abs/1712.00175
                if self.opt.pose_model_type == "posecnn":

                    axisangle = outputs[("axisangle", 0, frame_id)]
                    translation = outputs[("translation", 0, frame_id)]

                    inv_depth = 1 / depth
                    mean_inv_depth = inv_depth.mean(3, True).mean(2, True)

                    T = transformation_from_parameters(
                        axisangle[:, 0], translation[:, 0] * mean_inv_depth[:, 0], frame_id < 0)

                cam_points = self.backproject_depth[source_scale](
                    depth, inputs[("inv_K", source_scale)])
                pix_coords = self.project_3d[source_scale](
                    cam_points, inputs[("K", source_scale)], T)

                outputs[("sample", frame_id, scale)] = pix_coords

                if is_night:
                    outputs[("color", frame_id, scale)] = F.grid_sample(
                        inputs[("color_f", frame_id, source_scale)],
                        outputs[("sample", frame_id, scale)],
                        padding_mode="border")
                else:
                    outputs[("color", frame_id, scale)] = F.grid_sample(
                        inputs[("color", frame_id, source_scale)],
                        outputs[("sample", frame_id, scale)],
                        padding_mode="border")

                if not self.opt.disable_automasking:
                    if is_night:
                        outputs[("color_identity", frame_id, scale)] = \
                            inputs[("color_f", frame_id, source_scale)]
                    else:
                        outputs[("color_identity", frame_id, scale)] = \
                            inputs[("color", frame_id, source_scale)]

    def compute_reprojection_loss(self, pred, target):
        """Computes reprojection loss between a batch of predicted and target images
        """
        abs_diff = torch.abs(target - pred)
        l1_loss = abs_diff.mean(1, True)

        if self.opt.no_ssim:
            reprojection_loss = l1_loss
        else:
            ssim_loss = self.ssim(pred, target).mean(1, True)
            reprojection_loss = 0.85 * ssim_loss + 0.15 * l1_loss

        return reprojection_loss

    def compute_losses(self, inputs, outputs, is_night):
        """Compute the reprojection and smoothness losses for a minibatch
        """
        losses = {}
        total_loss = 0

        for scale in self.opt.scales:
            loss = 0
            reprojection_losses = []

            if self.opt.v1_multiscale:
                source_scale = scale
            else:
                source_scale = 0

            disp = outputs[("disp", scale)]
            if is_night:
                color = inputs[("color_f", 0, scale)]
                target = inputs[("color_f", 0, source_scale)]
            else:
                color = inputs[("color", 0, scale)]
                target = inputs[("color", 0, source_scale)]

            for frame_id in self.opt.frame_ids[1:]:
                pred = outputs[("color", frame_id, scale)]
                reprojection_losses.append(self.compute_reprojection_loss(pred, target))

            reprojection_losses = torch.cat(reprojection_losses, 1)

            if not self.opt.disable_automasking:
                identity_reprojection_losses = []
                for frame_id in self.opt.frame_ids[1:]:
                    if is_night:
                        pred = inputs[("color_f", frame_id, source_scale)]
                    else:
                        pred = inputs[("color", frame_id, source_scale)]
                    identity_reprojection_losses.append(
                        self.compute_reprojection_loss(pred, target))

                identity_reprojection_losses = torch.cat(identity_reprojection_losses, 1)

                if self.opt.avg_reprojection:
                    identity_reprojection_loss = identity_reprojection_losses.mean(1, keepdim=True)
                else:
                    # save both images, and do min all at once below
                    identity_reprojection_loss = identity_reprojection_losses

            elif self.opt.predictive_mask:
                # use the predicted mask
                mask = outputs["predictive_mask"]["disp", scale]
                if not self.opt.v1_multiscale:
                    mask = F.interpolate(
                        mask, [self.opt.height, self.opt.width],
                        mode="bilinear", align_corners=False)

                reprojection_losses *= mask

                # add a loss pushing mask to 1 (using nn.BCELoss for stability)
                weighting_loss = 0.2 * nn.BCELoss()(mask, torch.ones(mask.shape).cuda())
                loss += weighting_loss.mean()

            if self.opt.avg_reprojection:
                reprojection_loss = reprojection_losses.mean(1, keepdim=True)
            else:
                reprojection_loss = reprojection_losses

            if not self.opt.disable_automasking:
                # add random numbers to break ties
                identity_reprojection_loss += torch.randn(
                    identity_reprojection_loss.shape).cuda() * 0.00001

                combined = torch.cat((identity_reprojection_loss, reprojection_loss), dim=1)
            else:
                combined = reprojection_loss

            if combined.shape[1] == 1:
                to_optimise = combined
            else:
                to_optimise, idxs = torch.min(combined, dim=1)

            if not self.opt.disable_automasking:
                outputs["identity_selection/{}".format(scale)] = (
                    idxs > identity_reprojection_loss.shape[1] - 1).float()

            loss += to_optimise.mean()

            mean_disp = disp.mean(2, True).mean(3, True)
            norm_disp = disp / (mean_disp + 1e-7)
            smooth_loss = get_smooth_loss(norm_disp, color)

            loss += self.opt.disparity_smoothness * smooth_loss / (2 ** scale)
            total_loss += loss
            losses["loss/{}".format(scale)] = loss

        total_loss /= self.num_scales
        losses["loss"] = total_loss
        return losses

    def compute_depth_losses(self, inputs, outputs, losses):
        """Compute depth metrics, to allow monitoring during training

        This isn't particularly accurate as it averages over the entire batch,
        so is only used to give an indication of validation performance
        """
        depth_pred = outputs[("depth", 0, 0)]
        depth_pred = torch.clamp(F.interpolate(
            depth_pred, [375, 1242], mode="bilinear", align_corners=False), 1e-3, 80)
        depth_pred = depth_pred.detach()

        depth_gt = inputs["depth_gt"]
        mask = depth_gt > 0

        # garg/eigen crop
        crop_mask = torch.zeros_like(mask)
        crop_mask[:, :, 153:371, 44:1197] = 1
        mask = mask * crop_mask

        depth_gt = depth_gt[mask]
        depth_pred = depth_pred[mask]
        depth_pred *= torch.median(depth_gt) / torch.median(depth_pred)

        depth_pred = torch.clamp(depth_pred, min=1e-3, max=80)

        depth_errors = compute_depth_errors(depth_gt, depth_pred)

        for i, metric in enumerate(self.depth_metric_names):
            losses[metric] = np.array(depth_errors[i].cpu())

    def log_time(self, batch_idx, duration, losses):
        """Print a logging statement to the terminal
        """
        samples_per_sec = self.opt.batch_size / duration
        time_sofar = time.time() - self.start_time
        training_time_left = (
            self.num_total_steps / self.step - 1.0) * time_sofar if self.step > 0 else 0
        print_string = "epoch {:>3} | batch {:>6} | examples/s: {:5.1f}" + \
            " | loss: {:.5f} | loss day: {:.5f} | loss night: {:.5f}" + \
                " | loss depth sim: {:.5f} | loss feat sim: {:.5f} |  time elapsed: {} | time left: {}"
        loss = losses["loss"].cpu().data
        loss_day = losses["day"].cpu().data
        loss_night = 0
        loss_depth_sim = 0
        loss_feat_sim = 0
    
        if not self.opt.train_day_only:
            loss_night = losses["night"].cpu().data
            loss_depth_sim = losses["depth_sim"].cpu().data
            loss_feat_sim = losses["feat_sim"].cpu().data
        
        print(print_string.format(self.epoch, batch_idx, samples_per_sec, loss, \
                                loss_day, loss_night, loss_depth_sim, loss_feat_sim, \
                                sec_to_hm_str(time_sofar), sec_to_hm_str(training_time_left)))
            

    def log(self, mode, losses):
        """Write an event to the tensorboard events file
        """
        writer = self.writers[mode]
        for l, v in losses.items():
            writer.add_scalar("{}".format(l), v, self.step)

    def save_opts(self):
        """Save options to disk so we know what we ran this experiment with
        """
        models_dir = os.path.join(self.log_path, "models")
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        to_save = self.opt.__dict__.copy()

        with open(os.path.join(models_dir, 'opt.json'), 'w') as f:
            json.dump(to_save, f, indent=2)

    def save_model(self):
        """Save model weights to disk
        """
        save_folder = os.path.join(self.log_path, "models", "weights_{}".format(self.epoch))
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        for model_name, model in self.models.items():
            save_path = os.path.join(save_folder, "{}.pth".format(model_name))
            to_save = model.state_dict()
            if model_name == 'encoder':
                # save the sizes - these are needed at prediction time
                to_save['height'] = self.opt.height
                to_save['width'] = self.opt.width
                to_save['use_stereo'] = self.opt.use_stereo
            torch.save(to_save, save_path)

        save_path = os.path.join(save_folder, "{}.pth".format("adam"))
        torch.save(self.model_optimizer.state_dict(), save_path)

    def load_model(self):
        """Load model(s) from disk
        """
        self.opt.load_weights_folder = os.path.expanduser(self.opt.load_weights_folder)

        assert os.path.isdir(self.opt.load_weights_folder), \
            "Cannot find folder {}".format(self.opt.load_weights_folder)
        print("loading model from folder {}".format(self.opt.load_weights_folder))

        for n in self.opt.models_to_load:
            print("Loading {} weights...".format(n))
            path = os.path.join(self.opt.load_weights_folder, "{}.pth".format(n))
            model_dict = self.models[n].state_dict()
            pretrained_dict = torch.load(path)
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            self.models[n].load_state_dict(model_dict)

        # # loading adam state
        # optimizer_load_path = os.path.join(self.opt.load_weights_folder, "adam.pth")
        # if os.path.isfile(optimizer_load_path):
        #     print("Loading Adam weights")
        #     optimizer_dict = torch.load(optimizer_load_path)
        #     self.model_optimizer.load_state_dict(optimizer_dict)
        # else:
        #     print("Cannot find Adam weights so Adam is randomly initialized")
    
    def load_pseudo_model(self):
        """Load model(s) from disk
        """
        self.opt.load_pseudo_model = os.path.expanduser(self.opt.load_pseudo_model)

        assert os.path.isdir(self.opt.load_pseudo_model), \
            "Cannot find folder {}".format(self.opt.load_pseudo_model)
        print("loading model from folder {}".format(self.opt.load_pseudo_model))

        for n in ["encoder", "depth"]:
            print("Loading {} weights...".format(n))
            path = os.path.join(self.opt.load_pseudo_model, "{}.pth".format(n))
            model_dict = self.pseudo_models[n].state_dict()
            pretrained_dict = torch.load(path)
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            self.pseudo_models[n].load_state_dict(model_dict)
