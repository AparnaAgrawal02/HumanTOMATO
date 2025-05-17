import numpy as np
import torch

from tma.data.humanml.scripts.motion_process import process_file, recover_from_ric

from .base import BASEDataModule
from .humanml.data.dataset import (
    Text2MotionDatasetASL,
    Text2MotionDatasetMotionX_text_all,
)


class ASL_DataModule(BASEDataModule):

    def __init__(
        self, cfg, batch_size, num_workers, collate_fn=None, phase="train", **kwargs
    ):
        super().__init__(
            batch_size=batch_size, num_workers=num_workers, collate_fn=collate_fn
        )
        self.save_hyperparameters(logger=False)
        self.name = "asl"
        if cfg.DATASET.JOINT_TYPE == "humanml3d":
            self.njoints = 22
        elif cfg.DATASET.JOINT_TYPE == "motionx" or "asl":
            self.njoints = 52
        else:
            raise NotImplemented

        if phase == "text_only":
            self.Dataset = TextOnlyDataset
        else:
            if cfg.model.condition in [
                "text_all",
                "text_face",
                "text_body",
                "text_hand",
                "text_face_body",
                "text_seperate",
                "only_pose_concat",
                "only_pose_fusion",
            ]:
                self.Dataset = Text2MotionDatasetMotionX_text_all
            else:
                self.Dataset = Text2MotionDatasetASL

        self.cfg = cfg
        sample_overrides = {"split": "val", "tiny": True, "progress_bar": False}
        self._sample_set = self.get_sample_set(overrides=sample_overrides)
        
        # Get additional info of the dataset
        self.nfeats = self._sample_set.nfeats
        # self.transforms = self._sample_set.transforms

    def feats2joints(self, features, motion_type, smplx_model=None):
        # import pdb; pdb.set_trace()
        if motion_type in ["vector_263", "vector_623"]:
            mean = torch.tensor(self.hparams.mean).to(features)
            std = torch.tensor(self.hparams.std).to(features)
            features = features * std + mean
            
            return recover_from_ric(
                features, self.njoints
            )  # torch.Size([32, 92, 22, 3])
        elif motion_type == "smplx_212":
            assert smplx_model is not None
            mean = torch.tensor(self.hparams.mean).to(features)
            std = torch.tensor(self.hparams.std).to(features)
            features = features * (std + 1e-7) + mean
            bs = features.shape[0]
            features = features.reshape(-1, 212)
            output = smplx_model.smplx_model(
                pose_body=features[:, 3:66],
                pose_hand=features[:, 66:156],
                root_orient=features[:, :3],
            ).Jtr
            return output.reshape(bs, -1, 55, 3)  # torch.Size([32, 96, 55, 3])
        elif motion_type == "smplx_322":
            model_path = "/scratch/aparna/human_model_files/"
            mean = torch.tensor(self.hparams.mean).to(features)
            std = torch.tensor(self.hparams.std).to(features)
            features = features * (std + 1e-7) + mean
            bs = features.shape[0]
            features = features.reshape(-1, 322)

            """Convert [T, 322] axis-angle to [T, 55, 3] joint positions"""
            pose_tensor = torch.tensor(features, dtype=torch.float32).to('cuda')
            smplx_model = load_smplx_model(model_path=model_path, gender='neutral',device='cuda',batch_size=pose_tensor.shape[0])
            smplx_model.eval()
            assert smplx_model is not None
            pose_tensor = pose_tensor.view(-1, 322)
            global_orient = pose_tensor[:, :3]
            body_pose = pose_tensor[:, 3:66]
            left_hand_pose = pose_tensor[:, 66:111]
            right_hand_pose = pose_tensor[:, 111:156]
            jaw_pose = pose_tensor[:, 156:159]
            expression = pose_tensor[:, 159:209]
            trans = pose_tensor[:, 309:312]
            betas = pose_tensor[:, 312:322]
            
            output = output = smplx_model(
                    global_orient=global_orient,
                    body_pose=body_pose,
                    left_hand_pose=left_hand_pose,
                    right_hand_pose=right_hand_pose,
                    jaw_pose=jaw_pose,
                    expression=expression,
                    betas=betas,
                    transl=trans,
                    
                )
            del smplx_model
            return output.joints.detach().cpu().numpy().reshape(bs, -1, 144, 3)

        else:
            raise NotImplementedError

    def joints2feats(self, features):
        features = process_file(features, self.njoints)[0]
        # mean = torch.tensor(self.hparams.mean).to(features)
        # std = torch.tensor(self.hparams.std).to(features)
        # features = (features - mean) / std
        return features

    def renorm4t2m(self, features):
        # renorm to t2m norms for using t2m evaluators
        ori_mean = torch.tensor(self.hparams.mean).to(features)
        ori_std = torch.tensor(self.hparams.std).to(features)
        eval_mean = torch.tensor(self.hparams.mean_eval).to(features)
        eval_std = torch.tensor(self.hparams.std_eval).to(features)
        features = features * (ori_std + 1e-7) + ori_mean
        features = (features - eval_mean) / (eval_std + 1e-7)
        return features

    def renormt2m_back(self, features):
        eval_mean = torch.tensor(self.hparams.mean_eval).to(features)
        eval_std = torch.tensor(self.hparams.std_eval).to(features)
        features = features * (eval_std + 1e-7) + eval_mean
        return features

    def mm_mode(self, mm_on=True):
        # random select samples for mm
        if mm_on:
            self.is_mm = True
            self.name_list = self.test_dataset.name_list
            self.mm_list = np.random.choice(
                self.name_list, self.cfg.TEST.MM_NUM_SAMPLES, replace=False
            )
            self.test_dataset.name_list = self.mm_list
        else:
            self.is_mm = False
            self.test_dataset.name_list = self.name_list
