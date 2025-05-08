import os.path as osp
import os
import datetime
import time
from collections import OrderedDict
from einops import rearrange

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from dassl.engine import TRAINER_REGISTRY, TrainerXU
from dassl.metrics import compute_accuracy
from dassl.utils import MetricMeter, AverageMeter, load_pretrained_weights, load_checkpoint, save_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

_tokenizer = _Tokenizer()


def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url, cfg.MODEL.BACKBONE.PATH)

    try:
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict())

    return model


class AdaIN(nn.Module):
    def __init__(self):
        super().__init__()

    def mu(self, x):
        return torch.sum(x, (1)) / (x.shape[1])

    def sigma(self, x):
        return torch.sqrt(
            (torch.sum((x.permute([1, 0, 2]) - self.mu(x)).permute([1, 0, 2]) ** 2, (1)) + 0.000000023) / (x.shape[1]))


class domain_projector(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.ModuleList(nn.Linear(768, 256) for _ in range(12))
        self.linear2 = nn.ModuleList(nn.Linear(256, 512) for _ in range(12))
        self.adain = AdaIN()
        self.gap = nn.AdaptiveAvgPool2d((1, 768))

    def forward(self, data):
        data_prompt = []
        for i in range(len(data)):
            x_mu = self.adain.mu(data[i]).unsqueeze(1).to(torch.float32)
            x_sigma = self.adain.sigma(data[i]).unsqueeze(1).to(torch.float32)
            x_cat = torch.cat((x_mu, x_sigma), 1)
            x_cat = self.gap(x_cat).squeeze(1)
            x_out = self.linear1[i](x_cat)
            x_final = self.linear2[i](x_out)
            data_prompt.append(x_final)
        output = torch.stack(data_prompt, dim=1)

        return output


class image_projector(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.ModuleList(nn.Linear(768, 512) for _ in range(12))
        self.adain = AdaIN()
        self.lin = nn.Linear(12, 1)
        self.gap = nn.AdaptiveAvgPool2d((1, 768))

    def forward(self, data, n_imgctx):
        data_prompt = []
        for i in range(len(data)):
            x_gap = self.gap(data[i]).squeeze(1)
            x_lin = self.linear[i](x_gap)
            data_prompt.append(x_lin)
        feat = torch.stack(data_prompt, dim=1)
        output = []
        for i in range(n_imgctx):
            x = self.lin(feat.permute(0, 2, 1))
            x = x.permute(0, 2, 1)
            output.append(x)
        feat_tokens = torch.stack(output, dim=1).squeeze(2)
        return feat_tokens


class style_mapping_projector(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.ModuleList(nn.Linear(768, 384) for _ in range(12))
        self.linear2 = nn.ModuleList(nn.Linear(384, 512) for _ in range(12))
        self.adain = AdaIN()
        self.relu = nn.ReLU()
        self.gap = nn.AdaptiveAvgPool1d((768))

    def forward(self, data):
        data_prompt = []
        for i in range(len(data)):
            x_mu = self.adain.mu(data[i]).to(torch.float32)
            x_sigma = self.adain.sigma(data[i]).to(torch.float32)
            x_cat = torch.cat((x_mu, x_sigma), 1)
            x_gap = self.gap(x_cat)
            x_out = self.linear1[i](x_gap)
            x_relu = self.relu(x_out)
            x_final = self.linear2[i](x_relu)
            data_prompt.append(x_final)
        output = torch.stack(data_prompt, dim=1)
        return output


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    @autocast()
    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x[0].permute(1, 0, 2)
        x = self.ln_final(x).type(self.dtype)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
        return x


class PromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_imgctx = 4
        n_ctx = 24 + n_imgctx

        dtype = clip_model.dtype
        clip_imsize = clip_model.visual.input_resolution  # 224
        cfg_imsize = cfg.INPUT.SIZE[0]  # 224
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        self.domain_tokens = domain_projector()
        self.image_tokens = image_projector()
        self.style_mapping_tokens = style_mapping_projector()

        prompt_prefix = " ".join(["X"] * n_ctx)  # 'X X X X X X X X X X X X X X X X X X X X X X X X X X X X'
        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.K = 5
        self.dim = clip_model.text_projection.shape[1]  # 512
        self.n_imgctx = n_imgctx
        self.tokenized_prompts = tokenized_prompts
        self.name_lens = name_lens

    def construct_prompts(self, ctx, prefix, suffix, label=None):
        # dim0 is either batch_size (during training) or n_cls (during testing)
        # ctx: context tokens, with shape of (dim0, n_ctx, ctx_dim)
        # prefix: the sos token, with shape of (n_cls, 1, ctx_dim)
        # suffix: remaining tokens, with shape of (n_cls, *, ctx_dim)

        if label is not None:
            prefix = prefix[label]
            suffix = suffix[label]
        prompts = torch.cat(
            [
                prefix,
                ctx,
                suffix,
            ],
            dim=1,
        )

        return prompts

    @autocast()
    def forward(self, data):
        prefix = self.token_prefix
        suffix = self.token_suffix
        n_imgctx = self.n_imgctx

        domaintokens = self.domain_tokens(data)
        imagetokens = self.image_tokens(data, n_imgctx)

        tokens = torch.cat((domaintokens, domaintokens, imagetokens), dim=1)

        prompts = []
        for tokens_i in tokens:
            ctx_i = tokens_i.unsqueeze(0).expand(self.n_cls, -1, -1)
            pts_i = self.construct_prompts(ctx_i, prefix, suffix)
            prompts.append(pts_i)
        prompts = torch.stack(prompts)

        return prompts, domaintokens


class INTER_Module(nn.Module):
    """ IFT """

    def __init__(self, clip_model, beta_s=1.0, beta_t=1.0,
                 ):
        super().__init__()

        self.softmax = nn.Softmax(-1)
        input_dim = clip_model.text_projection.shape[1]
        pre_dim1 = input_dim // 8
        pre_dim2 = input_dim // 8

        self.beta_s = beta_s
        self.beta_t = beta_t
        self.scale = 0.1

        self.pre_project = nn.Sequential(
            nn.Linear(input_dim, pre_dim1),
            nn.BatchNorm1d(pre_dim1),
            nn.ReLU(inplace=True),

            nn.Linear(pre_dim1, pre_dim2),
            nn.BatchNorm1d(pre_dim2),
            nn.ReLU(inplace=True),

            nn.Linear(pre_dim2, input_dim * 3)
        ).half()

        self.post_project = nn.Sequential(
            nn.Linear(input_dim, input_dim)
        ).half()

        self.logit_scale = clip_model.logit_scale

    def forward(self, Fv, Fvs_bank, Fvt_bank):
        out_fv = self.pre_project(Fv)
        out_fvs = self.pre_project(Fvs_bank)
        out_fvt = self.pre_project(Fvt_bank)

        q_fv, k_fv, v_fv = tuple(rearrange(out_fv, 'b (d k) -> k b d ', k=3))
        q_fvs, k_fvs, v_fvs = tuple(rearrange(out_fvs, 'b (d k) -> k b d ', k=3))
        q_fvt, k_fvt, v_fvt = tuple(rearrange(out_fvt, 'b (d k) -> k b d ', k=3))

        As = self.softmax(self.scale * q_fv @ k_fvs.permute(1, 0))
        At = self.softmax(self.scale * q_fv @ k_fvt.permute(1, 0))

        Fsa = Fv + self.post_project(As @ v_fvs)
        Fta = Fv + self.post_project(At @ v_fvt)

        Fsa = Fsa / Fsa.norm(dim=-1, keepdim=True)
        Fta = Fta / Fta.norm(dim=-1, keepdim=True)

        return Fsa, Fta


class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.prompt_learner = PromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.attn_block = INTER_Module(clip_model, beta_s=0.1, beta_t=0.1)
        self.n_cls = self.prompt_learner.n_cls
        self.K = 5
        self.dim = clip_model.text_projection.shape[1]  # 512
        self.source_key_dict = {i: i for i in range(self.n_cls * self.K)}
        self.target_key_dict = {i: i for i in range(self.n_cls * self.K)}
        self.source_max_probs_list = [0.0 for i in range(self.n_cls * self.K)]
        self.target_max_probs_list = [0.0 for i in range(self.n_cls * self.K)]
        source_feat_bank = torch.zeros((self.n_cls * self.K, self.dim))
        target_feat_bank = torch.zeros((self.n_cls * self.K, self.dim))
        self.source_feat_bank = nn.Parameter(source_feat_bank)
        self.target_feat_bank = nn.Parameter(target_feat_bank)

    @autocast()
    def forward(self, s_image, t_image=None, label=None, domain=None):
        if t_image == None:
            image = s_image
            image_features, data = self.image_encoder(image.type(self.dtype))  # B,512  12,B,197,768
            prompts, domaintokens = self.prompt_learner(data)
            tokenized_prompts = self.tokenized_prompts  # 65,77
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)  # B,512
            logit_scale = self.logit_scale.exp()  # 100

            text_features = []
            for pts_i in prompts:  # 65,77,512
                tf = self.text_encoder(pts_i, tokenized_prompts)  # 65,512
                text_features.append(tf)
            text_features = torch.stack(text_features)  # B,65,512
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)  # B,65,512
            logits = []
            for txt, im in zip(text_features, image_features):  # 65,512   512
                l_i = logit_scale * im @ txt.t()  # 512  *   512,65   =  65
                logits.append(l_i)
            logits = torch.stack(logits)  # B,65
            if label == None:
                return logits  # B,65
            else:
                logits, feature = logits.detach(), image_features.detach()
                pseudo_label = torch.softmax(logits, dim=-1)
                max_probs, label_p = torch.max(pseudo_label, dim=-1)
                if domain == 'source':
                    for i, l in enumerate(label):
                        if l == label_p[i]:
                            index = l.item() * self.K
                            l_list = self.source_max_probs_list[index: index + self.K]
                            if max_probs[i] > min(l_list):
                                min_index = l_list.index(min(l_list))
                                self.source_max_probs_list[index + min_index] = max_probs[i]
                                self.source_feat_bank[index + min_index] = feature[i]
                                self.source_key_dict[index + min_index] = label_p[i]
                elif domain == 'target':
                    for i, l in enumerate(label_p):
                        index = l.item() * self.K
                        l_list = self.target_max_probs_list[index: index + self.K]
                        if max_probs[i] > min(l_list):
                            min_index = l_list.index(min(l_list))
                            self.target_max_probs_list[index + min_index] = max_probs[i]
                            self.target_feat_bank[index + min_index] = feature[i]
                            self.target_key_dict[index + min_index] = label_p[i]

        else:
            source_image_features, source_data = self.image_encoder(s_image.type(self.dtype))
            target_image_features, target_data = self.image_encoder(t_image.type(self.dtype))

            source_prompts, source_domaintokens = self.prompt_learner(source_data)
            target_prompts, target_domaintokens = self.prompt_learner(target_data)

            tokenized_prompts = self.tokenized_prompts

            source_image_features = source_image_features / source_image_features.norm(dim=-1, keepdim=True)
            target_image_features = target_image_features / target_image_features.norm(dim=-1, keepdim=True)
            logit_scale = self.logit_scale.exp()

            source_text_features = []
            for pts_i in source_prompts:
                tf = self.text_encoder(pts_i, tokenized_prompts)
                source_text_features.append(tf)
            source_text_features = torch.stack(source_text_features)
            source_text_features = source_text_features / source_text_features.norm(dim=-1, keepdim=True)

            target_text_features = []
            for pts_i in target_prompts:
                tf = self.text_encoder(pts_i, tokenized_prompts)
                target_text_features.append(tf)
            target_text_features = torch.stack(target_text_features)
            target_text_features = target_text_features / target_text_features.norm(dim=-1, keepdim=True)

            source_logits = []
            for txt, im in zip(source_text_features, source_image_features):
                l_i = logit_scale * im @ txt.t()
                source_logits.append(l_i)
            source_logits = torch.stack(source_logits)

            target_logits = []
            for txt, im in zip(target_text_features, target_image_features):
                l_i = logit_scale * im @ txt.t()
                target_logits.append(l_i)
            target_logits = torch.stack(target_logits)

            source_bank = torch.mean(self.source_feat_bank.reshape(self.n_cls, self.K, self.dim), dim=1)
            target_bank = torch.mean(self.target_feat_bank.reshape(self.n_cls, self.K, self.dim), dim=1)
            inter_s_image_features, inter_t_image_features = self.attn_block(source_image_features, source_bank,
                                                                             target_bank)

            inter_s_logits = []
            for txt, im in zip(source_text_features, inter_s_image_features):
                l_i = logit_scale * im @ txt.t()
                inter_s_logits.append(l_i)
            inter_s_logits = torch.stack(inter_s_logits)

            inter_t_logits = []
            for txt, im in zip(target_text_features, inter_t_image_features):
                l_i = logit_scale * im @ txt.t()
                inter_t_logits.append(l_i)
            inter_t_logits = torch.stack(inter_t_logits)

            return source_logits, target_logits, inter_s_logits, inter_t_logits, source_domaintokens, target_domaintokens, source_text_features, target_text_features


class entropy_loss(nn.Module):
    def __init__(self):
        super(entropy_loss, self).__init__()

    def forward(self, target_prob):
        full_enp = torch.zeros(target_prob.shape[0])
        target_prob = nn.functional.normalize(target_prob, dim=0)

        for i in range(len(target_prob)):
            total_en = 0
            for j in range(target_prob.shape[1]):
                total_en = total_en - target_prob[i][j] * torch.log(target_prob[i][j] + 1e-8)
            full_enp[i] = total_en
        avg_full_enp = torch.mean(full_enp)
        return avg_full_enp


@TRAINER_REGISTRY.register()
class IPCLIPB16(TrainerXU):
    def check_cfg(self, cfg):
        assert cfg.TRAINER.IPCLIPB16.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames
        print('******************************************')
        # print('classnames', classnames)
        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)
        self.dim = clip_model.text_projection.shape[1]  # 512

        if cfg.TRAINER.IPCLIPB16.PREC == "fp32" or cfg.TRAINER.IPCLIPB16V.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model.float()

        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, classnames, clip_model)

        self.n_cls = self.model.prompt_learner.n_cls

        name_to_update = "prompt_learner"

        # 冻结prompt_learner以外的层
        for name, param in self.model.named_parameters():
            if name_to_update not in name:
                param.requires_grad_(False)

        # Double check
        enabled = set()
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                enabled.add(name)
        print(f"Parameters to be updated: {enabled}")

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model.prompt_learner,
                                    cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)

        # transform the epoch to step schedule
        len_train_loader_x = len(self.train_loader_x)
        len_train_loader_u = len(self.train_loader_u)
        if self.cfg.TRAIN.COUNT_ITER == "train_x":
            self.num_batches = len_train_loader_x
        elif self.cfg.TRAIN.COUNT_ITER == "train_u":
            self.num_batches = len_train_loader_u
        elif self.cfg.TRAIN.COUNT_ITER == "smaller_one":
            self.num_batches = min(len_train_loader_x, len_train_loader_u)
        else:
            raise ValueError

        # NOTE: only give prompt_learner to the optimizer
        self.optim = build_optimizer(self.model.prompt_learner, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        '''
        register model could be updated. When new module needs to be updated
        register the module before use
        '''
        self.register_model("prompt_learner", self.model.prompt_learner,
                            self.optim, self.sched)

        self.scaler = GradScaler() if cfg.TRAINER.IPCLIPB16.PREC == "amp" else None  # 自动混合精度训练（Automatic Mixed Precision, AMP）
        self.construct_bank()

        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)

    def construct_bank(self):
        self.set_model_mode("eval")

        print("Constructing source feature bank...")
        data_loader_x = self.train_loader_x
        for batch_idx, batch in enumerate(data_loader_x):
            input, label = self.parse_batch_test(batch)
            self.model(input, label=label, domain='source')  # B,65
            if min(self.model.source_max_probs_list) > 0.99:
                break

        print("Constructing target feature bank...")
        data_loader_u = self.train_loader_u
        for batch_idx, batch in enumerate(data_loader_u):
            input, label = self.parse_batch_test(batch)
            self.model(input, label=label, domain='target')  # B,65
            if min(self.model.target_max_probs_list) > 0.99:
                break

        print('Feature banks are completed!')

    def save_model(self, epoch, directory, is_best=False, model_name=""):
        names = self.get_model_names()
        for name in names:
            model_dict = self._models[name].state_dict()

            optim_dict = None
            if self._optims[name] is not None:
                optim_dict = self._optims[name].state_dict()

            sched_dict = None
            if self._scheds[name] is not None:
                sched_dict = self._scheds[name].state_dict()

            save_checkpoint(
                {
                    "state_dict": model_dict,
                    "epoch": epoch + 1,
                    "optimizer": optim_dict,
                    "scheduler": sched_dict,
                },
                osp.join(directory, name),
                is_best=is_best,
                model_name=model_name,
            )

    def train(self):
        """Generic training loops."""
        self.before_train()
        for self.epoch in range(self.start_epoch, self.max_epoch):
            self.before_epoch()
            self.run_epoch()
            self.after_epoch()
        self.after_train()

    def run_epoch(self):  # print("take this run_epoch!!!!!!!!!!!!!!!!!!!")
        self.set_model_mode("train")
        losses = MetricMeter()
        batch_time = AverageMeter()
        data_time = AverageMeter()

        # Decide to iterate over labeled or unlabeled dataset
        len_train_loader_x = len(self.train_loader_x)
        len_train_loader_u = len(self.train_loader_u)
        if self.cfg.TRAIN.COUNT_ITER == "train_x":
            self.num_batches = len_train_loader_x
        elif self.cfg.TRAIN.COUNT_ITER == "train_u":
            self.num_batches = len_train_loader_u
        elif self.cfg.TRAIN.COUNT_ITER == "smaller_one":
            self.num_batches = min(len_train_loader_x, len_train_loader_u)
        else:
            raise ValueError

        train_loader_x_iter = iter(self.train_loader_x)
        train_loader_u_iter = iter(self.train_loader_u)

        end = time.time()
        for self.batch_idx in range(self.num_batches):
            try:
                batch_x = next(train_loader_x_iter)
            except StopIteration:
                train_loader_x_iter = iter(self.train_loader_x)
                batch_x = next(train_loader_x_iter)

            try:
                batch_u = next(train_loader_u_iter)
            except StopIteration:
                train_loader_u_iter = iter(self.train_loader_u)
                batch_u = next(train_loader_u_iter)

            data_time.update(time.time() - end)
            loss_summary = self.forward_backward(batch_x, batch_u)
            batch_time.update(time.time() - end)
            losses.update(loss_summary)

            if (
                    self.batch_idx + 1
            ) % self.cfg.TRAIN.PRINT_FREQ == 0 or self.num_batches < self.cfg.TRAIN.PRINT_FREQ:
                nb_remain = 0
                nb_remain += self.num_batches - self.batch_idx - 1
                nb_remain += (self.max_epoch - self.epoch -
                              1) * self.num_batches
                eta_seconds = batch_time.avg * nb_remain
                eta = str(datetime.timedelta(seconds=int(eta_seconds)))
                print("epoch [{0}/{1}][{2}/{3}]\t"
                      "time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                      "data {data_time.val:.3f} ({data_time.avg:.3f})\t"
                      "eta {eta}\t"
                      "{losses}\t"
                      "lr {lr:.6e}".format(
                    self.epoch + 1,
                    self.max_epoch,
                    self.batch_idx + 1,
                    self.num_batches,
                    batch_time=batch_time,
                    data_time=data_time,
                    eta=eta,
                    losses=losses,
                    lr=self.get_current_lr(),
                ))

            n_iter = self.epoch * self.num_batches + self.batch_idx
            for name, meter in losses.meters.items():
                self.write_scalar("train/" + name, meter.avg, n_iter)
            self.write_scalar("train/lr", self.get_current_lr(), n_iter)

            end = time.time()

    def forward_backward(self, batch_s, batch_t):
        # B,3,224,224
        self.entropy = entropy_loss()
        kl_loss = nn.KLDivLoss(reduction="batchmean")
        image_s, label_s, image_t, label_t = self.parse_batch_train(batch_s, batch_t)

        prec = self.cfg.TRAINER.IPCLIPB16.PREC
        if prec == "amp":
            with autocast():
                source_logits, target_logits, inter_s_logits, inter_t_logits, source_domaintokens, target_domaintokens, source_text_features, target_text_features = self.model(
                    image_s, image_t)
                loss_ce_s = F.cross_entropy(source_logits, label_s)
                loss_ce_is = F.cross_entropy(inter_s_logits, label_s)
                loss_ce_it = F.cross_entropy(inter_t_logits, label_s)
                loss_ce_t = F.cross_entropy(target_logits, label_t)
                source_textfeat = F.log_softmax(source_text_features, dim=1)
                target_textfeat = F.softmax(target_text_features, dim=1)
                loss_kl = kl_loss(source_textfeat, target_textfeat)
                loss_smn = F.mse_loss(source_domaintokens, target_domaintokens)
                target_probs = torch.nn.functional.softmax(target_logits, dim=1)  # B,65
                loss_entropy = self.entropy(target_probs)

                up_bounder = 5
                if loss_ce_t > up_bounder:
                    loss_ce_t = torch.clamp(loss_ce_t, 0, up_bounder)
                if loss_ce_is > up_bounder:
                    loss_ce_is = torch.clamp(loss_ce_is, 0, up_bounder)
                if loss_ce_it > up_bounder:
                    loss_ce_it = torch.clamp(loss_ce_it, 0, up_bounder)
                if loss_smn > up_bounder:
                    loss_smn = torch.clamp(loss_smn, 0, up_bounder)
                loss = loss_ce_s - loss_ce_t - loss_ce_is - loss_ce_it - 0.1 * loss_smn + 0.01 * loss_entropy - loss_kl
            self.optim.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optim)
            self.scaler.update()

        loss_summary = {
            "acc_x":
                compute_accuracy(source_logits[:, :self.n_cls], label_s)[0].item(),
            "loss":
                loss.item(),
            "loss_ce_s":
                loss_ce_s.item(),
            "loss_ce_is":
                loss_ce_is.item(),
            "loss_ce_it":
                loss_ce_it.item(),
            "loss_ce_t":
                loss_ce_t.item(),
            "loss_smn":
                loss_smn.item(),
            "loss_entropy":
                loss_entropy.item(),
            "loss_kl":
                loss_kl.item(),
        }

        self.update_lr()

        return loss_summary

    def after_epoch(self):  # print("take this after_epoch!!!!!!!!!!!!!!!!!!!")
        last_epoch = (self.epoch + 1) == self.max_epoch
        do_test = not self.cfg.TEST.NO_TEST
        meet_checkpoint_freq = ((self.epoch + 1) %
                                self.cfg.TRAIN.CHECKPOINT_FREQ == 0 if
                                self.cfg.TRAIN.CHECKPOINT_FREQ > 0 else False)

        if do_test:
            curr_result_train, curr_result_test = self.test()
            self.save_model(self.epoch,
                            self.output_dir,
                            model_name="model--{}--{:.2f}-->{:.2f}.pth.tar".format(self.epoch, curr_result_train,
                                                                                   curr_result_test))

            self.set_model_mode("train")

        if meet_checkpoint_freq or last_epoch:
            self.save_model(self.epoch, self.output_dir)

    def parse_batch_train(self, batch_s, batch_t):
        input_s = batch_s["img"]
        label_s = batch_s["label"]
        input_t = batch_t["img"]
        label_t = batch_t["label"]

        input_s = input_s.to(self.device)
        label_s = label_s.to(self.device)
        input_t = input_t.to(self.device)
        label_t = label_t.to(self.device)
        return input_s, label_s, input_t, label_t

    def load_model(self, directory, epoch=None):
        if not directory:
            print(
                "Note that load_model() is skipped as no pretrained model is given"
            )
            return

        names = self.get_model_names()

        # By default, the best model is loaded
        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError(
                    'Model not found at "{}"'.format(model_path))

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]

            # Ignore fixed token vectors
            if "token_prefix" in state_dict:
                del state_dict["token_prefix"]

            if "token_suffix" in state_dict:
                del state_dict["token_suffix"]

            print("Loading weights to {} "
                  'from "{}" (epoch = {})'.format(name, model_path, epoch))
            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=False)

    @torch.no_grad()
    def test(self):
        """A generic testing pipeline."""
        self.set_model_mode("eval")
        self.evaluator.reset()
        split = "train"
        test_x_data_loader = self.test_loader_x
        print(f"Evaluate on the *{split}* set")
        for batch_idx, batch in enumerate(tqdm(test_x_data_loader)):
            input, label = self.parse_batch_test(batch)
            output = self.model_inference(input)
            self.evaluator.process(output, label)

        results_test_x = self.evaluator.evaluate()

        for k, v in results_test_x.items():
            tag = f"{split}/{k}"
            self.write_scalar(tag, v, self.epoch)

        self.set_model_mode("eval")
        self.evaluator.reset()
        split = "test"
        test_u_data_loader = self.test_loader_u
        print(f"Evaluate on the *{split}* set")

        for batch_idx, batch in enumerate(tqdm(test_u_data_loader)):
            input, label = self.parse_batch_test(batch)
            output = self.model_inference(input)
            self.evaluator.process(output, label)

        results_test_u = self.evaluator.evaluate()

        for k, v in results_test_u.items():
            tag = f"{split}/{k}"
            self.write_scalar(tag, v, self.epoch)

        return list(results_test_x.values())[0], list(results_test_u.values())[0]

