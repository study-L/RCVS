import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.einops import rearrange

INF = 1e9

def mask_border(m, b: int, v):
    """ Mask borders with value
    Args:
        m (torch.Tensor): [N, H0, W0, H1, W1]
        b (int)
        v (m.dtype)
    """
    if b <= 0:
        return

    m[:, :b] = v
    m[:, :, :b] = v
    m[:, :, :, :b] = v
    m[:, :, :, :, :b] = v
    m[:, -b:] = v
    m[:, :, -b:] = v
    m[:, :, :, -b:] = v
    m[:, :, :, :, -b:] = v


def mask_border_with_padding(m, bd, v, p_m0, p_m1):
    if bd <= 0:
        return

    m[:, :bd] = v
    m[:, :, :bd] = v
    m[:, :, :, :bd] = v
    m[:, :, :, :, :bd] = v

    h0s, w0s = p_m0.sum(1).max(-1)[0].int(), p_m0.sum(-1).max(-1)[0].int()
    h1s, w1s = p_m1.sum(1).max(-1)[0].int(), p_m1.sum(-1).max(-1)[0].int()
    for b_idx, (h0, w0, h1, w1) in enumerate(zip(h0s, w0s, h1s, w1s)):
        m[b_idx, h0 - bd:] = v
        m[b_idx, :, w0 - bd:] = v
        m[b_idx, :, :, h1 - bd:] = v
        m[b_idx, :, :, :, w1 - bd:] = v


def compute_max_candidates(p_m0, p_m1):
    """Compute the max candidates of all pairs within a batch
    
    Args:
        p_m0, p_m1 (torch.Tensor): padded masks
    """
    h0s, w0s = p_m0.sum(1).max(-1)[0], p_m0.sum(-1).max(-1)[0]
    h1s, w1s = p_m1.sum(1).max(-1)[0], p_m1.sum(-1).max(-1)[0]
    max_cand = torch.sum(
        torch.min(torch.stack([h0s * w0s, h1s * w1s], -1), -1)[0])
    return max_cand


class CoarseMatching(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # general config
        self.thr = config['thr']
        self.border_rm = config['border_rm']
        # -- # for trainig fine-level LoFTR
        self.train_coarse_percent = config['train_coarse_percent']
        self.train_pad_num_gt_min = config['train_pad_num_gt_min']

        self.reset_threshold = config['reset_threshold'] = 15  # 默认阈值为15
        self.frame_counter = 0  # 帧计数器

        # we provide 2 options for differentiable matching
        self.match_type = config['match_type']
        if self.match_type == 'dual_softmax':
            self.temperature = config['dsmax_temperature']
        else:
            raise NotImplementedError()


    def forward(self, feat_c0, feat_c1, data, mask_c0=None, mask_c1=None):
        """
        Args:
            feat0 (torch.Tensor): [N, L, C]
            feat1 (torch.Tensor): [N, S, C]
            data (dict)
            mask_c0 (torch.Tensor): [N, L] (optional)
            mask_c1 (torch.Tensor): [N, S] (optional)
        Update:
            data (dict): {
                'b_ids' (torch.Tensor): [M'],
                'i_ids' (torch.Tensor): [M'],
                'j_ids' (torch.Tensor): [M'],
                'gt_mask' (torch.Tensor): [M'],
                'mkpts0_c' (torch.Tensor): [M, 2],
                'mkpts1_c' (torch.Tensor): [M, 2],
                'mconf' (torch.Tensor): [M]}
            NOTE: M' != M during training.
        """
        N, L, S, C = feat_c0.size(0), feat_c0.size(1), feat_c1.size(1), feat_c0.size(2)

        # normalize
        feat_c0, feat_c1 = map(lambda feat: feat / feat.shape[-1]**.5,
                               [feat_c0, feat_c1])

        # 增加帧计数器   实现STC时空校准模块
        self.frame_counter += 1

        # 检查是否需要重置
        if self.frame_counter > self.reset_threshold:
            # 重置计数器和特征
            self.frame_counter = 0
            data.update({'last_feat_c0': None, 'last_feat_c1': None})
            print(f"计数器达到阈值 {self.reset_threshold}，重置特征缓存")

        if 'last_feat_c0' in data and data['last_feat_c0'] is not None :        #原来的逻辑，感觉不方便写时空校准逻辑
        # if  data['last_feat_c0'] is not None:        #
            lastp0=data['last_mkpts0_c']/8
            lastp1=data['last_mkpts1_c']/8

            data.update({'i_ids': lastp0[:,1]*80+lastp0[:,0], 'j_ids': lastp1[:,1]*80+lastp1[:,0]})
            last_feat_c0=data['last_feat_c0'][:,data['i_ids'],:]
            last_feat_c1=data['last_feat_c1'][:,data['j_ids'],:]


            sim_matrix_vi = torch.einsum("nlc,nsc->nls", last_feat_c0,feat_c0) / self.temperature    #相同模态跨帧做匹配 l个和s个c长度的特征向量做点积
            sim_matrix_ir = torch.einsum("nlc,nsc->nls", last_feat_c1, feat_c1) / self.temperature

            # maskvi = sim_matrix_vi > self.thr
            # maskir = sim_matrix_ir > self.thr

            _, vi_all_j_ids = sim_matrix_vi.max(dim=2)
            _, ir_all_j_ids = sim_matrix_ir.max(dim=2)

            mkpts0_c = torch.stack(                              #stack函数把两个坐标拼接成完整坐标
                [vi_all_j_ids[0] % data['hw0_c'][1],   #这里vi_all_j_ids的维度是(b,l)，这个tensor索引为[0]是得到batch0的坐标索，引计算得到x坐标
                 vi_all_j_ids[0] // data['hw0_c'][1]],
                dim=1) * 8
            mkpts1_c = torch.stack(
                [ir_all_j_ids[0] % data['hw1_c'][1],
                 ir_all_j_ids[0] // data['hw1_c'][1]],
                dim=1) * 8



            data.update({'mkpts0_c': mkpts0_c,'mkpts1_c': mkpts1_c})
            data.update({'i_ids': vi_all_j_ids[0], 'j_ids': ir_all_j_ids[0]})



        else:
            sim_matrix = torch.einsum("nlc,nsc->nls", feat_c0,
                                      feat_c1) / self.temperature
            if mask_c0 is not None:
                sim_matrix.masked_fill_(
                    ~(mask_c0[..., None] * mask_c1[:, None]).bool(),
                    -INF)
            # conf_matrix = F.softmax(sim_matrix, 1) * F.softmax(sim_matrix, 2)

            data.update({'conf_matrix': sim_matrix})

            # predict coarse matches from conf_matrix
            data.update(**self.get_coarse_match(sim_matrix, data))


        data.update({'last_feat_c0': feat_c0, 'last_feat_c1': feat_c1})   #存储上一帧的信息，为下一次追踪做准备




    @torch.no_grad()
    def get_coarse_match(self, conf_matrix, data):
        """
        Args:
            conf_matrix (torch.Tensor): [N, L, S]
            data (dict): with keys ['hw0_i', 'hw1_i', 'hw0_c', 'hw1_c']
        Returns:
            coarse_matches (dict): {
                'b_ids' (torch.Tensor): [M'],
                'i_ids' (torch.Tensor): [M'],
                'j_ids' (torch.Tensor): [M'],
                'gt_mask' (torch.Tensor): [M'],
                'm_bids' (torch.Tensor): [M],
                'mkpts0_c' (torch.Tensor): [M, 2],
                'mkpts1_c' (torch.Tensor): [M, 2],
                'mconf' (torch.Tensor): [M]}
        """
        axes_lengths = {
            'h0c': data['hw0_c'][0],
            'w0c': data['hw0_c'][1],
            'h1c': data['hw1_c'][0],
            'w1c': data['hw1_c'][1]
        }
        _device = conf_matrix.device
        # 1. confidence thresholding
        mask = conf_matrix > self.thr
        mask = rearrange(mask, 'b (h0c w0c) (h1c w1c) -> b h0c w0c h1c w1c',
                         **axes_lengths)
        if 'mask0' not in data:
            mask_border(mask, self.border_rm, False)
        else:
            mask_border_with_padding(mask, self.border_rm, False,
                                     data['mask0'], data['mask1'])
        mask = rearrange(mask, 'b h0c w0c h1c w1c -> b (h0c w0c) (h1c w1c)',
                         **axes_lengths)

        # 2. mutual nearest
        mask = mask \
            * (conf_matrix == conf_matrix.max(dim=2, keepdim=True)[0]) \
            * (conf_matrix == conf_matrix.max(dim=1, keepdim=True)[0])

        # 3. find all valid coarse matches
        # this only works when at most one `True` in each row
        mask_v, all_j_ids = mask.max(dim=2)
        b_ids, i_ids = torch.where(mask_v)
        j_ids = all_j_ids[b_ids, i_ids]
        mconf = conf_matrix[b_ids, i_ids, j_ids]

        # 4. Random sampling of training samples for fine-level LoFTR
        # (optional) pad samples with gt coarse-level matches

        # These matches select patches that feed into fine-level network
        coarse_matches = {'b_ids': b_ids, 'i_ids': i_ids, 'j_ids': j_ids}

        # 4. Update with matches in original image resolution
        scale = data['hw0_i'][0] / data['hw0_c'][0]
        scale0 = scale * data['scale0'][b_ids] if 'scale0' in data else scale
        scale1 = scale * data['scale1'][b_ids] if 'scale1' in data else scale

        sem=data['sem']


        mkpts0_c = torch.stack(
            [i_ids % data['hw0_c'][1],
             i_ids // data['hw0_c'][1]],
            dim=1) * scale0
        mkpts1_c = torch.stack(
            [j_ids % data['hw1_c'][1],
             j_ids // data['hw1_c'][1]],
            dim=1) * scale1

        # These matches is the current prediction (for visualization)
        coarse_matches.update({
            'gt_mask': mconf == 0,
            'm_bids': b_ids[mconf != 0],  # mconf == 0 => gt matches
            'mkpts0_c': mkpts0_c[mconf != 0],
            'mkpts1_c': mkpts1_c[mconf != 0],
            'mconf': mconf[mconf != 0]
        })

        return coarse_matches
