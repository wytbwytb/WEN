import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import fvcore.nn.weight_init as weight_init
from .roi_heads.box_head import build_box_head
from fsdet.layers import ShapeSpec
import gc
import torch.nn.functional as F
import pickle

class NovelModule(nn.Module):
    
    def __init__(self, cfg, in_channel, pooler_resolution):
        super().__init__()
        
        self.num_classes = cfg.MODEL.ROI_HEADS.NUM_CLASSES + 1 #1 for background
        self.bg_clsid = self.num_classes - 1
        self.prototypes = { k : None for k in range(self.num_classes)}
        self.bg_bottom_k = cfg.MODEL.ROI_HEADS.NOVEL_MODULE.BG_BOTTOM_K
        self.prototypes_fuse_alpha = cfg.MODEL.ROI_HEADS.NOVEL_MODULE.PROTOTYPES_FUSE_ALPHA
        self.iou_thresh = cfg.MODEL.ROI_HEADS.NOVEL_MODULE.IOU_THRESH
        print('iou:',self.iou_thresh)
        self.feature_extractor =build_box_head(
            cfg, ShapeSpec(channels=in_channel, height=pooler_resolution, width=pooler_resolution)
        )
        self.prototypes_feature = { k : None for k in range(self.num_classes)}
        if cfg.MODEL.ROI_HEADS.NOVEL_MODULE.INIT_FEATURE_WEIGHT != None:
            with open(cfg.MODEL.ROI_HEADS.NOVEL_MODULE.INIT_FEATURE_WEIGHT, 'rb') as f:
                self.prototypes_feature = pickle.load(f)
        self.prototypes_feature_fuse_alpha = cfg.MODEL.ROI_HEADS.NOVEL_MODULE.PROTOTYPES_FUSE_ALPHA
        print('feature alpha:',self.prototypes_feature_fuse_alpha)



    def forward(self, box_features, proposals):
        gt_classes = torch.cat([x.gt_classes for x in proposals])
        ious = torch.cat([x.iou for x in proposals])

        bg_mask = gt_classes == self.bg_clsid
        bg_features = box_features[bg_mask]
        bg_ious = ious[bg_mask]
        
        #sorted by ious, choose the k lowest ious for bg
        sorted_bg_ious, sorted_bg_ids = torch.sort(bg_ious)
        retain_num = min(self.bg_bottom_k, bg_ious.shape[0])
        sorted_bg_ids_retained = sorted_bg_ids[:retain_num]
        bg_features = bg_features[sorted_bg_ids_retained]

        #merge new proposals into prototype for non-bg classes
        filter_mask = ious > self.iou_thresh  # R x K
        filter_inds = filter_mask.nonzero()
        num_filtered = filter_inds.shape[0]
        gt_classes = gt_classes[filter_mask]
        ious = ious[filter_mask]
        box_features = box_features[filter_mask]

        gt_classes = gt_classes.chunk(num_filtered, 0)
        ious = ious.chunk(num_filtered, 0)
        box_features = [torch.squeeze(x, dim=0) for x in box_features.chunk(num_filtered, 0)]
        # del ious, box_features
        # category each proposals into corresponding class
        proposals_per_class = { k : {'iou': [], 'feature': []} for k in range(self.num_classes)}
        for gt, iou, feature in zip(gt_classes, ious, box_features):
            ids = gt.item()
            proposals_per_class[ids]['iou'].append(iou)
            proposals_per_class[ids]['feature'].append(feature)

        #aggregate each prototype of this batch according to the iou weight 
        prototypes_per_batch = { k : None for k in range(self.num_classes)}
        for ids, proposals in proposals_per_class.items():
            if len(proposals['iou'])==0 and len(proposals['feature'])==0:
                continue
            ious = torch.cat(proposals['iou']).reshape(-1,1,1,1)
            features = torch.stack(proposals['feature'], dim=0)
            # a = (features * ious)
            # a = torch.sum(a,dim=0)
            # b = torch.sum(ious)
            prototypes_per_batch[ids] = torch.div(torch.sum(features * ious, dim=0), torch.sum(ious))
            del ious, features, proposals
            gc.collect()
        #build bg's prototype
        prototypes_per_batch[self.bg_clsid] = torch.mean(bg_features, dim=0)


        # #update global prototype and prototype feature
        for ids, prototype in prototypes_per_batch.items():
            if prototype == None:
                continue
            if self.prototypes[ids] == None:
                self.prototypes[ids] = prototype.clone()
            else:
                # self.prototypes[ids] = prototype.clone()
                current_prototype = self.prototypes[ids].detach() 
                
                self.prototypes[ids] = self.prototypes_fuse_alpha * prototype + (1 - self.prototypes_fuse_alpha) * current_prototype
            
            new_prototypes_feature = self.feature_extractor(self.prototypes[ids].unsqueeze(dim=0))
            if self.prototypes_feature[ids] == None:
                self.prototypes_feature[ids] = new_prototypes_feature.clone()
            else:
                # self.prototypes_feature[ids] = new_prototypes_feature.clone()
                current_prototype_feature = self.prototypes_feature[ids].detach() 
                self.prototypes_feature[ids] = self.prototypes_feature_fuse_alpha * new_prototypes_feature + (1 - self.prototypes_feature_fuse_alpha) * current_prototype_feature
            del new_prototypes_feature, prototype
        #         prototype = 
        #feat = self.head(x)
        #feat_normalized = F.normalize(feat, dim=1)
        # del proposals_per_class, prototypes_per_batch, box_features, bg_features
        # gc.collect()
        #print(sys.getsizeof(self.prototypes) / 1024, sys.getsizeof(self.prototypes_feature) / 1024)
        return
        
    

    def get_class_prototype_feature(self):
        features = [feature for feature in self.prototypes_feature.values()]
        if None in features:
            return None
        all_features = torch.cat(features, dim=0)
        return all_features
    
    def get_all_prototype_feature(self, labels):
        return torch.cat([self.prototypes_feature[i] for i in labels.tolist()], dim=0)

    def get_prototype_ids_cosinesim(self, box_features):
        all_features = self.get_class_prototype_feature()
        all_features = F.normalize(all_features, dim=1)
        box_features = F.normalize(box_features, dim=1)
        similarities = torch.exp(torch.matmul(box_features, all_features.T))
        sim_max, sim_max_ids = torch.max(similarities, dim=1, keepdim=True)
        return sim_max_ids.squeeze( )


def prototypes_loss(all_features, factor):
    if all_features == None:
        return None
    all_features = F.normalize(all_features, dim=1)
    similarities = torch.matmul(all_features, all_features.T)
    abs_sim = torch.abs(similarities)
    # for numerical stability

    mask = torch.ones_like(abs_sim)
    mask.fill_diagonal_(0)
    bool_mask = mask == 1
    abs_sim_masked = abs_sim[bool_mask]
    loss = torch.mean(abs_sim_masked)
    
    return factor * loss