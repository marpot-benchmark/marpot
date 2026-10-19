import inspect
import textwrap
from types import MethodType
import numpy as np
import torch
import torch.nn.functional as F
from mmdet.registry import DATASETS, TRANSFORMS
from mmdet.datasets import CocoPanopticDataset
from mmdet.datasets.transforms import LoadPanopticAnnotations
from mmdet.structures.mask import BitmapMasks
from m2f.valid_pixels import repair_coords, repair_matched_coords


@DATASETS.register_module()
class V601CocoPanopticDataset(CocoPanopticDataset):
    def parse_data_info(self, raw_data_info):
        result=super().parse_data_info(raw_data_info)
        anns={a['id']:a for a in raw_data_info['raw_ann_info'] if a['image_id']==result['img_id']}
        instances=[]
        for segment in result['segments_info']:
            ann=anns[segment['id']]
            crowd=bool(ann.get('iscrowd',0))
            isthing=bool(self.coco.load_cats(ids=ann['category_id'])[0]['isthing'])
            segment.update(iscrowd=crowd,is_thing=isthing)
            if isthing:
                x,y,w,h=ann['bbox']
                instances.append(dict(bbox=[x,y,x+w,y+h],bbox_label=segment['category'],ignore_flag=int(crowd)))
        result['instances']=instances
        return result


@TRANSFORMS.register_module()
class V601LoadPanopticAnnotations(LoadPanopticAnnotations):
    def _load_masks_and_semantic_segs(self, results):
        super()._load_masks_and_semantic_segs(results)
        if self.with_seg and 'gt_seg_map' in results:
            crowd=[s for s in results['segments_info'] if s.get('iscrowd',False)]
            if not crowd:return
            if self.with_mask:
                masks=results['gt_masks'].masks
                thing_segments=[s for s in results['segments_info'] if s['is_thing']]
                if len(masks)!=len(thing_segments):raise RuntimeError('panoptic instance/mask alignment changed')
                for segment,mask in zip(thing_segments,masks):
                    if segment.get('iscrowd',False):results['gt_seg_map'][mask.astype(bool)]=255
            else:
                from mmengine.fileio import get
                import mmcv
                pan=self.rgb2id(mmcv.imfrombytes(get(results['seg_map_path'],backend_args=self.backend_args),flag='color',channel_order='rgb').squeeze())
                for segment in crowd:results['gt_seg_map'][pan==segment['id']]=255


def _clone_method(original, substitutions, extra_globals):
    source=textwrap.dedent(inspect.getsource(original))
    for old,new in substitutions:
        if source.count(old)!=1:raise RuntimeError('pinned MMDet source drift: '+old[:100])
        source=source.replace(old,new)
    namespace=dict(original.__globals__);namespace.update(extra_globals)
    exec(compile(source,original.__code__.co_filename+'[v601]', 'exec'),namespace)
    return namespace[original.__name__]


def _repair_head_coords(coords, batch_gt_instances, mask_targets_list):
    targets=[dict(masks=gt.masks) for gt in batch_gt_instances]
    matches=[(None,range(len(t))) for t in mask_targets_list]
    return repair_matched_coords(coords,targets,matches,mask_targets_list[0].shape[-2:])


def install_query_head(head):
    if getattr(head,'_v601_valid_pixels',False):return
    cls=type(head)
    get_target=_clone_method(cls._get_targets_single,[
        ('    # shape (num_queries, num_points)',
         '    if len(gt_masks): point_coords = repair_coords(point_coords, gt_masks.bool().any(0))\n    # shape (num_queries, num_points)')],dict(repair_coords=repair_coords))
    loss=_clone_method(cls._loss_by_feat_single,[
        ('        # shape (num_total_gts, h, w) -> (num_total_gts, num_points)',
         '        points_coords = _repair_head_coords(points_coords, batch_gt_instances, mask_targets_list)\n        # shape (num_total_gts, h, w) -> (num_total_gts, num_points)')],dict(_repair_head_coords=_repair_head_coords))
    head._get_targets_single=MethodType(get_target,head)
    head._loss_by_feat_single=MethodType(loss,head)
    head._v601_valid_pixels=True


def install_fpn(model):
    roi=model.roi_head
    original_roi=roi.loss
    def roi_loss(self,x,rpn_results_list,batch_data_samples):
        self.mask_head._v601_valid=[sample.gt_sem_seg.sem_seg[0]!=255 for sample in batch_data_samples]
        try:return original_roi(x,rpn_results_list,batch_data_samples)
        finally:self.mask_head._v601_valid=None
    roi.loss=MethodType(roi_loss,roi)
    head=roi.mask_head;original_mask=head.loss_and_target
    def mask_loss(self,mask_preds,sampling_results,batch_gt_instances,rcnn_train_cfg):
        valid=getattr(self,'_v601_valid',None)
        if valid is None or all(bool(v.all()) for v in valid):
            return original_mask(mask_preds,sampling_results,batch_gt_instances,rcnn_train_cfg)
        from mmcv.ops import roi_align
        targets=self.get_targets(sampling_results,batch_gt_instances,rcnn_train_cfg)
        labels=torch.cat([r.pos_gt_labels for r in sampling_results])
        weights=[]
        for valid_map,res in zip(valid,sampling_results):
            boxes=res.pos_priors
            if hasattr(boxes,'tensor'):boxes=boxes.tensor
            boxes=boxes.to(mask_preds)
            rois=torch.cat((boxes.new_zeros((len(boxes),1)),boxes),dim=1)
            weights.append(roi_align(valid_map[None,None].to(mask_preds),rois,targets.shape[-2:],1.,0,'avg',True)[:,0]>=1.-1e-6)
        weight=torch.cat(weights).float()
        if mask_preds.shape[0]==0:loss=mask_preds.sum()
        else:
            pred=mask_preds[torch.arange(len(labels),device=labels.device),torch.zeros_like(labels) if self.class_agnostic else labels]
            loss=(F.binary_cross_entropy_with_logits(pred,targets,reduction='none')*weight).sum()/weight.sum().clamp_min(1.)
            loss=loss[None]*self.loss_mask.loss_weight
        return dict(loss_mask=dict(loss_mask=loss),mask_targets=targets)
    head.loss_and_target=MethodType(mask_loss,head)


def _boundary_loss(module, logits, labels, valid):
    if bool(valid.all()):return module(logits,labels)
    pred=logits.permute(0,2,3,1).reshape(-1)
    target=labels.reshape(-1).to(pred)
    keep=valid.reshape(-1)
    if not bool(keep.any()):return pred.sum()*0.
    pred=pred[keep];target=target[keep]
    pos=target==1;neg=target==0
    count=pos.sum()+neg.sum()
    weight=torch.zeros_like(pred)
    weight[pos]=(neg.sum()/count).to(weight)
    weight[neg]=(pos.sum()/count).to(weight)
    return module.loss_weight*F.binary_cross_entropy_with_logits(pred,target,weight,reduction='mean')


def install_pid(head):
    loss=_clone_method(type(head).loss_by_feat,[
        ("    loss['loss_bd'] = self.loss_decode[2](d_logit, bd_label)",
         "    loss['loss_bd'] = _boundary_loss(self.loss_decode[2], d_logit, bd_label, sem_label != self.ignore_index)")],dict(_boundary_loss=_boundary_loss))
    head.loss_by_feat=MethodType(loss,head)


def install(model,task):
    model=model.module if hasattr(model,'module') else model
    if task.startswith('pan_m2f_'):
        install_query_head(model.panoptic_head)
    elif task=='sem_mask2former_swin_t':
        install_query_head(model.decode_head)
    elif task=='pan_fpn_r50':
        install_fpn(model)
    elif task=='sem_pidnet_s':
        install_pid(model.decode_head)
    return dict(void_crowd_protocol='v601 ignore crowd targets and void/crowd spatial supervision; preserve native labels')
