from pathlib import Path
import types
import numpy as np
import torch
import torch.nn.functional as F


def install_backend(wrapper, options, device):
    import os
    config=os.environ.get('MARPOT_M2F_CONFIG')
    checkpoint=os.environ.get('MARPOT_M2F_CHECKPOINT')
    if not config or not checkpoint:
        raise RuntimeError('Mask2Former backend needs MARPOT_M2F_CONFIG and MARPOT_M2F_CHECKPOINT '
                           '(set by evaluation/run_eval.py --m2f-config/--m2f-checkpoint)')
    from mmdet.apis import init_detector
    from mmengine.dataset import Compose
    from mmdet.utils import register_all_modules
    register_all_modules(init_default_scope=True)
    previous=os.environ.get('TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD')
    os.environ['TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD']='1'
    try:
        detector=init_detector(str(config),str(checkpoint),device=device,palette="random")
    finally:
        if previous is None:os.environ.pop('TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD',None)
        else:os.environ['TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD']=previous
    pipeline=detector.cfg.test_dataloader.dataset.pipeline.copy()
    pipeline=[dict(x) for x in pipeline if 'Annotations' not in x['type']]
    pipeline[0]['type']='mmdet.LoadImageFromNDArray'
    transform=Compose(pipeline)
    state={}
    wrapper._m2f_detector=detector
    def resize(self, imgs):
        if len(imgs)!=1:raise ValueError('M2F supplement is batch=1')
        array=imgs[0].detach().cpu().permute(1,2,0).numpy()[:,:,::-1].copy()
        item=transform(dict(img=array,img_id=0))
        batch=detector.data_preprocessor({'inputs':[item['inputs']],'data_samples':[item['data_samples']]},False)
        state['samples']=batch['data_samples']
        self.img_size=tuple(batch['inputs'].shape[-2:])
        state['scaled']=tuple(batch['data_samples'][0].img_shape[:2])
        return batch['inputs']
    def forward(self, imgs):
        with torch.autocast(device_type='cuda',enabled=False):
            features=detector.extract_feat(imgs.float())
            cls,masks=detector.panoptic_head(features,state['samples'])
        return masks,cls
    def scaled(self, size):return state['scaled']
    def revert(self, logits, sizes):
        h,w=state['scaled']
        return [F.interpolate(logits[i:i+1,:,:h,:w],size=tuple(size),mode='bilinear',align_corners=False)[0] for i,size in enumerate(sizes)]
    def compose(self, masks, classes, stuff, threshold, overlap):
        fusion=detector.panoptic_fusion_head
        fusion.test_cfg['object_mask_thr']=float(threshold)
        fusion.test_cfg['iou_thr']=float(overlap)
        out=[]
        for m,c in zip(masks,classes):
            result=fusion.panoptic_postprocess(c,m).sem_seg[0]
            category=result % 1000
            seg=result.clone().long();seg[category==self.num_classes]=-1
            out.append(torch.stack([category.long(),seg],-1))
        return out
    wrapper.resize_and_pad_imgs_instance_panoptic=types.MethodType(resize,wrapper)
    wrapper.forward=types.MethodType(forward,wrapper)
    wrapper.scale_img_size_instance_panoptic=types.MethodType(scaled,wrapper)
    wrapper.revert_resize_and_pad_logits_instance_panoptic=types.MethodType(revert,wrapper)
    wrapper.to_per_pixel_preds_panoptic=types.MethodType(compose,wrapper)
    wrapper.mask_thresh=float(detector.panoptic_fusion_head.test_cfg.get('object_mask_thr',0.8))
    wrapper.overlap_thresh=float(detector.panoptic_fusion_head.test_cfg.get('iou_thr',0.8))
    wrapper.eval()
