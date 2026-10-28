from mmdet.registry import DATASETS, TRANSFORMS
from mmdet.datasets import CocoPanopticDataset
from mmdet.datasets.transforms import LoadPanopticAnnotations


@DATASETS.register_module()
class MarPOTCocoPanopticDataset(CocoPanopticDataset):
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
class MarPOTLoadPanopticAnnotations(LoadPanopticAnnotations):
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
