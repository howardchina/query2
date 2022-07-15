_base_ = './deformable_detr_r50_16x2_50e_stanford_cars_tiny.py'
load_from = 'work_dirs/pretrained/deformable_detr/deformable_detr_r50_16x2_50e_coco_20210419_220030-a12b9512.pth'
model = dict(
    bbox_head=dict(
        type='DeformableDETRHead',
        num_query=300,
        transformer=dict(
            two_stage_num_proposals=300)))
