dataset_type = 'CocoDataset'
data_root='/mnt/home1/workspace2/QueryInst/data/stanford_car/'
classes = [
    'Audi TTS Coupe 2012', 'Acura TL Sedan 2012', 'Dodge Dakota Club Cab 2007',
    'Hyundai Sonata Hybrid Sedan 2012', 'Ford F-450 Super Duty Crew Cab 2012',
    'Geo Metro Convertible 1993', 'Dodge Journey SUV 2012',
    'Dodge Charger Sedan 2012', 'Mitsubishi Lancer Sedan 2012',
    'Chevrolet Traverse SUV 2012', 'Buick Verano Sedan 2012',
    'Toyota Sequoia SUV 2012', 'Hyundai Elantra Sedan 2007',
    'Dodge Caravan Minivan 1997', 'Volvo C30 Hatchback 2012',
    'Plymouth Neon Coupe 1999', 'Chevrolet Malibu Sedan 2007',
    'Volkswagen Beetle Hatchback 2012',
    'Chevrolet Corvette Ron Fellows Edition Z06 2007',
    'Chrysler 300 SRT-8 2010', 'BMW M6 Convertible 2010',
    'GMC Yukon Hybrid SUV 2012', 'Nissan Juke Hatchback 2012',
    'Volvo 240 Sedan 1993', 'Suzuki SX4 Sedan 2012',
    'Dodge Ram Pickup 3500 Crew Cab 2010', 'Spyker C8 Coupe 2009',
    'Land Rover Range Rover SUV 2012',
    'Hyundai Elantra Touring Hatchback 2012', 'Chevrolet Cobalt SS 2010',
    'Hyundai Veracruz SUV 2012', 'Ferrari 458 Italia Coupe 2012',
    'BMW Z4 Convertible 2012', 'Dodge Charger SRT-8 2009',
    'Fisker Karma Sedan 2012', 'Infiniti QX56 SUV 2011', 'Audi A5 Coupe 2012',
    'Volkswagen Golf Hatchback 1991', 'GMC Savana Van 2012',
    'Audi TT RS Coupe 2012', 'Rolls-Royce Phantom Sedan 2012',
    'Porsche Panamera Sedan 2012', 'Bentley Continental GT Coupe 2012',
    'Jeep Grand Cherokee SUV 2012', 'Audi R8 Coupe 2012',
    'Cadillac Escalade EXT Crew Cab 2007',
    'Bentley Continental Flying Spur Sedan 2007',
    'Chevrolet Avalanche Crew Cab 2012', 'Dodge Dakota Crew Cab 2010',
    'HUMMER H3T Crew Cab 2010', 'Ford F-150 Regular Cab 2007',
    'Volkswagen Golf Hatchback 2012', 'Ferrari FF Coupe 2012',
    'Toyota Camry Sedan 2012', 'Aston Martin V8 Vantage Convertible 2012',
    'Audi 100 Sedan 1994', 'Ford Ranger SuperCab 2011',
    'GMC Canyon Extended Cab 2012', 'Acura TSX Sedan 2012',
    'BMW 3 Series Sedan 2012', 'Honda Odyssey Minivan 2012',
    'Dodge Durango SUV 2012', 'Toyota Corolla Sedan 2012',
    'Chevrolet Camaro Convertible 2012', 'Ford Edge SUV 2012',
    'Bentley Continental GT Coupe 2007', 'Audi 100 Wagon 1994',
    'Ford E-Series Wagon Van 2012', 'Jeep Patriot SUV 2012',
    'Audi S6 Sedan 2011', 'Mercedes-Benz S-Class Sedan 2012',
    'Hyundai Sonata Sedan 2012',
    'Rolls-Royce Phantom Drophead Coupe Convertible 2012',
    'Ford GT Coupe 2006', 'Cadillac CTS-V Sedan 2012', 'BMW X3 SUV 2012',
    'Chevrolet Express Van 2007', 'Chevrolet Impala Sedan 2007',
    'Chevrolet Silverado 1500 Extended Cab 2012',
    'Mercedes-Benz C-Class Sedan 2012', 'Hyundai Santa Fe SUV 2012',
    'Dodge Sprinter Cargo Van 2009', 'GMC Acadia SUV 2012',
    'Hyundai Genesis Sedan 2012', 'Dodge Caliber Wagon 2012',
    'Jeep Liberty SUV 2012', 'Mercedes-Benz 300-Class Convertible 1993',
    'Ford Expedition EL SUV 2009', 'BMW 1 Series Coupe 2012',
    'Jaguar XK XKR 2012', 'Hyundai Accent Sedan 2012',
    'Isuzu Ascender SUV 2008', 'Nissan 240SX Coupe 1998',
    'Scion xD Hatchback 2012', 'Chevrolet Corvette ZR1 2012',
    'Bentley Arnage Sedan 2009', 'Chevrolet HHR SS 2010',
    'Land Rover LR2 SUV 2012', 'Hyundai Azera Sedan 2012',
    'Chrysler Aspen SUV 2009', 'Buick Regal GS 2012',
    'BMW 3 Series Wagon 2012', 'Jeep Compass SUV 2012',
    'Ram C/V Cargo Van Minivan 2012', 'Spyker C8 Convertible 2009',
    'Audi S4 Sedan 2007', 'Rolls-Royce Ghost Sedan 2012',
    'AM General Hummer SUV 2000', 'Ford Freestar Minivan 2007',
    'Bentley Mulsanne Sedan 2011', 'Audi TT Hatchback 2011',
    'Mercedes-Benz SL-Class Coupe 2009',
    'Chevrolet Silverado 1500 Hybrid Crew Cab 2012', 'Buick Enclave SUV 2012',
    'Chevrolet TrailBlazer SS 2009', 'HUMMER H2 SUT Crew Cab 2009',
    'McLaren MP4-12C Coupe 2012', 'Dodge Challenger SRT8 2011',
    'Suzuki SX4 Hatchback 2012', 'Bugatti Veyron 16.4 Convertible 2009',
    'Toyota 4Runner SUV 2012', 'Buick Rainier SUV 2007',
    'Chrysler Sebring Convertible 2010', 'Acura Integra Type R 2001',
    'Audi V8 Sedan 1994', 'Audi RS 4 Convertible 2008',
    'Honda Accord Coupe 2012', 'Audi S4 Sedan 2012',
    'Aston Martin Virage Coupe 2012', 'Chevrolet Sonic Sedan 2012',
    'Chevrolet Monte Carlo Coupe 2007', 'Volvo XC90 SUV 2007',
    'Ford Mustang Convertible 2007', 'Aston Martin Virage Convertible 2012',
    'smart fortwo Convertible 2012', 'FIAT 500 Abarth 2012',
    'Infiniti G Coupe IPL 2012', 'Dodge Caliber Wagon 2007',
    'Hyundai Tucson SUV 2012', 'Acura ZDX Hatchback 2012',
    'BMW ActiveHybrid 5 Sedan 2012', 'Ferrari California Convertible 2012',
    'Nissan Leaf Hatchback 2012', 'Lamborghini Diablo Coupe 2001',
    'Audi S5 Convertible 2012', 'BMW 6 Series Convertible 2007',
    'Ferrari 458 Italia Convertible 2012',
    'Chevrolet Silverado 2500HD Regular Cab 2012',
    'Chevrolet Corvette Convertible 2012', 'Bugatti Veyron 16.4 Coupe 2009',
    'Tesla Model S Sedan 2012', 'FIAT 500 Convertible 2012',
    'Hyundai Veloster Hatchback 2012', 'Lincoln Town Car Sedan 2011',
    'Lamborghini Aventador Coupe 2012', 'Dodge Ram Pickup 3500 Quad Cab 2009',
    'Nissan NV Passenger Van 2012', 'Honda Odyssey Minivan 2007',
    'Maybach Landaulet Convertible 2012',
    'Chevrolet Silverado 1500 Regular Cab 2012', 'Suzuki Kizashi Sedan 2012',
    'Chevrolet Tahoe Hybrid SUV 2012', 'Mercedes-Benz Sprinter Van 2012',
    'Suzuki Aerio Sedan 2007', 'Audi S5 Coupe 2012',
    'Aston Martin V8 Vantage Coupe 2012', 'Chevrolet Malibu Hybrid Sedan 2010',
    'Ford F-150 Regular Cab 2012', 'Ford Fiesta Sedan 2012',
    'Ford Focus Sedan 2007',
    'Bentley Continental Supersports Conv. Convertible 2012',
    'Chevrolet Silverado 1500 Classic Extended Cab 2007', 'BMW X5 SUV 2007',
    'Jeep Wrangler SUV 2012', 'Acura TL Type-S 2008',
    'Chrysler Crossfire Convertible 2008',
    'Lamborghini Gallardo LP 570-4 Superleggera 2012',
    'Mercedes-Benz E-Class Sedan 2012', 'Chevrolet Express Cargo Van 2007',
    'GMC Terrain SUV 2012', 'Dodge Magnum Wagon 2008',
    'Honda Accord Sedan 2012', 'Chrysler PT Cruiser Convertible 2008',
    'Mazda Tribute SUV 2011', 'BMW M3 Coupe 2012',
    'Eagle Talon Hatchback 1998', 'Daewoo Nubira Wagon 2002',
    'BMW X6 SUV 2012', 'Lamborghini Reventon Coupe 2008',
    'Cadillac SRX SUV 2012', 'MINI Cooper Roadster Convertible 2012',
    'Acura RL Sedan 2012', 'BMW 1 Series Convertible 2012',
    'Dodge Durango SUV 2007', 'BMW M5 Sedan 2010',
    'Chrysler Town and Country Minivan 2012'
]
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file=data_root+'annotations/train.json',
        img_prefix=data_root+'cars_train/',
        pipeline=train_pipeline,
        classes=classes),
    val=dict(
        type=dataset_type,
        ann_file=data_root+'annotations/test.json',
        img_prefix=data_root+'cars_test/',
        pipeline=test_pipeline,
        classes=classes),
    test=dict(
        type=dataset_type,
        ann_file=data_root+'annotations/test.json',
        img_prefix=data_root+'cars_test/',
        pipeline=test_pipeline,
        classes=classes))
evaluation = dict(interval=1, metric='bbox')
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[8, 11])
runner = dict(type='EpochBasedRunner', max_epochs=12)
checkpoint_config = dict(interval=1)
log_config = dict(interval=50, hooks=[dict(type='TextLoggerHook')])
custom_hooks = [dict(type='NumClassCheckHook')]
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
# load_from = '/mnt/home1/workspace2/QueryInst/work_dirs/reppoints_minmax_r50_fpn_gn-neck+head_1x_coco/epoch_12.pth'
resume_from = None
workflow = [('train', 1)]
model = dict(
    type='RepPointsDetector',
    pretrained='torchvision://resnet50',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch'),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_input',
        num_outs=5,
        norm_cfg=dict(type='GN', num_groups=32, requires_grad=True)),
    bbox_head=dict(
        type='RepPointsHead',
        num_classes=196,
        in_channels=256,
        feat_channels=256,
        point_feat_channels=256,
        stacked_convs=3,
        num_points=9,
        gradient_mul=0.1,
        point_strides=[8, 16, 32, 64, 128],
        point_base_scale=4,
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox_init=dict(type='SmoothL1Loss', beta=0.11, loss_weight=0.5),
        loss_bbox_refine=dict(type='SmoothL1Loss', beta=0.11, loss_weight=1.0),
        transform_method='minmax',
        norm_cfg=dict(type='GN', num_groups=32, requires_grad=True)),
    train_cfg=dict(
        init=dict(
            assigner=dict(type='PointAssigner', scale=4, pos_num=1),
            allowed_border=-1,
            pos_weight=-1,
            debug=False),
        refine=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.5,
                neg_iou_thr=0.4,
                min_pos_iou=0,
                ignore_iof_thr=-1),
            allowed_border=-1,
            pos_weight=-1,
            debug=False)),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.5),
        max_per_img=100))
norm_cfg = dict(type='GN', num_groups=32, requires_grad=True)
gpu_ids = range(0, 4)
