# model settings
test_seen_classes = False
training_type = "Channel"
model = dict(
    type='BHRL',
    pretrained=None,
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
        stddev = 0.1,
        mean = 0,
        num_outs=5),
    rpn_head=dict(
        type='RPNHead',
        in_channels=384,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[8],
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[0.0, 0.0, 0.0, 0.0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
    roi_head=dict(
        type='BHRLRoIHead',
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(
                type='DeformRoIPoolPack',
                output_size=7,
                output_channels=256),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=[
            dict(
                type='BHRLConvFCBBoxHead',
                use_shared_fc = True,
                num_fcs=2,
                in_channels=384,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=1,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0.0, 0.0, 0.0, 0.0],
                    target_stds=[0.1, 0.1, 0.2, 0.2]),
                reg_class_agnostic=False,
                ihr = dict(
                    metric_module_in_channel=256,
                    metric_module_out_channel=384,
                ),
                loss_cls=dict(
                    type='RPLoss', use_sigmoid=False, loss_weight=1.0,alpha=0.25),
                loss_bbox=dict(type='L1Loss', loss_weight=1.0))]),
    train_cfg = dict(
        rpn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.7,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                match_low_quality=True,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=256,
                pos_fraction=0.5,
                neg_pos_ub=-1,
                add_gt_as_proposals=False),
            allowed_border=-1,
            pos_weight=-1,
            debug=False),
        rpn_proposal=dict(
            # nms_across_levels=False,
            nms_pre=2000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=[
            dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.5,
                    neg_iou_thr=0.5,
                    min_pos_iou=0.5,
                    match_low_quality=False,
                    ignore_iof_thr=-1),
                sampler=dict(
                    type='RandomSampler',
                    num=512,
                    pos_fraction=0.25,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=True),
                mask_size=28,
                pos_weight=-1,
                debug=False)]),
    test_cfg = dict(
        rpn=dict(
            # nms_across_levels=False,
            nms_pre=1000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            score_thr=0.05,
            nms=dict(type='nms', iou_threshold=0.5),
            max_per_img=100,
            mask_thr_binary=0.5))
            ,training_type=training_type
            )

# dataset settings
dataset_type = 'OneShotVOCDataset'
data_root = 'data/manga_dataset/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
# We use the same image size as the paper (One-Shot Instance Segmentation). It is the first to study one-shot object detection.
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=False),
    dict(type='Resize', img_scale=(1170, 1654), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='LoadSiameseReference'),
    dict(type='ReferenceTransform', img_scale=(192, 192), keep_ratio=True, **img_norm_cfg),
    dict(type='SiameseFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1024, 1024),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size=(1024, 1024)),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='LoadSiameseReference'),
            dict(type='ReferenceTransform', img_scale=(192, 192), keep_ratio=True, **img_norm_cfg),
            dict(type='SiameseFormatBundle'),
            dict(type='Collect', keys=['img'], meta_keys=['img_info', 'filename', 'ori_shape',
                                                          'img_shape', 'pad_shape', 'scale_factor',
                                                          'flip', 'img_norm_cfg', 'label']),
        ])
]


img_path = "/images"
config_path = "./configs/manga/BHRL_allpairs_mono.py"
threshold = 0


data = dict(
    samples_per_gpu=2,
    workers_per_gpu=8,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'voc_annotation/pairs_before.json',
        config_path = config_path,
        img_prefix=data_root+ img_path,
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'voc_annotation/pairs_after.json',
        img_prefix=data_root+ img_path,
        pipeline=test_pipeline,
        ),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'voc_annotation/pairs_after.json',
        config_path = config_path,
        threshold=threshold,
        img_prefix=data_root + img_path,
        pipeline=test_pipeline,
        test_seen_classes=test_seen_classes,
        position=0))
evaluation = dict(interval=1, metric='bbox')

# optimizer
optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=5000,
    warmup_ratio=0.001,
    step=[6])
runner = dict(type='EpochBasedRunner', max_epochs=1)
checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=20,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable


dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = 'work_dirs/manga/allpairs/trained_gaussian_tensor/channel'
load_from = 'resnet_model/res50_loadfrom.pth'
resume_from = None
workflow = [('train', 1)]
