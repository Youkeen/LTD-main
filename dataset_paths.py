import os

#UFD = [
#    dict(
#        real_path=os.path.join('/mnt/f3ac0c26-0f7c-4627-8eec-865c43675c61/yyw/datasets/detection/UnivFD_data/test', dataset_name),
#        fake_path=os.path.join('/mnt/f3ac0c26-0f7c-4627-8eec-865c43675c61/yyw/datasets/detection/UnivFD_data/test', dataset_name),
#        data_mode='wang2020',
#        key=dataset_name,
#        is_resize=False,
#    ) for dataset_name in [
#            "progan",
#            "stylegan",
#            "stylegan2",
#            "biggan",
#            "cyclegan",
#            "stargan",
#            "gaugan",
#            "deepfake",
#            "diffusion_datasets/guided",
#            "diffusion_datasets/ldm_200",
#            "diffusion_datasets/ldm_200_cfg",
#            "diffusion_datasets/ldm_100",
#            "diffusion_datasets/glide_100_27",
#            "diffusion_datasets/glide_50_27",
#            "diffusion_datasets/glide_100_10",
#            "diffusion_datasets/dalle",
#        ]
#]

GenImage = [
    dict(
        real_path=os.path.join('/data1/yinyongqing/Data/GenImage', dataset_name),
        fake_path=os.path.join('/data1/yinyongqing/Data/GenImage', dataset_name),
        data_mode='wang2020',
        key=dataset_name,
        is_resize=False,
    ) for dataset_name in [
            'adm',
            'midjourney',
            'sdv1.4',
            'glid',
            'sdv1.5',
            'vqdm',
            'wukong',
            'biggan'
        ]
]
DRCT = [
    dict(
        real_path=os.path.join('/data1/yinyongqing/LTD-main/datasets/DRCT', dataset_name, "val"),
        fake_path=os.path.join('/data1/yinyongqing/LTD-main/datasets/DRCT', dataset_name, "val"),
        data_mode='wang2020',
        key=dataset_name,
        is_resize=True,
    ) for dataset_name in os.listdir("/data1/yinyongqing/LTD-main/datasets/DRCT") if 'inpainting' not in dataset_name
]