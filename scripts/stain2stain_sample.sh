MODELS_PATH="--model_path ./pretrained_model/model770000.pt" 
DIFFUSION_ARGS="--image_size 256 --num_channels 128 --num_res_blocks 3 --diffusion_steps 50 --noise_schedule linear  --use_ddim True --rescale_timesteps True"
SAMPLE_ARGS="--target_domain MAS"
DATASET_ARGS="--data_dir /root/private_data/BingXiong/data/ANHIR_v2/val"
python main.py  $MODELS_PATH  $DIFFUSION_ARGS $DATASET_ARGS $SAMPLE_ARGS  --class_cond True