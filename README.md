### Training

Examples of json files with the model parameters can be found in the folder `/configs`. Use the following command to train a similarity metric:
```
CUDA_VISIBLE_DEVICES=<device_ids> python train.py --config <path/to/config.json> --exp-name <exp_name>
```

Use the following command for testing:
```
CUDA_VISIBLE_DEVICES=<device_id> python test.py --config <path/to/config.json> --exp-name <exp_name> --resume <path/to/checkkpoint.pt>
```
