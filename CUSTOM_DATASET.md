# Custom Dataset Training with ActionFormer

We provide a step-by-step tutorial on how to train your custom dataset with ActionFormer. In this tutorial, we use the THUMOS14 dataset as an example. The user can follow this guideline to use their custom datasets for training/testing ActionFormer.

1. Download the video dataset from the [original website](https://www.crcv.ucf.edu/THUMOS14/download.html).

2. Extract the RGB frames for THUMOS14. (Optional) Extract the Flow frames for THUMOS14. We recommend to use tools from [MMAction2](https://github.com/open-mmlab/mmaction2) to extract RGB and Flow frames:

   - Please follow the original instructions (before Step 4) in [MMAction2 README](https://github.com/open-mmlab/mmaction2/blob/master/tools/data/thumos14/README.md) to extract RGB (and Flow) Frames.
   - If you want to extract Flow frames, [denseflow](https://github.com/open-mmlab/denseflow) is needed. 

3. Extract the RGB (and Flow) features for THUMOS14.

   - In this part, we use the original [I3D network](https://github.com/Finspire13/pytorch-i3d-feature-extraction) as an example. The user can replace this part with any existing codebase (Please note that I3D network are trained with 25FPS videos, you may need to use ffmpeg to adjust the video's FPS before step 2).
   - The repo is depreated, we may use a newer repo to replace that part.
   - Prepare the environemnts according to the repo.
   - Run the following command to extract RGB features (stride is 4, oversample represents ten-crop) in the example. You can change the stride/sample strategy according to your requirements.
   ```shell
   python extract_features.py --mode rgb --load_model models/rgb_imagenet.pt \ 
   --input_dir [path_to_input_dir] --output_dir [path_to_rgbout_dir] \
   --sample_mode oversample --frequency 4 
   ```
   - (Optional) run the following command to extract Flow features:
   ```shell
   python extract_features.py --mode flow --load_model models/flow_imagenet.pt \ 
   --input_dir [path_to_input_dir] --output_dir [path_to_flowout_dir] \
   --sample_mode oversample --frequency 4 
   ```
   - (Optional) Combine RGB and Flow features into final features.
      - We need to combine the RGB and Flow features (two file) into final features (a single file). You need to write a simple python script to do that.

4. Prepare the annotations file for THUMOS14.
   - ActionFormer uses json format for annotations like ActivityNet v1.3. So we need to convert the table in THUMOS14 to json format.
   - Our json are like:
   ```json
   {"version":"0.0.1", "database": 
      {"video1": 
         {"subset": "Training", "duration": 33.83, "fps": 30.0, "annotations":
            [
               {
                  "label": "label1",
                  "segment": [0.3, 1.1],
                  "label_id": 5
               },
               {
                  "label": "label2",
                  "segment": [2.2, 3.3],
                  "label_id": 5
               },
               ...
            ]
         }
      }
   }
   ```
   - For each video, we need specify the `subset`, `duration` and the `fps`. If we do not set the `fps` for each video, we need to set it in step 5.
   - For each action in the video, we need to create the following annotations:
      - "label": the real_label (can be string) for this action.
      - "segment": the starting/ending of this action
      - "label_id": the label mapping (int) for this label (Should be calculated before creating the annotations.)

5. Create a custom dataset file for THUMOS14. 

   - You may refer to the `libs/datasets/thumos14.py` for details.
   - You may need to change some information in the dataset file, e.g., db_attributes to control the evaulation settings, e.g., the tIoU for evaluation.

6. Add the dataset name to the Dataset Registry.

   - In the `libs/__init__.py` file, include the dataset name you set in step 5 within `@register_dataset()`.

7. Create a config file for THUMOS14.
   - You may refer to the `configs/thumos14_i3d.yaml` for details.   
   - You may need to change some items in the config file, e.g., `train_split` and `val_split` for the subset name. `json_file` and `feat_folder` for dataset location, `feat_stride` (as `frequency` in step 3).

7. Start to train and eval on your dataset.
   - Please refer to `tools/run_all_exps.sh` to get some examples for training/testing.
