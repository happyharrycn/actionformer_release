import os
import json

import torch
import numpy as np

"""
Directory structure before processing:

This folder
│  convert_ego4d_trainval.py
│  ego4d_label_map.txt
│  ... 
│
└───features
│    └───slowfast8x8_r100_k400
│    └───omnivore_video_swinl
│
└───annotations
│    └───moments_train.json
│    └───moments_val.json
│  ...
"""

# full-video features downloaded from Ego4D website
slowfast_dir = 'features/slowfast8x8_r101_k400'
omnivore_dir = 'features/omnivore_video_swinl'

# annotation files downloaded from Ego4D website
train_annot_path = 'annotations/moments_train.json'
val_annot_path = 'annotations/moments_val.json'

# label mapping
label_map_path = 'ego4d_label_map.txt'

# where to save the processed features
slowfast_out_dir = 'features/slowfast_features'
omnivore_out_dir = 'features/omnivore_features'
os.makedirs(slowfast_out_dir, exist_ok=True)
os.makedirs(omnivore_out_dir, exist_ok=True)

# where to save the processed annotations
annot_out_path = 'annotations/ego4d.json'

# clip size / stride in feature extraction
clip_size = 32
stride = 16


with open(train_annot_path, 'r') as f:
    train_videos = json.load(f)['videos']
with open(val_annot_path, 'r') as f:
    val_videos = json.load(f)['videos']
videos = train_videos + val_videos

label_map = dict()
with open(label_map_path, 'r') as f:
    lines = [l.strip().split('\t') for l in f.readlines()]
    for v, k in lines:
        label_map[k] = int(v)

database = dict()

# parse video annotations
for video in videos:
    vid = video['video_uid']
    print('Processing video {:s} ...'.format(vid))
    subset = video['split']
    if subset == 'train':
        subset = 'training'
    elif subset == 'val':
        subset = 'validation'

    # load video features
    slowfast_path = os.path.join(slowfast_dir, vid + '.pt')
    omnivore_path = os.path.join(omnivore_dir, vid + '.pt')
    # skip video if feature does not exist
    if not os.path.exists(slowfast_path):
        print('> slowfast feature missing')
    if not os.path.exists(omnivore_path):
        print('> omnivore feature missing')
        continue
    slowfast_video = torch.load(slowfast_path).numpy()
    omnivore_video = torch.load(omnivore_path).numpy()

    # parse clip annotations
    clips = video['clips']
    for clip in clips:
        cid = clip['clip_uid']
        ss = max(float(clip['video_start_sec']), 0)
        es = float(clip['video_end_sec'])
        sf = max(int(clip['video_start_frame']), 0)
        ef = int(clip['video_end_frame'])
        duration = es - ss      # clip length in second
        frames = ef - sf        # clip length in frame
        fps = frames / duration
        if fps < 10 or fps > 100:
            continue
        
        # align event onsets and offsets with feature grid
        prepend_frames = sf % stride
        prepend_sec = prepend_frames / fps
        duration += prepend_sec
        frames += prepend_frames

        append_frames = append_sec = 0
        if (frames - clip_size) % stride:
            append_frames = stride - (frames - clip_size) % stride
            append_sec = append_frames / fps
            duration += append_sec
            frames += append_frames

        # save clip features
        si = (sf - prepend_frames) // stride
        ei = (ef + append_frames - clip_size) // stride
        if ei > len(slowfast_video):
            raise ValueError('end index exceeds slowfast feature length')
        if ei > len(omnivore_video):
            raise ValueError('end index exceeds omnivore feature length')
            
        slowfast_clip = slowfast_video[si:ei]
        omnivore_clip = omnivore_video[si:ei]
        np.save(
            os.path.join(slowfast_out_dir, cid + '.npy'), 
            slowfast_clip.astype(np.float32),
        )
        np.save(
            os.path.join(omnivore_out_dir, cid + '.npy'), 
            omnivore_clip.astype(np.float32),
        )
        
        annotations = []

        # parse annotations from different annotators
        annotators = clip['annotations']
        for annotator in annotators:
            
            # parse action items
            items = annotator['labels']
            for item in items:
                # skip items not from primary categories
                if not item['primary']:
                    continue
                
                ssi = item['video_start_time'] - ss + prepend_sec
                esi = item['video_end_time'] - ss + prepend_sec
                sfi = item['video_start_frame'] - sf + prepend_frames
                efi = item['video_end_frame'] - sf + prepend_frames
                
                # filter out very short actions
                if esi - ssi < 0.25:
                    continue
                
                label = item['label']
                annotations += [{
                    'label': label,
                    'segment': [round(ssi, 2), round(esi, 2)],
                    'segment(frames)': [sfi, efi],
                    'label_id': label_map[label],
                }]

        database[cid] = {
            'subset': subset,
            'duration': round(duration, 2),
            'fps': round(fps, 2),
            'annotations': annotations,
        }

    out = {'version': 'v1', 'database': database}
    with open(annot_out_path, 'w') as f:
        json.dump(out, f)