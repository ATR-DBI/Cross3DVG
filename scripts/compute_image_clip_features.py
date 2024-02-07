import sys,os,re
import pickle
import argparse
import torch
import pandas as pd
from PIL import Image
import glob
import collections
import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

_tokenizer = _Tokenizer()

def encode_image(model, image):
        return model.visual(image.type(model.dtype))
    
def encode_grid(model, images):
    activation = {}
    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook  
    model.visual.layer4.register_forward_hook(get_activation('visual.layer4'))
    _ = model.encode_image(images)
    return activation['visual.layer4']

def get_scanrefer_scene_ids(scanrefer_dir):
    train_df = pd.read_json(os.path.join(scanrefer_dir, 'ScanRefer_filtered_train.json'))
    val_df = pd.read_json(os.path.join(scanrefer_dir, 'ScanRefer_filtered_val.json'))
    test_df = pd.read_json(os.path.join(scanrefer_dir, 'ScanRefer_filtered_test.json'))
    scene_ids = sorted(set(list(train_df.scene_id) + list(val_df.scene_id) + list(test_df.scene_id)))
    return scene_ids

def get_riorefer_scene_ids(rio_dir):
    train_id = [line.strip() for line in open(os.path.join(rio_dir, 'meta_data/split', '3rscan_train.txt')).readlines()] 
    val_id = [line.strip() for line in open(os.path.join(rio_dir, 'meta_data/split', '3rscan_val.txt')).readlines()] 
    test_id = [line.strip() for line in open(os.path.join(rio_dir, 'meta_data/split', '3rscan_test.txt')).readlines()] 
    scene_ids = sorted(set(train_id + val_id + test_id))
    return scene_ids

def encode_scanrefer(args, model):
    feat_dict = collections.OrderedDict()
    data_dir = 'data/scannet/frames_square'
    scene_ids = get_scanrefer_scene_ids('data/scanrefer/meta_data')
    
    for scene_id in scene_ids:
        scene_dir = os.path.join(data_dir, scene_id)
        print('processing:', scene_dir)
        frame_dir = os.path.join(scene_dir, 'color')
        # Preprocessing
        images = []
        frame_ids = sorted([int(os.path.basename(frame_file).strip('.jpg')) for frame_file in glob.glob(os.path.join(frame_dir, '*jpg'))])
        
        for frame_id in frame_ids:
            frame_file = os.path.join(frame_dir, str(frame_id)+'.jpg')
            #print(frame_file)
            images.append(preprocess(Image.open(frame_file)))

        # images: (60, 3, 224, 224)
        images = torch.stack(images).to(device)

        # Feature Extraction
        with torch.no_grad():
            if args.encode_grid:
                assert 'ViT' not in args.clip_model
                # num_pos, clip_size, h, w (279, 2048, 7, 7)
                image_features = encode_grid(model, images)
            else:
                # image_features: num_pos, clip_size (60, 512)
                image_features = model.encode_image(images)                    

            if args.use_norm:
                image_features = image_features / image_features.norm(dim=1, keepdim=True)
            
        if scene_id not in feat_dict:
            feat_dict[scene_id] = collections.OrderedDict()
            
        for i, frame_id in enumerate(frame_ids):
            feat_dict[scene_id][frame_id] = image_features[i].cpu().numpy()    
    
    image_feat_dir = os.path.join('data/scanrefer', 'image_feature')
    image_feat_file = os.path.join(image_feat_dir, 'scanrefer_clip.'+args.clip_model+('.grid' if args.encode_grid else '')+('.norm' if args.use_norm else '')+'.pkl')
    os.makedirs(image_feat_dir, exist_ok=True)

    with open(image_feat_file, 'wb') as f:
        pickle.dump(feat_dict, f)            
    print('Done!')    

def encode_riorefer(args, model):
    feat_dict = collections.OrderedDict()
    data_dir = 'data/rio/scans'
    scene_ids = get_riorefer_scene_ids('data/rio')

    for k, scene_id in enumerate(scene_ids):
        scene_dir = os.path.join(data_dir, scene_id)
        print('processing:', k+1, '/', len(scene_ids), '\t', scene_dir)
        frame_dir = os.path.join(scene_dir, 'sequence')
        # Preprocessing
        images = []
        frame_ids = sorted([os.path.basename(frame_file).strip('.color.jpg') for frame_file in glob.glob(os.path.join(frame_dir, '*jpg'))])[::args.step]

        for frame_id in frame_ids:
            frame_file = os.path.join(frame_dir, str(frame_id)+'.color.jpg')
            # rotate 90 degrees to the right
            images.append(preprocess(Image.open(frame_file).rotate(-90, expand=True)))            

        # images: (60, 3, 224, 224)
        images = torch.stack(images).to(device)
        
        # Feature Extraction
        with torch.no_grad():
            if args.encode_grid:
                assert 'ViT' not in args.clip_model
                # num_pos, clip_size, h, w (279, 2048, 7, 7)
                image_features = encode_grid(model, images)
            else:
                # image_features: num_pos, clip_size (60, 512)
                image_features = model.encode_image(images)                    

            if args.use_norm:
                image_features = image_features / image_features.norm(dim=1, keepdim=True)
            
        if scene_id not in feat_dict:
            feat_dict[scene_id] = collections.OrderedDict()
        for i, frame_id in enumerate(frame_ids):
            feat_dict[scene_id][frame_id] = image_features[i].cpu().numpy()    
    
    image_feat_dir = os.path.join('data/riorefer', 'image_feature')
    image_feat_file = os.path.join(image_feat_dir, 'riorefer_clip.'+args.clip_model+('.grid' if args.encode_grid else '')+('.norm' if args.use_norm else '')+'.pkl')
    os.makedirs(image_feat_dir, exist_ok=True)

    with open(image_feat_file, 'wb') as f:
        pickle.dump(feat_dict, f)            
    print('Done!')    
    
    
def run(args, model):
    if args.encode_scanrefer:
        encode_scanrefer(args, model)         
    elif args.encode_riorefer:
        encode_riorefer(args, model)        

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--clip_model', type=str, default='ViT-B/32') # RN50, RN50x4
    parser.add_argument("--encode_scanrefer", action='store_true')    
    parser.add_argument("--encode_riorefer", action='store_true')
    parser.add_argument("--step", type=int, default=5)
    parser.add_argument("--encode_grid", action='store_true')    
    parser.add_argument("--use_norm", action='store_true')
    args = parser.parse_args()    

    device = "cuda" if torch.cuda.is_available() else "cpu"    
    model, preprocess = clip.load(args.clip_model, device=device)   
    args.clip_model = re.sub('/', '', args.clip_model)    
    run(args, model)    