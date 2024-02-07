import sys,os,re
import pickle
import argparse
import torch
import pandas as pd
from collections import defaultdict
from PIL import Image
import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

_tokenizer = _Tokenizer()
device = "cuda" if torch.cuda.is_available() else "cpu"



def encode_tokens(model, text):
    x = model.token_embedding(text).type(model.dtype)  # [batch_size, n_ctx, d_model] (8, 77, 512])
    x = x + model.positional_embedding.type(model.dtype)
    x = x.permute(1, 0, 2)  # NLD -> LND
    x = model.transformer(x)
    x = x.permute(1, 0, 2)  # LND -> NLD
    x = model.ln_final(x).type(model.dtype)
    x = x @ model.text_projection
    return x


def encode_refer(args, model, prefix='a photo of a ', suffix='', refer_type='scanrefer'): 
    span = args.span
    # limit with context_length
    context_length = args.context_length    
    feat_dict = {}    
    
    print('Encoding descrptions...')
    #for split in ['test', 'val', 'train']:
    for split in ['val', 'train']:
        if refer_type == 'scanrefer':        
            refer_file = os.path.join('data/scanrefer/meta_data', 'ScanRefer_filtered_'+split+'.json')
        elif refer_type == 'riorefer':        
            refer_file = os.path.join('data/riorefer/meta_data', 'RIORefer_'+split+'.json')            
        else:
            raise NotImplementedError
            
        ref_df = pd.read_json(refer_file)
        
        if args.caption_type == 'obj_desc':
            tokens_list = [_tokenizer.decode(_tokenizer.encode(record['description'])[:context_length-2]) for _, record in ref_df.iterrows()]
        elif args.caption_type == 'obj_name':                    
            tokens_list = [_tokenizer.decode(_tokenizer.encode(prefix+record['object_name'].replace('_', ' ')+suffix)[:context_length-2]) for _, record in ref_df.iterrows()]
        elif args.caption_type == 'obj_name_desc':                    
            tokens_list = [_tokenizer.decode(_tokenizer.encode(prefix+record['object_name'].replace('_', ' ')+suffix+'. '+record['description'])[:context_length-2]) for _, record in ref_df.iterrows()]
        else:
            raise NotImplementedError
        
        # feature extraction
        text_feats = []    
        inds = []
        for i in range(ref_df.shape[0] // span + 1):
            print(split, i*span, (i+1)*span)
            tokens = tokens_list[i*span:(i+1)*span] 
            
            with torch.no_grad():
                # batch, context_length
                text = clip.tokenize(tokens).to(device)        
                # batch
                inds.append(text.argmax(dim=-1))
                if args.encode_token:
                    # batch, context_length, clip_size (256, 77, 512)
                    text_feat = encode_tokens(model, text)
                else:
                    # batch, clip_size (256, 512)
                    text_feat = model.encode_text(text)
                # normalize
                if args.use_norm:
                    text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)                

                text_feats.append(text_feat)    
        # N, (context_length), clip_size
        text_feats = torch.cat(text_feats, dim=0)  
        # N
        inds = torch.cat(inds, dim=0) 

        # store feature
        for j, row in ref_df.iterrows():
            if row.scene_id not in feat_dict:
                feat_dict[row.scene_id] = {}
            if row.object_id not in feat_dict[row.scene_id]:
                feat_dict[row.scene_id][row.object_id] = {}                
            if args.encode_token:
                feat_dict[row.scene_id][row.object_id][row.ann_id] = text_feats[j][:inds[j]+1].cpu().numpy()
            else:              
                feat_dict[row.scene_id][row.object_id][row.ann_id] = text_feats[j].cpu().numpy()
            #break
            
    if args.encode_token:
        with torch.no_grad():
            unk_token="<|endoftext|>"
            bos_token="<|startoftext|>"
            eos_token="<|endoftext|>"
            pad_token="<|endoftext|>"

            special_tokens = [unk_token, bos_token, eos_token, pad_token]
            special_token_names = ['unk', 'bos', 'eos', 'pad']
            for special_token_name, special_token in zip(special_token_names, special_tokens):
                text_feat = model.encode_text(torch.tensor([[_tokenizer.encoder[special_token]]]).cuda())
                if args.use_norm:
                    text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)           
                text_feat = text_feat[0].cpu().numpy()
                feat_dict[special_token_name] = text_feat

    if refer_type == 'scanrefer':        
        text_feat_dir = os.path.join('data/scanrefer', 'text_feature')
        text_feat_file = os.path.join(text_feat_dir, 'scanrefer_clip.'+args.caption_type+'.'+args.clip_model+('.token' if args.encode_token else '')+('.norm' if args.use_norm else '')+'.pkl')
    elif refer_type == 'riorefer':
        text_feat_dir = os.path.join('data/riorefer', 'text_feature')
        text_feat_file = os.path.join(text_feat_dir, 'riorefer_clip.'+args.caption_type+'.'+args.clip_model+('.token' if args.encode_token else '')+('.norm' if args.use_norm else '')+'.pkl')
    else:
        raise NotImplementedError
    
    os.makedirs(text_feat_dir, exist_ok=True)
    with open(text_feat_file, 'wb') as f:
        pickle.dump(feat_dict, f) #, pickle.HIGHEST_PROTOCOL)
    print('Done!')
        

def encode_scannet200(args, model, prefix='a photo of a ', suffix=''):
    sys.path.append(os.path.join(os.getcwd(), os.pardir, "data/scannet200/"))
    from data.scannet.scannet200_constants import CLASS_LABELS_200, VALID_CLASS_IDS_200     
    descs = [prefix + x + suffix for x in CLASS_LABELS_200]
    descs += ['others']
    text = clip.tokenize(descs).to(device)        
    print('Encoding ScanNet200...')
    with torch.no_grad():
        text_feat = model.encode_text(text)
        if args.use_norm:
            text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)

    text_feat = text_feat.cpu().numpy()
    feat_dict = {object_id:text_feat[i] for i, object_id in enumerate(list(VALID_CLASS_IDS_200) + [0])}
        
    text_feat_dir = os.path.join('data/scanrefer', 'text_feature')
    text_feat_file = os.path.join(text_feat_dir, 'scannet200_clip.'+args.clip_model+('.token' if args.encode_token else '')+('.norm' if args.use_norm else '')+'.pkl')
    os.makedirs(text_feat_dir, exist_ok=True)

    with open(text_feat_file, 'wb') as f:
        pickle.dump(feat_dict, f) 
    print('Done!')
    
    
def encode_scannet_all(args, model, prefix='a photo of a ', suffix=''):
    sys.path.append(os.path.join(os.getcwd(), os.pardir, "data/scannet200/"))
    label_df = pd.read_csv('data/scannet/meta_data/scannetv2-labels.combined.tsv', sep='\t')    
    label_names = list(set(label_df.raw_category))
    texts = [prefix + x + suffix for x in label_names] # https://deepsquare.jp/2021/01/clip-openai/
    texts = clip.tokenize(texts).to(device)        
    print('Encoding ScanNet all...')
    with torch.no_grad():
        text_feat = model.encode_text(texts)  
        if args.use_norm:
            text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)

    text_feat = text_feat.cpu().numpy()
    feat_dict = {object_name:text_feat[i] for i, object_name in enumerate(label_names)}
    text_feat_dir = os.path.join('data/scanrefer', 'text_feature')
    text_feat_file = os.path.join(text_feat_dir, 'scannet_clip.'+args.clip_model+('.token' if args.encode_token else '')+('.norm' if args.use_norm else '')+'.pkl')
    os.makedirs(text_feat_dir, exist_ok=True)

    with open(text_feat_file, 'wb') as f:
        pickle.dump(feat_dict, f)            
    print('Done!')    
    

def encode_rio_all(args, model, prefix='a photo of a ', suffix=''): 
    label_df = pd.read_csv('data/rio/meta_data/3RScan.v2_Semantic-Classes-Mapping.csv', sep=',', skiprows=1)    
    label_names = list(set(label_df.Label))
    texts = [prefix + x + suffix for x in label_names] # https://deepsquare.jp/2021/01/clip-openai/
    texts = clip.tokenize(texts).to(device)        
    print('Encoding 3RScan all...')
    with torch.no_grad():
        text_feat = model.encode_text(texts)  
        if args.use_norm:
            text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)

    text_feat = text_feat.cpu().numpy()
    feat_dict = {object_name:text_feat[i] for i, object_name in enumerate(label_names)}
    text_feat_dir = os.path.join('data/riorefer', 'text_feature')
    text_feat_file = os.path.join(text_feat_dir, 'rio_clip.'+args.clip_model+('.token' if args.encode_token else '')+('.norm' if args.use_norm else '')+'.pkl')
    os.makedirs(text_feat_dir, exist_ok=True)

    with open(text_feat_file, 'wb') as f:
        pickle.dump(feat_dict, f)            
    print('Done!')        


def run(args, model):
    if args.encode_scanrefer:
        encode_refer(args, model, refer_type='scanrefer')
    if args.encode_scannet200:
        encode_scannet200(args, model)
    if args.encode_scannet_all:
        encode_scannet_all(args, model)        
        
    if args.encode_riorefer:
        encode_refer(args, model, refer_type='riorefer')           
    if args.encode_rio_all:
        encode_rio_all(args, model)    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    #parser.add_argument('--gpu', type=str, help='gpu', default='0')
    parser.add_argument('--clip_model', type=str, default='ViT-B/32') # RN50x4
    parser.add_argument('--span', type=int, default=256) 
    parser.add_argument('--context_length', type=int, default=77)
    parser.add_argument('--caption_type', type=str, default='obj_desc') # or obj_name, obj_name_desc
    parser.add_argument("--encode_scanrefer", action='store_true')
    parser.add_argument("--encode_scannet200", action='store_true')    
    parser.add_argument("--encode_scannet_all", action='store_true')
    parser.add_argument("--encode_riorefer", action='store_true')
    parser.add_argument("--encode_rio_all", action='store_true')    
    parser.add_argument("--encode_token", action='store_true')    
    parser.add_argument("--use_norm", action='store_true')
    args = parser.parse_args() 
    
    # ['RN50','RN101','RN50x4','RN50x16','RN50x64','ViT-B/32','ViT-B/16','ViT-L/14','ViT-L/14@336px']
    model, preprocess = clip.load(args.clip_model, device=device)   
    args.clip_model = re.sub('/', '', args.clip_model)
    run(args, model)
