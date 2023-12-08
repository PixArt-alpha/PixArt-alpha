import os, json, shutil

root = './data/MJData/'
dst_path = './data_local/MJImgs/'
os.makedirs(dst_path, exist_ok=True)

image_list_json = './data/MJData/partition/mj_1_new.json'
resolution = 1024

with open(f'{root}/not_exist.txt', 'r') as f:
    noe = set([line.strip() for line in f])
with open(f'{root}/not_exist_5-10.txt', 'r') as f:
    noe = noe.union([line.strip() for line in f])


def load_json(file_path):
    with open(file_path, 'r') as f:
        meta_data = json.load(f)

    return meta_data

ori_imgs_nums = 0


img_samples = []
txt_feat_samples = []
vae_feat_samples = []
hed_feat_sample = []

image_list_json = 'mj_1_new.json'
image_list_json = image_list_json if isinstance(image_list_json, list) else [image_list_json]
for json_file in image_list_json:
    meta_data = load_json(os.path.join(root, 'partition', json_file))
    ori_imgs_nums += len(meta_data)
    meta_data_clean = [item for item in meta_data if (item['path'] not in noe and item['ratio'] <= 4)]
    #img_samples.extend([os.path.join(root.replace('MJData', "MJImgs"), item['path']) for item in meta_data_clean])
    txt_feat_samples.extend([os.path.join(root, 'caption_features', '_'.join(item['path'].rsplit('/', 1)).replace('.png', '.npz')) for item in meta_data_clean])
    vae_feat_samples.extend([os.path.join(root, f'img_vae_features_{resolution}resolution/noflip', '_'.join(item['path'].rsplit('/', 1)).replace('.png', '.npy')) for item in meta_data_clean])
    hed_feat_sample.extend([os.path.join(root, f'hed_feature_{resolution}', item['path'].replace('.png', '.npz')) for item in meta_data_clean])

print("meta_data_clean", len(meta_data_clean), "txt_feat", len(txt_feat_samples), "vae_samples", len(vae_feat_samples), "hed samples", len(hed_feat_sample))
def make_dirs(source_path, destination_path):
    splits = destination_path.split("/")
    root = './data_local/'
    for i in range(2, len(splits) - 1):
        root = root + splits[i] + '/'
        os.makedirs(root, exist_ok=True)
    command = 'rsync -P %s %s'%(source_path, destination_path)
    os.system(command)
    # shutil.copyfile(source_path, destination_path)



for index in range(100):
    # if index == 16:
        # continue
    print("%d/%d"%(index, len(txt_feat_samples)))
    npz_path = txt_feat_samples[index]
    npz_path_destination_path = npz_path.replace("data", "data_local")
    print(npz_path, npz_path_destination_path)
    try:
        command = 'rsync -P %s %s'%(npz_path, npz_path_destination_path)
        os.system(command)
        # shutil.copyfile(npz_path, npz_path_destination_path)
    except:
        make_dirs(npz_path, npz_path_destination_path)
    assert os.path.exists(npz_path_destination_path), "error! copy failed %s"%npz_path
    
    
    npy_path = vae_feat_samples[index]
    npy_path_destination_path = npy_path.replace("data", "data_local")
    try:
        command = 'rsync -P %s %s'%(npy_path, npy_path_destination_path)
        os.system(command)
        # shutil.copyfile(npy_path, npy_path_destination_path)
    except:
        make_dirs(npy_path, npy_path_destination_path)
    assert os.path.exists(npy_path_destination_path), "error! copy failed %s"%npy_path
    
    
    hed_npz_path = hed_feat_sample[index]
    hed_npz_destination_path = hed_npz_path.replace("data", "data_local")
    try:
        command = 'rsync -P %s %s'%(hed_npz_path, hed_npz_destination_path)
        os.system(command)
        # shutil.copyfile(hed_npz_path, hed_npz_destination_path)
    except:
        make_dirs(hed_npz_path, hed_npz_destination_path)
    assert os.path.exists(hed_npz_destination_path), "error! copy failed %s"%hed_npz_path


    print(img_path, npz_path, npy_path, hed_npz_path)
    print(npz_path, npy_path, hed_npz_path)
    destination_path = img_path.replace("data", "data_local")
    
    
