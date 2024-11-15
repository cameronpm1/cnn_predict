import os
import json
import shutil
import numpy as np
from PIL import Image



def mkdir(folder):
    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.makedirs(folder)

def split_data(
    vid: str,
    sim: str,
    episode_len: int,
    set_split: list[float] = [0.8, 0.1, 0.1],
    seed: int = 1,
):
    seq_filepath = 'videos/'+sim+vid
    files = os.listdir(seq_filepath)

    data_split_dict = {}
    folder_list = list(range(int(len(files)/episode_len)))
    
    train_idx = np.random.choice(folder_list, size=int(len(folder_list)*set_split[0]), replace=False)
    folder_list = [i for i in folder_list if i not in train_idx]
    test_idx = np.random.choice(folder_list, size=int(len(folder_list)*(set_split[1])/(set_split[1]+set_split[2])), replace=False)
    folder_list = [i for i in folder_list if i not in test_idx]
    val_idx = folder_list

    data_split_dict['test'] = test_idx.tolist()
    data_split_dict['val'] = val_idx#.tolist()
    data_split_dict['train'] = train_idx.tolist()

    json_file_name = 'datainfo/' + sim + vid + 'data_split_dict_' + str(seed) + '.json'
    with open(json_file_name, "w") as outfile: 
        json.dump(data_split_dict, outfile,
          separators=(',', ':'), 
          sort_keys=True, 
          indent=4)

    data_dir = 'data/' + sim + vid
    j_count = 0
    for i in range(int(len(files)/episode_len)):
        data_sub_dir = data_dir + '/' + str(i) + '/'
        mkdir(data_sub_dir)
        for j in range(episode_len):
            file = 'frame' + str(j_count) + '.jpeg'
            dest = shutil.move(seq_filepath+file, data_sub_dir+file) 
            j_count += 1

def rename_data(
    data_dir: str,
    sim: str,
) -> None:

    seq_filepath = data_dir+sim
    folders = os.listdir(seq_filepath)
    i = 0

    for folder in folders:
        files = os.listdir(seq_filepath+folder+'/')
        idxs = []
        for file in files:
            start = 5
            end = file.find('.')
            idxs.append(int(file[start:end]))
        sorted_idxs = sorted(idxs)
        new_idxs = []
        for file in files:
            start = 5
            end = file.find('.')
            num = int(file[start:end])
            new_idx = sorted_idxs.index(num)
            new_idxs.append(new_idx)
        for i,file in enumerate(files):
            old_name = seq_filepath+folder+'/'+file
            new_name = seq_filepath+folder+'/'+str(new_idxs[i])+'.jpeg'
            os.rename(old_name,new_name)




if __name__ == '__main__':
    #split_data(vid='video2/',sim='cube_2d/',episode_len=60)
    rename_data(data_dir='data/',sim='cube_2d/video2/')
