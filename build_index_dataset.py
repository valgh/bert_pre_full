#########################################################
### Outputs a .txt file containing paths to dataset files
#########################################################

import os
import tqdm


def main():
    output_file = '/home/valerio/valerio/dataset/bert/dataset_index.txt'
    working_dir = '/home/valerio/valerio/dataset/multiverse/'    # main/multi/uni
    rw_text = '/randwalks_text_pro/'
    names_dir = '/names/'
    bins_d = os.listdir(working_dir)
    for d in tqdm.tqdm(bins_d):
        dir_to_inspect = working_dir+d+rw_text
        names_to_check = working_dir+d+names
        files_to_append = os.listdir(dir_to_inspect)
        for f in files_to_append:
            if os.path.exists(names_to_check+f):
                to_append = dir_to_inspect+f
                with open(output_file, 'a') as l:
                    l.write(to_append)
                    l.write('\n')
    print('Done.')


if __name__ == '__main__':
    main()

