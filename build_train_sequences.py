#########################################################
### Build .txt of string asm paths for BERT pretraining
#########################################################
import json
import tqdm

def main():
    train_index_file = '/home/valerio/valerio/dataset/bert/train_index.txt'
    train_file = '/home/valerio/valerio/dataset/bert/bert_train_paths.txt'
    print('Starting processing, this may take a while, please wait...\n')
    with open(train_index_file, 'r') as train_idx:
        for line in tqdm.tqdm(train_idx):
            f = open(line.strip(), 'r')
            paths = json.load(f)
            f.close()
            if paths != list(""): # some functions were just "RET", which is useless
                with open(train_file, 'a') as train_f:
                    for path in paths:
                        train_f.write(path)
                        train_f.write('\n')
                        train_f.write('\n')
    print('Done.')


if __name__ == '__main__':
    main()