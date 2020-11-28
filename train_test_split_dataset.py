#####################################################################
### Splits into 3 .txt, one for train one for test one for validation
#####################################################################

from sklearn.model_selection import train_test_split


def main():
    dataset_file = '/home/valerio/valerio/dataset/bert/dataset_index.txt'
    train_file = '/home/valerio/valerio/dataset/bert/train_index.txt'
    valid_file = '/home/valerio/valerio/dataset/bert/valid_index.txt'
    test_file = '/home/valerio/valerio/dataset/test_index.txt'
    dataset = []
    fake_labels = []
    print('Reading dataset file...\n')
    with open(dataset_file, 'r') as d:
        while True:
            line = d.readline()
            if not line:
                break
            else:
                dataset.append(line.strip())
                fake_labels.append(0)
    print('Done. Performing train_test_split...\n')
    X_train, X_test, y_train, y_test = train_test_split(dataset, fake_labels, test_size=0.10,
                                                        shuffle=True)
    X_train_def, X_valid, y_train_def, y_valid = train_test_split(X_train, y_train,
                                                                  test_size=0.10, shuffle=True)
    print('Done. Writing files...\n')
    with open(train_file, 'w') as tf:
        for el in X_train_def:
            tf.write(el)
            tf.write('\n')
    with open(test_file, 'w') as test_f:
        for el in X_test:
            test_f.write(el)
            test_f.write('\n')
    with open(valid_file, 'w') as valid_tf:
        for el in X_valid:
            valid_tf.write(el)
            valid_tf.write('\n')

    print('Done.')


if __name__ == '__main__':
    main()


def trial():
    dataset_file = 'C:/Users/trent/Desktop/thesis/bert_pretraining_build_dataset/out.txt'
    train_file = 'C:/Users/trent/Desktop/thesis/bert_pretraining_build_dataset/train_index.txt'
    valid_file = 'C:/Users/trent/Desktop/thesis/bert_pretraining_build_dataset/valid_index.txt'
    test_file = 'C:/Users/trent/Desktop/thesis/bert_pretraining_build_dataset/test_index.txt'
    dataset = []
    fake_labels = []
    with open(dataset_file, 'r') as d:
        while True:
            line = d.readline()
            if not line:
                break
            else:
                dataset.append(line.strip())
                fake_labels.append(0)

    X_train, X_test, y_train, y_test = train_test_split(dataset, fake_labels, test_size=0.10,
                                                        shuffle=True)
    X_train_def, X_valid, y_train_def, y_valid = train_test_split(X_train, y_train,
                                                                  test_size=0.10, shuffle=True)
    with open(train_file, 'w') as tf:
        for el in X_train_def:
            tf.write(el)
            tf.write('\n')
    with open(test_file, 'w') as test_f:
        for el in X_test:
            test_f.write(el)
            test_f.write('\n')
    with open(valid_file, 'w') as valid_tf:
        for el in X_valid:
            valid_tf.write(el)
            valid_tf.write('\n')

    print('Done')
