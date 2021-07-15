import os

for dataset in ['amazon', 'dslr', 'webcam']:
    with open(dataset + '.txt', 'r') as f:
        lines = f.readlines()

    classes = {}
    for line in lines:
        a, b = line.split(' ')
        b = b.strip()
        if b not in classes:
            classes[b] = []
        classes[b].append(a)

    train = ''
    val = ''
    for k in classes:
        n = int(len(classes[k]) * 0.9)
        for item in classes[k][:n]:
            train += str(item) + ' ' + str(k) + '\n'
        for item in classes[k][n:]:
            val += str(item) + ' ' + str(k) + '\n'


    with open(dataset + '_train.txt', 'w') as f:
        f.write(train)
    with open(dataset + '_val.txt', 'w') as f:
        f.write(val)
