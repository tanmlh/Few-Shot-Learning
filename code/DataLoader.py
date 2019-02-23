import random
import torch
import torchnet as tnt

def get_cls2idxes(labels):
    """
    Given a list of labels, return a map of label to list of indexes of samples
    """
    cls2idxes = {}
    for i, ele in enumerate(labels):
        if cls2idxes.get(ele) is None:
            cls2idxes[ele] = []
        cls2idxes[ele].append(i)
    return cls2idxes

class EpisodeLoader(object):
    """
    Return a Episode loader based on the given dataset
    """
    def __init__(self, dataset, episode_param, batch_size=1, num_workers=4, epoch_size=2000):
        self.dataset = dataset
        self.num_cats = episode_param['num_cats']
        self.num_sup_per_cat = episode_param['num_sup_per_cat']
        self.num_que_per_cat = episode_param['num_que_per_cat']
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.epoch_size = epoch_size


        self.cls2idxes = get_cls2idxes(self.dataset.labels)
        self.labels = self.dataset.labels
        self.label_set = self.cls2idxes.keys()

    def sample_episode(self):
        # sample support examplars
        cats = random.sample(self.label_set, self.num_cats)
        sup_examps = []
        que_examps = []

        for i in range(len(cats)):
            examps = random.sample(self.cls2idxes[cats[i]], self.num_sup_per_cat + self.num_que_per_cat)
            sup_examps += [(examp, i) for examp in examps[:self.num_sup_per_cat]]
            que_examps += [(examp, i) for examp in examps[self.num_sup_per_cat:]]

        random.shuffle(sup_examps)
        random.shuffle(que_examps)

        return sup_examps, que_examps, cats

    def idx2tensor(self, examps):
        """
        Return tensors of images and labels with the given indexes of images and list of labels
        """
        data = torch.stack([self.dataset[idx][0] for idx, _ in examps])
        labels = torch.LongTensor([label for _, label in examps])

        return data, labels

    def get_iterator(self, seed=0):
        """
        Return a iterator of the data loader
        """
        # random.seed(seed)
        def load_func(idx):
            sup_examps, que_examps, cats = self.sample_episode()
            sup_data, sup_labels = self.idx2tensor(sup_examps)
            que_data, que_labels = self.idx2tensor(que_examps)
            cats = [cats[i] for i in que_labels]
            cats = torch.LongTensor(cats)
            return sup_data, sup_labels, que_data, que_labels, cats

        tnt_dataset = tnt.dataset.ListDataset(range(self.epoch_size), load_func)
        data_loader = tnt_dataset.parallel(batch_size=self.batch_size,
                                           num_workers=self.num_workers,
                                           shuffle=True)
        return data_loader

    def __call__(self, seed=0):
        return self.get_iterator(seed)

    def __len__(self):
        return self.epoch_size / self.batch_size

class BatchLoader(object):

    def __init__(self, dataset, batch_size, num_workers):

        self.length = len(dataset)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dataset = dataset


    def get_loader(self):

        def load_func(idx):
            return self.dataset[idx]

        tnt_dataset = tnt.dataset.ListDataset(range(self.length), load_func)
        data_loader = tnt_dataset.parallel(batch_size=self.batch_size,
                                           num_workers=self.num_workers,
                                           shuffle=True)
        self.data_loader = data_loader
        return self.data_loader

    def __call__(self, seed=0):
        return self.get_loader()

def get_loader(dataset):
    from dataloader import FewShotDataloader
    dloader = FewShotDataloader(
        dataset=dataset,
        nKnovel=5,
        nKbase=0,
        nExemplars=1, # num training examples per novel category
        nTestNovel=30, # num test examples for all the novel categories
        nTestBase=0, # num test examples for all the base categories
        batch_size=8,
        num_workers=7,
        epoch_size=2000, # num of batches per epoch
    )
    return dloader

