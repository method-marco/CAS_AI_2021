from sdv.timeseries.deepecho import PAR

from dataset import MyDataset
from env import MyEnv

if __name__ == '__main__':

    dataset = MyDataset()
    train, test = dataset.get_train_test()

    train.index.name = 'date'
    train_sdv = train.reset_index(drop=False)
    model = PAR(sequence_index='date')

    model.fit(train_sdv)
    model.save('train_sdv_model.pkl')