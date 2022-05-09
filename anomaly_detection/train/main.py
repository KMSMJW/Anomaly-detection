from . import utils
from . import processing as pr

if __name__ == "__main__":

    dataloader = pr.data_loader()

    train_set, test_set, label_set = dataloader.MSL()

    train_set, test_set = dataloader.min_max(train_set, test_set)

    train_set_overlap = dataloader.window_overlap(train_set,100,1)
    test_set_overlap = dataloader.window_overlap(test_set,100,1)

    trainer = utils.trainer()

    trainer.init_trainset(train_set_overlap)

    trainer.build_ae()

    trainer.train(1,512)

    # trainer.save_model(name='save_test')

    # trainer.load_model(name='save_test')

    # predict_test = trainer.predict(test_set_overlap[0])
    # print("predict:",predict_test)

    trainer.ROC_score(test_set_overlap,test_set,label_set, 1, name="test")