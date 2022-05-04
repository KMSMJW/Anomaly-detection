import TadGAN_trainer
import processing as pr

dataloader = pr.data_loader()

train_set, test_set, label_set = dataloader.MSL()

train_set, test_set = dataloader.min_max(train_set, test_set)

train_set_overlap = dataloader.window_overlap(train_set,100,1)
test_set_overlap = dataloader.window_overlap(test_set,100,1)

trainer = TadGAN_trainer.TadGAN_trainer()

trainer.init_trainset(train_set_overlap)

trainer.build_tadgan()

trainer.train(1,128,5)

trainer.ROC_score(test_set_overlap,test_set,label_set,1,name='test')