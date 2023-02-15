import transformer_trainer
import processing as pr
import matplotlib.pyplot as plt

dataloader = pr.data_loader()
train_set, test_set, label_set = dataloader.MSL()

train_set, test_set = dataloader.min_max(train_set, test_set)

train_set_overlap = dataloader.window_overlap(train_set,100,0)
test_set_overlap = dataloader.window_overlap(test_set,100,0)

trainer = transformer_trainer.transformer_trainer()

trainer.init_trainset(train_set_overlap)

trainer.build_transformer()

trainer.train(10,256)

score_set = trainer.scoring(test_set_overlap[:333])

plt.plot(score_set)
plt.show()