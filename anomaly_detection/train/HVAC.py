import pandas as pd
df = pd.read_csv("C:/Users/minseok/Desktop/archive/MZVAV-1.csv")
train_set = df[:10000]
train_set = train_set.to_numpy()
train_set = train_set[:,1:-2]
test_set = df.to_numpy()
test_set = test_set[:,1:-2]
import TadGAN_trainer
import processing as pr
import utils
import transformer_trainer

dataloader = pr.data_loader()

train_set, test_set = dataloader.min_max(train_set, test_set)

train_set_overlap = dataloader.window_overlap(train_set,100,0)
test_set_overlap = dataloader.window_overlap(test_set,100,0)

trainer = transformer_trainer.transformer_trainer()

trainer.init_trainset(train_set_overlap)

trainer.build_transformer()

trainer.train(10,256)