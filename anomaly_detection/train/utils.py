from model import AE, CONV_VAE, LSTM, LSTM_VAE
from tqdm.autonotebook import tqdm
import tensorflow as tf
import numpy as np
from IPython import display
import processing as pr
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
import os

class trainer:
    def __init__(self):
        self.mse_loss_fn = tf.keras.losses.MeanSquaredError()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)

    def init_trainset(self,train_set):
        """
        Decision model input size

        Args:
            train_set (ndarray):
                train set
        
        Returns:

        """
        self.train_set = train_set
        self.seqs = train_set.shape[1]
        self.channels = train_set.shape[2]

    def build_ae(self):
        """
        build autoencoder model
        """
        self.model = AE.ae(self.seqs, self.channels)

    def build_conv_vae(self):
        """
        build convolutional variational autoencoder model
        """
        self.model = CONV_VAE.VariationalAutoEncoder(self.seqs, self.channels)

    def build_lstm(self):
        """
        build lstm model
        """
        self.model = LSTM.LSTM(self.seqs, self.channels)

    def build_lstm_vae(self):
        """
        build lstm variational autoencoder model
        """
        self.model = LSTM_VAE.VariationalAutoEncoder(self.channels)

    def train(self, epochs, batch_size):
        """
        training model

        Args:
            epochs (int):
                number of epoch
            batch_size (int):
                number of batch size
        """
        for epoch in range(epochs):
            print("\nStart of epoch %d" % (epoch,))
            copy_set = np.copy(self.train_set)
            for k in tqdm(range(copy_set.shape[0]//batch_size)):
                batch_mask = np.random.choice(copy_set.shape[0], batch_size, replace=False)
                batch = copy_set[batch_mask]
                with tf.GradientTape() as tape:
                    input = tf.constant(batch, dtype='float32')
                    prediction = self.model(input)
                    reconstruction_error = self.mse_loss_fn(input,prediction)
                    latent_loss = sum(self.model.losses)
                    loss_value = reconstruction_error + latent_loss
                gradients = tape.gradient(loss_value, self.model.trainable_weights)
                self.optimizer.apply_gradients(zip(gradients, self.model.trainable_weights))
                copy_set = np.delete(copy_set,batch_mask,axis=0)
            display.clear_output()
            if float(tf.reduce_mean(reconstruction_error)) != 0.0:
                print("Reconstruction error at epoch %d: %.4f"% (epoch, float(tf.reduce_mean(reconstruction_error))))
            if float(tf.reduce_mean(latent_loss)) != 0.0:
                print("latent loss at epoch %d: %.4f"% (epoch, float(tf.reduce_mean(latent_loss))))
            if float(tf.reduce_mean(loss_value)) != 0.0:
                print("total loss at epoch %d: %.4f"% (epoch, float(tf.reduce_mean(loss_value))))

    def save_model(self, name):
        """
        save model

        Args:
            name (str):
                model name
        """
        name = str(name)
        self.model.save('./model_parameter/' + name)

    def load_model(self, name):
        """
        load model

        Args:
            name (str):
                model name
        """
        name = str(name)
        self.model = tf.keras.models.load_model('./model_parameter/' + name)

    def predict(self, data):
        """
        prediction

        Args:
            data (ndarray):
                input data
        
        Returns:
            ndarray:
                predict data
        """
        if len(data.shape) == 3:
            predict = self.model(data)
        else:
            input = np.array([data])
            predict = self.model(input)
        return predict

    def ROC_score(self, test_set, test_original, label_set, step_size, name):
        """
        get ROC score and save figure

        Args:
            test_set (ndarray):
                test set with smoothing window applied
            test_original (ndarray):
                test set without smoothing window applied
            label_set (ndarray, 0 or 1):
                label set
            step_size (int):
                step size for smoothing window
            name (str):
                figure save name
        """
        name = str(name)
        num = test_set.shape[0] // 100
        reconstruct_set = list()
        latent_set = list()
        self.seqs = test_set.shape[1]
        print("----------------  scoring  ------------------")
        for i in tqdm(range(0,num)):
            input = test_set[i*100:(i+1)*100]
            recon = self.model(input)
            reconstruct_set.extend(recon)
            if len(self.model.losses) == 0:
                latent_score = self.model.losses
            else:
                latent_score = self.model.losses[0]
            latent_set.extend(latent_score)
        input = test_set[num*100:]
        recon = self.model(input)
        reconstruct_set.extend(recon)
        if len(self.model.losses) == 0:
                latent_score = self.model.losses
        else:
            latent_score = self.model.losses[0]
        latent_set.extend(latent_score)
        reconstruct_set = np.array(reconstruct_set)
        latent_set = np.array(latent_set)
        print("--------------- reconstructing  ----------------")
        if step_size == 1:
            print("--------kde scoring---------")
            kde_score = pr.data_loader().kde_score(latent_set, self.seqs)
            print("--------average predict----------")
            predict = pr.data_loader().pred(reconstruct_set)
            recon_error = (predict - test_original)**2
            error_score = np.mean(recon_error, axis=1)
        else:
            kde_score = latent_set
            recon_error = (reconstruct_set - test_set)**2
            label_set = pr.data_loader().label_window(label_set,self.seqs)
            error_score = np.mean(np.mean(recon_error, axis=1), axis=1)
        error_score = (error_score - np.mean(error_score)) / np.std(error_score)
        if np.isnan(np.sum(kde_score)) or np.sum(kde_score) == 0:
            score_set = error_score
        else:
            kde_score = (kde_score - np.mean(kde_score)) / np.std(kde_score)
            score_set = error_score + kde_score
        false_positive_rate, true_positive_rate, thresholds = roc_curve(label_set, score_set)
        roc_auc = metrics.roc_auc_score(label_set, score_set)
        plt.title('ROC-AUC')
        plt.xlabel('False Positive Rate(1 - Specificity)')
        plt.ylabel('True Positive Rate(Sensitivity)')

        plt.plot(false_positive_rate, true_positive_rate, 'b', label='Model (AUC = %0.2f)'% roc_auc)
        plt.plot([0,1],[1,1],'y--')
        plt.plot([0,1],[0,1],'r--')

        plt.legend(loc='lower right')
        plt.savefig(os.path.dirname(os.path.abspath(__file__)) + '/graph/' + name + '.png', dpi=300)
        # plt.show()