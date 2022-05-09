from . import processing as pr
import tensorflow as tf
from . import TadGAN_util as util
import numpy as np
from tqdm.autonotebook import tqdm
from IPython import display
from sklearn.metrics import roc_curve
from sklearn import metrics
import matplotlib.pyplot as plt

class TadGAN_trainer:
    def __init__(self):
        learning_rate = 0.0005
        self.encoder_optimizer = tf.keras.optimizers.Adam(learning_rate)
        self.generator_optimizer = tf.keras.optimizers.Adam(learning_rate)
        self.critic_x_optimizer = tf.keras.optimizers.Adam(learning_rate)
        self.critic_z_optimizer = tf.keras.optimizers.Adam(learning_rate)

    def init_trainset(self, train_set):
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
        self.latent_dim = self.seqs//5

        self.encoder_input_shape = (self.seqs, self.channels)
        self.generator_input_shape = (self.seqs//5, self.channels)
        self.critic_x_input_shape = (self.seqs, self.channels)
        self.critic_z_input_shape = (self.seqs//5, self.channels)
        self.encoder_reshape_shape = (self.seqs//5, self.channels)
        self.generator_reshape_shape = (self.seqs//2, self.channels)

    def build_tadgan(self):
        """
        build TadGAN model
        """
        self.encoder = util.build_encoder_layer(input_shape=self.encoder_input_shape, encoder_reshape_shape=self.encoder_reshape_shape)
        self.generator = util.build_generator_layer(input_shape=self.generator_input_shape, generator_reshape_shape=self.generator_reshape_shape)
        self.critic_x = util.build_critic_x_layer(input_shape=self.critic_x_input_shape)
        self.critic_z = util.build_critic_z_layer(input_shape=self.critic_z_input_shape)

    def train(self, epochs, batch_size, n_critics):
        """
        training model

        Args:
            epochs (int):
                number of epoch
            batch_size (int):
                number of batch size
            n_critics (int):
                number of critics
        """

        X_ = np.copy(self.train_set)

        for epoch in range(1, epochs+1):
            epoch_e_loss = list()
            epoch_g_loss = list()
            epoch_cx_loss = list()
            epoch_cz_loss = list()
    
            np.random.shuffle(X_)

            minibatches_size = batch_size*n_critics
            num_minibatches = int(X_.shape[0]//minibatches_size)

            self.encoder.trainable = False
            self.generator.trainable = False

            for i in tqdm(range(num_minibatches), total=num_minibatches):
                minibatch = X_[i*minibatches_size:(i+1)*minibatches_size]

                for j in range(n_critics):

                    x = minibatch[j*batch_size:(j+1)*batch_size]
                    z = tf.random.normal(shape=(batch_size, self.latent_dim, self.channels), mean=0.0, stddev=1, dtype=tf.dtypes.float32, seed=1748)

                    self.critic_x.trainable = True
                    self.critic_z.trainable = False
                    epoch_cx_loss.append(util.critic_x_train_on_batch(x,z,self.critic_x, self.generator, batch_size, self.critic_x_optimizer))
                    self.critic_x.trainable = False
                    self.critic_z.trainable = True
                    epoch_cz_loss.append(util.critic_z_train_on_batch(x,z,self.encoder, self.critic_z, batch_size,self.critic_z_optimizer))

                self.critic_z.trainable = False
                self.critic_x.trainable = False
                self.encoder.trainable = True
                self.generator.trainable = True

                enc_loss, gen_loss = util.enc_gen_train_on_batch(x, z,self.encoder, self.generator, self.critic_x, self.critic_z, self.encoder_optimizer, self.generator_optimizer)
                epoch_e_loss.append(enc_loss)
                epoch_g_loss.append(gen_loss)

            cx_loss = np.mean(np.array(epoch_cx_loss), axis=0)
            cz_loss = np.mean(np.array(epoch_cz_loss), axis=0)
            e_loss = np.mean(np.array(epoch_e_loss), axis=0)
            g_loss = np.mean(np.array(epoch_g_loss), axis=0)
            display.clear_output()

            print('Epoch: {}/{}, [Dx loss: {}] [Dz loss: {}] [E loss: {}] [G loss: {}]'.format(epoch, epochs, cx_loss, cz_loss, e_loss, g_loss))

    def save_model(self, name):
        """
        save model

        Args:
            name (str):
                model name
        """
        self.generator.save('./model_parameter/' + name + '/generator')
        self.encoder.save('./model_parameter/' + name + '/encoder')
        self.critic_x.save('./model_parameter/' + name + '/critic_x')
        self.critic_z.save('./model_parameter/' + name + '/critic_z')

    def load_model(self, name):
        """
        load model

        Args:
            name (str):
                model name
        """
        self.generator = tf.keras.models.load_model('./model_parameter/' + name + '/generator')
        self.encoder = tf.keras.models.load_model('./model_parameter/' + name + '/encoder')
        self.critic_x = tf.keras.models.load_model('./model_parameter/' + name + '/critic_x')
        self.critic_z = tf.keras.models.load_model('./model_parameter/' + name + '/critic_z')

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
            predict = self.generator(self.encoder(data))
        else:
            input = np.array([data])
            predict = self.generator(self.encoder(input))
        return predict

    def ROC_score(self, test_set, test_original, label_set, step_size, name):
        """
        get ROC score and save figure

        Args:
            test_set (ndarray):
                test set with smoothing window applied
            test_original (ndarray):
                test_set without smoothing window applied
            label_set (ndarray, 0 or 1):
                label set
            step_size (int):
                step size for smoothing window
            name (str):
                figure save name
        """
        num = test_set.shape[0]//100
        reconstruct_set = list()
        critic_set = list()
        self.seqs = test_set.shape[1]
        print("---------scoring----------")
        for i in tqdm(range(0,num)):
            i = test_set[i*100:(i+1)*100]
            recon = self.predict(i)
            reconstruct_set.extend(recon)
            critic_score = self.critic_x(i)
            critic_set.extend(critic_score)
        i = test_set[num*100:]
        recon = self.predict(i)
        reconstruct_set.extend(recon)
        critic_score = self.critic_x(i)
        critic_set.extend(critic_score)
        reconstruct_set = np.array(reconstruct_set)
        critic_set = np.array(critic_set)
        print("-----------reconstructing-------------")
        if step_size == 1:
            kde_score = pr.data_loader().kde_score(critic_set, self.seqs)
            predict = pr.data_loader().pred(reconstruct_set)
            recon_error = np.abs(predict - test_original)
            error_score = np.mean(recon_error, axis=-1)
        else:
            kde_score = critic_set
            recon_error = np.abs(reconstruct_set - test_set)
            label_set = pr.data_loader().label_window(label_set, self.seqs)
            error_score = np.mean(np.mean(recon_error, axis=-1), axis=-1)
        error_score = (error_score - np.mean(error_score)) / np.std(error_score)
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
        plt.savefig('./graph/' + name + '.png', dpi=300)
        plt.show()