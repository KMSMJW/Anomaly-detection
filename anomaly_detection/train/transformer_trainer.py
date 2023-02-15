import tensorflow as tf
import transformer_model as model
import numpy as np
from tqdm.autonotebook import tqdm
from IPython import display

class transformer_trainer:
    def __init__(self):
        self.optimizer = tf.keras.optimizers.Adam(learning_rate = 0.0001)
        self.e_layers = 3
        self.mse = tf.keras.losses.MeanSquaredError()
    
    def mse_test(self,x,y):
        return (x-y)**2

    def kl_loss(self,p,q):
        res = p*(tf.math.log(p+0.0001) - tf.math.log(q+0.0001))
        return tf.reduce_mean(tf.reduce_sum(res, axis=-1), axis=1)

    def init_trainset(self, train_set):
        """
        Decision model input size

        Args:
            train_set (ndarray):
                train set
        
        Returns:

        """
        self.train_set = train_set
        self.win_size = train_set.shape[1]
        self.c_out = train_set.shape[2]

    def build_transformer(self):
        """
        build TadGAN model
        """
        self.model = model.AnomalyTransformer(win_size=self.win_size, c_out=self.c_out, e_layers=3)

    # def train_batch(self, minibatch):
    #     with tf.GradientTape(persistent=True) as tape:
    #         prediction = self.model(minibatch)
    #         reconstruction_error = tf.reduce_sum(tf.reduce_mean((prediction-minibatch)**2, axis=-1), axis=-1)
    #         Assdis = self.model.losses[0]
    #         Assdis = tf.reduce_sum(tf.abs(Assdis), axis=-1)
    #         loss_value1 = reconstruction_error + self.weights*Assdis
    #         loss_value2 = reconstruction_error - self.weights*Assdis
    #     gradients1 = tape.gradient(loss_value1, self.model.trainable_weights)
    #     gradients2 = tape.gradient(loss_value2, self.model.trainable_weights)
    #     self.optimizer.apply_gradients(zip(gradients1, self.model.trainable_weights))
    #     self.optimizer.apply_gradients(zip(gradients2, self.model.trainable_weights))
    #     return float(tf.reduce_mean(reconstruction_error)), float(tf.reduce_mean(Assdis))

    def train(self, epochs, batch_size):
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

        for epoch in range(epochs):
            print("\nStart of epoch %d" % (epoch,))
            copy_set = np.copy(self.train_set)
            for k in tqdm(range(copy_set.shape[0]//batch_size)):
                loss1_list = list()
                rec_loss_list = list()
                prior_loss_list = list()
                batch_mask = np.random.choice(copy_set.shape[0], batch_size, replace=False)
                batch = copy_set[batch_mask]
                series_loss = 0.0
                prior_loss = 0.0
                with tf.GradientTape(persistent=True) as tape:
                    input = tf.constant(batch, dtype='float32')
                    output, series, prior, _ = self.model(input)
                    # output = model(input)
                    for u in range(len(prior)):
                        series_loss += tf.reduce_mean(self.kl_loss(series[u],
                        tf.stop_gradient((prior[u] / tf.tile(tf.reduce_sum(prior[u], axis=-1)[...,tf.newaxis], [1,1,1,self.win_size]))))) \
                        + tf.reduce_mean(self.kl_loss(tf.stop_gradient((prior[u] / tf.tile(tf.reduce_sum(prior[u], axis=-1)[...,tf.newaxis], [1,1,1,self.win_size]))), series[u]))
                        prior_loss += tf.reduce_mean(self.kl_loss(tf.stop_gradient(series[u]), \
                        (prior[u] / tf.tile(tf.reduce_sum(prior[u], axis=-1)[...,tf.newaxis], [1,1,1,self.win_size])))) \
                        + tf.reduce_mean(self.kl_loss((prior[u] / tf.tile(tf.reduce_sum(prior[u], axis=-1)[...,tf.newaxis], [1,1,1,self.win_size])), tf.stop_gradient(series[u])))
                    series_loss = series_loss/len(series)
                    prior_loss = prior_loss/len(prior)
                    rec_loss = self.mse(output, input)
                    loss1 = rec_loss - 3*series_loss
                    loss2 = rec_loss + 3*prior_loss
                    loss1_list.append(loss1.numpy())
                    rec_loss_list.append(rec_loss.numpy())
                    prior_loss_list.append(prior_loss.numpy())
                gradients1 = tape.gradient(loss1, self.model.trainable_variables)
                gradients2 = tape.gradient(loss2, self.model.trainable_variables)
                self.optimizer.apply_gradients((grad,var) for (grad,var) in zip(gradients1, self.model.trainable_variables) if grad is not None)
                self.optimizer.apply_gradients((grad,var) for (grad,var) in zip(gradients2, self.model.trainable_variables) if grad is not None)
                copy_set = np.delete(copy_set,batch_mask,axis=0)
            display.clear_output()
            print("loss1:", np.average(loss1_list))
            print("rec:", np.average(rec_loss_list))
            print("prior:", np.average(prior_loss_list))

    def save_model(self, name):
        """
        save model

        Args:
            name (str):
                model name
        """
        self.model.save('./model_parameter/' + name)

    def load_model(self, name):
        """
        load model

        Args:
            name (str):
                model name
        """
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

    # def softmax(self, x, axis):
    #     log = tf.math.exp(x - tf.reduce_max(x, keepdims=True, axis=axis))
    #     sum = tf.reduce_sum(log, keepdims=True, axis=axis)
    #     return log/sum

    # def ROC_score(self, test_set, test_original, label_set, step_size, name):
    #     """
    #     get ROC score and save figure

    #     Args:
    #         test_set (ndarray):
    #             test set with smoothing window applied
    #         test_original (ndarray):
    #             test set without smoothing window applied
    #         label_set (ndarray, 0 or 1):
    #             label set
    #         step_size (int):
    #             step size for smoothing window
    #         name (str):
    #             figure save name
    #     """
    #     name = str(name)
    #     num = test_set.shape[0] // 100
    #     reconstruct_set = list()
    #     latent_set = list()
    #     self.seqs = test_set.shape[1]
    #     print("----------------  scoring  ------------------")
    #     for i in tqdm(range(0,num)):
    #         input = test_set[i*100:(i+1)*100]
    #         recon = self.model(input)
    #         reconstruct_set.extend(recon)
    #         if len(self.model.losses) == 0:
    #             latent_score = self.model.losses
    #         else:
    #             latent_score = self.model.losses[0]
    #         latent_set.extend(latent_score)
    #     input = test_set[num*100:]
    #     recon = self.model(input)
    #     reconstruct_set.extend(recon)
    #     if len(self.model.losses) == 0:
    #             latent_score = self.model.losses
    #     else:
    #         latent_score = self.model.losses[0]
    #     latent_set.extend(latent_score)
    #     reconstruct_set = np.array(reconstruct_set)
    #     latent_set = np.array(latent_set)
    #     print("--------------- reconstructing  ----------------")
    #     if step_size == 1:
    #         print("--------kde scoring---------")
    #         kde_score = pr.data_loader().kde_score(latent_set, self.seqs)
    #         print("--------average predict----------")
    #         predict = pr.data_loader().pred(reconstruct_set)
    #         recon_error = (predict - test_original)**2
    #         error_score = np.mean(recon_error, axis=1)
    #     else:
    #         kde_score = latent_set
    #         recon_error = (reconstruct_set - test_set)**2
    #         label_set = pr.data_loader().label_window(label_set,self.seqs)
    #         error_score = np.mean(np.mean(recon_error, axis=1), axis=1)
    #     error_score = (error_score - np.mean(error_score)) / np.std(error_score)
    #     if np.isnan(np.sum(kde_score)) or np.sum(kde_score) == 0:
    #         score_set = error_score
    #     else:
    #         kde_score = (kde_score - np.mean(kde_score)) / np.std(kde_score)
    #         score_set = error_score + kde_score
        # false_positive_rate, true_positive_rate, thresholds = roc_curve(label_set, score_set)
        # roc_auc = metrics.roc_auc_score(label_set, score_set)
        # plt.title('ROC-AUC')
        # plt.xlabel('False Positive Rate(1 - Specificity)')
        # plt.ylabel('True Positive Rate(Sensitivity)')

        # plt.plot(false_positive_rate, true_positive_rate, 'b', label='Model (AUC = %0.2f)'% roc_auc)
        # plt.plot([0,1],[1,1],'y--')
        # plt.plot([0,1],[0,1],'r--')

        # plt.legend(loc='lower right')
        # plt.savefig(os.path.dirname(os.path.abspath(__file__)) + '/graph/' + name + '.png', dpi=300)
        # # plt.show()

    def scoring(self, test_set_overlap):
        attens_energy = list()
        test_batch_size = 1
        num = test_set_overlap.shape[0] // test_batch_size
        for i in range(num):
            test_input = test_set_overlap[i*test_batch_size:(i+1)*test_batch_size]
            test_output, test_series, test_prior, _ = self.model(test_input)
            test_loss = tf.reduce_mean(self.mse_test(test_input, test_output), axis=-1)
            for u in range(len(test_prior)):
                if u == 0:
                    test_series_loss = self.kl_loss(test_series[u], (test_prior[u] / tf.tile(tf.reduce_sum(test_prior[u], axis=-1)[...,tf.newaxis], [1,1,1,self.win_size])))*50
                    test_prior_loss = self.kl_loss(test_prior[u] / tf.tile(tf.reduce_sum(test_prior[u], axis=-1)[...,tf.newaxis], [1,1,1,self.win_size]), test_series[u])*50
                else:
                    test_series_loss += self.kl_loss(test_series[u], (test_prior[u] / tf.tile(tf.reduce_sum(test_prior[u], axis=-1)[...,tf.newaxis], [1,1,1,self.win_size])))*50
                    test_prior_loss += self.kl_loss(test_prior[u] / tf.tile(tf.reduce_sum(test_prior[u], axis=-1)[...,tf.newaxis], [1,1,1,self.win_size]), test_series[u])*50
            metric = tf.nn.softmax((-test_prior_loss-test_series_loss), axis=-1)
            cri = metric*test_loss
            attens_energy.append(cri)
        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        return attens_energy