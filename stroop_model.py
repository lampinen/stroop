from itertools import permutations

from copy import deepcopy

import numpy as np
import tensorflow as tf

from HoMM import HoMM_model
from HoMM.configs import default_run_config, default_architecture_config


## stroop task

stroop_inputs = np.array([[1., 0., 1. ,0.],
                          [1., 0., 0. ,1.],
                          [0., 1., 1. ,0.],
                          [0., 1., 0. ,1.]])

stroop_targets_word = stroop_inputs[:, :2]
stroop_targets_color = stroop_inputs[:, 2:]

incongruent_stimuli = [1, 2]



## model

run_config = default_run_config.default_run_config
run_config.update({
    "output_dir_format": "stroop_mixing_results_pw_%.1f/",

    "train_meta": False,

    "proportion_word_training": 0.9,
    
    "num_epochs": 10000,
    "early_stopping_thresh": 1e-1,

    "init_learning_rate": 1e-2,
    "lr_decay": 1.,

    "num_runs": 25,
})

architecture_config = default_architecture_config.default_architecture_config

architecture_config.update({
    "input_shape": [4],
    "output_shape": [2],

    "z_dim": 32,

    "M_num_hidden": 64,
    "H_num_hidden": 64,

    "F_num_hidden_layers": 0,  # linear task network

    "optimizer": "SGD",
    
    "meta_batch_size": 2,
    #"F_num_hidden": 32,
})

if False:  # enable for task conditioned
    architecture_config.update({
        "task_conditioned_not_hyper": True,

        "F_num_hidden_layers": 2,
    })

    run_config.update({
        "output_dir_format": "stroop_mixing_results_cnh_pw_%.1f/"
    })

def xe_loss(output_logits, targets):
    return tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(logits=output_logits,
                                                    labels=targets))


class stroop_model(HoMM_model.HoMM_model):
    def __init__(self, run_config=None):
        super(stroop_model, self).__init__(
            architecture_config=architecture_config, run_config=run_config,
            base_loss=xe_loss)

    def _pre_build_calls(self):
        self.base_train_tasks = []
        self.base_eval_tasks = []

        # set up the meta tasks

        self.meta_class_train_tasks = []
        self.meta_class_eval_tasks = [] 

        self.meta_map_train_tasks = []
        self.meta_map_eval_tasks = [] 

        # set up the meta pairings 
        self.meta_pairings = {}

    def fill_buffers(self, num_data_points=1):
        pass

    def _pre_loss_calls(self):
        def _logits_to_accuracy(x, labels=self.base_target_ph):
            #mask = tf.logical_not(backward_mask)  # only held out examples
            #masked_x = tf.boolean_mask(x, mask)
            #masked_labels = tf.boolean_mask(labels, mask)
            x_top_inds = tf.argmax(x, axis=-1)
            label_top_inds = tf.argmax(labels, axis=-1)
            
            return tf.reduce_mean(tf.cast(tf.equal(x_top_inds, label_top_inds),
                                          tf.float32))
        self.base_accuracy = _logits_to_accuracy(self.base_output)
        self.base_fed_emb_accuracy =  _logits_to_accuracy(self.base_fed_emb_output)


    def build_feed_dict(self, task, from_zeros=False, lr=None, mix_num=0,
                        call_type="base_standard_train"):
        feed_dict = {}
        base_or_meta, call_type, train_or_eval = call_type.split("_")

        if task == "word":
            task_data = stroop_targets_word 
            other_task_data = stroop_targets_color 
        else:
            task_data = stroop_targets_word 
            other_task_data = stroop_targets_color 
        
        if mix_num > 0:
            this_dataset_y = np.zeros([4 + mix_num % 2, 2], dtype=np.float32)
            this_dataset_x = np.zeros([4 + mix_num % 2, 4], dtype=np.float32)
            num_other_task = mix_num // 2 + mix_num % 2 
            num_this_task = 4 - (mix_num // 2)
            this_task_indices = np.random.choice(4, num_this_task, 
                                                 replace=False)
            other_task_indices = np.random.choice(4, num_other_task,
                                                  replace=False)
            this_dataset_y[:num_this_task] = task_data[this_task_indices, :] 
            this_dataset_y[num_this_task:] = other_task_data[other_task_indices, :] 
            this_dataset_x[:num_this_task] = stroop_inputs[this_task_indices, :] 
            this_dataset_x[num_this_task:] = stroop_inputs[other_task_indices, :] 
        else:
            this_dataset_y = task_data
            this_dataset_x = stroop_inputs 

        feed_dict[self.base_input_ph] = this_dataset_x 
        feed_dict[self.base_target_ph] = this_dataset_y

        if from_zeros:
            feed_dict[self.feed_embedding_ph] = np.zeros(
                [1, self.architecture_config["z_dim"]], dtype=np.float32)

        if train_or_eval == "train":
            if task == "word":  # weighting learning rate ~= training more often
                lr *= self.run_config["proportion_word_training"]
            else:
                lr *= 1 - self.run_config["proportion_word_training"]
            feed_dict[self.lr_ph] = lr
            feed_dict[self.keep_prob_ph] = self.tkp
            feed_dict[self.guess_input_mask_ph] = self._random_guess_mask(
                len(stroop_inputs))
        else:
            feed_dict[self.keep_prob_ph] = 1.
            feed_dict[self.guess_input_mask_ph] = np.ones(len(stroop_inputs), 
                                                          dtype=np.bool) 

        return feed_dict

    def base_eval(self, task):

        name = str(task) + ":{}:{}:{}"
        res = []
        names = []
        for from_zeros in [False, True]:
            for mix_num in range(9):
                if from_zeros == True and mix_num > 0:
                    break
                feed_dict = self.build_feed_dict(task, from_zeros=from_zeros, 
                                                 mix_num=mix_num,
                                                 call_type="base_standard_eval")
                if from_zeros:
                    call_type = "from_zeros"
                    fetches = [self.total_base_fed_emb_loss, self.base_fed_emb_accuracy]
                else:
                    call_type = "standard"
                    fetches = [self.total_base_loss, self.base_accuracy]
                res += self.sess.run(fetches, feed_dict=feed_dict)

                names += [name.format("standard", mix_num, "loss"),
                          name.format("standard", mix_num, "accuracy")]


        return names, res

    def run_base_eval(self):
        """Run evaluation on basic tasks."""
        names = []
        losses = []
        for task in ["word", "color"]:
            these_names, these_losses = self.base_eval(task)
            names += these_names
            losses += these_losses

        return names, losses

    def run_eval(self, epoch, print_losses=True):
        epoch_s = "%i, " % epoch

        base_names, base_losses = self.run_base_eval()
        if epoch == 0:
            # set up format string
            self.loss_format = ", ".join(["%f" for _ in base_names]) + "\n"

            # write headers and overwrite existing file 
            with open(self.run_config["loss_filename"], "w") as fout:
                fout.write("epoch, " + ", ".join(base_names) + "\n")
        with open(self.run_config["loss_filename"], "a") as fout:
            formatted_losses = epoch_s + (self.loss_format % tuple(base_losses))
            fout.write(formatted_losses)

        if print_losses:
            print(formatted_losses)

        return base_losses

    def run_training(self):
        """Train model."""

        eval_every = self.run_config["eval_every"]

        learning_rate = self.run_config["init_learning_rate"]

        num_epochs = self.run_config["num_epochs"]
        lr_decays_every = self.run_config["lr_decays_every"]
        lr_decay = self.run_config["lr_decay"]
        min_learning_rate = self.run_config["min_learning_rate"]

        self.run_eval(epoch=0)

        tasks = ["word", "color"] 

        word_only = run_config["proportion_word_training"] == 1.
        color_only = run_config["proportion_word_training"] == 0.

        if word_only:
            tasks = ["word"]
        elif color_only:
            tasks = ["color"]

        for epoch in range(1, num_epochs+1):
            order = np.random.permutation(len(tasks))
            for task_i in order:
                task = tasks[task_i]
                self.base_train_step(task, learning_rate)

            if epoch % eval_every == 0:
                scores = self.run_eval(epoch)
                words_learned = scores[0] < run_config["early_stopping_thresh"] and scores[1] == 1.
                colors_learned = scores[4] < run_config["early_stopping_thresh"] and scores[5] == 1.
                if (words_learned and colors_learned) or (words_learned and word_only) or (colors_learned and color_only):
                    print("Early stop!")
                    break

            if epoch % lr_decays_every == 0 and epoch > 0:
                if learning_rate > min_learning_rate:
                    learning_rate *= lr_decay


## running stuff
for run_i in range(run_config["num_runs"]):
    for pwt in [0.5, 0.9, 0.1]:
        run_config["proportion_word_training"] = pwt
        run_config["output_dir"] = run_config["output_dir_format"] % run_config["proportion_word_training"]
        np.random.seed(run_i)
        tf.set_random_seed(run_i)
        run_config["this_run"] = run_i

        model = stroop_model(run_config=run_config)
        model.run_training()

        tf.reset_default_graph()
