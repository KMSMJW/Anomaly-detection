
�7root"_tf_keras_network*�7{"name": "critic_z", "trainable": false, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "Functional", "config": {"name": "critic_z", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 20, 25]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_4"}, "name": "input_4", "inbound_nodes": []}, {"class_name": "Flatten", "config": {"name": "flatten_3", "trainable": false, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_3", "inbound_nodes": [[["input_4", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_4", "trainable": false, "dtype": "float32", "units": 4000, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_4", "inbound_nodes": [[["flatten_3", 0, 0, {}]]]}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_4", "trainable": false, "dtype": "float32", "alpha": 0.20000000298023224}, "name": "leaky_re_lu_4", "inbound_nodes": [[["dense_4", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_4", "trainable": false, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}, "name": "dropout_4", "inbound_nodes": [[["leaky_re_lu_4", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_5", "trainable": false, "dtype": "float32", "units": 4000, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_5", "inbound_nodes": [[["dropout_4", 0, 0, {}]]]}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_5", "trainable": false, "dtype": "float32", "alpha": 0.20000000298023224}, "name": "leaky_re_lu_5", "inbound_nodes": [[["dense_5", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_5", "trainable": false, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}, "name": "dropout_5", "inbound_nodes": [[["leaky_re_lu_5", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_6", "trainable": false, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_6", "inbound_nodes": [[["dropout_5", 0, 0, {}]]]}], "input_layers": [["input_4", 0, 0]], "output_layers": [["dense_6", 0, 0]]}, "shared_object_id": 15, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 20, 25]}, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 20, 25]}, "is_graph_network": true, "full_save_spec": {"class_name": "__tuple__", "items": [[{"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 20, 25]}, "float32", "input_4"]}], {}]}, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 20, 25]}, "float32", "input_4"]}, "keras_version": "2.7.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "critic_z", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 20, 25]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_4"}, "name": "input_4", "inbound_nodes": [], "shared_object_id": 0}, {"class_name": "Flatten", "config": {"name": "flatten_3", "trainable": false, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_3", "inbound_nodes": [[["input_4", 0, 0, {}]]], "shared_object_id": 1}, {"class_name": "Dense", "config": {"name": "dense_4", "trainable": false, "dtype": "float32", "units": 4000, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 2}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 3}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_4", "inbound_nodes": [[["flatten_3", 0, 0, {}]]], "shared_object_id": 4}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_4", "trainable": false, "dtype": "float32", "alpha": 0.20000000298023224}, "name": "leaky_re_lu_4", "inbound_nodes": [[["dense_4", 0, 0, {}]]], "shared_object_id": 5}, {"class_name": "Dropout", "config": {"name": "dropout_4", "trainable": false, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}, "name": "dropout_4", "inbound_nodes": [[["leaky_re_lu_4", 0, 0, {}]]], "shared_object_id": 6}, {"class_name": "Dense", "config": {"name": "dense_5", "trainable": false, "dtype": "float32", "units": 4000, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 7}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 8}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_5", "inbound_nodes": [[["dropout_4", 0, 0, {}]]], "shared_object_id": 9}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_5", "trainable": false, "dtype": "float32", "alpha": 0.20000000298023224}, "name": "leaky_re_lu_5", "inbound_nodes": [[["dense_5", 0, 0, {}]]], "shared_object_id": 10}, {"class_name": "Dropout", "config": {"name": "dropout_5", "trainable": false, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}, "name": "dropout_5", "inbound_nodes": [[["leaky_re_lu_5", 0, 0, {}]]], "shared_object_id": 11}, {"class_name": "Dense", "config": {"name": "dense_6", "trainable": false, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 12}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 13}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_6", "inbound_nodes": [[["dropout_5", 0, 0, {}]]], "shared_object_id": 14}], "input_layers": [["input_4", 0, 0]], "output_layers": [["dense_6", 0, 0]]}}}2
�root.layer-0"_tf_keras_input_layer*�{"class_name": "InputLayer", "name": "input_4", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 20, 25]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 20, 25]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_4"}}2
�root.layer-1"_tf_keras_layer*�{"name": "flatten_3", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Flatten", "config": {"name": "flatten_3", "trainable": false, "dtype": "float32", "data_format": "channels_last"}, "inbound_nodes": [[["input_4", 0, 0, {}]]], "shared_object_id": 1, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}, "shared_object_id": 17}}2
�root.layer_with_weights-0"_tf_keras_layer*�{"name": "dense_4", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_4", "trainable": false, "dtype": "float32", "units": 4000, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 2}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 3}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["flatten_3", 0, 0, {}]]], "shared_object_id": 4, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 500}}, "shared_object_id": 18}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 500]}}2
�root.layer-3"_tf_keras_layer*�{"name": "leaky_re_lu_4", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_4", "trainable": false, "dtype": "float32", "alpha": 0.20000000298023224}, "inbound_nodes": [[["dense_4", 0, 0, {}]]], "shared_object_id": 5}2
�root.layer-4"_tf_keras_layer*�{"name": "dropout_4", "trainable": false, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dropout", "config": {"name": "dropout_4", "trainable": false, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}, "inbound_nodes": [[["leaky_re_lu_4", 0, 0, {}]]], "shared_object_id": 6}2
�root.layer_with_weights-1"_tf_keras_layer*�{"name": "dense_5", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_5", "trainable": false, "dtype": "float32", "units": 4000, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 7}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 8}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["dropout_4", 0, 0, {}]]], "shared_object_id": 9, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 4000}}, "shared_object_id": 19}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 4000]}}2
�root.layer-6"_tf_keras_layer*�{"name": "leaky_re_lu_5", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_5", "trainable": false, "dtype": "float32", "alpha": 0.20000000298023224}, "inbound_nodes": [[["dense_5", 0, 0, {}]]], "shared_object_id": 10}2
�root.layer-7"_tf_keras_layer*�{"name": "dropout_5", "trainable": false, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dropout", "config": {"name": "dropout_5", "trainable": false, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}, "inbound_nodes": [[["leaky_re_lu_5", 0, 0, {}]]], "shared_object_id": 11}2
�	root.layer_with_weights-2"_tf_keras_layer*�{"name": "dense_6", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_6", "trainable": false, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 12}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 13}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["dropout_5", 0, 0, {}]]], "shared_object_id": 14, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 4000}}, "shared_object_id": 20}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 4000]}}2