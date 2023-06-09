from __future__ import print_function, division
import utils
import CGAN
import tensorflow as tf
import numpy as np
import csv
import tqdm
import os
import json

def train(cgan, X_train, y_train, X_test, y_test, epochs, batch_size, sample_interval, class_num):
    """
    Train the CGAN model.
    
    Args:
        cgan (CGAN): The CGAN model.
        X_train (ndarray): Training data features.
        y_train (ndarray): Training data labels.
        X_test (ndarray): Testing data features.
        y_test (ndarray): Testing data labels.
        epochs (int): Number of training epochs.
        batch_size (int): Batch size for training.
        sample_interval (int): Interval for saving generated  samples.
        channel_name (str): Name of the channels.
        class_num (int): Number of classes.
    """
    print("----------------------Training begins--------------------")
    # Adversarial ground truths
    valid = np.ones((batch_size, 1))
    fake = np.zeros((batch_size, 1))
    
    # Set callbacks
    callbacks = tf.keras.callbacks.CallbackList(
        None,
        add_history=True,
        add_progbar=True,
        model=cgan.discriminator,
        epochs=epochs,
        verbose=1,
        steps=1
    )
    callbacks.on_train_begin()
    
    # Set up training record file
    record = {
        'sample_interval': sample_interval,
        'recorded_ckpts': 0
    }
    with open('training_record.json', "w") as outfile:
        json.dump(record, outfile)
    
    for epoch in range(epochs):
        callbacks.on_epoch_begin(epoch)
        
        # Initialize for Discriminator
        idx = np.random.randint(0, X_train.shape[0] - 1, batch_size)
        features, labels = X_train[idx], y_train[idx]
        noise = np.random.normal(0, 1, (batch_size, 100))
        gen_features = cgan.generator.predict([noise, labels])

        # Train Discriminator
        d_loss_real = cgan.discriminator.train_on_batch([features, labels], valid)
        d_loss_fake = cgan.discriminator.train_on_batch([gen_features, labels], fake)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        # Initialize for Discriminator
        sampled_labels = np.random.randint(0, class_num - 1, batch_size).reshape(-1, 1)
        
        # Train Generator
        g_loss = cgan.combined.train_on_batch([noise, sampled_labels], valid)

        # Print the training progress
        print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch+1, d_loss[0], 100 * d_loss[1], g_loss))

        logs = {"train_loss": d_loss[0], "train_acc": d_loss[1]}
        
        # Save model and statistical result every $sample_interval$ epochs
        if (epoch+1) % sample_interval == 0:
            table_name = 'tabular_result/train/' + cgan.channel_name + '_test.csv'
            # Create the table if it doesn't exist
            if not os.path.exists(table_name):
                open(table_name, 'w').close()
            with open(table_name, 'a', newline='') as csvfile:
                csv_writer = csv.writer(csvfile, dialect='unix')
                #save model
                ckpt_save_path = cgan.manager.save()
                print (f'Saving checkpoint for epoch {epoch+1} at {ckpt_save_path}')
                # Opening training record file, increment the recorded ckpts number
                with open('training_record.json', 'r') as openfile:
                    record = json.load(openfile)
                    record['recorded_ckpts'] += 1
                with open('training_record.json', 'w') as outfile:
                    json.dump(record, outfile)
                
                # Perform prediction for the testing samples
                valid_all = np.ones((y_test.shape[0], 1))
                csv_row = cgan.discriminator.evaluate(x=[X_test, y_test], y=valid_all)
                csv_row.insert(0, str(epoch+1))
                csv_writer.writerow(csv_row)
                csvfile.close()
        
        callbacks.on_epoch_end(epoch, logs)
    callbacks.on_train_end()

    # Plot the training accuracy and loss
    history = cgan.discriminator.history
    if history is not None:
        utils.plot_history(history, cgan.channel_name)
    print("----------------------Training complete--------------------")

def test(cgan, X_test, y_test, class_num):
    """
    Test the CGAN model.
    
    Args:
        cgan (CGAN): The CGAN model.
        X_test (ndarray): Testing data features.
        y_test (ndarray): Testing data labels.
        cgan.channel_name (str): Name of the channel.
        class_num (int): Number of classes.
    """
    print("----------------------Test begins--------------------")
    
    with open(f'tabular_result/test/prediction_confidence_{cgan.channel_name}.csv', 'w', newline='') as csvfile:
        csv_writer = csv.DictWriter(csvfile, dialect='unix', fieldnames=['idx', 'true_label', 'predicted_label', 'score'])
        csv_writer.writeheader()
        
        for j in tqdm.tqdm(range(y_test.shape[0])):
            highest = None
            highest_label = None
            
            for label in range(class_num):
                X_test_j = np.expand_dims(X_test[j], axis=0)
                sample_confs = cgan.discriminator.predict([X_test_j, np.array([label])], verbose=0)
                
                if highest is None:
                    highest = sample_confs[0][0]
                    highest_label = label
                elif highest < sample_confs[0][0]:
                    highest = sample_confs[0][0]
                    highest_label = label
            
        csv_writer.writerow({"idx": j, "true_label": y_test[j], "predicted_label": highest_label, "score": highest})
    
    print("----------------------Test complete--------------------")


if __name__ == '__main__':
    # Load the arguments
    args = utils.parse_arguments()
    model_path = args.load_ckpt
    npz_dir_path = args.load_npz_dir
    channel_name = '_'.join(args.selected_features)

    # Check if the provided npz file path is valid
    if npz_dir_path is None:
        raise ValueError(f"No input for npz directory name.")
    elif not os.path.exists(npz_dir_path):
        raise ValueError(f"Wrong npz directory name: '{npz_dir_path}' does not exist.")
    elif not os.path.exists(os.path.join(npz_dir_path, 'train.npz')):
        raise FileNotFoundError(f"File 'train.npz' does not exist in {npz_dir_path}.")
    elif not os.path.exists(os.path.join(npz_dir_path, 'test.npz')):
        raise FileNotFoundError(f"File 'test.npz' does not exist in {npz_dir_path}.")
    else:
        # Load the training and testing data
        with np.load(os.path.join(npz_dir_path, 'train.npz')) as data:
            X_train = data["X_train"]
            y_train = data["y_train"]
        with np.load(os.path.join(npz_dir_path, 'test.npz')) as data:
            X_test = data["X_test"]
            y_test = data["y_test"]
        y_test = np.array([y for y in y_test])

    # Perform reduction remensionality
    if (args.L_reduced_dim is None) ^ (args.F_reduced_dim is None):
        if (args.L_reduced_dim is not None and args.L_reduced_dim >= X_train.shape[1]):
            raise ValueError(f"Wrong reduced dimension input for {args.L_reduced_dim}: the reduced dimension must be smaller than the original dimension.")
        elif (args.F_reduced_dim is not None and args.F_reduced_dim >= X_train.shape[2]):
            raise ValueError(f"Wrong reduced dimension input for {args.F_reduced_dim}: the reduced dimension must be smaller than the original dimension.")
        X_train, _ = utils.reduce_dimension(X_train, args.L_reduced_dim, args.F_reduced_dim)
        X_test, _ = utils.reduce_dimension(X_test, args.L_reduced_dim, args.F_reduced_dim)
    else:
        raise ValueError(f"Wrong reduced dimension input: only one of L-reduced-dim and R-reduced-dim should be None.")

    # Channel selection
    feature_indices = {'MFCC': 0, 'DMFCC': 1, 'FBC': 2}
    selected_indices = [feature_indices[feature] for feature in args.selected_features if feature in feature_indices]
    X_train = X_train[:, :, :, selected_indices]
    X_test = X_test[:, :, :, selected_indices]
    
    # Load existing model (if provided)
    # Initialize model
    cgan = CGAN.CGAN(L_num=X_train.shape[1], F_num=X_train.shape[2], channels_num=len(args.selected_features),
                        num_classes=args.class_num, channel_name=channel_name)
    if os.path.exists(model_path):
        resume_epoch = int(args.ckpt_epoch)
        # Load the sample interval and recorded ckpts of the previous training model
        with open('training_record.json', 'r') as openfile:
            record = json.load(openfile)
            num_ckpts = record['recorded_ckpts']
            prev_sample_interval = record['sample_interval']
        ind_ckpt = int(resume_epoch / prev_sample_interval)
        cgan.update_model(model_path, num_ckpts, ind_ckpt)
    elif model_path != '':
        raise FileNotFoundError(f"Model '{model_path}' does not exist")

    # Create empty directory to save models
    if not os.path.exists('saved_model'):
        os.makedirs('saved_model')
        
    # Start training or testing
    if args.mode == 'train':
        # Create empty directory to log statistics
        if not os.path.exists('tabular_result/train'):
            os.makedirs('tabular_result/train')
        if not os.path.exists('graphical_result'):
            os.makedirs('graphical_result')
        train(cgan, X_train, y_train, X_test, y_test, args.epoch_num, args.batch_size, args.sample_size, args.class_num)
    elif args.mode == 'test':
        # Create empty directory to log statistics
        if not os.path.exists('tabular_result/test'):
            os.makedirs('tabular_result/test')
        test(cgan, X_test, y_test, args.class_num)
    else:
        raise ValueError(f"You need to set the mode to either train or test.")
