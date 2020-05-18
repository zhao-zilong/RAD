import numpy as np

def gennoise(train_labels, n_classes, n_init, last_clean, batch_size, time_horizon, noise_min, noise_max, noise_type, noise_steps):
    # make a copy of the labels before chaning them
    train_labels_noisy = train_labels.copy()
    start_next_batch = last_clean # skip initial labels


    noise_per_batch = np.arange(int(time_horizon))
    train_labels_good = np.ones(len(train_labels_noisy))
    j = 0
    noise_interval = noise_steps[0]
    noise_interval_idx = -1
    for i in np.arange(int(time_horizon)):
        # generate dynamic noise pattern
        noise_interval_idx += 1

        if i > noise_steps[j]:
            j += 1
            noise_interval_idx = 1
            noise_interval = noise_steps[j] - noise_steps[j - 1]

        if noise_type[j] == 'constant':
            noise_mean = (noise_max[j] + noise_min[j]) / 2.0
            noise_per_batch[i] = int(noise_mean * batch_size)  # number of noisy samples for each batch
        elif noise_type[j] == 'uniform':
            noise_per_batch[i] = int(
                np.random.uniform(noise_min[j], noise_max[j]) * batch_size)  # number of noisy samples for each batch
        elif noise_type[j] == 'gaussian':
            noise_mean = noise_min[j]#(noise_max[j] + noise_min[j]) / 2.0
            noise_std = noise_max[j]#(noise_max[j] - noise_min[j]) / 2.0 / 3.0  # we use the three sigma-rule
            noise_per_batch[i] = min(max(int(np.random.normal(noise_mean, noise_std) * batch_size),0),batch_size)  # number of noisy samples for each batch
        elif noise_type[j] == 'sin':
            noise_mean = (noise_max[j] + noise_min[j]) / 2.0
            noise_half_range = (noise_max[j] - noise_min[j]) / 2.0
            noise_step = 2.0 * np.pi * noise_interval_idx / noise_interval
            noise_per_batch[i] = int((noise_mean + np.sin(noise_step) * noise_half_range) * batch_size)
        elif noise_type[j] == 'linear':
            noise_range = noise_max[j] - noise_min[j]
            noise_step = noise_interval_idx / noise_interval
            noise_per_batch[i] = int((noise_min[j] + noise_step * noise_range) * batch_size)
        else:
            print("Unsupported noise type: %s" % noise_type[j])
            exit(1)

        # apply pattern to the labels
        if start_next_batch < n_init:
            end_next_batch = n_init
        else:
            end_next_batch = start_next_batch + batch_size


        noisy_idx = np.random.choice(np.arange(start_next_batch, end_next_batch), noise_per_batch[i], replace=False)
        for k in noisy_idx:
            other_class_list = [ 2.0, 3.0, 4.0, 5.0] # the difference between noise.py and noise_cluster.py is here
            other_class_list.remove(train_labels[k])
            train_labels_noisy[k] = np.random.choice(other_class_list)
            train_labels_good[k] = 0.0 # mark label as not good

        start_next_batch = end_next_batch

    return train_labels_noisy, train_labels_good, noise_per_batch
