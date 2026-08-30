
import numpy as np

def smooth_probabilities(probs, window=5):
    if len(probs) < window:
        return probs
    kernel = np.ones(window) / window
    return np.convolve(probs, kernel, mode="same")

def persistence_alarm(probs, threshold, consecutive_windows=3,
                      refractory_windows=30):
    alarms = []
    count = 0
    cooldown = 0

    for i, p in enumerate(probs):
        if cooldown > 0:
            cooldown -= 1
            continue

        if p >= threshold:
            count += 1
        else:
            count = 0

        if count >= consecutive_windows:
            alarms.append(i)
            cooldown = refractory_windows
            count = 0

    return np.asarray(alarms, dtype=np.int64)
