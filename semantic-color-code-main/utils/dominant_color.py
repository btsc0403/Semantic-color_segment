# Get the dominant color perceived by human from the extracted color palette
# Use k-means algorithm applied on each color palette
import math
import matplotlib.pyplot as plt
import os.path
import json
import dbscan
from collections import Counter
import numpy as np
import util as util
import sys
sys.path.append('E:\Test\Testcode\semantic-color-code-main')
from scripts.visualization import plot_matrix
# import extract_dominant_color as extract_dominant_color

# measure difference between two HSVs:
# H difference is measured as the angle between two hues
# S and V differences are measured as absolute differences
def DistHSV(HSV1, HSV2):
    assert(util.ValidHSV(HSV1))
    assert(util.ValidHSV(HSV2))
    
    hue_diff = DistHue(HSV1, HSV2)
    sat_diff = abs(HSV1[1] - HSV2[1])
    val_diff = abs(HSV1[2] - HSV2[2])

    return math.sqrt(hue_diff * hue_diff + sat_diff * sat_diff + val_diff * val_diff)

def DistHue(HSV1, HSV2):
    assert(util.ValidHSV(HSV1))
    assert(util.ValidHSV(HSV2))

    # convert hues to angular degrees [0, 180] -> [0, 360] -> [0, 2Pi]
    ad1 = 2 * math.pi * (HSV1[0] * 2 / 360)
    ad2 = 2 * math.pi * (HSV2[0] * 2 / 360)

    temp = math.cos(ad1) * math.cos(ad2) + math.sin(ad1) * math.sin(ad2)
    if temp >= 1:
        temp = 1
    elif temp <= -1:
        temp = -1
    angle = math.acos(temp)

    # convert anglar degress to hue differences [0, 2*Pi] -> [0, 360] -> [0, 180]
    hue_diff = angle / (2 * math.pi) * 360 / 2
    
    return hue_diff

def MeanHue(H1, H2, w1 = 1, w2 = 1):
    assert(0 <= H1 <= 180)
    assert(0 <= H2 <= 180)
    
    # convert hues to angular degrees [0, 180] -> [0, 360] -> [0, 2Pi]
    ad1 = 2 * math.pi * (H1 * 2 / 360)
    ad2 = 2 * math.pi * (H2 * 2 / 360)

    mean_dir = [math.cos(ad1) * w1 + math.cos(ad2) * w2, math.sin(ad1) * w1 + math.sin(ad2) * w2]
    mean_dir_angle = math.atan2(mean_dir[1], mean_dir[0])

    if mean_dir_angle < 0: # atan2 values range in [-Pi, Pi], convert to [0, 2Pi]
        mean_dir_angle = 2 * math.pi + mean_dir_angle

    mean_dir_degree = mean_dir_angle / (2 * math.pi) * 360 / 2

    return mean_dir_degree


if __name__ == '__main__':
    PALETTE_OUTPUT_FOLDER = r'E:\Test\Testoutput\palette_12_12'
    prompts = ["banana, fruit"]
    for prompt in prompts:
        palette_dir = f"{PALETTE_OUTPUT_FOLDER}/{prompt}"
        hsv_points_list = np.load(f"{palette_dir}/hsv_points_full_list.npy")
        print(hsv_points_list.shape)
        
        # raise OSError
        clustering = dbscan(eps=10,min_samples=300, metric=DistHue).fit(hsv_points_list)
        labels = clustering.labels_

        # Number of clusters in labels, ignoring noise if present.
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise_ = list(labels).count(-1)
        print(f"DBSCAN Info\nnum of points:{hsv_points_list.shape[0]}\tn_clusters_:{n_clusters_}\tn_noise_:{n_noise_}")
        
        print(Counter(clustering.labels_).most_common())
        # raise OSError
        main_label = Counter(clustering.labels_).most_common()[0][0] \
            if Counter(clustering.labels_).most_common()[0][0] != -1 else Counter(clustering.labels_).most_common()[1][0]
        print(main_label)
        cluster0_hsvs = hsv_points_list[clustering.labels_ == main_label]
        
        for i in range(len(cluster0_hsvs)):
            if i == 0:
                ave_hue = cluster0_hsvs[i][0]
            else:
                ave_hue = MeanHue(ave_hue, cluster0_hsvs[i][0], i, 1)
        cluster0_hues = cluster0_hsvs[:,0]
        cluster0_sats = cluster0_hsvs[:,1]
        cluster0_vals = cluster0_hsvs[:,2]
        print("mean hue", ave_hue, "std", np.std(cluster0_hues))
        print("mean sats", np.mean(cluster0_sats), "std", np.std(cluster0_sats))
        print("mean vals", np.mean(cluster0_vals), "std", np.std(cluster0_vals))
        
        print(cluster0_hsvs.shape)
        plot_matrix(cluster0_hsvs, save_path=r"./banana_test.png")
        raise OSError
        cluster0_label = Counter(clustering.labels_).most_common()[0][0]
        cluster0_hsvs = []

        for i in range(len(clustering.labels_)):
            if clustering.labels_[i] == cluster0_label:
                cluster0_hsvs.append(hsvs[i])

        cluster0_hues = [row[0] for row in cluster0_hsvs]
        cluster0_sats = [row[1] for row in cluster0_hsvs]
        cluster0_vals = [row[2] for row in cluster0_hsvs]
        print("all hsv", cluster0_hsvs)

        ave_hue = 0
        for i in range(len(cluster0_hues)):
            if i == 0:
                ave_hue = cluster0_hues[i]
            else:
                ave_hue = MeanHue(ave_hue, cluster0_hues[i], i, 1)

        print("mean hue", ave_hue, "std", np.std(cluster0_hues))
        print("mean sats", np.mean(cluster0_sats), "std", np.std(cluster0_sats))
        print("mean vals", np.mean(cluster0_vals), "std", np.std(cluster0_vals))
    
        #  hues = [None] * len(extract_dominant_color.file_indices)

        # img_count = 0
        # hsvs = [] # stores all hsv values in the file
        # for file_index in range(len(extract_dominant_color.file_indices)):
        #     # open the file
        #     filename = os.path.join(extract_dominant_color.folder, 'palette_'+"{:05d}".format(file_index)+'.txt')

        #     with open(filename, 'r') as fin:
        #         lines = fin.readlines()
        #         # each line is a palette, has 5 colors
        #         for line in lines:
        #             data = json.loads(line)

        #             if data == None:
        #                 continue

        #             for i in range(len(data)):
        #                 hsvs.append(data[i]['hsv'])

        #         # print(len(hsvs), hsvs)
        #  print("hsv size", len(hsvs))
        # clustering = DBSCAN(eps=10, min_samples=1200, metric=DistHue).fit(hsvs)
        # cluster0_label = Counter(clustering.labels_).most_common()[0][0]
        #   cluster0_hsvs = []

        #  for i in range(len(clustering.labels_)):
        #     if clustering.labels_[i] == cluster0_label:
        #         cluster0_hsvs.append(hsvs[i])

        # cluster0_hues = [row[0] for row in cluster0_hsvs]
        # cluster0_sats = [row[1] for row in cluster0_hsvs]
        # cluster0_vals = [row[2] for row in cluster0_hsvs]
    # print("all hsv", cluster0_hsvs)

    # ave_hue = 0
    # for i in range(len(cluster0_hues)):
    #     if i == 0:
    #         ave_hue = cluster0_hues[i]
    #     else:
    #         ave_hue = MeanHue(ave_hue, cluster0_hues[i], i, 1)

    # print("mean hue", ave_hue, "std", np.std(cluster0_hues))
    # print("mean sats", np.mean(cluster0_sats), "std", np.std(cluster0_sats))
    # print("mean vals", np.mean(cluster0_vals), "std", np.std(cluster0_vals))

    # plt.plot(palette_extraction.file_indices, hues)
    # plt.xlabel('Image Number')
    # plt.ylabel('Hue')
    # plt.show()
