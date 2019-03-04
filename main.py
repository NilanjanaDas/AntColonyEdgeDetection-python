import itertools
import random

import cv2
import numpy as np


class Ant:
    def __init__(self, x, y, max_memory):
        self.max_memory = max_memory
        self.x = x
        self.y = y
        self.memory = [(x, y)]
        self.sleep = False

    def set_point(self, x, y):
        self.x = x
        self.y = y
        self.memory.append((x, y))
        if len(self.memory) > self.max_memory:
            del self.memory[0]


def main():
    img = cv2.imread("lena.jpg", cv2.IMREAD_GRAYSCALE)
    x = cv2.Sobel(img, cv2.CV_16S, 1, 0)
    y = cv2.Sobel(img, cv2.CV_16S, 0, 1)
    x = cv2.convertScaleAbs(x)
    y = cv2.convertScaleAbs(y)
    gradient = cv2.addWeighted(x, 0.5, y, 0.5, 0)

    num_ant = int(img.size / 5)
    alpha = 1
    beta = 1
    max_memory = 50
    delta_p = 0.01
    evaporation_p = 0.1
    epoch_num = 500

    info = np.zeros_like(img, dtype='double') + 0.0001
    random_list = list(itertools.product(range(0, img.shape[0]), range(0, img.shape[1])))
    random_list = random.sample(random_list, num_ant)

    ant_list = []

    for c in random_list:
        ant_list.append(Ant(c[0], c[1], max_memory))

    # neighbor_pos_map = {0: (-1, 0), 1: (1, 0), 2: (0, 1), 3: (0, -1), 4: (-1, 1), 5: (1, 1), 6: (-1, -1), 7: (1, -1)}
    neighbor_pos_map = np.array([(-1, 0), (1, 0), (0, 1), (0, -1), (-1, 1), (1, 1), (-1, -1), (1, -1)])

    # 每轮增加的信息素
    epoch_info = np.zeros_like(info, dtype='double')

    for i in range(epoch_num):
        for ant in ant_list:
            if ant.sleep:
                continue
            # 计算领域概率并移动
            neighbor_pos = neighbor_pos_map + (ant.x, ant.y)
            neighbor_pos[neighbor_pos > 255] = 255
            neighbor_pos[neighbor_pos < 0] = 0

            neighbor_pos = set([(t[0], t[1]) for t in neighbor_pos.tolist()]) - set(ant.memory)
            neighbor_pos = np.array([[t[0], t[1]] for t in neighbor_pos])
            if len(neighbor_pos) == 0:
                ant.sleep = True
                continue
            rows = neighbor_pos[:, 0]
            cols = neighbor_pos[:, 1]
            neighbor_info = info[rows, cols]
            neighbor_gradient = gradient[rows, cols]

            t1 = neighbor_info ** alpha + 0.0000001
            t2 = neighbor_gradient ** beta + 0.0000001
            prob = t1 * t2 / (sum(t1 * t2))
            next_point_index = np.random.choice(a=len(neighbor_pos), size=1, replace=False, p=prob)
            next_point = neighbor_pos[next_point_index][0]
            ant.set_point(next_point[0], next_point[1])

            epoch_info[next_point[0], next_point[1]] += delta_p

        info = (1 - evaporation_p) * info + epoch_info

        info_min, info_max = info.min(), info.max()  # 求最大最小值
        info_tmp = (info - info_min) / (info_max - info_min) * 256
        cv2.imshow("s1", info_tmp.astype(np.int8))

        _, info_tmp = cv2.threshold(info_tmp, int(255 * 0.35), 255, 0)
        cv2.imshow("s2", info_tmp)
        cv2.waitKey(1)
        # cv2.waitKey()
        print("epoch %d" % i)
    print()
    cv2.waitKey()


if __name__ == '__main__':
    main()
