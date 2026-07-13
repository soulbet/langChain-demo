# bin_seeding 将数据离散化到网格，减少初始点数量
ms = MeanShift(bandwidth=bw, bin_seeding=True, min_bin_freq=10)