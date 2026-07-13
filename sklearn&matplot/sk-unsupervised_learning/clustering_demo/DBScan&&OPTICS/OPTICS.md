# OPTICS
> OPTICS 是 DBSCAN 的一个推广版本，它解决了 DBSCAN 最致命的弱点——对密度不均匀的数据表现差。OPTICS 不直接输出一个固定的聚类结果，而是生成一个可达性图（Reachability Plot），让你可以观察数据在不同密度层次上的聚类结构。
## 与DBScan对比
                  DBSCAN 的表现	                    OPTICS 的改进
密度不均匀	单一 eps 无法同时适应稀松和密集区域	✅ 通过可达距离自动适应密度变化
不同密度层次	只能看到一种粒度的聚类	✅ 可达性图展示所有密度层次的聚类结构
参数敏感	对 eps 非常敏感	✅ 不需要指定 eps（只需最大半径上限）

## 核心概念
- 核心距离:对于点 p，core-distance(p) = 使 |N_ε(p)| ≥ minPts 的最小半径。N_ε(p)表示minPts个数
  - 符号：core-distance(p)	        
  - 定义：使 p 成为核心点的最小半径	               
  - 含义：p 变成“核心”所需的最小 eps
- 可达距离：	
  - 符号：reachability-distance(p, q)		
  - 定义：max(core-distance(q), distance(p, q))
  - 含义：从 q 到达 p 的“代价”
> 注意：core-distance(q) 是 q 的属性，reachability-distance(p, q) 是从 q 到 p 的单向距离（非对称）。
- 从点 q 到点 p 的可达距离：
reachability-distance(p,q)=max(core-distance(q),distance(p,q))
- 如果 q 是核心点，可达距离 = max(核心距离, 欧氏距离)
- 如果 q 不是核心点，则无法从 q 到达任何点
## 核心距离
输入：样本集 X，最大半径 ε_max，最小点数 minPts
输出：有序列表（点的处理顺序）+ 每个点的可达距离

1. 初始化：所有点标记为"未处理"
2. 对于每个未处理的点 p：
   a) 标记 p 为"已处理"  
   b) 将 p 加入有序列表  
   c) 计算 p 的 ε_max 邻域 N(p)  
   d) 如果 |N(p)| < minPts：  
        设置 p 的可达距离 = 未定义（∞）  
        continue  
   e) 否则（p 是核心点）：  
        创建"种子集合" seeds = {}  
        对于 N(p) 中的每个点 q（未处理）：  
            计算从 p 到 q 的可达距离  
            seeds.add(q, reachability)  
        while seeds 非空：  
            从 seeds 中取出可达距离最小的点 q  
            标记 q 为"已处理"  
            将 q 加入有序列表  
            如果 q 是核心点：  
                对于 q 的 ε_max 邻域中的每个点 r（未处理）：  
                    计算从 q 到 r 的可达距离 new_dist  
                    如果 r 不在 seeds 中：  
                        seeds.add(r, new_dist)  
                    否则如果 new_dist < 当前记录：  
                        更新 seeds 中 r 的可达距离  
3. 返回有序列表（点的顺序）和每个点的可达距离
## 提取聚类
你可以设置一个阈值 ε'（≤ ε_max），在可达性图中画一条水平线：
- 可达距离 < ε' 的点 → 属于同一个簇
- 可达距离 ≥ ε' 的点 → 簇边界或噪声
这样，不同的 ε' 对应不同密度的聚类结果，而无需重新运行算法。