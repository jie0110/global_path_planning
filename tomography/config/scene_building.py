from .scene import ScenePCD, SceneMap, SceneTrav


class SceneBuilding():
    pcd = ScenePCD()
    pcd.file_name = 'output.pcd'

    map = SceneMap()
    map.resolution = 0.1    # 网格分辨率                       
    map.ground_h = -5.0      # 地图最低高度：建筑物地下室起始高度                               
    map.slice_dh = 1.0       # 切片间隔：每0.5m一个新切片  

    trav = SceneTrav()                                                                                       
    trav.kernel_size = 1         # 穿越代价计算的卷积核大小（n×n个邻格）                         
    trav.interval_min = 1.0     # 最小通行间隔                                 
    trav.interval_free = 1.2    # 自由通行间隔                             
    trav.slope_max = 0.40        # 最大可站立坡度：tan(角度)，约21.8°                               
    trav.step_max = 0.22         # 最大可跨越台阶高度                                      
    trav.standable_ratio = 0.01  # 落脚点要求              
    trav.cost_barrier = 50.0     # 障碍物代价值（高于此值视为不可通行）                             
    trav.safe_margin = 0.20      # 安全距离：障碍物外 x m内有线性惩罚                               
    trav.inflation = 0.20        # 硬安全距离：x m内代价最大（=cost_barrier）