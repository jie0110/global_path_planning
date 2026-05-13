import os
import sys
import pickle
import numpy as np

from utils import transTrajGrid2Map

sys.path.append('../')
from lib import a_star, ele_planner, traj_opt

rsg_root = os.path.dirname(os.path.abspath(__file__)) + '/../..'


class TomogramPlanner(object):
    def __init__(self, cfg):
        self.cfg = cfg

        self.use_quintic = self.cfg.planner.use_quintic
        self.max_heading_rate = self.cfg.planner.max_heading_rate

        self.tomo_dir = rsg_root + self.cfg.wrapper.tomo_dir

        self.resolution = None
        self.center = None
        self.n_slice = None
        self.slice_h0 = None
        self.slice_dh = None
        self.map_dim = []
        self.offset = None

        self.start_idx = np.zeros(3, dtype=np.int32)
        self.end_idx = np.zeros(3, dtype=np.int32)

    def loadTomogram(self, tomo_file):
        with open(self.tomo_dir + tomo_file + '.pickle', 'rb') as handle:
            data_dict = pickle.load(handle)

            tomogram = np.asarray(data_dict['data'], dtype=np.float32)

            self.resolution = float(data_dict['resolution'])
            self.center = np.asarray(data_dict['center'], dtype=np.double)
            self.n_slice = tomogram.shape[1]
            self.slice_h0 = float(data_dict['slice_h0'])
            self.slice_dh = float(data_dict['slice_dh'])
            self.map_dim = [tomogram.shape[2], tomogram.shape[3]]
            self.offset = np.array([int(self.map_dim[0] / 2), int(self.map_dim[1] / 2)], dtype=np.int32)

        trav = tomogram[0]
        trav_gx = tomogram[1]
        trav_gy = tomogram[2]
        elev_g = tomogram[3]
        elev_g = np.nan_to_num(elev_g, nan=-100)
        elev_c = tomogram[4]
        elev_c = np.nan_to_num(elev_c, nan=1e6)

        self.elev_g = elev_g  # [n_slice, map_dim_x, map_dim_y], used for layer lookup
        self.initPlanner(trav, trav_gx, trav_gy, elev_g, elev_c)
        
    def initPlanner(self, trav, trav_gx, trav_gy, elev_g, elev_c):
        diff_t = trav[1:] - trav[:-1]
        diff_g = np.abs(elev_g[1:] - elev_g[:-1])

        gateway_up = np.zeros_like(trav, dtype=bool)
        mask_t = diff_t < -8.0
        mask_g = (diff_g < 0.1) & (~np.isnan(elev_g[1:]))
        gateway_up[:-1] = np.logical_and(mask_t, mask_g)

        gateway_dn = np.zeros_like(trav, dtype=bool)
        mask_t = diff_t > 8.0
        mask_g = (diff_g < 0.1) & (~np.isnan(elev_g[:-1]))
        gateway_dn[1:] = np.logical_and(mask_t, mask_g)
        
        # ── 诊断：统计每层 gateway 数量 ───────────────────────────────                        
        for i in range(trav.shape[0]):
            print(f"layer {i}: gateway_up={gateway_up[i].sum()}, "                              
                    f"gateway_dn={gateway_dn[i].sum()}, "                                         
                    f"trav_valid={(trav[i] < 20).sum()}")

        gateway = np.zeros_like(trav, dtype=np.int32)
        gateway[gateway_up] = 2
        gateway[gateway_dn] = -2

        self.planner = ele_planner.OfflineElePlanner(
            max_heading_rate=self.max_heading_rate, use_quintic=self.use_quintic
        )
        self.planner.init_map(
            35, 15, self.resolution, self.n_slice, 1,
            trav.reshape(-1, trav.shape[-1]).astype(np.double),
            elev_g.reshape(-1, elev_g.shape[-1]).astype(np.double),
            elev_c.reshape(-1, elev_c.shape[-1]).astype(np.double),
            gateway.reshape(-1, gateway.shape[-1]),
            trav_gy.reshape(-1, trav_gy.shape[-1]).astype(np.double),
            -trav_gx.reshape(-1, trav_gx.shape[-1]).astype(np.double)
        )

    def pos3d_to_layer(self, pos_3d):
        """Given world position [x, y, z], return the layer index whose ground height
        is closest to z. Falls back to layer 0 if the cell has no valid data."""
        xy_idx = self.pos2idx(pos_3d[:2])
        x_row = int(np.clip(xy_idx[1], 0, self.map_dim[0] - 1))
        y_col = int(np.clip(xy_idx[0], 0, self.map_dim[1] - 1))

        z = float(pos_3d[2])
        ground_heights = self.elev_g[:, x_row, y_col]  # [n_slice]

        valid = ground_heights > -99
        if not np.any(valid):
            return 0

        diffs = np.where(valid, np.abs(ground_heights - z), 1e9)
        layer = int(np.argmin(diffs))
        print(f"pos3d_to_layer: z={z:.2f}m → layer {layer} "
              f"(ground={ground_heights[layer]:.2f}m)")
        return layer

    def plan(self, start_pos, end_pos):
        start_pos = np.asarray(start_pos, dtype=np.float32)
        end_pos   = np.asarray(end_pos,   dtype=np.float32)

        if start_pos.shape[0] == 3:
            self.start_idx[0] = self.pos3d_to_layer(start_pos)
            self.end_idx[0]   = self.pos3d_to_layer(end_pos)
        else:
            self.start_idx[0] = 0
            self.end_idx[0]   = 0

        self.start_idx[1:] = self.pos2idx(start_pos[:2])
        self.end_idx[1:]   = self.pos2idx(end_pos[:2])

        self.planner.plan(self.start_idx, self.end_idx, True)
        path_finder: a_star.Astar = self.planner.get_path_finder()
        path = path_finder.get_result_matrix()
        if len(path) == 0:
            return None

        optimizer: traj_opt.GPMPOptimizer = (
            self.planner.get_trajectory_optimizer()
            if not self.use_quintic
            else self.planner.get_trajectory_optimizer_wnoj()
        )

        opt_init = optimizer.get_opt_init_value()
        init_layer = optimizer.get_opt_init_layer()
        traj_raw = optimizer.get_result_matrix()
        layers = optimizer.get_layers()
        heights = optimizer.get_heights()

        opt_init = np.concatenate([opt_init.transpose(1, 0), init_layer.reshape(-1, 1)], axis=-1)
        traj = np.concatenate([traj_raw, layers.reshape(-1, 1)], axis=-1)
        y_idx = (traj.shape[-1] - 1) // 2

        # ── 诊断输出 ──────────────────────────────────────────────
        a_star_layers = path[:, 0].astype(int)
        print(f"[DEBUG] A* path layers: {np.unique(a_star_layers)} "
              f"(start={a_star_layers[0]}, end={a_star_layers[-1]})")
        opt_layer_ints = layers.astype(int)
        print(f"[DEBUG] Optimizer layers: unique={np.unique(opt_layer_ints)}")
        print(f"[DEBUG] Smoother heights (m): min={heights.min():.3f}, max={heights.max():.3f}, "
              f"mean={heights.mean():.3f}")
        # ─────────────────────────────────────────────────────────

        # Replace smoother heights with direct elev_g lookups to prevent
        # trajectory from floating above the surface at layer transitions.
        # traj[:,0]=col, traj[:,y_idx]=row in the internal grid (from C++ GetHeight).
        layer_int = np.clip(layers.astype(int), 0, self.n_slice - 1)
        row_int   = np.clip(traj[:, y_idx].astype(int), 0, self.map_dim[0] - 1)
        col_int   = np.clip(traj[:, 0].astype(int),     0, self.map_dim[1] - 1)
        heights_map = self.elev_g[layer_int, row_int, col_int]
        invalid = heights_map < -99.0
        heights_fixed = np.where(invalid, heights, heights_map)
        print(f"[DEBUG] Map-snapped heights (m): min={heights_fixed.min():.3f}, "
              f"max={heights_fixed.max():.3f}, invalid_frac={invalid.mean():.2%}")

        traj_3d = np.stack([traj[:, 0], traj[:, y_idx], heights_fixed / self.resolution], axis=1)
        traj_3d = transTrajGrid2Map(self.map_dim, self.center, self.resolution, traj_3d)

        print(f"[DEBUG] World traj z: min={traj_3d[:,2].min():.3f}, "
              f"max={traj_3d[:,2].max():.3f}")

        return traj_3d
    
    def pos2idx(self, pos):
        pos = pos - self.center
        idx = np.round(pos / self.resolution).astype(np.int32) + self.offset
        idx = np.array([idx[1], idx[0]], dtype=np.float32)
        return idx