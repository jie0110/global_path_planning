import sys
import argparse
import numpy as np

import rospy
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseWithCovarianceStamped, PoseStamped

from utils import traj2ros
from planner_wrapper import TomogramPlanner

sys.path.append('../')
from config import Config

parser = argparse.ArgumentParser()
parser.add_argument('--scene', type=str, default='Spiral', help='Name of the scene. Available: [\'Spiral\', \'Building\', \'Plaza\']')
args = parser.parse_args()

cfg = Config()

if args.scene == 'Spiral':
    tomo_file = 'spiral0.3_2'
elif args.scene == 'Building':
    tomo_file = 'output'
else:
    tomo_file = 'plaza3_10'

planner = TomogramPlanner(cfg)
planner.loadTomogram(tomo_file)

start_pos = None
end_pos = None
path_pub = None


def try_plan():
    if start_pos is None or end_pos is None:
        return
    traj_3d = planner.plan(start_pos, end_pos)
    if traj_3d is not None:
        path_pub.publish(traj2ros(traj_3d))
        print("Trajectory published")
    else:
        print("Planning failed: no path found")


def initialpose_cb(msg):
    global start_pos
    p = msg.pose.pose.position
    start_pos = np.array([p.x, p.y, p.z], dtype=np.float32)
    rospy.loginfo(f"Start updated: {start_pos}")
    try_plan()


def goal_cb(msg):
    global end_pos
    p = msg.pose.position
    end_pos = np.array([p.x, p.y, p.z], dtype=np.float32)
    rospy.loginfo(f"Goal updated: {end_pos}")
    try_plan()


if __name__ == '__main__':
    rospy.init_node("pct_planner", anonymous=True)

    path_pub = rospy.Publisher("/pct_path", Path, latch=True, queue_size=1)
    rospy.Subscriber("/initialpose", PoseWithCovarianceStamped, initialpose_cb)
    rospy.Subscriber("/goal_3d", PoseStamped, goal_cb)

    rospy.loginfo("PCT Planner ready. Waiting for /initialpose and /goal_3d ...")
    rospy.spin()
