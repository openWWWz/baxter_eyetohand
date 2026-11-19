#!/usr/bin/env bash
# 自动加载 Baxter ROS 消息和 eyetohand Python；默认仅读取和预览。
set -e
capture_args=("$@")
set --
grasp_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
baxter_master="${ROS_MASTER_URI:-http://192.168.1.102:11311}"
case "$baxter_master" in
  http://localhost:11311|http://localhost:11311/|http://127.0.0.1:11311|http://127.0.0.1:11311/)
    baxter_master=http://192.168.1.102:11311 ;;
esac
baxter_host_ip="${ROS_IP:-192.168.1.116}"
source /opt/ros/noetic/setup.bash
source /home/robot24/wzy/Baxter_ROS/ros_ws/devel/setup.bash
export ROS_MASTER_URI="$baxter_master"
export ROS_IP="$baxter_host_ip"
exec /home/robot24/miniconda3/envs/eyetohand/bin/python -u \
  "$grasp_dir/test_grasp_baxter.py" "${capture_args[@]}"
