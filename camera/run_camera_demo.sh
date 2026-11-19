#!/usr/bin/env bash
# 用现有 eyetohand Python 运行独立预览；ROS 只用于复用已运行发布器。
set -e
camera_args=("$@")
set --
camera_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
camera_master="${ROS_MASTER_URI:-http://192.168.1.102:11311}"
case "$camera_master" in
  http://localhost:11311|http://localhost:11311/|http://127.0.0.1:11311|http://127.0.0.1:11311/)
    camera_master=http://192.168.1.102:11311 ;;
esac
camera_ip="${ROS_IP:-192.168.1.116}"
if [[ -f /opt/ros/noetic/setup.bash ]]; then source /opt/ros/noetic/setup.bash; fi
export ROS_MASTER_URI="$camera_master"
export ROS_IP="$camera_ip"
exec /home/robot24/miniconda3/envs/eyetohand/bin/python -u "$camera_dir/camera_demo.py" "${camera_args[@]}"
