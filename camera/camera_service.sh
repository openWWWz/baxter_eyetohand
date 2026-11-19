#!/usr/bin/env bash
# 管理本机 RealSense ROS 发布器；启动必须收到新帧，停止必须等进程退出。
set -euo pipefail

camera_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
publisher="${camera_dir}/publish_realsense_ros.py"
pid_file="${camera_dir}/publisher.pid"
log_file="${camera_dir}/publisher.log"
camera_python=/home/robot24/miniconda3/envs/eyetohand/bin/python

running() {
  [[ -f "$pid_file" ]] || return 1
  read -r pub_pid < "$pid_file" || return 1
  [[ "$pub_pid" =~ ^[0-9]+$ ]] || return 1
  kill -0 "$pub_pid" 2>/dev/null || return 1
  [[ "$(ps -p "$pub_pid" -o stat=)" != Z* ]] || return 1
  ps -p "$pub_pid" -o args= | rg -F -q -- "$publisher"
}

configure_ros() {
  source /opt/ros/noetic/setup.bash
  source /home/robot24/wzy/Baxter_ROS/ros_ws/devel/setup.bash
  export ROS_MASTER_URI=http://192.168.1.102:11311
  export ROS_IP=192.168.1.116
}

check_stream() {
  # 不以 rostopic list 的旧注册信息判断成功；实际订阅三路消息并检查持续更新。
  timeout 30s "$camera_python" - "$pub_pid" <<'PY'
import pathlib
import sys
import threading
import time
import rospy
from sensor_msgs.msg import CameraInfo, Image

args = pathlib.Path('/proc/' + sys.argv[1] + '/cmdline').read_bytes().decode().split('\0')
namespace = '/baxter_camera'
for index, argument in enumerate(args):
    if argument == '--namespace' and index + 1 < len(args):
        namespace = args[index + 1]
    elif argument.startswith('--namespace='):
        namespace = argument.split('=', 1)[1]
namespace = '/' + namespace.strip('/')
rospy.init_node('check_local_camera_stream', anonymous=True, disable_signals=True)
condition = threading.Condition()
counts, messages = {}, {}

def receive(msg, key):
    with condition:
        messages[key] = msg
        counts[key] = counts.get(key, 0) + 1
        condition.notify_all()

subs = [rospy.Subscriber(namespace + suffix, kind, receive, callback_args=key, queue_size=1,
                         buff_size=2**23)
        for suffix, kind, key in [('/color/image_raw', Image, 'RGB'),
                                  ('/aligned_depth_to_color/image_raw', Image, 'depth'),
                                  ('/color/camera_info', CameraInfo, 'camera_info')]]
try:
    deadline = time.monotonic() + 20
    with condition:
        while not all(counts.get(key, 0) >= 2 for key in ('RGB', 'depth', 'camera_info')):
            if time.monotonic() >= deadline or rospy.is_shutdown():
                missing = [key for key in ('RGB', 'depth', 'camera_info') if counts.get(key, 0) < 2]
                raise RuntimeError('等待连续消息超时: ' + ', '.join(missing))
            condition.wait(.1)
        color, depth, info = [messages[key] for key in ('RGB', 'depth', 'camera_info')]
        if ((color.width, color.height) != (depth.width, depth.height)
                or (color.width, color.height) != (info.width, info.height)
                or info.K[0] <= 0 or info.K[4] <= 0
                or not color.data or not depth.data or depth.encoding != '16UC1'):
            raise RuntimeError('RGB、深度、内参的尺寸或内容不正确')
    print(f'已确认连续发布：RGB {color.width}x{color.height}，深度 {depth.encoding}，'
          f'内参 {namespace}/color/camera_info', flush=True)
except RuntimeError as exc:
    print(str(exc), file=sys.stderr, flush=True)
    sys.exit(1)
finally:
    for sub in subs:
        sub.unregister()
    rospy.signal_shutdown('stream checked')
PY
}

stop_publisher() {
  if running; then
    # 后台 shell 的 SIGINT 可能被继承为忽略；SIGTERM 交给 rospy 正常关闭。
    echo "正在停止相机发布器 PID $pub_pid……"
    kill -TERM "$pub_pid" 2>/dev/null || true
    for ((attempt=0; attempt<40; attempt++)); do
      if ! running; then break; fi
      sleep .2
    done
    if running; then
      echo '正常退出超时，结束残留的相机发布器进程。'
      kill -KILL "$pub_pid" 2>/dev/null || true
      for ((attempt=0; attempt<10; attempt++)); do
        if ! running; then break; fi
        sleep .1
      done
    fi
    if running; then
      echo '相机发布器仍未退出，请检查设备状态。' >&2
      return 1
    fi
    echo '相机发布器已停止。'
  else
    echo '相机发布器未运行，清理残留 PID。'
  fi
  rm -f -- "$pid_file"
}

start_publisher() {
  configure_ros
  if running; then
    echo "相机发布器进程存在：PID $pub_pid，正在检查实际消息……"
    if check_stream; then return 0; fi
    echo '进程未持续发布有效数据，重新启动相机发布器。'
    stop_publisher
  fi
  nohup setsid "$camera_python" -u "$publisher" "$@" \
    > "$log_file" 2>&1 < /dev/null &
  pub_pid=$!
  echo "$pub_pid" > "$pid_file"
  sleep 1
  if ! running; then
    echo "启动失败，请查看日志：$log_file" >&2
    tail -n 12 "$log_file" >&2
    return 1
  fi
  echo "已创建发布器 PID $pub_pid，等待 RGB、深度和内参……"
  if ! check_stream; then
    echo "启动后没有正常发布，请查看日志：$log_file" >&2
    tail -n 12 "$log_file" >&2
    stop_publisher
    return 1
  fi
  echo "相机发布器已就绪：PID $pub_pid；日志 $log_file"
}

action="${1:-status}"
if [[ $# -gt 0 ]]; then shift; fi
case "$action" in
  start) start_publisher "$@" ;;
  restart) stop_publisher; start_publisher "$@" ;;
  stop) stop_publisher ;;
  check)
    if ! running; then echo '相机发布器未运行' >&2; exit 1; fi
    configure_ros
    check_stream ;;
  status)
    if running; then
      echo "相机发布器进程存在：PID $pub_pid；日志 $log_file"
      echo '确认图像和内参持续更新：bash camera_service.sh check'
    else
      echo '相机发布器未运行'
    fi ;;
  *) echo "用法：bash $0 start|restart [发布器参数] | stop | status | check" >&2; exit 2 ;;
esac
