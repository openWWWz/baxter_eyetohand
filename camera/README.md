> 当前机器的完整命令见 [本机运行说明](../本机运行.md)。以下包含原设备的历史验证记录。

# 独立相机预览 demo

只想看相机画面，直接执行：

```bash
cd /home/robot24/wzy_ws/baxter_worksapce/camera
bash run_camera_demo.sh
```

窗口显示彩色图；RealSense 有深度时同时显示深度，按 `q` / Esc 退出。脚本不固定相机序列号，不读取标定文件，不检查旧标定内参，也不控制机器人。自动模式优先复用本目录正在运行的相机发布器；没有发布器则自动枚举并打开本机 RealSense，再尝试普通 USB/UVC 彩色相机。更换设备后重启 demo 即可。USB 摄像头不提供深度时只显示彩色图。

使用现有 `eyetohand` Python，启动脚本自动准备环境；也可以在已激活的环境里执行 `python camera_demo.py`。常用命令：

```bash
bash run_camera_demo.sh --list                       # 显示已接入相机
bash run_camera_demo.sh --source usb                 # 自动选择普通 USB 摄像头
bash run_camera_demo.sh --source usb --device /dev/video6  # 手动指定 USB 节点
bash run_camera_demo.sh --source ros                 # 仅读取现有 RGB/深度话题
bash run_camera_demo.sh --check                      # 读取三帧检查，不弹窗
```

直接打开 RealSense 可以用 `--source realsense`，无需序列号。若这台相机已经由本目录的发布器读取，默认模式可直接复用；若坚持直接读取，先停止占用程序。其他 ROS 图像可通过 `--source ros --rgb-topic /你的图像话题 --depth-topic /你的深度话题` 指定，无深度话题也可显示彩色图。RGB/深度在此 demo 中用于预览，不承诺硬件同步。

# 在本机发布固定 RealSense 相机（Baxter 标定）

发布脚本自动枚举当前本机 RealSense；无需序列号即可启动，多台时可选用 `--serial` 指定。它读取本机 USB 相机，把彩色图、对齐到彩色图像素的深度图及各自的 `CameraInfo` 发布到 Baxter 所用的 ROS master；不会控制机械臂。相机需要**固定在机器人基座之外或固定在完全不动的头部**，不能安装在标定的手臂上。

在这台电脑上打开**第一个终端**：

```bash
source /opt/ros/noetic/setup.bash
source /home/robot24/wzy/Baxter_ROS/ros_ws/devel/setup.bash
export ROS_MASTER_URI=http://192.168.1.102:11311
export ROS_IP=192.168.1.116
source /home/robot24/miniconda3/etc/profile.d/conda.sh
conda activate eyetohand
cd /home/robot24/wzy_ws/baxter_worksapce/camera
python publish_realsense_ros.py --width 640 --height 480 --fps 30 --preview
```

画面预览左侧彩色、右侧对齐深度；按 `q` 退出发布。如果不需要弹窗，省略 `--preview`。如果提示设备被占用，先退出正在读取该 D455 的其他相机程序；不要同时让标定采集程序的 `--realsense` 模式占用同一台相机。

也可以在后台运行，采集标定时无需一直保留发布终端；本机已用 **15 FPS** 模式做过图像、深度及内参的实际订阅检查：

```bash
cd /home/robot24/wzy_ws/baxter_worksapce/camera
bash camera_service.sh start --fps 15
bash camera_service.sh status
bash camera_service.sh check
tail -n 20 publisher.log
# 标定结束后：bash camera_service.sh stop
```

更换相机、拔插设备或采集提示 `camera_info` 超时时，执行 `bash camera_service.sh restart --fps 15`。脚本先等待旧进程退出，再枚举当前设备；实际收到连续的彩色、深度和内参才报告“已就绪”。`status` 只表示进程是否存在，`check` 会验证数据。独立 `camera_demo.py` 是预览器，不负责发布 `CameraInfo`；ROS 标定采集时保持此发布器运行。采集脚本等待内参最多 30 秒。

持续发布的话题：

| 话题                                                  | 类型                       | 内容                                                |
| ----------------------------------------------------- | -------------------------- | --------------------------------------------------- |
| `/baxter_camera/color/image_raw`                    | `sensor_msgs/Image`      | 已矫正彩色图，`bgr8`                              |
| `/baxter_camera/color/camera_info`                  | `sensor_msgs/CameraInfo` | 上述已矫正彩色图的内参                              |
| `/baxter_camera/aligned_depth_to_color/image_raw`   | `sensor_msgs/Image`      | 已对齐且矫正的深度图，`16UC1`，单位毫米；0 为无效 |
| `/baxter_camera/aligned_depth_to_color/camera_info` | `sensor_msgs/CameraInfo` | 与彩色图相同的像素内参                              |

此 D455 当前彩色原始内参为 `inverse_brown_conrady` 且畸变系数非零。脚本对彩色和已对齐深度应用同一个矫正映射，再发布**零畸变**的 `CameraInfo`，因此采集必须使用上述已矫正图像及对应 `camera_info`；不可把未矫正的 USB 原图或以前 Realman 的内参与这些话题混用。图像和内参的宽、高都随脚本参数变化，标定过程中不能改动分辨率。

在**第二个终端**执行同样的 ROS/conda 环境配置，然后验证本机能订阅到连续图像和内参：

```bash
rostopic hz /baxter_camera/color/image_raw
rostopic echo -n 1 /baxter_camera/color/camera_info
rostopic echo -n 1 /robot/limb/right/endpoint_state/header
```

检查后按 `Ctrl+C` 停止 `rostopic hz`，**保持第一个发布终端运行**。在第二个终端启动采集：

```bash
cd /home/robot24/wzy_ws/baxter_worksapce/eyetohand
python record_baxter.py --arm right \
  --image-topic /baxter_camera/color/image_raw \
  --camera-info-topic /baxter_camera/color/camera_info
```

随后在采集窗口按 `a` 保存不同右臂静止姿态，按 `q` 结束。标定求解步骤见 [Baxter 眼在手外标定](../eyetohand/README.md)。机器人末端消息的 `header.frame_id` 当前为空，采集脚本会原样保存这一事实；Baxter SDK 通常将末端位姿视为基座坐标，应结合机器人配置核对后再使用计算得到的变换。
