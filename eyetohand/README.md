> 当前机器的完整命令见 [本机运行说明](../本机运行.md)。以下包含原设备的历史验证记录。

# Baxter 眼在手外标定

沿用 `eyetohand/accurate_eye_to_hand.py` 的手眼求解与有效棋盘格筛选逻辑。`record_baxter.py` 从 `/robot/limb/right/endpoint_state` 和 `/robot/joint_states` 读取末端及七个关节位姿。手动采集模式只读；**只有指定 `--action` 才会控制 Baxter 手臂回放**。将棋盘格固定在所选手臂末端，固定相机及 Baxter 底座在整个采集期间均不能移动。手腕相机随手臂运动，不适用于此流程。

在 **192.168.1.116 的图形桌面** 打开终端，按顺序执行：

```bash
source /opt/ros/noetic/setup.bash
source /home/robot24/wzy/Baxter_ROS/ros_ws/devel/setup.bash
export ROS_MASTER_URI=http://192.168.1.102:11311
export ROS_IP=192.168.1.116
source /home/robot24/miniconda3/etc/profile.d/conda.sh
conda activate eyetohand
cd /home/robot24/wzy_ws/baxter_worksapce/eyetohand

python record_baxter.py --list
rostopic echo -n 1 /robot/limb/right/endpoint_state/header
```

`--list` 显示可用的图像及内参话题。先按 [本机 D455 发布说明](../camera/README.md) 启动本机 USB 相机的 ROS 话题，再用同一台相机的彩色图与内参采集：

```bash
python record_baxter.py --arm right \
  --image-topic /baxter_camera/color/image_raw \
  --camera-info-topic /baxter_camera/color/camera_info
```

如果想用 Baxter **头部相机**且头部在整个采集过程中保持不动，可尝试启动它。已在工作空间内确认 `CameraController` 支持 `head_camera`；是否能打开取决于当前相机占用和机器人服务状态：

```bash
rosservice call /cameras/list
python - <<'PY'
import rospy
from baxter_interface.camera import CameraController
rospy.init_node('open_baxter_head_camera', anonymous=True)
camera = CameraController('head_camera')
camera.resolution = (640, 400)  # 设置分辨率时会打开相机
print('已请求启动 head_camera')
PY
python record_baxter.py --list
```

确认出现 `/cameras/head_camera/image` 及**该图像对应**的 `/cameras/head_camera/camera_info_std` 后，可将它们分别用作 `--image-topic` 和 `--camera-info-topic`。如果报相机占用，先确认是否有不再使用的手腕相机占据了相机资源；不要随意关闭正在被其他程序使用的相机。头部或躯干若发生转动，须重新采集该姿态的数据。

若相机没有 `CameraInfo`，但已经有该相机**当前彩色流分辨率**的内参文件，也可以改成 `--intrinsics /绝对路径/intrinsics.json`。内参 JSON 应含 `width`、`height`、`camera_matrix`（3×3）和 `dist_coeffs`，不要复用 Realman 的 D435 内参。ROS 话题须是真实的彩色 `sensor_msgs/Image`；程序不会把深度图当作彩色图处理。

如果固定 RealSense **直接接到这台本机 USB**，且没有运行 ROS 相机发布器，可以让采集程序直接读 RealSense，并自动保存这一台相机实际彩色流的内参：

```bash
python record_baxter.py --arm right --realsense --width 640 --height 480 --fps 30
# 有多台 RealSense 时指定正确设备：--serial 设备序列号
```

本机现已检测到一台 D455（序列号 `043422251096`）；直接采集时程序会矫正其非零 `inverse_brown_conrady` 彩色畸变，再保存与矫正图片相配的零畸变内参。若要先把本机的彩色、对齐深度及内参发布为 ROS 话题，再使用上面的 `--image-topic` 采集，请按 [本机 D455 发布说明](../camera/README.md) 操作。**同一台相机不能同时运行发布程序和采集程序的 `--realsense` 模式**。运行程序前先确认终端能收到持续更新的 `/robot/limb/right/endpoint_state`；仅 `rostopic list` 出现话题不代表数据已到达本机。

将标定板固定到右臂末端，棋盘格始终完整入镜。**不传 `--action`** 时，每次改变右臂位置与倾斜角并停稳后，在图像窗口按 `a` 保存一组；按 `q` 结束。默认按顺序新建 `1/`、`2/`……，首次目录结构如下，永不覆盖旧数据：

```text
1/
├── picture/0.png      # 第一张照片；随后 1.png、2.png……
├── action/0.json      # 同组的七个关节角（弧度）和末端位姿
├── record.txt         # 每行 x,y,z,rx,ry,rz；第一行对应 0.png
├── intrinsics.json    # 本次彩色图内参
└── session.json       # 相机与机器人接口信息
```

下次想复现这批姿态，传入整个动作文件夹（按编号依次执行，自动生成 `2/picture` 和 `2/action`），或指定其中一组动作文件（仅执行这一组）：

```bash
# 先确认机器人已使能，标定板和相机固定，手臂运动范围内没有障碍物。
python record_baxter.py --arm right \
  --image-topic /baxter_camera/color/image_raw \
  --camera-info-topic /baxter_camera/color/camera_info \
  --action /home/robot24/wzy_ws/baxter_worksapce/eyetohand/1/action

# 若只回放一组，可改用 --action /home/robot24/wzy_ws/baxter_worksapce/eyetohand/1/action/0.json
# 也可用 --output /home/robot24/wzy_ws/baxter_worksapce/eyetohand/2 指定新目录。
```

程序先验证机器人已使能、运动命令接口已连接，才开始回放。每一组调用 Baxter SDK 的关节位置运动接口，**最多等待 10 秒确认到位**，随后再等待 **10 秒** 停稳并自动拍照；新的实际关节角和末端位姿也会保存到输出目录。到位超时、当前姿态偏离目标、图像与关节前后不稳定时停止或拒绝保存，不会将上一轮的末端位姿直接复用。按 `q` 可中止后续动作。运动速度比例默认 `0.15`，结束后恢复 Baxter SDK 默认 `0.3`；可通过 `--move-timeout`、`--settle-seconds`、`--speed` 调整。启动前请确保运动路径没有障碍，操作时保持急停可用。

使用输出目录运行求解（第一轮为 `1`，回放得到下一轮则改为 `2`）：

```bash
python accurate_eye_to_hand.py \
  --dataset /home/robot24/wzy_ws/baxter_worksapce/eyetohand/1 \
  --cols 11 --rows 8 --square-size 0.015
```

`--cols` / `--rows` 是标定板**内角点**数，`--square-size` 是单格边长（米）；`11×8、0.015 m` 是之前那块板子的参数，更换板子应重新测量。默认优先使用 OpenCV 的 SB 棋盘格检测（失败时回退普通检测）；这批 `1/` 数据中普通检测会出现角点偏移和棋盘方向翻转，导致错误标定。仅识别到棋盘角点的图片会参与求解，屏幕打印有效图片编号及拟合误差；结果保存为该目录下的 `hand_eye_result.json`。其中 `T_base_camera` 为**相机坐标系到 Baxter 末端位姿所属基座坐标系**的变换，`T_gripper_target` 是末端到标定板的估计；这不是夹爪中心补偿，也不应直接用于驱动机器人抓取。先核对末端 `header.frame_id` 及相机固定方式，再用独立测试点验证实际误差。

新相机的 `5/` 共 31 组，经逐图检查，本次排除 `5、7、8、14、20、25`，用其余 25 张计算并保存为 `5/hand_eye_result.json`。逐图原因、指标、角点总览和留出检验在 `5/quality/`；原始数据全部保留。复算时必须带同样的筛选参数，图像编号仍与原始 `record.txt` 对应：

```bash
python accurate_eye_to_hand.py --dataset 5 --cols 11 --rows 8 --square-size 0.015 \
  --exclude-images 5 7 8 14 20 25
```

点选抓取改为 `bash run_baxter_grasp.sh --execute --calibration 5/hand_eye_result.json --stop-above-mm 7`，使用新相机结果并保留目标抬高 7 mm。

**用新姿态验证（只读、不驱动手臂）：**保留原来的相机位置、分辨率和右臂上的标定板。重新运行上面的 `record_baxter.py` **不带 `--action`**，手动摆出至少 5 组和 `1/` 不同的右臂位置与倾斜角，停稳后每组按 `a`，最后按 `q`。会自动保存到 `2/`（若 `2/` 已存在，则用实际打印的新目录编号）。然后执行：

```bash
cd /home/robot24/wzy_ws/baxter_worksapce/eyetohand
python validate_baxter.py --dataset 2 --calibration 1/hand_eye_result.json
```

该脚本逐张比较相机观察到的棋盘在基座的位置与右臂末端预测的棋盘位置，输出平移毫米、旋转角度和单张图的 PnP 像素误差；新图不参与 `1/` 的求解。相机/板子/底座必须保持固定，不要拿同一批图片、同一组回放关节姿态冒充独立验证。此检查只能检验相机、末端、标定板在新姿态下的一致性；**不能单独证明彩色对齐深度的毫米精度、机器人基座绝对坐标或夹爪中心 TCP 偏移**。实际抓取前还要用单独测量的物理标记点和夹爪中心做低速、可中断的到点检查。

本机 D455 已通过 [本机相机发布脚本](../camera/README.md) 提供固定相机话题。现场右臂位姿话题首次连接可能耗时约 20 秒，此后约 100 Hz 更新；采集脚本等待末端与关节状态首次连接。机器人消息 `header.frame_id` 是空字符串，采集脚本原样记录；图像和位姿配对使用本机单调时钟及停稳检查。**动作回放只通过 `--action` 显式触发**；开发和验证脚本时没有向 Baxter 发送运动指令。

## 点选控制右臂到目标、开合夹爪

无需重新拍标定照片，直接使用 `1/hand_eye_result.json` 和本机发布的 RGB/对齐深度，流程与 Realman 相同。先确保相机发布器运行：

```bash
cd /home/robot24/wzy_ws/baxter_worksapce/camera
bash camera_service.sh start --fps 15
```

在本机图形桌面的终端启动抓取界面：

```bash
cd /home/robot24/wzy_ws/baxter_worksapce/eyetohand
bash run_baxter_grasp.sh --execute
```

入口脚本自动加载 Noetic、`ros_ws` 消息和 `eyetohand` Python，并使用 `ROS_MASTER_URI=http://192.168.1.102:11311`、`ROS_IP=192.168.1.116`。保留终端显式设置的其他远程 master/IP，将 ROS 默认 localhost master 改为上述 Baxter master。省略 `--execute` 只预览、计算和保存选点，不发送关节或夹爪命令。`bash run_baxter_grasp.sh --headless-check` 为只读联机检查，订阅图像、核对内参、读取右臂并求当前姿态的 IK 后退出。

窗口左侧 RGB、右侧对齐深度；点击和按键都在窗口操作：

| 操作 | 行为 |
| --- | --- |
| `o` | 张开右侧电动夹爪 |
| RGB 图上左键 | 读取点击点周围 5×5 深度，显示相机坐标和基座目标坐标；点击本身不运动 |
| `p` | 读取**按键时右臂当前四元数**，以该目标姿态移动夹爪点到选点；不自动闭合 |
| `c` | 到达目标后闭合右夹爪，打印 gripping/position 反馈 |
| `r` | 保持上一步目标姿态，沿到点方向退回约 8 cm |
| `x` | 取消后续路径，并请求关节保持当前位置及夹爪停止 |
| `q` / Esc | 退出；正在执行动作时先取消 |

启动前手动调整右臂朝向，移除夹爪上的标定板；相机、机器人底座保持标定时的位置。流程为 `o` 张开 → 点击 → `p` 到点 → 观察 → `c` 闭合 → `r` 退回。程序不自动估计抓取姿态。默认 `--grasp-inset 0` 且 `--stop-above-mm 7`：在可见表面点的基座 Z 坐标上加 7 mm，X/Y 不变，例如原本下降 70 mm 现在下降 63 mm。这是目标位置修正，与夹爪 TCP 偏移分开；选点预览、按 p 执行和退回均使用修正后的位置，日志同时保留原始表面点。用 `--stop-above-mm 0` 可恢复直接到表面。可见表面点不一定是物体几何中心。

深度使用 `16UC1` 毫米，RGB/深度/CameraInfo 必须同时间戳和光学坐标系，并与标定内参匹配。无效深度、深度边缘跳变、过期图像/状态、IK 无解或关节大跳变会拒绝动作。先预检全部路径点（默认每 4 cm 一个固定目标姿态的 IK 点），再通过 Baxter SDK 的正常关节位置模式逐段执行。单段内部是关节运动，不能保证严格笛卡尔直线或全程精确恒姿态，也不包含桌面/物体的环境碰撞规划。速度比例默认 `0.1`，每段到位超时 20 秒，不设单次总移动距离上限。程序核对使能/停止状态，不自动使能、校准或重置硬件；软件停止依赖 ROS 网络，紧急情况使用机器人急停。

**Baxter 不使用 Realman 的 184.3 mm 补偿。** 已核对在线模型：`right_gripper_base -> right_gripper` z=132.7 mm，`right_hand -> right_gripper_base` z=25 mm；当前 `endpoint_state` 与 `base -> right_gripper` TF 的位置差约 0.3 mm，endpoint 已包含所配置的夹爪长度。额外偏移默认 `--tcp-offset-mm 0 0 0`。若改装指尖中心与模型不同，可用该参数指定相对当前 endpoint 局部坐标系的额外偏移，不能再次加入整段夹爪长度。换算为 `endpoint_target = selected_point_base - R_current @ tcp_offset`。

选点彩色图、毫米深度图、目标坐标、按 p 时姿态、预检关节路径和操作记录保存在独立的 `grasp_baxter_时间戳/`。另一份标定可通过 `--calibration /绝对路径/2/hand_eye_result.json` 指定。本次开发只执行了离线测试与只读联机检查，没有驱动实际右臂或开合夹爪。
