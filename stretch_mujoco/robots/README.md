# 多机器人统一接口

`stretch_mujoco.robots` 为 Stretch 3、Google Robot 和 Stanford TidyBot 提供统一的
MuJoCo 控制、状态、相机、传感器、位姿和抓取接口。统一的是调用方式和数据语义；各机器人仍保留
符合自身 MJCF 与控制方式的 actuator、camera、sensor 和运行后端。

## 快速开始

```python
from stretch_mujoco.robots import RobotType, create_simulator

sim = create_simulator(
    RobotType.GOOGLE_ROBOT,
    cameras_to_use=[],  # 相机默认关闭；按需传 sim.Cameras.all()
)

try:
    sim.start(headless=True)
    sim.move_to(sim.Actuators.joint_shoulder, 0.2)
    sim.wait_until_at_setpoint(sim.Actuators.joint_shoulder)
    print(sim.pull_status())
finally:
    sim.stop()
```

需要启用全部相机时，先取得机器人类型，再把对应枚举传给工厂：

```python
probe = create_simulator(RobotType.TIDYBOT)
sim = create_simulator(
    RobotType.TIDYBOT,
    cameras_to_use=probe.Cameras.all(),
    camera_hz=20,
)
```

`cameras_to_use=None` 和空列表都表示不创建 renderer，以避免不需要视觉时占用 GPU。

## 目录结构

```text
robots/
├── README.md                 # 本文档
├── __init__.py               # RobotType 与 create_simulator 工厂
├── base.py                   # 公共 ABC、枚举协议和数据接口
├── base_controllers.py       # 差速与 planar/omni 底盘控制器
├── generic_simulator.py      # Google Robot、TidyBot 的线程后端
├── stretch3/
│   ├── robot.py              # Stretch3RobotSimulator 适配器
│   ├── actuators.py          # 兼容导出
│   ├── cameras.py            # 兼容导出
│   ├── sensors.py            # 兼容导出
│   ├── sites.py              # 抓取、IMU、Lidar、marker frame
│   └── *_data.py / joints.py # 兼容导出和状态类型
├── google_robot/
│   ├── robot.py              # GoogleRobotSimulator 适配器
│   ├── actuators.py          # planar base、arm、gripper、head
│   ├── cameras.py            # SimplerEnv head RGB + simulated depth
│   ├── sensors.py            # IMU、ToF、cliff sensor
│   ├── sites.py              # gripper、camera、IMU frame
│   ├── joints.py             # StatusGoogleRobotJoints
│   ├── camera_data.py        # StatusGoogleRobotCameraData
│   ├── sensor_data.py        # StatusGoogleRobotSensorData
│   └── config.py             # 关节范围、速度和相机标定
└── tidybot/
    ├── robot.py              # TidyBotRobotSimulator 适配器
    ├── actuators.py          # planar base、Gen3、2F-85 tendon
    ├── cameras.py            # calibrated base RGB + simulated RGB-D
    ├── sensors.py            # 当前模型为空
    ├── sites.py              # pinch_site
    ├── joints.py             # StatusTidyBotJoints
    ├── camera_data.py        # StatusTidyBotCameraData
    ├── sensor_data.py        # 空传感器快照
    └── config.py             # keyframe、关节范围和 C930e 标定
```

## 分层结构

```text
用户代码
   │
   ▼
create_simulator(RobotType, ...)
   │
   ├── Stretch3RobotSimulator ──► StretchMujocoSimulator
   │                              └── 多进程 MujocoServer + Manager proxies
   │
   ├── GoogleRobotSimulator ──┐
   │                          ├──► GenericRobotBackend
   └── TidyBotRobotSimulator ─┘    └── 进程内 physics thread + RLock
```

所有适配器实现 [`RobotSimulator`](base.py)，并通过以下四个属性暴露机器人专用类型：

```python
sim.Actuators  # type[RobotActuators]
sim.Cameras    # type[RobotCameras]
sim.Sensors    # type[RobotSensors]
sim.Sites      # type[RobotBodySites]
```

因此业务代码可以复用生命周期和查询逻辑，同时仍由类型系统阻止把 TidyBot actuator 传给
Google Robot。

## 公共接口

### 生命周期

```python
sim.start(show_viewer_ui=False, headless=False, use_passive_viewer=True)
sim.is_running()
sim.stop()
```

- `headless=True` 不创建交互式 MuJoCo viewer。
- Stretch 3 支持原有 passive/managed viewer 路径。
- `GenericRobotBackend` 当前只支持 passive viewer；传入
  `headless=False, use_passive_viewer=False` 会抛出 `NotImplementedError`。
- Stretch 3 使用 multiprocessing，调用入口必须置于
  `if __name__ == "__main__":` 内。

### 关节控制

```python
sim.move_to(actuator, absolute_position)
sim.move_by(actuator, increment)

sim.is_reached_set_position(actuator, position_tolerance=0.05)
sim.wait_until_at_setpoint(actuator, timeout=5.0)
sim.wait_while_is_moving(actuator, timeout=5.0)
```

`RobotActuators.actuator_type()` 返回：

| 类型 | 语义 |
|---|---|
| `POSITION` | `move_to` 参数为绝对关节位置 |
| `VELOCITY` | actuator 本身由速度控制，例如 Stretch wheel actuator |
| `GENERAL` | 参数为原始 `ctrl`，例如 TidyBot 夹爪的 `0..255` |
| `MOTOR` | 为未来 torque/current actuator 预留 |

适配器会校验 actuator 的枚举类型。传错机器人枚举会抛出 `TypeError`；有限 actuator 的目标值会
依据 MJCF `ctrlrange` 截断。

### 底盘控制

```python
sim.set_base_velocity(
    v_linear=0.3,    # 前向速度，m/s
    omega=0.2,       # 角速度，rad/s
    v_lateral=0.0,   # 横向速度，m/s
)
```

- Stretch 3 是差速底盘，`v_lateral` 必须为 `0`，否则抛出 `ValueError`。
- Google Robot 和 TidyBot 当前使用 position-controlled planar base；后端每个 physics step
  将 body-frame 速度积分到 `x/y/theta` position target。
- Google Robot 为兼容现有统一接口保留 planar compatibility layer，并未直接模拟
  SimplerEnv URDF 的轮地接触动力学。

### Keyframe

| 机器人 | `home()` | `stow()` |
|---|---|---|
| Stretch 3 | `home` | `stow` |
| Google Robot | 未定义，抛出 `NotImplementedError` | 未定义，抛出 `NotImplementedError` |
| TidyBot | `home` | 映射到 `retract` |

### 状态与位姿

```python
status = sim.pull_status()
base_xyt = sim.get_base_pose()
world_T_ee = sim.get_ee_pose()
world_T_link = sim.get_link_pose("link_name")
world_T_site = sim.get_site_pose(sim.Sites.GRIPPER)
limits = sim.pull_joint_limits()
```

- 位姿统一返回 `4×4`、world-frame 齐次变换矩阵。
- `get_site_pose()` 要求传入当前机器人的 site 枚举。
- Google Robot/TidyBot 直接读取 MuJoCo body/site pose，不经过 Stretch URDF 正运动学。
- `pull_status()` 返回机器人专用 dataclass。每个普通 actuator 对应一个
  `PositionVelocity(pos, vel)` 字段。
- 状态对象采用 `default_factory` 创建嵌套字段，实例之间不共享可变状态。

### 抓取与场景工具

```python
sim.attach_object_to_gripper("object_body_name")
sim.release_grasped_object()
sim.set_object_visibility("object_body_name", False)
sim.add_world_frame((1.0, 0.0, 0.5), (0.0, 0.0, 0.0))
```

`attach_object_to_gripper()` 是稳定抓取辅助功能。Generic backend 要求目标 body 存在且由 free
joint 驱动，并保持首次附着时的 gripper-relative transform。它不是接触物理抓取判定器。

Generic backend 的 `add_world_frame()` 只向 passive viewer 的 user scene 添加坐标轴；headless
模式下不会产生可见输出。

## 相机接口

### 枚举语义

每个 `RobotCameras` 成员提供：

```python
camera.camera_name_in_mjcf
camera.is_rgb
camera.is_depth
camera.is_simulated
camera.initial_camera_settings
camera.post_processing_callback
```

同一个 MJCF `<camera>` 可以对应 RGB 和 simulated depth 两个枚举成员。例如 Google Robot 的
`overhead_camera` 与 `overhead_depth` 共享相同 optical frame，但分别使用 RGB renderer 和 depth
renderer。

### 获取图像

```python
snapshot = sim.pull_camera_data()

rgb = snapshot.get_camera_data(
    sim.Cameras.rgb()[0],
    auto_correct_rgb=True,  # RGB -> OpenCV BGR
)
depth_m = snapshot.get_camera_data(sim.Cameras.depth()[0])
all_frames = snapshot.get_all(auto_correct_rgb=True)
K = snapshot.get_intrinsic_matrix(sim.Cameras.rgb()[0])
```

- RGB 数组为 `uint8 [H,W,3]`。
- Generic backend 的 depth 数组为 `float32 [H,W]`，单位为米。
- `CameraSettings` 保存 `fovy`、`fx/fy`、分辨率、可选 `cx/cy`、crop 和 distortion
  coefficients。
- 相机默认关闭，必须在构造 simulator 时通过 `cameras_to_use` 启用。
- Stretch 3 会按当前 `MjModel` 自动跳过不存在的可选相机。例如默认场景没有
  `office_overview`，而 `office_scene.xml` 中会启用它。

### 相机能力

| 机器人 | RGB | Depth | 说明 |
|---|---:|---:|---|
| Stretch 3 | D405、D435i、Nav；可选 office overview | D405、D435i | D405/D435i 为成对 RGB-D |
| Google Robot | `overhead_camera`，640×512 | `overhead_depth`，640×512 | RGB 使用 SimplerEnv 调优 K；depth 为仿真流 |
| TidyBot | `base` 640×360、`wrist` 640×480 | 对应两路 simulated depth | base RGB 使用官方 C930e 标定；wrist 为仿真相机 |

TidyBot 官方项目的两台 ceiling camera 属于环境传感器，不属于 robot-mounted camera。公开源码
没有提供可复用于任意场景的 6DoF 外参，因此未把它们硬编码到默认机器人模型中。

## 传感器接口

```python
for sensor in sim.available_sensors:
    value = sim.pull_sensor_data().get_data(sensor)
    metadata = sim.get_sensor_metadata(sensor)
    print(sensor.name, value.shape, metadata.units, metadata.frame)
```

`SensorMetadata` 包含：

- `sensor_type`
- `shape`
- `dtype`
- `units`
- `frame`
- `description`

一个逻辑传感器可以聚合多个 MuJoCo sensor。Google Robot 的 `TIME_OF_FLIGHT` 由 8 个
`rangefinder` 聚合为 shape `(8,)`；`CLIFF` 由 2 个向下射线聚合为 `(2,)`。

| 机器人 | 逻辑传感器 |
|---|---|
| Stretch 3 | base gyro `(3,)`、base accel `(3,)`、360-beam lidar `(360,)` |
| Google Robot | base/neck gyro 和 accel、8-beam ToF、2-beam cliff |
| TidyBot | 当前 MJCF 无内部 sensor，返回空集合 |

`available_sensors` 根据实际编译后的 `MjModel` 检测，不应只根据枚举推断某个自定义场景一定
包含对应 sensor。

## Body/Site 引用

`BodySiteRef` 用三个字段描述坐标参考：

```python
BodySiteRef(
    name="gripper",
    mjcf_kind="site",   # "body" 或 "site"
    purpose="ee_grasp",
)
```

`purpose` 当前常用值为 `ee_grasp`、`camera`、`imu`、`lidar`、`marker` 和
`docking`。`grasp_sites()` 与 `camera_sites()` 提供语义子集，避免业务代码硬编码 MJCF 名称。

主要抓取参考如下：

| 机器人 | 抓取参考 | MJCF kind |
|---|---|---|
| Stretch 3 | `link_grasp_center` | body |
| Google Robot | `gripper` | site |
| TidyBot | `pinch_site` | site |

## 三种机器人的运行后端

### Stretch 3

[`Stretch3RobotSimulator`](stretch3/robot.py) 是现有 `StretchMujocoSimulator` 的适配层。
物理、相机和传感器运行在子进程，通过 `multiprocessing.Manager` proxies 向调用进程发布快照和
接收命令。这样保持 Stretch 原有示例、ROS/Web Teleop 语义和 viewer 路径。

### Google Robot 与 TidyBot

[`GenericRobotBackend`](generic_simulator.py) 在当前进程创建 `MjModel/MjData`，后台 physics
thread 持续 step。所有 MuJoCo data 访问由 `threading.RLock` 串行化；调用者获得 dataclass
副本，不直接持有后台可变状态。

每个适配器向后端注入：

- actuator enum 与 status dataclass
- camera enum 与 camera-data dataclass
- sensor enum 与 sensor-data dataclass
- planar base actuator 名称
- grasp reference
- 最大线速度、角速度
- 可选初始关节位置

该后端负责 ctrlrange 截断、position target、planar velocity 积分、keyframe、camera rendering、
sensor snapshot、body/site pose、可见性和稳定附着。

## 机器人能力矩阵

| 能力 | Stretch 3 | Google Robot | TidyBot |
|---|---|---|---|
| `move_to/move_by` | ✓ | ✓ | ✓ |
| 底盘 | 差速 | planar compatibility | planar/omni |
| 横移 `v_lateral` | ✗ | ✓ | ✓ |
| actuator 类型 | Position + Velocity | Position | Position + General |
| RGB-D | D405、D435i | head RGB + simulated depth | base/wrist RGB + simulated depth |
| 内部 sensor | IMU + lidar | 2×IMU + ToF + cliff | 无 |
| 抓取 frame | `link_grasp_center` | `gripper` | `pinch_site` |
| keyframe | home、stow | 无 | home、retract |
| 运行模型 | 多进程 | physics thread | physics thread |

## 工厂参数

```python
create_simulator(
    robot_type,
    scene_xml_path=None,
    model=None,
    camera_hz=30,
    cameras_to_use=None,
    start_translation=None,
    start_rotation_quat=None,
)
```

- `robot_type` 接受 `RobotType` 或字符串 `stretch3/google_robot/tidybot`。
- `scene_xml_path` 优先于机器人默认 scene。
- `model` 可直接传入已编译的 `mujoco.MjModel`。
- `start_translation` 为 `[x,y,z]`。
- `start_rotation_quat` 为 `[w,x,y,z]`。
- 对 free-joint Stretch base，完整 translation/quaternion 写入 `qpos0`。
- 对 planar Google Robot/TidyBot，只使用 translation 的 `x/y`，并从 quaternion 提取 yaw；不会
  把 `z` 或 quaternion 分量误写到机械臂关节。

## 交互示例

Stretch 3 键盘控制和全部相机：

```bash
.venv/bin/python examples/keyboard_teleop.py --imagery
```

Google Robot 或 TidyBot 键盘控制与 RGB-D：

```bash
python examples/robot_keyboard_rgbd.py --robot google_robot
python examples/robot_keyboard_rgbd.py --robot tidybot
```

只开 OpenCV RGB-D 窗口、不启动 3D viewer：

```bash
MUJOCO_GL=egl python examples/robot_keyboard_rgbd.py --robot tidybot --no-viewer
```

## 添加新机器人

1. 在 `robots/<robot_name>/` 创建以下文件：
   `actuators.py`、`cameras.py`、`sensors.py`、`sites.py`、`joints.py`、
   `camera_data.py`、`sensor_data.py`、`config.py` 和 `robot.py`。
2. 实现 `RobotActuators`：MJCF joint 映射、类型、base 标记、状态读数和反向查找。
3. 实现 `RobotCameras`：MJCF camera 名、RGB/depth 分类、标定和仿真标记。
4. 实现 `RobotSensors`：名称、metadata、replicated sensor 聚合及 `from_mjmodel()`。
5. 实现 `RobotBodySites`：抓取、相机及其他语义 frame。
6. 为 status/camera/sensor snapshot 创建 dataclass；嵌套可变字段必须使用
   `field(default_factory=...)`。
7. 实现 `RobotSimulator` 适配器。position-controlled planar 机器人可复用
   `GenericRobotBackend`；其他底盘或 torque controller 应实现专用后端/控制器。
8. 在 `RobotType` 和 `create_simulator()` 中注册新类型。
9. 添加以下测试：
   - 包导入和工厂构造
   - MJCF actuator/camera/sensor 名称一致性
   - 状态 dataclass 不共享默认对象
   - `move_to/move_by` 与 ctrlrange
   - base pose/velocity
   - camera shape、dtype、K 和 depth 单位
   - sensor shape 与 metadata
   - site/EE pose
   - keyframe 和错误类型

## 模型来源与保真边界

- Stretch 3 使用项目原生 MJCF/URDF 资产和标定。
- Google Robot 保留 Menagerie geometry 与稳定的 planar base，同时从
  SimplerEnv/ManiSkill2_real2sim URDF 移植 head、camera、IMU、ToF 和 cliff frame。它不是
  Google 官方制造级 CAD/标定包。
- TidyBot 使用 Menagerie 的 base CAD、Kinova Gen3 和 Robotiq 2F-85；base RGB 使用官方
  TidyBot robot #2 C930e 标定。wrist 与 depth stream 为仿真能力。
- Generic backend 的稳定附着、planar base 和 simulated depth 面向任务仿真，不等价于完整
  wheel/contact、真实相机噪声、畸变和抓取接触动力学。

## 测试

机器人接口和 RGB-D 回归测试主要位于：

- `tests/test_robot_interfaces.py`
- `tests/test_robot_keyboard_rgbd.py`
- `tests/test_stretch_camera_availability.py`
- `tests/test_keyboard_control.py`

运行项目测试：

```bash
.venv/bin/pytest -q tests
```

仓库根目录下还包含 `third_party` 自带测试；直接运行无路径限制的 `pytest` 可能需要 Robocasa、
Robosuite 等额外依赖。
