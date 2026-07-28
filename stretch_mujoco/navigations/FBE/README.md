# Frontier-Based Exploration（FBE）

本目录实现二维占用栅格上的前沿探索（Frontier-Based Exploration）。机器人从一张全未知的局部地图开始，通过观测源逐步标记自由空间和障碍，检测已知自由区与未知区之间的前沿，并使用 A* 导航到合适的前沿，直到达到探索阈值，或地图停止变化且不存在可达前沿。

当前实现主要用于 MuJoCo/Habitat 仿真验证，但观测层已经与 FBE 核心解耦。仿真使用真值地图模拟二维激光；未来接入真机时，可以用 ROS `LaserScan`、SLAM 地图或 Nav2 costmap 后端替换仿真观测源，而无需修改前沿检测和状态机。

## 方法概览

局部地图使用三种栅格值：

| 值 | 含义 | 规划时是否可通行 |
|---:|---|---|
| `-1` | 未知、尚未观测 | 否 |
| `0` | 已知自由空间 | 是 |
| `1` | 已知障碍 | 否 |

一个自由栅格只要与至少一个未知栅格八邻接，就被定义为前沿栅格。相邻前沿通过八邻域 BFS 聚类，形成候选前沿簇。

前沿选择使用近似的信息增益/移动代价评分：

```text
score = cluster_size / (1 + euclidean_distance)
```

这会优先选择信息量较大且距离较近的前沿。当前距离是欧氏距离，并非 A* 的真实路径代价。

被选中的目标不是前沿簇几何质心本身，而是距离质心最近的真实前沿栅格，避免质心落入未知区域或障碍内部。A* 只允许经过已知自由栅格。

## 探索流程

```text
观测源更新局部地图
        ↓
检测并聚类前沿
        ↓
过滤当前地图版本中不可达的前沿
        ↓
选择信息量大且较近的前沿
        ↓
A* 在已知自由区域内规划
        ↓
根据机器人真实位姿逐路点执行
        ↓
地图变化时重新验证剩余路径
        ↓
继续探索或结束
```

如果某个前沿无法规划到达，该前沿簇会被加入当前地图版本的黑名单。地图出现新观测后，黑名单会清除，因为新发现的自由空间可能使原先不可达的区域变得可达。

正在执行的路径会在以下时机重新检查：

- 新观测改变局部地图时；
- 路点索引因为到达容差而前进后；
- 向外部控制器发布下一个目标之前。

如果剩余路径与最新地图冲突，FBE 会立即放弃路径、拉黑当前前沿并重新检测目标。

## 状态机

`FBEPlanner` 使用以下状态：

| 状态 | 作用 |
|---|---|
| `SCANNING` | 更新地图并检测候选前沿 |
| `PLANNING` | 使用 A* 规划到前沿目标 |
| `MOVING` | 等待机器人根据位姿反馈到达当前路点 |
| `ROTATING` | 预留的旋转扫描状态，当前实现会返回 `MOVING` |
| `FINISHED` | 探索结束 |

探索可能因以下原因结束：

- 探索比例达到 `explore_threshold`，并且没有剩余前沿；
- 连续 `max_stagnant_scans` 次扫描没有新增地图信息，并且没有可达前沿；
- 连续停滞后，地图中已经没有任何前沿。

具体原因保存在：

```python
fbe.diag.finish_reason
```

由于障碍内部、墙后空间和不可达区域可能永远无法被传感器观测，最终探索比例不一定达到 100%。

## 代码结构

```text
FBE/
├── README.md       # 本文档
├── __init__.py     # 公共接口导出
├── local_map.py    # 局部地图、观测源协议、仿真激光后端
├── frontier.py     # 前沿检测、聚类和评分
└── planner.py      # FBE 状态机、A* 调用、黑名单和路径验证
```

### `local_map.py`

- `LocalOccupancyGrid`
  - 保存局部三值占用地图；
  - 不读取仿真真值地图；
  - 提供世界坐标与栅格坐标转换；
  - `update_cells()` 用于融合自由/障碍栅格；
  - `update_from_array()` 用于融合已对齐的 SLAM 或 costmap 数组。
- `MapObservationSource`
  - 可替换观测后端的协议；
  - 后端只需实现 `update(local_map, robot_xy, robot_yaw)`。
- `SimulatedLaserObservationSource`
  - 当前 MuJoCo/Habitat 使用的理想二维激光；
  - 只有该类会读取仿真真值 occupancy；
  - 使用 Bresenham 射线更新局部地图。

### `frontier.py`

- `detect_frontier_cells()`：检测自由/未知边界；
- `cluster_frontiers()`：通过 BFS 聚类前沿；
- `best_frontier()`：根据前沿大小和距离选择目标；
- `FrontierCluster`：保存前沿栅格、质心和簇大小。

### `planner.py`

- `FBEPlanner`：探索状态机；
- `FBEState`：状态枚举；
- `FBEDiagnostics`：探索率、前沿数、当前路径和结束原因；
- 管理不可达前沿黑名单；
- 将局部地图转换为 A* 使用的 `OccupancyGrid`；
- 根据实时位姿推进路点并验证剩余路径。

## 直接使用 FBE

```python
from stretch_mujoco.navigations.FBE import FBEPlanner

fbe = FBEPlanner(
    god_grid,
    robot_xy=(0.0, 0.0),
    robot_yaw=0.0,
    num_rays=180,
    max_range_m=5.0,
    fov_degrees=270.0,
    min_cluster_size=3,
    min_frontier_dist=0.5,
    explore_threshold=0.85,
    max_stagnant_scans=30,
)

while not fbe.is_finished():
    # 位姿必须来自机器人或仿真器，而不是命令值。
    fbe.step(robot_xy, robot_yaw)

    waypoint = fbe.current_target()
    if waypoint is not None:
        send_waypoint_to_controller(waypoint)

print(fbe.diag.finish_reason)
```

`pop_command()` 为兼容旧接口而保留，但不会在发布命令时推进路径。路径进度只根据传入 `step()` 的真实机器人位姿更新。

## 通过统一控制器使用

```python
from stretch_mujoco.navigations import Algorithm, NavigationController

navigation = NavigationController(
    model,
    data,
    algorithm=Algorithm.FBE,
    planner_kwargs={
        "robot_xy": (0.0, 0.0),
        "explore_threshold": 0.85,
    },
)

diagnostics = navigation.step_exploration(robot_xy, robot_yaw)
waypoint = navigation.explorer.current_target()
```

FBE 没有固定终点，因此不能使用点到点接口：

```python
navigation.plan(start, goal)  # Algorithm.FBE 下会抛出 RuntimeError
```

如果任务是导航到一个已知绿色目标点，应使用 `Algorithm.ASTAR` 或 `Algorithm.FMM`，而不是 FBE。

## 自定义观测源

下面是一个不依赖仿真真值的最小后端：

```python
class AlignedMapObservationSource:
    def __init__(self, get_map):
        self.get_map = get_map

    def update(self, local_map, robot_xy, robot_yaw):
        # 返回值必须已经转换到与 local_map 相同的 frame、origin、
        # resolution 和 shape，数值为 -1、0、1。
        observation = self.get_map()
        local_map.update_from_array(observation)


source = AlignedMapObservationSource(get_aligned_occupancy_array)
fbe = FBEPlanner(
    reference_grid,
    robot_xy=(0.0, 0.0),
    observation_source=source,
)
```

接入 ROS 时，适配器还需要处理：

- `map`、`odom`、`base_link` 和传感器坐标系之间的 TF；
- `nav_msgs/OccupancyGrid` 的 origin、resolution 和栅格方向；
- ROS occupancy 概率到 `-1/0/1` 的阈值转换；
- 时间戳、过期消息和定位跳变；
- 机器人半径膨胀，避免与 Nav2 costmap 重复膨胀。

当前仓库尚未实现 ROS 后端。

## 仿真示例

二维 Matplotlib 演示：

```bash
.venv/bin/python examples/fbe_demo.py --steps 30000
```

MuJoCo 3D Viewer：

```bash
.venv/bin/python examples/fbe_viewer.py
```

Viewer 控制：

- `G`：开始或暂停；
- `R`：重置探索；
- `T`：俯视相机；
- `O`：全局相机；
- `Esc`：退出。

可视化中红色前沿标记才是 FBE 当前目标。绿色固定点只是演示参考点，不参与 FBE 的目标选择。

## 测试

```bash
.venv/bin/pytest -q tests/test_fbe_navigation.py tests/test_navigations.py
```

测试覆盖：

- 激光方向与局部地图更新；
- 可替换观测源；
- 未知区域不可通行；
- 前沿评分；
- 不可达前沿黑名单；
- 地图停滞终止；
- 路点反馈和路径失效；
- A* 安全回归；
- `Algorithm.FBE` 控制器集成。

## 当前限制

- 默认仿真激光是理想模型，没有噪声、延迟和定位误差；
- 前沿评分使用欧氏距离，而非真实路径代价；
- `ROTATING` 尚未实现完整的主动旋转扫描控制；
- FBE 只负责选择探索目标和生成路径，不负责底盘速度控制；
- 动态避障、恢复行为和急停应由 Nav2 或独立局部控制器负责；
- ROS `LaserScan`、TF 和 Nav2 costmap 适配器尚未实现。

