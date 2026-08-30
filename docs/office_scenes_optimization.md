# 十个办公室场景优化说明

## 1. 当前版本效果

仓库已有 10 个单区域开放式办公室，集中在 stretch_mujoco/models/assets/office_scenes/：

| 编号 | 文件前缀 | 当前布局 |
|---|---|---|
| 1 | office_01_linear_bench | 线性工作台 |
| 2 | office_02_cross_axis | 十字轴向工作区 |
| 3 | office_03_long_gallery | 长廊式工作区 |
| 4 | office_04_central_meeting | 中央会议区 |
| 5 | office_05_team_clusters | 团队簇状布局 |
| 6 | office_06_diagonal_flow | 对角动线 |
| 7 | office_07_u_bench | U 形工作台 |
| 8 | office_08_dual_island | 双岛台 |
| 9 | office_09_staggered_rows | 错列排布 |
| 10 | office_10_social_core | 社交核心区 |

每个场景有 XML、JSON manifest 和 catalog 条目，包含工作区、会议区、休息区、零食台和 Stretch 初始位姿；使用共享 office_assets 资源和简化碰撞代理。Scene 2 另有 office_02_cross_axis_robot.xml、office_scene2_multi_npc.xml 用于机器人和多 NPC 语义演示。

## 2. 当前版本使用方式

    uv run examples/generated_office_scene.py --scene 1
    uv run examples/generated_office_scene.py --scene 10 --top-view
    uv run examples/generated_office_scene.py --scene 3 --auto-orbit
    uv run examples/generated_office_scene.py --scene 1 --headless

--scene 取值 1-10；窗口中可用鼠标旋转、平移、缩放，按 1 切换 overview、2 切换 top view、O 切换自动环绕。程序从 catalog.json 读取 XML 并打印纹理、材质、网格和 geom 数量。

查看带机器人/NPC 的 Scene 2：

    uv run examples/office_multi_npc_day.py --end 18 --step 0.25 --chat-log

基础 Stretch 办公室（含零食和单 NPC 动画）：

    uv run examples/office_scene.py

## 3. 已知问题

### 3.1 几何和摆放

- 区域边界、工作区与会议/休息区间距不一致；机器人 clearance 数值未形成统一可行走性校验。
- 桌面电脑、显示器、键盘和支架的局部坐标/高度没有统一约定，已出现电脑不在桌面、漂浮在空中的问题；部分物体与桌面、墙面或椅子穿插。
- 视觉 mesh 与碰撞 proxy 的尺寸、旋转和原点可能不一致，导致看起来接触但抓取/导航判定失败。
- 站立点、坐下点、机器人操作点命名和朝向不统一，NPC 可能从桌后穿行或坐到错误一侧。
- 相机中心、视距和灯光未按 bounds 自适应，部分场景首屏看不全或出现过曝/死角。

### 3.2 资产和语义

- 十个 XML 存在重复结构，资产、材质、碰撞体和 site 容易漂移，缺少统一 schema 校验。
- manifest 主要记录数量和初始位姿，未描述家具锚点、占用空间、交互点、可通行宽度和可抓取面。
- 电脑、会议桌、零食台等关键物体的 semantic ID 与 XML name 需要逐场景核对；缺失绑定会让 Agent 逻辑位置与画面不一致。

## 4. 优化方案

### A. 统一坐标和布局规范

- 规定 Z=0 为地面，桌面、显示器底座、椅面和机器人操作高度使用集中常量；桌面物品通过 on_surface 锚点计算 Z，不再手写绝对高度。
- 为每个区域定义 zone、入口、出口、最小通道宽度和安全缓冲；脚本检查家具 AABB 不重叠、通道可达、机器人和 NPC 起点可达。
- 每个交互对象提供 approach_site、interaction_site、place_site 和朝向；命名与 office_semantics*.json 保持一致。

### B. 修复桌面电脑和物体

- 为 desk、monitor、stand、keyboard 建立父 body 和局部锚点；显示器底部固定在桌面上方，键盘/鼠标投影到桌面平面，固定设备禁止 freejoint。
- 增加 headless 几何审计：检查物体最低/最高点及与桌面、墙面的穿透量，发现漂浮或穿插时报告 XML body、geom 和建议。
- 视觉 mesh 只负责外观，碰撞 proxy 与真实尺寸共享 manifest 参数；抓取物体保留 freejoint，固定设备使用父 body/weld。

### C. 十场景专项

- 逐场景标注工作、会议、休息、零食和机器人停靠区，确保连续主通道和备用通道；5、6、9 重点检查长距离/对角转弯半径。
- 4、5、8、10 的多人聚集区增加椅子拉出与 handover 空间；1、2、7 重点检查桌后 NPC 站立点。
- 每个场景提供统一 overview/top 相机和灯光基线；catalog 记录面积、资产数量、最小通道宽度和机器人 clearance。

### D. 资源与生成流程

- 共享家具和材质保留在 office_assets，场景只保存布局参数；生成器输出 XML、manifest 和预览图，避免手工 XML 分叉。
- 增加 validate_office_scenes 命令，一次编译 10 个 XML，执行命名、AABB、支撑关系、碰撞层、语义绑定和相机检查。
- 生成 top-view 预览和问题标记，提交前人工抽查电脑、椅子、门口、零食台和机器人起点。

## 5. 分工与验收标准

同事负责 stretch_mujoco/models/assets/office_scenes/、共享 office_assets、场景生成/校验脚本和 office_scene*.json；NPC 同事只依赖稳定的 site、semantic ID 和 manifest schema，不直接改布局 XML。

验收要求：10 个场景均可 headless 编译；电脑和桌面物品均落在支撑面且无穿插/漂浮；NPC 与 Stretch 能从初始点到达所有交互点；AABB、通道宽度和语义绑定检查无错误；overview/top 预览完整，机器人操作与多人 handover 空间满足阈值。

