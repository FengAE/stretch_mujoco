# Git 协作与环境配置方案

## 1. 推荐结论

采用“一个稳定主分支 + 两条职责分支 + Pull Request”模式：

- main：可运行、可测试的基线，只接受 PR，不直接提交。
- feat/npc-system：NPC 动作、外观、状态、对话和 NPC 测试。
- feat/office-scenes：10 个办公室 XML/JSON、共享资产、布局生成器、几何校验和场景测试。

不建议给每个人一个长期独立仓库，也不建议两个人直接在 main 上开发。两个功能分支可以在本阶段保持数周，但每天都应从 main 同步；功能完成后尽快合并并删除。若两条分支需要联调，再临时建立 integration/office-npc，它不是发布分支。

## 2. 首次上传远程仓库

当前工作区存在大量未提交修改和未跟踪文件，首次上传前不要直接 git add .。先逐项确认哪些是项目源码、测试、场景资产，哪些是数据集、视频、运行日志、私有模型或临时文件。

建议流程：

    # 1. 在当前工作区创建基线检查分支
    git switch -c chore/prepare-repository
    git status --short

    # 2. 确认大文件和敏感文件不会进入 Git
    git check-ignore -v models_smplx_v1_1.zip stretch_mujoco/models/assets/humanoid/private/SMPLX_NEUTRAL.npz
    git ls-files | sort > /tmp/tracked-files.txt

    # 3. 分批加入源码、配置、文档和必要测试，检查暂存内容
    git add README.md pyproject.toml uv.lock docs stretch_mujoco examples tools tests
    git diff --cached --stat
    git diff --cached --name-only
    git commit -m "chore: establish collaborative project baseline"

    # 4. 绑定远程仓库并推送
    git remote add origin <远程仓库地址>
    git push -u origin chore/prepare-repository

确认基线能安装和运行后，将该分支合并为 main，并在远程仓库开启：

- main 分支保护：禁止强制推送和直接推送。
- PR 至少 1 人审核，CI 必须通过。
- 合并方式使用 squash merge，提交信息保持一个功能一个提交。
- CODEOWNERS：NPC 路径由 NPC 负责人审核，场景路径由场景负责人审核。

不要提交以下内容：*.local.json 中的 API 密钥、SMPL-X 私有参数、超大 zip、录制视频、运行输出、个人 IDE 配置。需要共享的私有模型放在受控对象存储或公司文件服务器，并在文档中记录校验和。

## 3. 创建和使用两条分支

在 main 基线合并后执行：

    git switch main
    git pull --ff-only origin main
    git switch -c feat/npc-system
    git push -u origin feat/npc-system

    git switch main
    git switch -c feat/office-scenes
    git push -u origin feat/office-scenes

同事首次开发：

    git clone --recurse-submodules <远程仓库地址>
    cd stretch_mujoco
    git switch --track origin/feat/npc-system
    # 或：git switch --track origin/feat/office-scenes

每天开始工作前同步主分支：

    git fetch origin
    git rebase origin/main

提交前先运行格式化和测试，再推送：

    git status
    git diff
    uv run pytest -q
    git push

不要使用 git push --force 覆盖共享分支。若 rebase 后确实需要更新个人分支，使用 git push --force-with-lease，并先通知另一位同事。

## 4. 文件边界和冲突规则

NPC 同事主要修改：

- stretch_mujoco/humanoid/
- stretch_mujoco/agents/
- stretch_mujoco/semantics/
- NPC 配置、对话协议和相关测试

场景同事主要修改：

- stretch_mujoco/models/assets/office_scenes/
- stretch_mujoco/models/assets/office_assets/
- tools/generate_office_scenes.py 和场景校验工具
- office_scene*.json、场景回归测试

双方共同修改 README.md、pyproject.toml、uv.lock 或公共接口时，先在 issue/群组中说明；依赖变化由一人修改后另一人立即重新执行 uv lock 和安装验证。NPC 依赖的 site 名、semantic ID 和 manifest schema 由场景同事先冻结，变更必须通过 PR 通知。

联调顺序建议为：场景同事先提交稳定的 XML/site/manifest，NPC 同事基于该接口开发；需要提前联调时，从场景分支合并到临时 integration/office-npc，不要互相 cherry-pick 未完成提交。

## 5. 快速环境配置

### 5.1 通用 Python/uv 环境

项目声明 Python >=3.10，Robocasa 额外依赖要求 Python 3.10。两位同事建议统一使用 Python 3.10.x 和 uv：

    # 安装 uv 后，在仓库根目录执行
    uv python install 3.10
    uv venv --python 3.10
    uv sync --extra dev

    # 验证解释器、依赖和基础场景
    uv run python -V
    uv run python -c "import mujoco; print(mujoco.__version__)"
    uv run examples/generated_office_scene.py --scene 1 --headless
    uv run examples/office_scene.py --headless
    uv run pytest -q

uv.lock 必须提交并作为依赖唯一来源；日常使用 uv run ...，不要在系统 Python 上单独 pip install。只做 NPC/场景开发时无需安装 Robocasa、VLM 等可选 extra；需要时再执行 uv sync --extra robocasa 或 uv sync --extra vlfm。

### 5.2 openpi-client 路径依赖

当前 pyproject.toml 将 openpi-client 指向仓库上级目录的 ../openpi/packages/openpi-client。因此需要 OpenPI 的同事在同级目录准备：

    cd ..
    git clone <openpi 仓库地址> openpi
    cd stretch_mujoco
    uv sync --extra dev

如果本阶段不使用 OpenPI，应将该本地路径依赖改为可安装版本或拆到可选 extra，并由 PR 明确记录，避免新成员因目录不存在而无法同步环境。

### 5.3 MuJoCo 图形依赖

Linux 无窗口/headless 测试通常只需要 Python 依赖；打开 viewer 还需要可用的 GLFW/OpenGL 驱动。若出现 evdev 编译错误，安装 python3-dev；若出现 GLFW/OpenGL 错误，检查显卡驱动或使用 --headless。Windows/macOS 使用同一套 uv 命令，图形驱动问题按本机系统处理。

### 5.4 私有 SMPL-X 资产

仓库忽略 stretch_mujoco/models/assets/humanoid/private/ 和 generated 输出，因此新环境不会自动拥有 NPC 网格。由项目负责人通过受控存储发放已授权文件，并在本地执行：

    unzip smplx_assets.zip -d stretch_mujoco/models/assets/humanoid/private/
    uv run prepare_smplx_npc --help
    uv run bake_smplx_animations --help
    uv run tools/generate_npc_textures.py

生成的 OBJ/PNG 只保留在本地或 CI 缓存，不提交远程仓库。资产包应附 SHA-256 校验值和许可证说明。

## 6. PR、CI 与发布节奏

每个 PR 只解决一个主题，标题使用 feat:、fix:、test: 或 docs:。PR 描述至少包含：改动目录、运行命令、测试结果、是否改变 XML/site/semantic ID、是否需要重新生成资产。

最低 CI 检查：uv sync --extra dev、uv run pytest -q、Black/flake8 检查、10 个办公室 XML headless 编译、NPC 场景 headless 编译。涉及私有模型的 CI 使用受控 runner 或跳过资产生成但必须检查引用路径。

建议每周从两条功能分支各合并一次可运行增量；合并前由另一位同事完成一次跨模块 smoke test。发布或演示前从 main 打 tag，例如 v0.6.0-npc-office，并保留对应的 uv.lock、场景 catalog 和资产清单。

## 7. 出现问题时的处理

- 环境无法安装：先确认 Python 版本、uv 版本、uv.lock 是否最新，再检查 ../openpi 路径。
- 场景能编译但物体漂浮：在场景分支运行 headless 几何校验，不要直接修改 NPC 代码掩盖问题。
- NPC 逻辑与画面不一致：检查 semantic ID、site 名和物理控制器完成回报，使用事件日志复现。
- 合并冲突：优先保留已冻结的公共接口；XML 大范围冲突由场景负责人解决，公共配置冲突由双方共同确认后再合并。

