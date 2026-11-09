# 维护说明

唯一维护目录：D:\graduate\meereis-sr，对应 GitHub wxh042/meereis-sr。

## 日常使用

1. 双击根目录“启动平台.cmd”。首次启动会自动生成本机演示密码，查看 runtime/local-config.json，用户名 researcher。
2. 打开 http://127.0.0.1:8501 。使用期间保持启动窗口打开，Ctrl+C 统一停止前端、API 和 Worker。
3. 双击“验证项目.cmd”检查两个模型、单张/批量推理、鉴权、错误输入及历史记录。验证使用独立临时数据库，不改日常任务数据。

默认演示模式为 SQLite + Celery 文件队列，无需安装 MySQL/Redis；正式研究部署可用环境变量切换。前后端仅监听本机地址。

## 代码归属

- platform/model.py：推理网络；改结构必须同步训练和权重兼容性验证。
- platform/celery_worker.py：推理、批处理、任务执行及模型缓存。
- platform/app.py：API；platform/database.py：持久化；platform/security.py：登录。
- platform/web_gui.py：交互界面。
- training/：原研究训练实现，保留网络版本差异用于复现，不当作多个可运行产品维护。
- experiments/：原实验评估脚本、逐样本指标。
- scripts/manage.py：统一服务生命周期；scripts/verify.py：端到端检查。

## 环境恢复

Python 3.11，新建 .venv 后安装 requirements.lock.txt。CPU PyTorch 使用官方 CPU wheel 源：

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.lock.txt
```

requirements.txt 表达直接依赖，requirements.lock.txt 记录本机验证过的确切版本。每次升级依赖后先运行验证，再更新锁定清单。

## 数据与文件

runtime/ 内为本机密码、数据库、队列、输出和日志，不提交 GitHub。
原科研大数据保存在 D:\graduate\code，旧工程压缩备份与清理清单保存在 D:\graduate\archives。
本机旧文件路径迁移时，先检查训练/评估路径和 README，不直接删除唯一的数据或模型权重。

## 提交流程

修改 → 运行验证 → 查看 git diff → commit → push。论文指标是历史实验结果；未经全量重跑不能用演示图像更新论文结论。

10 km 训练数据在 datasets/10km，25 km 训练切片在 datasets/25km；通过 MEEREIS_TRAIN_HR / MEEREIS_TRAIN_LR / MEEREIS_TRAIN_PATCHES 切换位置。训练数据不提交 Git。首次配置可运行 scripts/setup.ps1。
