# TacGate-DP 本地验证记录

验证日期：2026-10-02。环境：Python 3.9.23、PyTorch 2.8.0+cpu、
Torchvision 0.23.0+cpu、Diffusers 0.36.0、NumPy 1.26.4、Zarr 2.18.2。

- `python -m unittest discover -s tests -v`：23 项全部通过。
- `python scripts/smoke_test.py`：真实 PNG/Parquet/Zarr fixture 经 Stage A、
  checkpoint 恢复、Stage B、逆扩散评估和 Toy 闭环，输出 `status: OK`。
- 中断恢复的模型 state_dict 与同一环境下连续训练的 state_dict 逐项一致。
- 默认 4 帧、5 模态、256 维 token 和 8 步/8 维动作的模型前向通过。
- 两种 GroupNorm / U-Net 宽度组合的前向、反向与有限梯度验证通过。
- 离线 wheel 构建成功；安装到独立临时目标后，模型构造和训练 CLI 通过，
  未导入外部 `diffusion_policy` 包。
- 原研究目录 23 个 Python 源文件 SHA-256 保持一致。

## 更名与文档更新复核

2026-10-02 在 `TacGate-DP` 目录完成复核：

- 重新运行上述 23 项测试与完整 smoke，全部通过。
- 原研究目录 23 个、发布目录 `src` 下 18 个 Python 源文件的 SHA-256
  均与更名前一致。模型、数据和训练实现未因命名与文档更新而改变。
- Python 发行名为 `tacgate-dp`，导入名仍为 `tactile_gate_policy`。
  离线 wheel 安装到独立临时目录后，默认模型构造通过；三个 `tacgate-*`
  命令的 `--help` 均可运行，三个原 `tgp-*` 入口也可加载。
- 离线源码包构建通过；中英文 README、方法与相关工作文档、配置、
  smoke 脚本及 PDF/SVG/PNG/LaTeX 论文图均被收录。
  wheel 保留两份 DP 第三方许可证。
- 六份 Markdown 文档的 44 个本地链接全部有效；发布文档没有旧目录引用。
- 重新生成论文图并渲染 PDF 进行目视检查。PDF 为单页 180 × 140 mm
  矢量图，五种字体全部嵌入，没有内嵌位图；配套英文图注与当前默认配置一致。

验证生成的安装目录、数据、构建产物和渲染预览均已清理，不随发布目录保留。

## 验证范围

测试数据是程序生成的工程 fixture，未验证实际操作任务成功率或真实 FR3
硬件控制。GitHub Actions 已配置，远程 CI 尚未执行。Zarr 3 与 CUDA 路径
尚未在本机运行验证。数据和 checkpoint 不随仓库发布。
