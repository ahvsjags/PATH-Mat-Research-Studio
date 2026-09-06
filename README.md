# PATH-Mat Research Studio

公开网页演示：材料路线分析、协作流程展示、模型指导实验筛选和实验结果记录。

## 在线体验

GitHub Pages：<https://ahvsjags.github.io/PATH-Mat-Research-Studio/>

公开体验期限：2027-03-06。到期后网页会停止计算，源码仍保留在本仓库中。

## 网页版包含的功能

- 路线资料输入和结果总览
- 可拖动旋转的材料路线视图
- 12个协作模块的工作关系和说明
- XRD、Raman、SEM 的下一步检测建议
- 温度 × 时长 × 气氛候选组合筛选
- 第一轮实验条件推荐和设备时间估算
- 实验记录填写、CSV/JSON下载
- 亮色/暗色模式和手机端布局

网页计算在浏览器中完成，模型指导实验页的分数用于同一批候选条件排序，实验结论以实际测量为准。

## 完整 Python 版本

仓库根目录保留了 Gradio/Python 版本、EC-PathWM模型文件、12个模块的源码和测试文件，可按 `README` 与 `requirements.txt` 在本地启动。
