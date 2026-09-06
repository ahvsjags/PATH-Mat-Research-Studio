from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import gradio as gr
import qrcode

from backend import PRESETS, analyze, health_check, load_preset, screen_experiments


APP_ROOT = Path(__file__).resolve().parent
ASSET_DIR = APP_ROOT / "assets"
ASSET_DIR.mkdir(exist_ok=True)


CSS = """
:root {
  --pm-ink: #0b1f33;
  --pm-navy: #0b2d4f;
  --pm-blue: #1769aa;
  --pm-sky: #eaf5fc;
  --pm-red: #c0392b;
  --pm-green: #18815f;
  --pm-paper: #f7fafc;
}
.gradio-container { width:100% !important; max-width: 1540px !important; min-width:0 !important; margin: 0 auto !important; color: var(--pm-ink); }
.pm-shell { background: linear-gradient(135deg,#071b2e 0%,#0b3358 62%,#1769aa 100%); border-radius: 18px; padding: 24px 28px; color: white; margin: 8px 0 16px; box-shadow: 0 18px 50px rgba(7,27,46,.18); }
.pm-shell { position:relative; overflow:hidden; isolation:isolate; }
.pm-shell:before { content:""; position:absolute; width:420px; height:420px; border-radius:50%; right:-120px; top:-245px; background:radial-gradient(circle,rgba(99,230,255,.38),rgba(99,230,255,0) 68%); z-index:-1; animation:pm-drift 7s ease-in-out infinite alternate; }
@keyframes pm-drift { from { transform:translate3d(-18px,8px,0) scale(.94); } to { transform:translate3d(15px,-4px,0) scale(1.08); } }
.pm-shell h1 { margin: 0; font-size: 31px; letter-spacing: -.02em; color:#ffffff !important; }
.pm-shell p { margin: 7px 0 0; color: #d9efff; font-size: 14px; }
.pm-eyebrow { font-size:11px; letter-spacing:.18em; text-transform:uppercase; color:#79e8f2; font-weight:800; margin-bottom:7px; }
.pm-stat-grid { display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:10px; margin-top:19px; max-width:760px; }
.pm-stat { padding:10px 12px; border-radius:12px; background:rgba(4,17,30,.28); border:1px solid rgba(183,236,255,.18); backdrop-filter:blur(8px); }
.pm-stat strong { display:block; color:#fff; font-size:18px; line-height:1; }
.pm-stat span { display:block; color:#b9d7e8; font-size:10px; margin-top:5px; letter-spacing:.03em; }
.pm-shell .pm-live { display:inline-flex; align-items:center; gap:7px; background:rgba(255,255,255,.13); border:1px solid rgba(255,255,255,.24); padding:5px 10px; border-radius:999px; font-size:12px; float:right; }
.pm-shell .pm-dot { width:8px; height:8px; border-radius:99px; background:#58e1ad; box-shadow:0 0 0 4px rgba(88,225,173,.15); }
.pm-panel { border: 1px solid #d9e5ef !important; border-radius: 16px !important; background: white !important; box-shadow: 0 10px 30px rgba(30,65,95,.06); }
.pm-pagehead { border-radius:16px; padding:18px 20px; margin:4px 0 14px; color:#eaf7ff; background:linear-gradient(125deg,#071421,#0d2d43 65%,#124866); border:1px solid #1e526f; }
.pm-pagehead .pm-kicker { color:#62d8df; text-transform:uppercase; letter-spacing:.16em; font-size:10px; font-weight:800; }
.pm-pagehead h2 { color:white !important; margin:5px 0 4px; font-size:23px; }
.pm-pagehead p { color:#a9c5d6; margin:0; font-size:13px; }
.pm-visual-note { border:1px solid #d9e5ef; border-radius:14px; padding:14px 16px; background:linear-gradient(145deg,#f8fbfd,#eef7fb); color:#355269; }
.pm-visual-note strong { color:#0a3454; }
.pm-plot { border-radius:16px !important; overflow:hidden !important; border:1px solid #183850 !important; box-shadow:0 16px 46px rgba(7,31,48,.18); }
#pm-pages > .tab-nav { position:sticky; top:0; z-index:20; background:rgba(247,250,252,.94); backdrop-filter:blur(12px); border-radius:13px; padding:5px; }
#pm-pages > .tab-nav button { min-height:45px; font-weight:700; }
.pm-status { border-radius: 15px; padding: 17px 19px; border-left: 5px solid; min-height: 108px; }
.pm-status-ok { background:#edf9f4; border-color:var(--pm-green); }
.pm-status-warn { background:#fff8e8; border-color:#d68a00; }
.pm-status-bad { background:#fff0ef; border-color:var(--pm-red); }
.pm-status-kicker { text-transform:uppercase; letter-spacing:.12em; font-size:11px; font-weight:700; opacity:.62; }
.pm-status-title { font-size:20px; font-weight:800; margin:4px 0 2px; }
.pm-status-detail { font-size:13px; opacity:.78; }
.pm-gate-grid { display:grid; grid-template-columns:repeat(5,minmax(0,1fr)); gap:10px; }
.pm-gate { border:1px solid #dfe8ef; border-radius:13px; padding:12px; background:#fff; min-height:120px; }
.pm-gate h4 { margin:8px 0 4px; font-size:13px; }
.pm-gate p { margin:0; font-size:11px; line-height:1.45; color:#52677a; overflow-wrap:anywhere; }
.pm-badge { font-size:10px; font-weight:800; padding:3px 7px; border-radius:999px; }
.pm-ok { background:#d9f4e9; color:#10684b; }
.pm-warn { background:#fff1c9; color:#8b5c00; }
.pm-bad { background:#ffe0dd; color:#a22b20; }
.pm-note { border-left:3px solid #79b7df; padding:9px 12px; background:#eef8ff; border-radius:8px; font-size:12px; color:#38556e; }
.pm-process-grid { display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:10px; margin:0 0 16px; }
.pm-process-step { position:relative; min-height:124px; padding:15px 15px 14px; border:1px solid #d5e5ee; border-radius:15px; background:linear-gradient(145deg,#ffffff,#f0f8fb); box-shadow:0 8px 24px rgba(25,67,92,.06); }
.pm-process-step:after { content:"→"; position:absolute; right:-10px; top:43px; z-index:2; width:20px; height:20px; border-radius:50%; background:#0f4b6c; color:white; text-align:center; line-height:19px; font-weight:900; }
.pm-process-step:last-child:after { display:none; }
.pm-process-step .pm-step-number { display:inline-flex; width:27px; height:27px; align-items:center; justify-content:center; border-radius:9px; background:#0c3958; color:#74ebdf; font-weight:900; font-size:12px; }
.pm-process-step h3 { margin:10px 0 5px; color:#0b2d4f; font-size:15px; }
.pm-process-step p { margin:0; color:#526c7d; font-size:12px; line-height:1.55; }
.pm-panel, .pm-status, .pm-gate, .pm-note, .pm-visual-note, .pm-screen-input, .pm-screen-output { color:#0b1f33 !important; }
.pm-gate h4, .pm-status-title { color:#0b1f33 !important; }
.pm-screen-input { color:#0b1f33 !important; border:1px solid #d9e5ef !important; border-radius:16px !important; background:linear-gradient(145deg,#fff,#f7fbfd) !important; }
.pm-screen-input .form, .pm-screen-input .block, .pm-screen-input fieldset { background:#ffffff !important; color:#0b1f33 !important; }
.pm-screen-input .prose h3, .pm-screen-input .prose p, .pm-screen-input label, .pm-screen-input span, .pm-screen-input .block-title, .pm-screen-input input { color:#0b1f33 !important; opacity:1 !important; }
.pm-screen-input .prose h3 { background:#ffffff !important; border-radius:10px !important; padding:8px 10px !important; }
.pm-screen-input fieldset label { background:#f6fafc !important; border-color:#d7e3ec !important; }
.pm-screen-input fieldset label.selected { background:#e5f5ef !important; border-color:#8bcdb7 !important; }
.pm-screen-input input { background:#ffffff !important; }
.pm-screen-input .file-preview .file, .pm-screen-input .file-preview .filename { background:#f6fafc !important; color:#0b1f33 !important; }
.pm-screen-input .file-preview .filename span { color:#0b1f33 !important; }
.pm-screen-input .download-link { color:#1769aa !important; }
.pm-screen-input [data-testid="block-label"] { background:#ffffff !important; color:#0b1f33 !important; }
.pm-screen-output { color:#0b1f33 !important; border:1px solid #b9dcd2 !important; border-radius:16px !important; background:linear-gradient(145deg,#ffffff,#e9f7f2) !important; }
.pm-screen-output .prose, .pm-screen-output .prose h1, .pm-screen-output .prose h2, .pm-screen-output .prose h3, .pm-screen-output .prose h4, .pm-screen-output .prose p, .pm-screen-output .prose li, .pm-screen-output .prose ol, .pm-screen-output .prose ul { color:#0b1f33 !important; opacity:1 !important; }
.pm-screen-output .prose strong { color:#073b50 !important; }
.pm-screen-output .prose blockquote { color:#123b48 !important; background:#ddf2ea !important; border-left:4px solid #18815f !important; border-radius:0 10px 10px 0; padding:10px 13px; }
.pm-screen-output .prose blockquote p, .pm-screen-output .prose blockquote strong { color:#123b48 !important; }
.pm-footer { text-align:center; color:#657a8e; font-size:11px; margin:14px 0 8px; }
button.primary { background:linear-gradient(135deg,#0b365d,#1769aa) !important; border:none !important; }
@media (max-width: 900px) {
  html, body, .gradio-container, main.app, main.app > .contain, main.app .wrap {
    width:100% !important; min-width:0 !important; max-width:100vw !important;
  }
  main.app .row { flex-wrap:wrap !important; min-width:0 !important; max-width:100% !important; }
  main.app .row > .column { flex:1 1 100% !important; min-width:0 !important; max-width:100% !important; }
  .tabs, .tab-wrapper { min-width:0 !important; max-width:100% !important; }
  .tab-container { max-width:calc(100vw - 64px) !important; overflow-x:auto !important; }
  .pm-gate-grid { grid-template-columns:1fr 1fr; }
  .pm-process-grid { grid-template-columns:1fr; }
  .pm-process-step:after { content:"↓"; right:calc(50% - 10px); top:auto; bottom:-11px; }
  .pm-shell h1 { font-size:25px; }
  .pm-shell .pm-live { float:none; margin-bottom:10px; }
  .pm-stat-grid { grid-template-columns:1fr 1fr; }
  .pm-pagehead h2 { font-size:20px; }
  #pm-pages > .tab-nav { position:relative; overflow-x:auto; flex-wrap:nowrap; }
}
"""


def build_app() -> gr.Blocks:
    with gr.Blocks(title="PATH-Mat Research Console", css=CSS, fill_width=True) as demo:
        gr.HTML(
            """
            <div class="pm-shell">
              <span class="pm-live"><span class="pm-dot"></span> 模型已连接 · 仅分析资料</span>
              <div class="pm-eyebrow">材料路线分析与实验建议</div>
              <h1>PATH-Mat Research Studio</h1>
              <p>输入一条材料制备路线，系统查找相关资料、比较可能结果、检查信息是否充分，并给出下一步建议</p>
              <div class="pm-stat-grid">
                <div class="pm-stat"><strong>12</strong><span>分工协作模块</span></div>
                <div class="pm-stat"><strong>10</strong><span>可查看的处理步骤</span></div>
                <div class="pm-stat"><strong>5</strong><span>结果检查项目</span></div>
                <div class="pm-stat"><strong>2×3D</strong><span>可旋转交互视图</span></div>
              </div>
            </div>
            """
        )

        with gr.Tabs(elem_id="pm-pages"):
            with gr.Tab("01 · 结果总览"):
                with gr.Row(equal_height=False):
                    with gr.Column(scale=4, min_width=320, elem_classes=["pm-panel"]):
                        gr.Markdown("### 输入研究路线")
                        preset = gr.Dropdown(
                            choices=[
                                ("SrFe₁₂O₁₉｜完整示例", "SrFe₁₂O₁₉｜覆盖域内功能演示"),
                                ("SrFe₁₂O₁₉｜资料较少示例", "SrFe₁₂O₁₉｜历史资料不足示例"),
                                ("ZnO｜新资料来源示例", "ZnO｜外部来源与缺模态挑战"),
                                ("填写自己的路线", "自定义路线"),
                            ],
                            value=list(PRESETS)[0], label="快速体验案例"
                        )
                        preset_note = gr.Markdown(PRESETS[list(PRESETS)[0]]["note"], elem_classes=["pm-note"])
                        question = gr.Textbox(label="研究问题", lines=3, value=PRESETS[list(PRESETS)[0]]["question"])
                        with gr.Row():
                            target = gr.Textbox(label="目标化学式", value=PRESETS[list(PRESETS)[0]]["target"])
                            family = gr.Dropdown(
                                choices=[
                                    ("氧化物", "oxide"),
                                    ("其他无机材料", "other_inorganic"),
                                    ("磷酸盐或磷化物", "phosphate_or_phosphide"),
                                    ("自定义", "custom"),
                                ],
                                value=PRESETS[list(PRESETS)[0]]["family"], label="材料类别", allow_custom_value=True,
                            )
                        precursors = gr.Textbox(label="前驱体（逗号分隔）", value=PRESETS[list(PRESETS)[0]]["precursors"])
                        source_id = gr.Textbox(label="资料来源编号", value=PRESETS[list(PRESETS)[0]]["source_id"])
                        actions = gr.Textbox(label="工艺步骤（按顺序，用逗号分隔）", value=PRESETS[list(PRESETS)[0]]["actions"])
                        with gr.Row():
                            temperature = gr.Number(label="温度 °C", value=PRESETS[list(PRESETS)[0]]["temperature"], precision=1)
                            duration = gr.Number(label="时长 h", value=PRESETS[list(PRESETS)[0]]["duration"], precision=1)
                            atmosphere = gr.Dropdown(
                                choices=[("空气", "air"), ("惰性气氛", "inert"), ("真空", "vacuum"), ("氧气 O₂", "O2"), ("氩气 Ar", "Ar"), ("氮气 N₂", "N2")],
                                value=PRESETS[list(PRESETS)[0]]["atmosphere"], label="气氛", allow_custom_value=True,
                            )
                        modalities = gr.CheckboxGroup(
                            choices=["XRD", "Raman", "SEM", "PL", "XPS"],
                            value=PRESETS[list(PRESETS)[0]]["modalities"], label="已有表征",
                        )
                        with gr.Accordion("资料与测试选项", open=False):
                            license_status = gr.Dropdown(
                                choices=[
                                    ("来源允许使用", "source-licensed"),
                                    ("CC BY 4.0", "CC BY 4.0"),
                                    ("需要查看来源许可", "source_license_required"),
                                    ("尚不清楚", "unknown"),
                                    ("商业使用受限", "commercial-restricted"),
                                ],
                                value=PRESETS[list(PRESETS)[0]]["license_status"], label="资料使用条件",
                            )
                            evidence_linked = gr.Checkbox(value=True, label="资料来源可以追踪")
                            protocol_complete = gr.Checkbox(value=True, label="输入内容已经逐项检查")
                            device_allows = gr.Checkbox(value=True, label="设备支持当前气氛")
                            challenge = gr.Radio(
                                choices=[
                                    ("正常分析", "无"),
                                    ("模拟：没有资料来源", "移除来源"),
                                    ("模拟：缺少关键检测", "缺少关键表征"),
                                    ("模拟：请求控制设备", "请求物理执行"),
                                    ("模拟：异常输入", "对抗输入"),
                                ],
                                value="无", label="异常情况演示",
                                info="可以演示资料缺失、关键检测缺失或设备操作请求出现时，系统会怎样回应。",
                            )
                            uploaded_file = gr.File(
                                label="上传自己的 JSON / CSV（可选，读取第一条记录）",
                                file_types=[".json", ".csv"], type="filepath",
                            )
                        run_button = gr.Button("开始分析", variant="primary", size="lg")

                    with gr.Column(scale=7, min_width=520):
                        gr.HTML('<div class="pm-pagehead"><div class="pm-kicker">结果总览</div><h2>当前路线更可能得到什么结果</h2><p>这里依次显示路线判断、判断依据、缺少的数据和下一步建议。</p></div>')
                        status_output = gr.HTML(
                            '<div class="pm-status pm-status-warn"><div class="pm-status-kicker">准备就绪</div><div class="pm-status-title">选择案例并启动分析</div><div class="pm-status-detail">所有功能页会同时更新。</div></div>'
                        )
                        decision_output = gr.Markdown("运行后显示路线判断、证据缺口和下一步动作。")

            with gr.Tab("02 · 3D 路线图"):
                gr.HTML('<div class="pm-pagehead"><div class="pm-kicker">3D 路线图</div><h2>观察当前路线在模型判断中的位置</h2><p>旋转图形，查看目标产物倾向、资料覆盖程度和综合比较分数之间的关系。</p></div>')
                route_plot = gr.Plot(show_label=False, elem_classes=["pm-plot"])
                gr.HTML('<div class="pm-visual-note"><strong>读图方法：</strong>X轴越高，模型越偏向目标产物；Y轴越高，可供参考的相近资料越少；Z轴用于比较候选路线。白色节点表示当前输入。</div>')

            with gr.Tab("03 · 协作流程"):
                gr.HTML('<div class="pm-pagehead"><div class="pm-kicker">协作流程</div><h2>查看12个模块如何共同完成一次分析</h2><p>旋转网络并悬停节点，可以看到每个部分负责什么、运行是否完成以及发现了什么。</p></div>')
                agent_plot = gr.Plot(show_label=False, elem_classes=["pm-plot"])
                gr.Markdown("### 各模块运行结果")
                trace_output = gr.Dataframe(
                    headers=["模块", "状态", "本次完成的工作"],
                    datatype=["str"] * 3, interactive=False, wrap=True,
                )

            with gr.Tab("04 · 下一步实验"):
                gr.HTML('<div class="pm-pagehead"><div class="pm-kicker">下一步实验</div><h2>用哪些数据区分目标产物和其他可能产物</h2><p>系统列出三种可能结果，并给出XRD、Raman、SEM和单变量对照实验的建议顺序。</p></div>')
                hypothesis_output = gr.Markdown("运行后生成分阶段验证路径和判定标准。")

            with gr.Tab("05 · 模型指导实验"):
                gr.HTML(
                    '<div class="pm-pagehead"><div class="pm-kicker">模型指导实验</div>'
                    '<h2>先让模型筛选条件，再安排第一轮实验</h2>'
                    '<p>输入候选温度、时长和气氛，系统逐组运行模型，选择第一轮条件，并计算可以暂缓多少实验和设备时间。</p></div>'
                )
                gr.HTML(
                    """
                    <div class="pm-process-grid">
                      <div class="pm-process-step"><span class="pm-step-number">1</span><h3>生成候选组合</h3><p>把温度、时长和气氛展开成完整的实验条件表。</p></div>
                      <div class="pm-process-step"><span class="pm-step-number">2</span><h3>逐组运行模型</h3><p>EC-PathWM 读取每组温度和时长，给出模型排序分。</p></div>
                      <div class="pm-process-step"><span class="pm-step-number">3</span><h3>安排第一轮</h3><p>优先选择模型分数较高或彼此差异较大的条件，暂缓相近组合。</p></div>
                      <div class="pm-process-step"><span class="pm-step-number">4</span><h3>带回实验结果</h3><p>把 XRD、Raman 或 SEM 结果补回系统，再决定第二轮条件。</p></div>
                    </div>
                    """
                )
                with gr.Row(equal_height=False):
                    with gr.Column(scale=4, min_width=320, elem_classes=["pm-screen-input"]):
                        gr.Markdown("### 设置候选实验范围")
                        screening_temperatures = gr.Textbox(
                            value="950, 1100, 1250",
                            label="候选温度 °C（用逗号分隔）",
                        )
                        screening_durations = gr.Textbox(
                            value="4, 8, 12",
                            label="候选时长 h（用逗号分隔）",
                        )
                        screening_atmospheres = gr.CheckboxGroup(
                            choices=[
                                ("空气", "air"),
                                ("惰性气氛", "inert"),
                                ("真空", "vacuum"),
                                ("氧气 O₂", "O2"),
                                ("氩气 Ar", "Ar"),
                                ("氮气 N₂", "N2"),
                            ],
                            value=["air", "inert"],
                            label="候选气氛",
                        )
                        with gr.Row():
                            screening_repeats = gr.Slider(
                                minimum=1, maximum=5, value=2, step=1,
                                label="每个条件重复次数",
                            )
                            screening_count = gr.Slider(
                                minimum=1, maximum=12, value=4, step=1,
                                label="第一轮先做几组",
                            )
                        screening_characterization = gr.Number(
                            value=2, minimum=0, maximum=100, precision=1,
                            label="每次表征预计占用时间 h",
                        )
                        screening_button = gr.Button(
                            "生成模型指导实验计划", variant="primary", size="lg"
                        )
                        screening_download = gr.File(
                            label="下载实验安排与结果记录表（CSV）",
                            interactive=False,
                        )
                    with gr.Column(scale=7, min_width=520, elem_classes=["pm-screen-output"]):
                        screening_summary = gr.Markdown(
                            "### 等待生成计划\n\n填写候选条件后点击按钮。这里会显示第一轮建议、计算过程和预计暂缓的实验时间。"
                        )
                screening_plot = gr.Plot(show_label=False, elem_classes=["pm-plot"])
                gr.Markdown("### 每一组条件怎么安排")
                screening_table = gr.Dataframe(
                    headers=[
                        "顺序", "温度 °C", "时长 h", "气氛",
                        "模型排序分", "在18组中的位置", "第一轮安排", "为什么这样排",
                    ],
                    datatype=["number", "str", "str", "str", "str", "str", "str", "str"],
                    interactive=False,
                    wrap=True,
                )
                gr.HTML(
                    '<div class="pm-visual-note"><strong>怎样现场演示：</strong>先用默认的18组条件生成计划；'
                    '再把“第一轮先做几组”改为2或6，或增加一个气氛，重新生成。实验次数、设备时间和推荐组合会立即变化。</div>'
                )

            with gr.Tab("06 · 结果检查与记录"):
                gr.HTML('<div class="pm-pagehead"><div class="pm-kicker">结果检查</div><h2>查看资料、模型、设备条件和输入内容是否完整</h2><p>每一项都会显示通过状态和具体原因，方便找到需要补充的内容。</p></div>')
                gate_output = gr.HTML('<div class="pm-gate-grid"></div>')
                with gr.Tabs():
                    with gr.Tab("模型详细数据"):
                        raw_output = gr.JSON(label="模型输出与检查结果")
                    with gr.Tab("完整运行记录"):
                        events_output = gr.JSON(label="按顺序保存的处理记录")

            with gr.Tab("07 · 下载完整记录"):
                gr.HTML('<div class="pm-pagehead"><div class="pm-kicker">下载完整记录</div><h2>保存本次输入、结果、各模块状态和处理过程</h2><p>下载文件可以用于复查本次分析，也可以与之后的新实验结果进行比较。</p></div>')
                report_output = gr.File(label="下载本次分析记录（Markdown）")
                gr.Markdown("文件包含主结论、模型详细数据、各模块运行结果和完整处理记录。")

        gr.HTML(
            '<div class="pm-footer">PATH-Mat 根据现有资料生成路线比较和实验建议；网页不连接实验设备。</div>'
        )

        preset_outputs = [
            question, target, precursors, source_id, family, actions, temperature, duration,
            atmosphere, modalities, license_status, evidence_linked, protocol_complete,
            device_allows, challenge, preset_note,
        ]
        preset.change(load_preset, inputs=preset, outputs=preset_outputs, show_progress="hidden")

        analysis_inputs = [
            question, target, precursors, source_id, family, actions, temperature, duration,
            atmosphere, modalities, license_status, evidence_linked, protocol_complete,
            device_allows, challenge, uploaded_file,
        ]
        analysis_outputs = [
            status_output, gate_output, decision_output, hypothesis_output, trace_output,
            raw_output, events_output, report_output, route_plot, agent_plot,
        ]
        run_button.click(
            analyze, inputs=analysis_inputs, outputs=analysis_outputs,
            api_name="analyze_route", api_description="Run source-bound PATH-Mat route analysis.",
        )
        screening_button.click(
            screen_experiments,
            inputs=[
                target, precursors, source_id, family, actions, modalities, evidence_linked,
                screening_temperatures, screening_durations, screening_atmospheres,
                screening_repeats, screening_count, screening_characterization,
            ],
            outputs=[screening_summary, screening_table, screening_plot, screening_download],
            api_name="screen_experiments",
            api_description="Rank a candidate condition grid and estimate first-round experiment workload.",
        )

    return demo


def _write_qr(url: str) -> Path:
    target = ASSET_DIR / "pathmat_console_qr.png"
    image = qrcode.make(url)
    image.save(target)
    (ASSET_DIR / "share_url.txt").write_text(url, encoding="utf-8")
    return target


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--share", action="store_true", help="Create a temporary public Gradio share URL.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", default=7860, type=int)
    parser.add_argument("--no-block", action="store_true")
    args = parser.parse_args()

    print("HEALTH", json.dumps(health_check(), ensure_ascii=False), flush=True)
    demo = build_app()
    launched = demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        prevent_thread_lock=args.no_block,
        show_error=True,
        allowed_paths=[str(APP_ROOT / "reports")],
    )
    try:
        _, local_url, share_url = launched
    except (TypeError, ValueError):
        local_url, share_url = f"http://127.0.0.1:{args.port}", None
    public_url = share_url or local_url
    qr_path = _write_qr(public_url)
    print(f"LOCAL_URL={local_url}", flush=True)
    print(f"PUBLIC_URL={share_url or ''}", flush=True)
    print(f"QR_PATH={qr_path}", flush=True)
    if args.no_block:
        while True:
            time.sleep(3600)


if __name__ == "__main__":
    main()
