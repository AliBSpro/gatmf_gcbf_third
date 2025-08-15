# 严格对齐版：GAT-MF × GCBF+（PyTorch）

与原 GCBF+ 论文/代码范式对齐的关键点：
- **QP 教师（式17）**：使用 ProxQP（优先）或 OSQP，**不提供近似/投影回退**。`safety/qp.py`
- **严格掩码（定义1/IV-A）**：半径外邻居数值与梯度均被屏蔽。`safety/encoders.py`
- **目标网络 + 停梯度（IV-B）**：h̄、π̄ 软更新；在 CBF 导数项与 QP 教师中均使用 h̄。`safety/cbf_net.py`、`trainer.py`
- **有限时域标注（IV-C）**：滚动 T 步生成 D_C/D_A；训练时参与集合项。`safety/labeler.py`
- **CBF 铰链（式20–21）**：导数项 + D_C/D_A 集合项；差分用环境真实 Δt。`safety/losses.py`
- **控制蒸馏（式22）**：πϕ 与 RL Actor 均蒸馏到 π_QP(h̄)。`trainer.py`

## 运行
```bash
pip install -r requirements.txt
python trainer.py        # 训练（默认不启用执行期护盾）
# 若需在线护盾探索，在 main 或调用处把 shield=True
```
