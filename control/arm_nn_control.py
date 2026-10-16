"""
Topic 1: Neural Network-Based Control Systems
Нейросетевое управление 2-звенным манипулятором (плоскость, без гравитации).

Что делает скрипт:
  1. Симулирует динамику руки  M(q) q'' + C(q,q') q' = tau  (RK4).
  2. Три контроллера:
       - PID                  (классический, без модели)
       - CTC                  (computed torque, использует номинальную модель; "учитель")
       - NN                   (MLP, обучена повторять CTC по данным -> imitation learning)
  3. Собирает данные на случайных траекториях, обучает MLP (PyTorch).
  4. Сравнивает PID / NN / CTC в сценариях:
       nominal, payload (+0.5 кг на конце), sensor noise, payload+noise,
       unseen (быстрая траектория вне обучающего диапазона),
       и point-to-point (перерегулирование, время установления).
  5. Сохраняет графики (PNG) и таблицу метрик (CSV).

Запуск:  python arm_nn_control.py
Зависимости: numpy, matplotlib, torch
"""
import csv
import os
import sys
from dataclasses import dataclass, replace

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

OUT = os.path.dirname(os.path.abspath(__file__))
SEED = 0
rng_global = np.random.default_rng(SEED)
torch.manual_seed(SEED)

DT = 0.002          # шаг симуляции, с
TAU_MAX = 30.0      # ограничение момента, Н·м (одинаково для всех контроллеров)


# --------------------------------------------------------------------------
# 1. Модель манипулятора
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Arm:
    m1: float = 1.0
    m2: float = 1.0
    l1: float = 0.5
    l2: float = 0.5
    lc1: float = 0.25
    lc2: float = 0.25
    I1: float = 0.02
    I2: float = 0.02

    def with_payload(self, mp):
        """Точечная масса mp на конце второго звена -> пересчёт m2, lc2, I2."""
        m = self.m2 + mp
        lc = (self.m2 * self.lc2 + mp * self.l2) / m
        I = self.I2 + self.m2 * (lc - self.lc2) ** 2 + mp * (self.l2 - lc) ** 2
        return replace(self, m2=m, lc2=lc, I2=I)

    def MC(self, q, dq):
        c2, s2 = np.cos(q[1]), np.sin(q[1])
        a = self.m2 * self.l1 * self.lc2
        M11 = self.m1 * self.lc1**2 + self.I1 + self.m2 * (self.l1**2 + self.lc2**2) + self.I2 + 2 * a * c2
        M12 = self.m2 * self.lc2**2 + self.I2 + a * c2
        M22 = self.m2 * self.lc2**2 + self.I2
        h = -a * s2
        M = np.array([[M11, M12], [M12, M22]])
        C = np.array([[h * dq[1], h * (dq[0] + dq[1])], [-h * dq[0], 0.0]])
        return M, C

    def f(self, x, tau):
        q, dq = x[:2], x[2:]
        M, C = self.MC(q, dq)
        # решение 2x2 системы аналитически (быстрее np.linalg.solve)
        b = tau - C @ dq
        det = M[0, 0] * M[1, 1] - M[0, 1] ** 2
        ddq = np.array([(M[1, 1] * b[0] - M[0, 1] * b[1]) / det,
                        (-M[0, 1] * b[0] + M[0, 0] * b[1]) / det])
        return np.r_[dq, ddq]

    def step(self, x, tau, dt=DT):
        k1 = self.f(x, tau)
        k2 = self.f(x + dt / 2 * k1, tau)
        k3 = self.f(x + dt / 2 * k2, tau)
        k4 = self.f(x + dt * k3, tau)
        return x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


NOMINAL = Arm()


# --------------------------------------------------------------------------
# 2. Референсные траектории: (t) -> (qr, dqr, ddqr)
# --------------------------------------------------------------------------
def sines(c, A, w, ph):
    c, A, w, ph = map(np.asarray, (c, A, w, ph))

    def ref(t):
        return (c + A * np.sin(w * t + ph),
                A * w * np.cos(w * t + ph),
                -A * w**2 * np.sin(w * t + ph))
    return ref


def min_jerk(q0, q1, T=1.0):
    q0, q1 = np.asarray(q0, float), np.asarray(q1, float)

    def ref(t):
        s = np.clip(t / T, 0, 1)
        p = 10 * s**3 - 15 * s**4 + 6 * s**5
        dp = (30 * s**2 - 60 * s**3 + 30 * s**4) / T
        ddp = (60 * s - 180 * s**2 + 120 * s**3) / T**2
        if t >= T:
            dp = ddp = 0.0
        d = q1 - q0
        return q0 + d * p, d * dp, d * ddp
    return ref


def random_traj(rng):
    return sines(c=rng.uniform(-0.5, 0.5, 2),
                 A=rng.uniform(0.2, 0.8, 2),
                 w=rng.uniform(0.5, 2.5, 2),
                 ph=rng.uniform(0, 2 * np.pi, 2))


# --------------------------------------------------------------------------
# 3. Контроллеры. Интерфейс: ctrl(t, q_meas, dq_meas, ref) -> tau
# --------------------------------------------------------------------------
class PID:
    name = "PID"

    def __init__(self, Kp=(150., 60.), Ki=(30., 10.), Kd=(25., 8.), dt=DT):
        self.Kp, self.Ki, self.Kd = map(np.array, (Kp, Ki, Kd))
        self.dt = dt
        self.ei = np.zeros(2)

    def reset(self):
        self.ei = np.zeros(2)

    def __call__(self, t, q, dq, ref):
        qr, dqr, _ = ref(t)
        e, de = qr - q, dqr - dq
        self.ei = np.clip(self.ei + e * self.dt, -1.0, 1.0)     # anti-windup
        return self.Kp * e + self.Ki * self.ei + self.Kd * de


class CTC:
    """Computed torque: tau = M(q)(ddqr + Kd de + Kp e) + C dq. Модель — НОМИНАЛЬНАЯ."""
    name = "CTC"

    def __init__(self, arm=NOMINAL, Kp=400., Kd=40.):
        self.arm, self.Kp, self.Kd = arm, Kp, Kd

    def reset(self):
        pass

    def __call__(self, t, q, dq, ref):
        qr, dqr, ddqr = ref(t)
        M, C = self.arm.MC(q, dq)
        return M @ (ddqr + self.Kd * (dqr - dq) + self.Kp * (qr - q)) + C @ dq


def features(q, dq, ref_t):
    qr, dqr, ddqr = ref_t
    return np.r_[q, dq, qr - q, dqr - dq, ddqr]          # 10 входов


class MLP(nn.Module):
    def __init__(self, n_in=10, n_out=2, h=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_in, h), nn.Tanh(),
                                 nn.Linear(h, h), nn.Tanh(),
                                 nn.Linear(h, n_out))

    def forward(self, x):
        return self.net(x)


class NNController:
    """Нейроконтроллер. Прямой проход на numpy (в петле симуляции быстрее, чем torch)."""
    name = "NN"

    def __init__(self, model, x_mu, x_sd, y_mu, y_sd):
        sd = model.state_dict()
        self.W = [sd[f"net.{i}.weight"].double().numpy() for i in (0, 2, 4)]
        self.b = [sd[f"net.{i}.bias"].double().numpy() for i in (0, 2, 4)]
        self.x_mu, self.x_sd, self.y_mu, self.y_sd = x_mu, x_sd, y_mu, y_sd

    def reset(self):
        pass

    def __call__(self, t, q, dq, ref):
        z = (features(q, dq, ref(t)) - self.x_mu) / self.x_sd
        z = np.tanh(self.W[0] @ z + self.b[0])
        z = np.tanh(self.W[1] @ z + self.b[1])
        y = self.W[2] @ z + self.b[2]
        return y * self.y_sd + self.y_mu

    def save(self, path):
        """Сохранить веса и нормировку (для arm_live.py)."""
        np.savez(path, W0=self.W[0], W1=self.W[1], W2=self.W[2],
                 b0=self.b[0], b1=self.b[1], b2=self.b[2],
                 x_mu=self.x_mu, x_sd=self.x_sd, y_mu=self.y_mu, y_sd=self.y_sd)

    @classmethod
    def from_npz(cls, path):
        d = np.load(path)
        o = cls.__new__(cls)
        o.W = [d["W0"], d["W1"], d["W2"]]
        o.b = [d["b0"], d["b1"], d["b2"]]
        o.x_mu, o.x_sd, o.y_mu, o.y_sd = d["x_mu"], d["x_sd"], d["y_mu"], d["y_sd"]
        return o


# --------------------------------------------------------------------------
# 4. Замкнутый контур
# --------------------------------------------------------------------------
def simulate(ctrl, ref, arm=NOMINAL, T=5.0, x0=None, q_noise=0.0, dq_noise=0.0,
             act_noise=0.0, label_ctrl=None, seed=1, record_data=False):
    """Симуляция. Датчики шумят (q_noise, dq_noise), act_noise — шум в исполнительном
    механизме (для сбора данных). label_ctrl — контроллер, чьи моменты пишем как метки."""
    rng = np.random.default_rng(seed)
    n = int(round(T / DT))
    qr0, dqr0, _ = ref(0.0)
    x = np.r_[qr0, dqr0] if x0 is None else np.array(x0, float)
    ctrl.reset()
    if label_ctrl is not None:
        label_ctrl.reset()
    t_log, x_log, qr_log, tau_log = [], [], [], []
    X, Y = [], []
    for k in range(n):
        t = k * DT
        qm = x[:2] + rng.normal(0, q_noise, 2) if q_noise else x[:2]
        dqm = x[2:] + rng.normal(0, dq_noise, 2) if dq_noise else x[2:]
        tau = np.clip(ctrl(t, qm, dqm, ref), -TAU_MAX, TAU_MAX)
        if record_data and k % 5 == 0:
            X.append(features(qm, dqm, ref(t)))
            Y.append(np.clip(label_ctrl(t, qm, dqm, ref), -TAU_MAX, TAU_MAX))
        tau_applied = tau + rng.normal(0, act_noise, 2) if act_noise else tau
        t_log.append(t); x_log.append(x.copy()); qr_log.append(ref(t)[0]); tau_log.append(tau)
        x = arm.step(x, np.clip(tau_applied, -TAU_MAX, TAU_MAX))
    out = dict(t=np.array(t_log), x=np.array(x_log), qr=np.array(qr_log), tau=np.array(tau_log))
    if record_data:
        out["X"], out["Y"] = np.array(X), np.array(Y)
    return out


# --------------------------------------------------------------------------
# 5. Сбор данных и обучение
# --------------------------------------------------------------------------
def collect_data(n_traj=150, T=4.0):
    print(f"[data] собираю {n_traj} траекторий по {T} c (учитель = CTC, шум в приводе для охвата ошибок)...")
    teacher = CTC()
    Xs, Ys = [], []
    for i in range(n_traj):
        ref = random_traj(rng_global)
        qr0, dqr0, _ = ref(0.0)
        x0 = np.r_[qr0 + rng_global.uniform(-0.2, 0.2, 2), dqr0 + rng_global.uniform(-0.3, 0.3, 2)]
        r = simulate(teacher, ref, T=T, x0=x0, act_noise=2.0, label_ctrl=teacher,
                     seed=1000 + i, record_data=True)
        Xs.append(r["X"]); Ys.append(r["Y"])
    X, Y = np.vstack(Xs), np.vstack(Ys)
    print(f"[data] {len(X)} обучающих пар (вход 10, выход 2)")
    return X, Y


def train_nn(X, Y, epochs=80, bs=256, lr=3e-3):
    x_mu, x_sd = X.mean(0), X.std(0) + 1e-8
    y_mu, y_sd = Y.mean(0), Y.std(0) + 1e-8
    Xn = torch.tensor((X - x_mu) / x_sd, dtype=torch.float32)
    Yn = torch.tensor((Y - y_mu) / y_sd, dtype=torch.float32)
    idx = torch.randperm(len(Xn))
    n_val = len(Xn) // 10
    va, tr = idx[:n_val], idx[n_val:]
    model = MLP()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    loss_fn = nn.MSELoss()
    tr_hist, va_hist = [], []
    for ep in range(epochs):
        model.train()
        perm = tr[torch.randperm(len(tr))]
        tot = 0.0
        for i in range(0, len(perm), bs):
            b = perm[i:i + bs]
            opt.zero_grad()
            loss = loss_fn(model(Xn[b]), Yn[b])
            loss.backward()
            opt.step()
            tot += loss.item() * len(b)
        sched.step()
        model.eval()
        with torch.no_grad():
            v = loss_fn(model(Xn[va]), Yn[va]).item()
        tr_hist.append(tot / len(tr)); va_hist.append(v)
        if ep % 10 == 0 or ep == epochs - 1:
            print(f"[train] epoch {ep+1:3d}/{epochs}  train MSE {tr_hist[-1]:.5f}  val MSE {v:.5f}")
    return NNController(model, x_mu, x_sd, y_mu, y_sd), tr_hist, va_hist


# --------------------------------------------------------------------------
# 6. Метрики
# --------------------------------------------------------------------------
def tip_xy(q):
    """Прямая кинематика: положение кончика руки (м) по углам суставов."""
    l1, l2 = NOMINAL.l1, NOMINAL.l2
    return np.c_[l1 * np.cos(q[:, 0]) + l2 * np.cos(q[:, 0] + q[:, 1]),
                 l1 * np.sin(q[:, 0]) + l2 * np.sin(q[:, 0] + q[:, 1])]


def tracking_metrics(r):
    e = r["qr"] - r["x"][:, :2]
    en = np.linalg.norm(e, axis=1)
    tip_e = np.linalg.norm(tip_xy(r["x"][:, :2]) - tip_xy(r["qr"]), axis=1)
    return dict(rmse=float(np.sqrt(np.mean(en**2))),
                tip_rmse_cm=float(np.sqrt(np.mean(tip_e**2)) * 100),
                max_err=float(en.max()),
                energy=float(np.sum(np.sum(r["tau"] ** 2, axis=1)) * DT))


def p2p_metrics(r, q_target, q_start, tol=0.02):
    q = r["x"][:, :2]
    step = np.abs(q_target - q_start)
    ov, ts = [], []
    for j in range(2):
        d = np.sign(q_target[j] - q_start[j])
        ov.append(max(0.0, float(np.max(d * (q[:, j] - q_target[j])) / step[j] * 100)))
        outside = np.where(np.abs(q[:, j] - q_target[j]) > tol * step[j])[0]
        ts.append(float(r["t"][outside[-1]]) if len(outside) else 0.0)
    final = float(np.linalg.norm(q[-1] - q_target))
    return dict(overshoot_pct=max(ov), settling_s=max(ts), final_err=final)


# --------------------------------------------------------------------------
# 7. Эксперименты и графики
# --------------------------------------------------------------------------
# палитра: slot1 blue = NN, slot2 orange = PID, slot3 aqua = CTC; reference = серый
C_NN, C_PID, C_CTC, C_REF = "#2a78d6", "#eb6834", "#1baf7a", "#52514e"
INK, GRID = "#0b0b0b", "#e4e3df"

plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": "#b9b8b2", "axes.labelcolor": "#52514e",
    "xtick.color": "#52514e", "ytick.color": "#52514e", "axes.titlecolor": INK,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.axisbelow": True, "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
    "savefig.facecolor": "#fcfcfb",
})
COL = {"PID": C_PID, "NN": C_NN, "CTC": C_CTC}


def label_end(ax, x, y, text, color, dy=0.0):
    ax.annotate(text, (x, y), xytext=(4, dy), textcoords="offset points",
                color=INK, fontsize=9, va="center",
                bbox=dict(boxstyle="square,pad=0.1", fc="none", ec="none"))
    ax.plot([x], [y], "o", ms=0)  # keep limits


def animate_arm(results, scen, path, fps=30):
    """GIF: рука под управлением PID и NN едет по одной траектории.
    Слева: серым — эталонная рука (куда надо), оранжевая — PID, синяя — NN, хвосты — след кончика.
    Справа сверху: ошибка кончика (см); справа снизу: моменты, которые выдаёт нейросеть."""
    from matplotlib.animation import FuncAnimation, PillowWriter

    rp, rn = results[(scen, "PID")], results[(scen, "NN")]
    t = rp["t"]
    step = int(round(1.0 / (fps * DT)))
    idx = np.arange(0, len(t), step)
    l1, l2 = NOMINAL.l1, NOMINAL.l2

    def fk(q):
        elbow = np.c_[l1 * np.cos(q[:, 0]), l1 * np.sin(q[:, 0])]
        tip = elbow + np.c_[l2 * np.cos(q[:, 0] + q[:, 1]), l2 * np.sin(q[:, 0] + q[:, 1])]
        return elbow, tip

    e_ref, p_ref = fk(rp["qr"])
    e_pid, p_pid = fk(rp["x"][:, :2])
    e_nn, p_nn = fk(rn["x"][:, :2])
    err_pid = np.linalg.norm(p_pid - p_ref, axis=1) * 100
    err_nn = np.linalg.norm(p_nn - p_ref, axis=1) * 100
    tau = rn["tau"]

    fig = plt.figure(figsize=(11, 5.4))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.05, 1])
    ax = fig.add_subplot(gs[:, 0]); axe = fig.add_subplot(gs[0, 1]); axt = fig.add_subplot(gs[1, 1])

    ax.set_aspect("equal"); ax.set_xlim(-1.1, 1.1); ax.set_ylim(-1.1, 1.1)
    ax.plot(p_ref[:, 0], p_ref[:, 1], color=C_REF, lw=0.8, ls=":", label="reference tip path")
    ghost, = ax.plot([], [], color=C_REF, lw=1.6, ls="--", label="reference arm")
    arm_pid, = ax.plot([], [], "-o", color=C_PID, lw=3, ms=5, label="PID")
    arm_nn, = ax.plot([], [], "-o", color=C_NN, lw=3, ms=5, label="NN")
    tr_pid, = ax.plot([], [], color=C_PID, lw=1.2, alpha=0.7)
    tr_nn, = ax.plot([], [], color=C_NN, lw=1.2, alpha=0.7)
    ax.plot(0, 0, "s", color=INK, ms=6)
    ax.legend(frameon=False, loc="upper left", fontsize=8)
    ax.set_xlabel("x, m"); ax.set_ylabel("y, m")

    axe.set_xlim(0, t[-1]); axe.set_ylim(0, max(err_pid.max(), err_nn.max()) * 1.15)
    axe.set_ylabel("tip error, cm"); axe.set_title("How far the tip is from the target", loc="left", fontsize=10)
    ln_ep, = axe.plot([], [], color=C_PID, lw=1.5, label="PID")
    ln_en, = axe.plot([], [], color=C_NN, lw=1.5, label="NN")
    axe.legend(frameon=False, loc="upper left")

    axt.set_xlim(0, t[-1]); pad = 0.1 * (tau.max() - tau.min() + 1e-9)
    axt.set_ylim(tau.min() - pad, tau.max() + pad)
    axt.set_xlabel("time, s"); axt.set_ylabel("torque, N·m")
    axt.set_title("What the neural network outputs", loc="left", fontsize=10)
    ln_t1, = axt.plot([], [], color=C_NN, lw=1.5, label="joint 1")
    ln_t2, = axt.plot([], [], color=C_NN, lw=1.5, ls="--", label="joint 2")
    axt.legend(frameon=False, loc="upper left", ncol=2)

    trail = int(1.0 / DT)

    def update(i):
        k = idx[i]
        ghost.set_data([0, e_ref[k, 0], p_ref[k, 0]], [0, e_ref[k, 1], p_ref[k, 1]])
        arm_pid.set_data([0, e_pid[k, 0], p_pid[k, 0]], [0, e_pid[k, 1], p_pid[k, 1]])
        arm_nn.set_data([0, e_nn[k, 0], p_nn[k, 0]], [0, e_nn[k, 1], p_nn[k, 1]])
        s = slice(max(0, k - trail), k + 1, step)
        tr_pid.set_data(p_pid[s, 0], p_pid[s, 1]); tr_nn.set_data(p_nn[s, 0], p_nn[s, 1])
        ax.set_title(f"{scen}   t = {t[k]:.2f} s", loc="left", fontsize=10)
        sl = slice(0, k + 1, step)
        ln_ep.set_data(t[sl], err_pid[sl]); ln_en.set_data(t[sl], err_nn[sl])
        ln_t1.set_data(t[sl], tau[sl, 0]); ln_t2.set_data(t[sl], tau[sl, 1])
        return ghost, arm_pid, arm_nn, tr_pid, tr_nn, ln_ep, ln_en, ln_t1, ln_t2

    fig.tight_layout()
    ani = FuncAnimation(fig, update, frames=len(idx), blit=False)
    ani.save(path, writer=PillowWriter(fps=fps), dpi=85)
    plt.close(fig)


def main():
    # ---- данные + обучение ----
    try:                           # консоль Windows (cp1251/cp866) не должна ронять print
        sys.stdout.reconfigure(errors="replace")
    except Exception:
        pass
    plt.switch_backend("Agg")      # графики сохраняем в файлы, окно не нужно
    X, Y = collect_data()
    nn_ctrl, tr_hist, va_hist = train_nn(X, Y)
    nn_ctrl.save(os.path.join(OUT, "nn_weights.npz"))
    pid, ctc = PID(), CTC()
    ctrls = [pid, nn_ctrl, ctc]

    # ---- сценарии слежения ----
    ref_nom = sines(c=[0, 0], A=[1.0, 0.5], w=[1.0, 1.0], ph=[0, np.pi / 2])   # sin t, 0.5 cos t
    ref_fast = sines(c=[0, 0.3], A=[0.8, 0.6], w=[3.5, 3.0], ph=[0, 0.5])      # вне обучающего диапазона w∈[0.5,2.5]
    loaded = NOMINAL.with_payload(0.5)
    scenarios = [
        ("nominal",        ref_nom,  NOMINAL, 0.0,   0.0),
        ("payload +0.5kg", ref_nom,  loaded,  0.0,   0.0),
        ("sensor noise",   ref_nom,  NOMINAL, 0.002, 0.02),
        ("payload+noise",  ref_nom,  loaded,  0.002, 0.02),
        ("unseen fast",    ref_fast, NOMINAL, 0.0,   0.0),
    ]
    results, rows = {}, []
    print("\n[eval] сценарии слежения за траекторией (RMSE по норме ошибки, рад)")
    for name, ref, arm, qn, dqn in scenarios:
        for c in ctrls:
            r = simulate(c, ref, arm=arm, T=5.0, q_noise=qn, dq_noise=dqn, seed=7)
            m = tracking_metrics(r)
            results[(name, c.name)] = r
            rows.append(dict(scenario=name, controller=c.name, **{k: round(v, 5) for k, v in m.items()}))
            print(f"  {name:15s} {c.name:4s} RMSE={m['rmse']:.4f}  tip={m['tip_rmse_cm']:.3f} cm  "
                  f"max={m['max_err']:.4f}  energy={m['energy']:.1f}")

    # ---- point-to-point ----
    q_start, q_target = np.array([0.0, 0.0]), np.array([1.0, -0.8])
    ref_p2p = min_jerk(q_start, q_target, T=1.0)
    print("\n[eval] point-to-point 0 -> (1.0, -0.8) рад за 1 с (min-jerk), затем удержание")
    p2p = {}
    for arm_name, arm in (("nominal", NOMINAL), ("payload +0.5kg", loaded)):
        for c in ctrls:
            r = simulate(c, ref_p2p, arm=arm, T=3.0, seed=7)
            m = p2p_metrics(r, q_target, q_start)
            p2p[(arm_name, c.name)] = r
            rows.append(dict(scenario=f"p2p {arm_name}", controller=c.name,
                             rmse=round(tracking_metrics(r)["rmse"], 5),
                             max_err=round(tracking_metrics(r)["max_err"], 5),
                             energy=round(tracking_metrics(r)["energy"], 5),
                             overshoot_pct=round(m["overshoot_pct"], 2),
                             settling_s=round(m["settling_s"], 3),
                             final_err=round(m["final_err"], 5)))
            print(f"  {arm_name:15s} {c.name:4s} overshoot={m['overshoot_pct']:.1f}%  "
                  f"settling(2%)={m['settling_s']:.2f}s  final_err={m['final_err']:.4f}")

    # ---- CSV ----
    keys = ["scenario", "controller", "rmse", "tip_rmse_cm", "max_err", "energy",
            "overshoot_pct", "settling_s", "final_err"]
    with open(os.path.join(OUT, "metrics.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in keys})

    # ---- Рис. 1: слежение, nominal ----
    fig, axs = plt.subplots(2, 2, figsize=(11, 6), sharex=True)
    t = results[("nominal", "PID")]["t"]
    for j in range(2):
        ax = axs[0, j]
        ax.plot(t, results[("nominal", "PID")]["qr"][:, j], color=C_REF, lw=1.5, ls="--", label="reference")
        ax.plot(t, results[("nominal", "PID")]["x"][:, j], color=C_PID, lw=1.5, label="PID")
        ax.plot(t, results[("nominal", "NN")]["x"][:, j], color=C_NN, lw=1.5, label="NN")
        ax.set_ylabel(f"q{j+1}, rad")
        ax.set_title(f"Joint {j+1}: trajectory (nominal plant)", loc="left", fontsize=10)
        ax = axs[1, j]
        for c in (pid, nn_ctrl):
            r = results[("nominal", c.name)]
            ax.plot(t, r["qr"][:, j] - r["x"][:, j], color=COL[c.name], lw=1.3, label=c.name)
        ax.set_ylabel(f"error q{j+1}, rad")
        ax.set_xlabel("time, s")
        ax.set_title(f"Joint {j+1}: tracking error", loc="left", fontsize=10)
    axs[0, 0].legend(frameon=False, ncol=3, loc="lower left")
    axs[1, 0].legend(frameon=False, ncol=2, loc="lower right")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig1_tracking_nominal.png"), dpi=160); plt.close(fig)

    # ---- Рис. 2: ошибка при нагрузке и шуме датчиков ----
    fig, ax = plt.subplots(figsize=(8, 3.8))
    for c in (pid, nn_ctrl):
        r = results[("payload+noise", c.name)]
        e = np.linalg.norm(r["qr"] - r["x"][:, :2], axis=1)
        ax.plot(r["t"], e, color=COL[c.name], lw=1.4, label=c.name)
    ax.set_xlabel("time, s"); ax.set_ylabel("‖error‖, rad")
    ax.set_title("Tracking error with +0.5 kg payload and sensor noise", loc="left", fontsize=10)
    ax.legend(frameon=False, ncol=2, loc="upper right")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig2_error_payload_noise.png"), dpi=160); plt.close(fig)

    # ---- Рис. 3: RMSE по сценариям ----
    names = [s[0] for s in scenarios]
    fig, ax = plt.subplots(figsize=(9, 4.2))
    width = 0.26
    xs = np.arange(len(names))
    for i, c in enumerate(ctrls):
        vals = [tracking_metrics(results[(n, c.name)])["rmse"] for n in names]
        pos = xs + (i - 1) * (width + 0.02)
        ax.bar(pos, vals, width, color=COL[c.name], label=c.name)
        for p, v in zip(pos, vals):
            ax.text(p, v, f"{v:.3f}", ha="center", va="bottom", fontsize=7.5, color=INK)
    ax.set_xticks(xs); ax.set_xticklabels(names)
    ax.set_ylabel("RMSE, rad (lower is better)")
    ax.set_title("Tracking RMSE by scenario", loc="left", fontsize=10)
    ax.legend(frameon=False, ncol=3, loc="upper left")
    ax.grid(axis="x", visible=False)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig3_rmse_by_scenario.png"), dpi=160); plt.close(fig)

    # ---- Рис. 4: point-to-point ----
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.8), sharey=False)
    for ax, arm_name in zip(axs, ("nominal", "payload +0.5kg")):
        t = p2p[(arm_name, "PID")]["t"]
        ax.plot(t, p2p[(arm_name, "PID")]["qr"][:, 0], color=C_REF, lw=1.3, ls="--", label="reference")
        for c in (pid, nn_ctrl):
            ax.plot(t, p2p[(arm_name, c.name)]["x"][:, 0], color=COL[c.name], lw=1.5, label=c.name)
        ax.set_title(f"Joint 1 step move, {arm_name}", loc="left", fontsize=10)
        ax.set_xlabel("time, s"); ax.set_ylabel("q1, rad")
    axs[0].legend(frameon=False, loc="lower right")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig4_point_to_point.png"), dpi=160); plt.close(fig)

    # ---- Рис. 5: обучение ----
    fig, ax = plt.subplots(figsize=(6, 3.4))
    ax.plot(tr_hist, color=C_NN, lw=1.5, label="train")
    ax.plot(va_hist, color=C_REF, lw=1.5, ls="--", label="validation")
    ax.set_yscale("log"); ax.set_xlabel("epoch"); ax.set_ylabel("MSE (normalized)")
    ax.set_title("NN training curve", loc="left", fontsize=10)
    ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig5_training_curve.png"), dpi=160); plt.close(fig)

    # ---- анимации ----
    for scen, fname in (("unseen fast", "anim_unseen_fast.gif"),
                        ("payload+noise", "anim_payload_noise.gif")):
        animate_arm(results, scen, os.path.join(OUT, fname))
        print(f"[anim] {fname}")

    print(f"\n[done] метрики: {os.path.join(OUT, 'metrics.csv')}; графики: {OUT}/fig*.png")


if __name__ == "__main__":
    main()