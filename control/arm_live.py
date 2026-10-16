"""
Живая визуализация: PID и нейросеть управляют двумя одинаковыми руками В РЕАЛЬНОМ ВРЕМЕНИ.

Запуск (сначала один раз python arm_nn_control.py — он сохранит nn_weights.npz):
    python arm_live.py           интерактивное окно
    python arm_live.py --gif     без окна: записать arm_live_demo.gif (сценарий по таймеру)

Управление в окне:
    клик мышью    новая цель для кончика руки (цели ставятся в очередь)
    p             груз +0.5 кг на конце (плант меняется, контроллеры об этом не знают)
    n             шум датчиков вкл/выкл
    пробел        пауза
    r             сбросить руки в исходное положение
    q             выход

Серая область — рабочая зона, на которой обучалась сеть (|q1|,|q2| <= 1.3 рад).
Клик за её пределами проецируется внутрь: вне обучающего диапазона нейросеть не должна работать.
"""
import os
import sys
import time
from collections import deque

import numpy as np

GIF_MODE = "--gif" in sys.argv

import matplotlib
if GIF_MODE:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

from arm_nn_control import (NOMINAL, DT, TAU_MAX, PID, NNController, min_jerk,
                            C_PID, C_NN, C_REF, INK)

HERE = os.path.dirname(os.path.abspath(__file__))
L1, L2 = NOMINAL.l1, NOMINAL.l2
Q_LIM = 1.3                      # диапазон углов, на котором обучалась сеть
PAYLOAD_KG = 0.5
Q_START = np.array([0.0, 0.4])
DEMO_WAYPOINTS = [(0.8, 0.9), (-0.5, 1.0), (0.1, -0.9), (-0.9, 0.6), (0.9, -0.5), (0.0, 0.4)]


# ---------------------------- кинематика -----------------------------------
def fk(q):
    """(локоть, кончик) в метрах."""
    ex, ey = L1 * np.cos(q[0]), L1 * np.sin(q[0])
    return (ex, ey), (ex + L2 * np.cos(q[0] + q[1]), ey + L2 * np.sin(q[0] + q[1]))


def ik(x, y, q_hint):
    """Обратная кинематика 2-звенника; из двух решений берём ближе к обучающему диапазону."""
    c2 = np.clip((x * x + y * y - L1**2 - L2**2) / (2 * L1 * L2), -1, 1)
    best = None
    for s in (1, -1):
        q2 = s * np.arccos(c2)
        q1 = np.arctan2(y, x) - np.arctan2(L2 * np.sin(q2), L1 + L2 * np.cos(q2))
        q1 = (q1 + np.pi) % (2 * np.pi) - np.pi
        q = np.array([q1, q2])
        cost = 100 * np.sum(np.maximum(np.abs(q) - Q_LIM, 0) ** 2) + 0.01 * np.sum((q - q_hint) ** 2)
        if best is None or cost < best[0]:
            best = (cost, q)
    return np.clip(best[1], -Q_LIM, Q_LIM)


# ---------------------------- планировщик цели -----------------------------
class Planner:
    """Ставит цели в очередь и отдаёт плавную min-jerk траекторию (qr, dqr, ddqr)."""

    def __init__(self, q0):
        self.q_goal = np.array(q0, float)
        self.fn, self.t0, self.T = min_jerk(q0, q0, 1.0), 0.0, 1.0
        self.queue = deque()
        self.idle_since = 0.0
        self.demo_i = 0

    def add(self, q):
        self.queue.append(np.array(q, float))

    def busy(self, t):
        return t - self.t0 < self.T

    def ref(self, t):
        return self.fn(t - self.t0)

    def update(self, t, auto=True):
        if self.busy(t):
            return
        if not self.queue and auto and t - (self.t0 + self.T) > 0.6:
            self.add(DEMO_WAYPOINTS[self.demo_i % len(DEMO_WAYPOINTS)])
            self.demo_i += 1
        if self.queue:
            q_prev, q_new = self.q_goal, self.queue.popleft()
            d = float(np.max(np.abs(q_new - q_prev)))
            # длительность подбираем так, чтобы скорость/ускорение остались в обучающем диапазоне
            self.T = max(1.0, np.sqrt(1.2 * d), 0.95 * d)
            self.fn, self.t0, self.q_goal = min_jerk(q_prev, q_new, self.T), t, q_new


# ---------------------------- состояние симуляции --------------------------
class Rig:
    def __init__(self, ctrl):
        self.ctrl = ctrl
        self.arm = NOMINAL
        self.x = np.r_[Q_START, 0.0, 0.0]
        self.trail = deque(maxlen=120)

    def reset(self):
        self.x = np.r_[Q_START, 0.0, 0.0]
        self.trail.clear()
        self.ctrl.reset()


class World:
    def __init__(self):
        nn_path = os.path.join(HERE, "nn_weights.npz")
        if not os.path.exists(nn_path):
            sys.exit("Нет nn_weights.npz — сначала запустите:  python arm_nn_control.py")
        self.rigs = {"PID": Rig(PID()), "NN": Rig(NNController.from_npz(nn_path))}
        self.rng = np.random.default_rng(3)
        self.reset()

    def reset(self):
        self.t = 0.0
        self.planner = Planner(Q_START)
        self.payload = False
        self.noise = False
        self.paused = False
        self.err_hist = {k: deque(maxlen=int(6 / 0.03)) for k in self.rigs}
        self.t_hist = deque(maxlen=int(6 / 0.03))
        for r in self.rigs.values():
            r.arm = NOMINAL
            r.reset()

    def set_payload(self, on):
        self.payload = on
        for r in self.rigs.values():
            r.arm = NOMINAL.with_payload(PAYLOAD_KG) if on else NOMINAL

    def advance(self, sim_seconds, auto=True):
        """Продвинуть симуляцию на sim_seconds (шагами DT)."""
        for _ in range(max(1, int(round(sim_seconds / DT)))):
            self.planner.update(self.t, auto)
            for r in self.rigs.values():
                qm, dqm = r.x[:2], r.x[2:]
                if self.noise:
                    qm = qm + self.rng.normal(0, 0.002, 2)
                    dqm = dqm + self.rng.normal(0, 0.02, 2)
                tau = np.clip(r.ctrl(self.t, qm, dqm, self.planner.ref), -TAU_MAX, TAU_MAX)
                r.x = r.arm.step(r.x, tau)
            self.t += DT
        qr = self.planner.ref(self.t)[0]
        tip_ref = np.array(fk(qr)[1])
        self.t_hist.append(self.t)
        for k, r in self.rigs.items():
            tip = np.array(fk(r.x[:2])[1])
            r.trail.append(tip)
            self.err_hist[k].append(np.linalg.norm(tip - tip_ref) * 1000)   # мм

    def click(self, x, y):
        q = ik(x, y, self.planner.q_goal)
        self.planner.queue.appendleft(q)


# ---------------------------- рисование ------------------------------------
def build_figure(world):
    fig = plt.figure(figsize=(11, 6.2))
    gs = fig.add_gridspec(2, 2, height_ratios=[3, 1.25])
    axes = {"PID": fig.add_subplot(gs[0, 0]), "NN": fig.add_subplot(gs[0, 1])}
    ax_err = fig.add_subplot(gs[1, :])
    colors = {"PID": C_PID, "NN": C_NN}

    # рабочая зона обучения (серым)
    g1, g2 = np.meshgrid(np.linspace(-Q_LIM, Q_LIM, 70), np.linspace(-Q_LIM, Q_LIM, 70))
    wx = L1 * np.cos(g1) + L2 * np.cos(g1 + g2)
    wy = L1 * np.sin(g1) + L2 * np.sin(g1 + g2)

    art = {}
    for name, ax in axes.items():
        ax.set_aspect("equal"); ax.set_xlim(-1.1, 1.1); ax.set_ylim(-1.1, 1.1)
        ax.scatter(wx, wy, s=6, color="#e4e3df", zorder=0, linewidths=0)
        ax.plot(0, 0, "s", color=INK, ms=6, zorder=5)
        ax.set_title(name + ("  (classical, no model)" if name == "PID" else "  (neural network)"),
                     loc="left", fontsize=10)
        trail, = ax.plot([], [], color=colors[name], lw=1.2, alpha=0.6, zorder=2)
        link, = ax.plot([], [], "-", color=colors[name], lw=5, solid_capstyle="round", zorder=4)
        joints, = ax.plot([], [], "o", color=colors[name], ms=7, mec="#fcfcfb", mew=1.5, zorder=6)
        tip, = ax.plot([], [], "o", color=colors[name], ms=7, mec="#fcfcfb", mew=1.5, zorder=6)
        goal, = ax.plot([], [], "x", color=C_REF, ms=9, mew=2, zorder=3)
        ghost, = ax.plot([], [], "o", mfc="none", mec=C_REF, ms=11, mew=1.4, zorder=3)
        txt = ax.text(0.02, 0.03, "", transform=ax.transAxes, fontsize=10, color=INK)
        art[name] = dict(trail=trail, link=link, joints=joints, tip=tip, goal=goal, ghost=ghost, txt=txt)

    # ось ошибки неподвижна (лог-шкала, время «сколько секунд назад»), чтобы работала быстрая перерисовка
    # верхняя декада оставлена пустой — там живёт легенда
    ax_err.set_yscale("log"); ax_err.set_ylim(0.05, 500); ax_err.set_xlim(-6, 0)
    ax_err.set_yticks([0.1, 1, 10, 100]); ax_err.set_yticklabels(["0.1", "1", "10", "100"])
    ax_err.set_ylabel("tip error, mm (log)"); ax_err.set_xlabel("seconds ago")
    ln = {k: ax_err.plot([], [], color=colors[k], lw=1.6, label=k)[0] for k in ("PID", "NN")}
    ax_err.legend(frameon=False, loc="upper left", ncol=2)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    # Строка статуса живёт в своей невидимой оси: у текста на уровне фигуры (fig.text) нет .axes,
    # и FuncAnimation(blit=True) на нём падает с AttributeError — окно остаётся пустым.
    ax_status = fig.add_axes([0, 0, 1, 0.04])
    ax_status.axis("off")
    status = ax_status.text(0.5, 0.5, "", ha="center", va="center", fontsize=9, color=C_REF)
    return fig, axes, art, ax_err, ln, status


def make_update(world, axes, art, ax_err, ln, status, fixed_dt=None, script=None):
    last = {"wall": time.perf_counter()}

    def update(_frame):
        if script is not None:
            script(world)
        if not world.paused:
            if fixed_dt is not None:
                world.advance(fixed_dt)
            else:
                now = time.perf_counter()
                dt_wall = min(now - last["wall"], 0.05)       # реальное время, но не больше 50 мс за кадр
                last["wall"] = now
                world.advance(dt_wall)
        else:
            last["wall"] = time.perf_counter()

        q_goal = world.planner.q_goal
        goal_xy = fk(q_goal)[1]
        ref_xy = fk(world.planner.ref(world.t)[0])[1]
        for name, r in world.rigs.items():
            a = art[name]
            (ex, ey), (tx, ty) = fk(r.x[:2])
            a["link"].set_data([0, ex, tx], [0, ey, ty])
            a["joints"].set_data([ex], [ey])
            a["tip"].set_data([tx], [ty])
            a["tip"].set_markersize(13 if world.payload else 7)
            tr = np.array(r.trail) if len(r.trail) else np.zeros((0, 2))
            a["trail"].set_data(tr[:, 0], tr[:, 1])
            a["goal"].set_data([goal_xy[0]], [goal_xy[1]])
            a["ghost"].set_data([ref_xy[0]], [ref_xy[1]])
            e = world.err_hist[name][-1] if world.err_hist[name] else 0.0
            a["txt"].set_text(f"tip error: {e:5.1f} mm")
        t_arr = np.array(world.t_hist)
        if len(t_arr):
            for k in ln:
                ln[k].set_data(t_arr - t_arr[-1], np.maximum(np.array(world.err_hist[k]), 0.05))
        status.set_text(f"payload +0.5 kg: {'ON' if world.payload else 'off'}   |   "
                        f"sensor noise: {'ON' if world.noise else 'off'}   |   "
                        f"click = new target   [p] payload  [n] noise  [space] pause  [r] reset")
        dyn = [status, *ln.values()]
        for a in art.values():
            dyn += [a["trail"], a["link"], a["joints"], a["tip"], a["goal"], a["ghost"], a["txt"]]
        return dyn

    return update


def main():
    world = World()
    fig, axes, art, ax_err, ln, status = build_figure(world)

    if GIF_MODE:
        fps, total = 20, 14.0

        def script(w):
            if abs(w.t - 5.0) < 0.03 and not w.payload:
                w.set_payload(True)
            if abs(w.t - 10.0) < 0.03 and not w.noise:
                w.noise = True

        update = make_update(world, axes, art, ax_err, ln, status, fixed_dt=1.0 / fps, script=script)
        out = os.path.join(HERE, "arm_live_demo.gif")
        ani = FuncAnimation(fig, update, frames=int(total * fps), blit=False)
        ani.save(out, writer=PillowWriter(fps=fps), dpi=80)
        print("[gif]", out)
        return

    def on_click(ev):
        if ev.inaxes in axes.values() and ev.button == 1:
            world.click(ev.xdata, ev.ydata)

    def on_key(ev):
        if ev.key == "p":
            world.set_payload(not world.payload)
        elif ev.key == "n":
            world.noise = not world.noise
        elif ev.key == " ":
            world.paused = not world.paused
        elif ev.key == "r":
            world.reset()

    fig.canvas.mpl_connect("button_press_event", on_click)
    fig.canvas.mpl_connect("key_press_event", on_key)
    update = make_update(world, axes, art, ax_err, ln, status)
    # blit=True: перерисовываются только движущиеся элементы -> реальное время без тормозов
    _ani = FuncAnimation(fig, update, interval=30, blit=True, cache_frame_data=False)
    try:                                   # консоль Windows (cp1251/cp866) не должна ронять print
        sys.stdout.reconfigure(errors="replace")
    except Exception:
        pass
    print(__doc__)
    plt.show()


if __name__ == "__main__":
    main()