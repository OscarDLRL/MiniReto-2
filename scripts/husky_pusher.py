"""
husky_pusher.py  —  Fase 1: Husky A200 despeja el corredor
===========================================================
Fases: SCAN → ALIGN → APPROACH → PUSH → NEXT → DONE
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import animation
from pathlib import Path

# ── Constantes ────────────────────────────────────────────────────────────────
HUSKY_W    = 0.90
HUSKY_H    = 0.58
BOX_S      = 0.55                          # lado de la caja [m]
LANE_Y     = +0.72                         # carril superior
CONTACT    = HUSKY_W/2 + BOX_S/2          # 0.725 m  (centro-centro al contactar)
STAGE      = CONTACT  + 0.30              # 1.025 m  (staging a la derecha)
PUSH_V     = 0.35                          # m/s de empuje
EXIT_X     = -0.35                         # caja "fuera" cuando box.x < EXIT_X
COL_MARGIN = 0.03                          # holgura en colisión [m]

# Carril central
def _draw_world(ax, show_side_zones=True):
    ax.set_xlim(-4.5, 12.5)
    ax.set_ylim(-2.0, 2.0)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)
    ax.axvline(0.0, color="gray", lw=1.0, ls="--", alpha=0.5)
    ax.axvline(6.0, color="gray", lw=1.0, ls="--", alpha=0.5)
    ax.axhline(-1.0, color="gray", lw=1.0, ls=":", alpha=0.6)
    ax.axhline(1.0, color="gray", lw=1.0, ls=":", alpha=0.6)
    ax.axhline(LANE_Y, color="#1f77b4", lw=1.2, ls="-.", alpha=0.7)

    ax.add_patch(mpatches.Rectangle((0.0, -1.0), 6.0, 2.0,
                                    facecolor="#fff8e8", edgecolor="none", alpha=0.45))
    if show_side_zones:
        ax.add_patch(mpatches.Rectangle((-4.0, -2.0), 4.0, 4.0,
                                        facecolor="#e9f5ff", edgecolor="none", alpha=0.35))
        ax.add_patch(mpatches.Rectangle((6.0, -2.0), 6.0, 4.0,
                                        facecolor="#ecffef", edgecolor="none", alpha=0.35))

    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")

# POligono orientado del Husky para visualización
def _robot_corners(x, y, theta):
    local = np.array([
        [-HUSKY_W/2, -HUSKY_H/2],
        [ HUSKY_W/2, -HUSKY_H/2],
        [ HUSKY_W/2,  HUSKY_H/2],
        [-HUSKY_W/2,  HUSKY_H/2],
    ])
    ct, st = np.cos(theta), np.sin(theta)
    rot = np.array([[ct, -st], [st, ct]])
    return local @ rot.T + np.array([x, y])


# Polígono orientado del Husky para visualización (patch de Matplotlib)
def _robot_patch(x, y, theta, facecolor="#2b2d42"):
    return mpatches.Polygon(
        _robot_corners(x, y, theta),
        closed=True,
        facecolor=facecolor,
        edgecolor="black",
        linewidth=1.2,
        zorder=4,
    )

# Guardar imagen resumen con trayectoria y estado final
def save_husky_snapshot(log, boxes, out_path):
    """Guarda una imagen resumen con trayectoria y estado final."""
    fig, ax = plt.subplots(figsize=(12, 5))
    _draw_world(ax)
    ax.set_title("Husky A200 - Fase 1 (trayectoria y estado final)")

    x = np.array(log["x"])
    y = np.array(log["y"])
    th = np.array(log["theta"])

    ax.plot(x, y, color="#0b5d1e", lw=2.0, label="Trayectoria Husky", zorder=3)
    ax.scatter([x[0]], [y[0]], s=50, color="#1f77b4", zorder=5, label="Inicio Husky")
    ax.scatter([x[-1]], [y[-1]], s=55, color="#d62728", zorder=5, label="Fin Husky")

    robot = _robot_patch(x[-1], y[-1], th[-1])
    ax.add_patch(robot)

    for b in boxes:
        color = "#2ca02c" if not b.active else "#ff7f0e"
        label = f"Caja {b.name} ({'fuera' if not b.active else 'activa'})"
        ax.add_patch(mpatches.Rectangle(
            (b.x - b.w/2, b.y - b.h/2), b.w, b.h,
            facecolor=color, edgecolor="black", alpha=0.75, zorder=4, label=label
        ))

    handles, labels = ax.get_legend_handles_labels()
    uniq = {}
    for h, l in zip(handles, labels):
        if l not in uniq:
            uniq[l] = h
    ax.legend(uniq.values(), uniq.keys(), loc="upper right", fontsize=9)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


# Guardar animación GIF de la simulación
def save_husky_gif(log, out_path, step_stride=4, fps=15):
    """Guarda GIF de la simulación usando muestras del log."""
    x = np.array(log["x"])
    y = np.array(log["y"])
    th = np.array(log["theta"])
    box_names = list(log["box_x"].keys())
    bx = {k: np.array(v) for k, v in log["box_x"].items()}
    by = {k: np.array(v) for k, v in log["box_y"].items()}

    idx = np.arange(0, len(x), max(1, step_stride), dtype=int)
    if idx[-1] != len(x) - 1:
        idx = np.append(idx, len(x) - 1)

    fig, ax = plt.subplots(figsize=(12, 5))
    _draw_world(ax, show_side_zones=False)
    ax.set_title("Husky A200 - Fase 1 (animacion)")

    path_line, = ax.plot([], [], color="#0b5d1e", lw=2.0, zorder=3)
    husky = _robot_patch(x[0], y[0], th[0])
    ax.add_patch(husky)
    heading, = ax.plot([], [], color="#111111", lw=2.0, zorder=5)
    txt = ax.text(0.02, 0.96, "", transform=ax.transAxes, va="top", fontsize=10)

    box_patches = {}
    for name in box_names:
        patch = mpatches.Rectangle((bx[name][0]-BOX_S/2, by[name][0]-BOX_S/2), BOX_S, BOX_S,
                                   facecolor="#ff7f0e", edgecolor="black", alpha=0.8, zorder=4)
        ax.add_patch(patch)
        box_patches[name] = patch

    def _update(k):
        i = idx[k]
        path_line.set_data(x[:i+1], y[:i+1])

        husky.set_xy(_robot_corners(x[i], y[i], th[i]))

        hx2 = x[i] + 0.45*np.cos(th[i])
        hy2 = y[i] + 0.45*np.sin(th[i])
        heading.set_data([x[i], hx2], [y[i], hy2])

        for name in box_names:
            box_patches[name].set_xy((bx[name][i]-BOX_S/2, by[name][i]-BOX_S/2))
            if bx[name][i] < EXIT_X:
                box_patches[name].set_facecolor("#2ca02c")

        txt.set_text(f"t = {log['t'][i]:.1f} s | estado = {log['state'][i]}")
        return [path_line, husky, heading, txt, *box_patches.values()]

    ani = animation.FuncAnimation(fig, _update, frames=len(idx), interval=1000/fps, blit=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ani.save(out_path, writer=animation.PillowWriter(fps=fps))
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# 1.  Husky A200
# ══════════════════════════════════════════════════════════════════════════════
class HuskyA200:
    """Skid-steer 4 ruedas con factor de deslizamiento."""

    def __init__(self, r=0.1651, B=0.555, mass=50.0):
        self.r, self.B, self.mass = r, B, mass
        self.x = self.y = self.theta = 0.0
        self.v = self.omega = 0.0
        self.terrain = "asphalt"
        self._slip = {"asphalt":1.00,"grass":0.85,
                      "gravel":0.78,"sand":0.65,"mud":0.50}

    def forward_kinematics(self, wR1, wR2, wL1, wL2):
        s = self._slip.get(self.terrain, 0.8)
        v = self.r/2 * ((wR1+wR2)/2 + (wL1+wL2)/2) * s
        w = self.r/self.B * ((wR1+wR2)/2 - (wL1+wL2)/2)
        return v, w

    def inverse_kinematics(self, v, omega):
        wR = (2*v + omega*self.B) / (2*self.r)
        wL = (2*v - omega*self.B) / (2*self.r)
        return wR, wR, wL, wL

    def update_pose(self, v, omega, dt):
        tm = self.theta + omega*dt/2
        self.x     += v*np.cos(tm)*dt
        self.y     += v*np.sin(tm)*dt
        self.theta += omega*dt
        self.theta  = np.arctan2(np.sin(self.theta), np.cos(self.theta))
        self.v, self.omega = v, omega

    def get_pose(self):  return (self.x, self.y, self.theta)
    def reset(self, x=0., y=0., theta=0.):
        self.x, self.y, self.theta = x, y, theta
        self.v = self.omega = 0.0


# ══════════════════════════════════════════════════════════════════════════════
# 2.  LiDAR 2D simulado
# ══════════════════════════════════════════════════════════════════════════════
class LiDAR2D:
    def __init__(self, n_beams=360, fov_deg=270., max_range=9., noise_std=0.015):
        self.n_beams   = n_beams
        self.fov       = np.radians(fov_deg)
        self.max_range = max_range
        self.noise_std = noise_std
        self.angles    = np.linspace(-self.fov/2, self.fov/2, n_beams, endpoint=False)

    def scan(self, pose, boxes):
        rx, ry, rt = pose
        ranges = np.full(self.n_beams, self.max_range)
        for i, a in enumerate(self.angles):
            dx, dy = np.cos(rt+a), np.sin(rt+a)
            for b in boxes:
                if not b.active: continue
                t = _ray_aabb(rx, ry, dx, dy,
                              b.x-b.w/2, b.y-b.h/2, b.x+b.w/2, b.y+b.h/2)
                if t is not None and t < ranges[i]:
                    ranges[i] = t
        return np.clip(ranges + np.random.normal(0, self.noise_std, self.n_beams),
                       0, self.max_range)

    def nearest_box(self, pose, ranges, boxes):
        """Caja activa más cercana según el retorno LiDAR."""
        rx, ry, rt = pose
        active = [b for b in boxes if b.active]
        if not active: return None
        bi = int(np.argmin(ranges))
        if ranges[bi] >= self.max_range - 0.05:
            return min(active, key=lambda b: np.hypot(b.x-rx, b.y-ry))
        ang = self.angles[bi] + rt
        wx  = rx + ranges[bi]*np.cos(ang)
        wy  = ry + ranges[bi]*np.sin(ang)
        return min(active, key=lambda b: np.hypot(b.x-wx, b.y-wy))


def _ray_aabb(ox, oy, dx, dy, x0, y0, x1, y1):
    tmin, tmax = 0., 1e9
    for o, d, lo, hi in [(ox,dx,x0,x1),(oy,dy,y0,y1)]:
        if abs(d)<1e-10:
            if o<lo or o>hi: return None
        else:
            t1,t2 = sorted([(lo-o)/d,(hi-o)/d])
            tmin,tmax = max(tmin,t1), min(tmax,t2)
            if tmin>tmax: return None
    return tmin if tmin>1e-6 else None


# ══════════════════════════════════════════════════════════════════════════════
# 3.  Caja 2D  (obstáculo físico)
# ══════════════════════════════════════════════════════════════════════════════
class Box2D:
    def __init__(self, name, x, y, w=BOX_S, h=BOX_S, mass=25.):
        self.name = name
        self.x, self.y = float(x), float(y)
        self.w, self.h = w, h
        self.mass   = mass
        self.active = True

    @property
    def xmin(self): return self.x - self.w/2
    @property
    def xmax(self): return self.x + self.w/2
    @property
    def ymin(self): return self.y - self.h/2
    @property
    def ymax(self): return self.y + self.h/2

    def track_husky(self, hx):
        """Durante PUSH la caja sigue al frente (izquierdo) del Husky."""
        self.x = hx - CONTACT

    def is_outside(self): return self.x < EXIT_X
    def __repr__(self):
        return f"Box2D({self.name}, x={self.x:.2f}, y={self.y:.2f}, active={self.active})"


# ══════════════════════════════════════════════════════════════════════════════
# 4.  Controlador P  (goto point)
# ══════════════════════════════════════════════════════════════════════════════
class GoTo:
    """Navegar a un punto: primero gira, luego avanza."""
    def __init__(self, k_v=0.80, k_w=2.50,
                 v_max=0.70, w_max=1.50, tol=0.10):
        self.k_v, self.k_w     = k_v, k_w
        self.v_max, self.w_max = v_max, w_max
        self.tol = tol

    def __call__(self, pose, goal):
        """(v, omega, reached)"""
        x, y, theta = pose
        gx, gy = goal
        dist = np.hypot(gx-x, gy-y)
        if dist < self.tol: return 0., 0., True
        ag = np.arctan2(gy-y, gx-x)
        ae = np.arctan2(np.sin(ag-theta), np.cos(ag-theta))
        if abs(ae) > 0.08:
            return 0., float(np.clip(self.k_w*ae, -self.w_max, self.w_max)), False
        v = float(np.clip(self.k_v*dist, 0, self.v_max))
        w = float(np.clip(self.k_w*ae,  -self.w_max, self.w_max))
        return v, w, False


# ══════════════════════════════════════════════════════════════════════════════
# 5.  Máquina de estados: HuskyPusher
# ══════════════════════════════════════════════════════════════════════════════
class HuskyPusher:
    """
    SCAN → ALIGN → APPROACH → PUSH → NEXT → DONE

    El Husky rodea las cajas por el carril superior (LANE_Y = +0.72 m).
    Empuje en dirección −X (←).
    """

    def __init__(self, husky, lidar, boxes, dt=0.05):
        self.husky, self.lidar, self.boxes = husky, lidar, boxes
        self.dt     = dt
        self.goto   = GoTo()
        self.state  = "SCAN"
        self.target = None
        self._ag    = None   # align_goal
        self._apg   = None   # approach_goal
        self._last_cmd  = (0., 0.)
        self._last_meas = (0., 0.)
        self.t = 0.
        self.log = {
            't':[], 'x':[], 'y':[], 'theta':[],
            'v_cmd':[], 'omega_cmd':[],
            'v_meas':[], 'omega_meas':[],
            'state':[],
            'box_x': {b.name:[] for b in boxes},
            'box_y': {b.name:[] for b in boxes},
        }

    # ── API ──────────────────────────────────────────────────────────────────
    def step(self) -> bool:
        if self.state == "DONE": return True
        pose = self.husky.get_pose()
        { "SCAN":     self._scan,
          "ALIGN":    self._align,
          "APPROACH": self._approach,
          "PUSH":     self._push,
          "NEXT":     self._next,
        }[self.state](pose)
        self._log(pose)
        self.t += self.dt
        return self.state == "DONE"

    def run(self, max_steps=12000):
        for _ in range(max_steps):
            if self.step():
                print("[HuskyPusher] DONE ✓ — corredor despejado")
                return True, self.log
        print("[HuskyPusher] TIMEOUT")
        return False, self.log

    def check_success(self):
        return all(not (0<=b.x<=6 and -1<=b.y<=1) for b in self.boxes)

    def _active_boxes(self):
        return [b for b in self.boxes if b.active]

    # ── SCAN ─────────────────────────────────────────────────────────────────
    def _scan(self, pose):
        """Subir al carril (control directo) y luego detectar con LiDAR."""
        _, y, theta = pose

        # ─ Entrada al carril: control directo (sin PointFollower) ─
        y_err = LANE_Y - y
        if abs(y_err) > 0.06:
            # Orientar hacia ±Y según la dirección necesaria
            theta_target = np.pi/2 if y_err > 0 else -np.pi/2
            t_err = np.arctan2(np.sin(theta_target - theta),
                               np.cos(theta_target - theta))
            if abs(t_err) > 0.08:
                self._apply(0., float(np.clip(2.5*t_err, -1.5, 1.5)))
            else:
                v = float(np.clip(0.9 * abs(y_err), 0.08, 0.6))
                self._apply(v, float(np.clip(1.0*t_err, -0.5, 0.5)))
            return

        # ─ En el carril: detectar la siguiente caja ─
        active = self._active_boxes()
        if not active:
            self.state = "DONE"; return

        ranges = self.lidar.scan(pose, self.boxes)
        box    = self.lidar.nearest_box(pose, ranges, self.boxes)
        if box is None:
            box = min(active, key=lambda b: b.x)

        self.target = box
        self._ag    = (box.x + STAGE,   LANE_Y)    # staging en el carril
        self._apg   = (box.x + CONTACT, box.y)     # punto de contacto
        print(f"[SCAN]     Caja {box.name} ({box.x:.2f},{box.y:.2f}) | "
              f"align=({self._ag[0]:.2f},{self._ag[1]:.2f})")
        self.state = "ALIGN"

    # ── ALIGN ────────────────────────────────────────────────────────────────
    def _align(self, pose):
        """Avanzar por el carril hasta el staging point."""
        x, y, _ = pose
        v, w, _ = self.goto(pose, self._ag)
        v, w    = self._col(v, w, pose)
        self._apply(v, w)
        if np.hypot(x-self._ag[0], y-self._ag[1]) < 0.14:
            print("[ALIGN]    Staging OK → bajando a contacto")
            self.state = "APPROACH"

    # ── APPROACH ─────────────────────────────────────────────────────────────
    def _approach(self, pose):
        """Bajar desde el carril hasta contactar el lado derecho de la caja."""
        x, y, _ = pose
        v, w, reached = self.goto(pose, self._apg)
        v, w = self._col(v, w, pose, skip=self.target)
        self._apply(v, w)
        if reached or np.hypot(x-self._apg[0], y-self._apg[1]) < 0.10:
            print(f"[APPROACH] Contacto con {self.target.name} → PUSH")
            self.state = "PUSH"

    # ── PUSH ─────────────────────────────────────────────────────────────────
    def _push(self, pose):
        """
        1. Girar hasta theta = π  (mirar −X).
        2. Avanzar en −X arrastrando la caja.
        """
        _, y, theta = pose
        box = self.target

        # Alinear orientación a π
        err = np.arctan2(np.sin(np.pi - theta), np.cos(np.pi - theta))
        if abs(err) > 0.06:
            self._apply(0., float(np.clip(2.5*err, -1.5, 1.5)))
            return

        # Avanzar: v>0 con θ=π → dx/dt = −v  (hacia la izquierda ✓)
        y_corr = float(np.clip(1.5*(box.y - y), -0.3, 0.3))
        self._apply(PUSH_V, y_corr)
        box.track_husky(self.husky.x)        # caja pegada al frente
        self._resolve_box_contacts(box)

        if box.is_outside():
            print(f"[PUSH]     Caja {box.name} fuera → x={box.x:.2f}")
            box.active = False
            self.state = "NEXT"

    def _resolve_box_contacts(self, source_box):
        """Evita solapes caja-caja y propaga empuje entre cajas en contacto."""
        moved = True
        min_dx = BOX_S
        y_overlap_tol = BOX_S * 0.9

        while moved:
            moved = False
            for other in self.boxes:
                if other is source_box:
                    continue

                # Solo propaga empuje si ambas cajas comparten prácticamente carril en Y.
                if abs(other.y - source_box.y) > y_overlap_tol:
                    continue

                dx = source_box.x - other.x
                if abs(dx) < min_dx:
                    # Mantener separación mínima para cuerpos sólidos.
                    if dx <= 0:
                        other.x = source_box.x + min_dx
                    else:
                        other.x = source_box.x - min_dx
                    source_box = other
                    moved = True
                    break

    # ── NEXT ─────────────────────────────────────────────────────────────────
    def _next(self, pose):
        remaining = self._active_boxes()
        if remaining:
            print(f"[NEXT]     Quedan {len(remaining)} caja(s) → SCAN")
            self.state = "SCAN"
        else:
            self.state = "DONE"

    # ── Colisión AABB ────────────────────────────────────────────────────────
    def _col(self, v, w, pose, skip=None):
        """Cancela avance si el siguiente paso intersecta una caja activa."""
        if v <= 0: return v, w
        x, y, theta = pose
        nx = x + v*np.cos(theta)*self.dt
        ny = y + v*np.sin(theta)*self.dt
        hw, hh = HUSKY_W/2, HUSKY_H/2
        m = COL_MARGIN
        for b in self.boxes:
            if not b.active or b is skip: continue
            if (nx+hw+m > b.xmin and nx-hw-m < b.xmax and
                ny+hh+m > b.ymin and ny-hh-m < b.ymax):
                return 0., w
        return v, w

    # ── Helpers ──────────────────────────────────────────────────────────────
    def _apply(self, v_cmd, w_cmd):
        wR, _, wL, _ = self.husky.inverse_kinematics(v_cmd, w_cmd)
        n = np.random.normal(0, 0.008, 4)
        vm, wm = self.husky.forward_kinematics(wR+n[0],wR+n[1],wL+n[2],wL+n[3])
        self.husky.update_pose(vm, wm, self.dt)
        self._last_cmd  = (v_cmd, w_cmd)
        self._last_meas = (vm, wm)

    def _log(self, pose):
        x, y, theta = pose
        self.log['t'].append(self.t)
        self.log['x'].append(x);  self.log['y'].append(y)
        self.log['theta'].append(theta)
        self.log['v_cmd'].append(self._last_cmd[0])
        self.log['omega_cmd'].append(self._last_cmd[1])
        self.log['v_meas'].append(self._last_meas[0])
        self.log['omega_meas'].append(self._last_meas[1])
        self.log['state'].append(self.state)
        for b in self.boxes:
            self.log['box_x'][b.name].append(b.x)
            self.log['box_y'][b.name].append(b.y)


# ══════════════════════════════════════════════════════════════════════════════
# 6.  Escenario y demo
# ══════════════════════════════════════════════════════════════════════════════
def make_scenario():
    np.random.seed(42)
    husky = HuskyA200()
    husky.reset(x=-2.0, y=0.0, theta=0.0)
    lidar = LiDAR2D()
    boxes = [Box2D("1",1.,0.), Box2D("2",3.,0.), Box2D("3",5.,0.)]
    return husky, lidar, boxes


def demo_husky_pusher():
    print("="*65)
    print("DEMO  Husky A200 — Fase 1  (empuje ←)")
    print("="*65)
    husky, lidar, boxes = make_scenario()
    pusher = HuskyPusher(husky, lidar, boxes, dt=0.05)
    ok, log = pusher.run(max_steps=15000)
    print(f"\nÉxito: {ok}  |  Despejado: {pusher.check_success()}")
    for b in boxes: print(f"  {b}")
    print(f"  Tiempo sim: {log['t'][-1]:.1f} s")

    out_dir = Path(__file__).resolve().parent.parent / "results"
    png_path = out_dir / "husky_pusher_snapshot.png"
    gif_path = out_dir / "husky_pusher_run.gif"
    save_husky_snapshot(log, boxes, png_path)
    save_husky_gif(log, gif_path)
    print(f"  Imagen: {png_path}")
    print(f"  GIF:    {gif_path}")

    return log, boxes


if __name__ == "__main__":
    demo_husky_pusher()