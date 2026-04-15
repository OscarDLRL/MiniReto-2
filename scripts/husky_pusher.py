"""
husky_pusher.py  —  Fase 1: Husky A200 despeja el corredor
===========================================================
Fases: SCAN → ALIGN → APPROACH → PUSH → NEXT → DONE
"""

import numpy as np

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
                print("[HuskyPusher] DONE — corredor despejado")
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

    return log, boxes


if __name__ == "__main__":
    demo_husky_pusher()