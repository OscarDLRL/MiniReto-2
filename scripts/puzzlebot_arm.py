"""
puzzlebot_arm.py

Modelo simple del mini brazo 3 DoF del PuzzleBot y de un PuzzleBot móvil
para la fase 3 del mini reto.

La vista principal del simulador es 2D (desde arriba), pero el brazo conserva:
- FK 3D
- IK geométrica cerrada
- Jacobiano analítico
- mapeo de fuerza a torque con tau = J^T f

Además, se incluye una clase PuzzleBot2D con un modelo diferencial sencillo
para mover cada robot, tomar una caja con el brazo, llevarla al punto de apilado
y regresar a una zona segura.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import StandardScaler
    import joblib
    _RF_AVAILABLE = True
except ImportError:  # pragma: no cover
    _RF_AVAILABLE = False


@dataclass
class SmallBox:
    """Caja pequeña que será manipulada por un PuzzleBot."""

    label: str
    color: str
    pickup_xy: np.ndarray
    size_xy: np.ndarray = field(default_factory=lambda: np.array([0.16, 0.16], dtype=float))
    height: float = 0.06
    world_xy: Optional[np.ndarray] = None
    carried_by: Optional[str] = None
    placed: bool = False
    stack_level: Optional[int] = None

    def __post_init__(self) -> None:
        self.pickup_xy = np.array(self.pickup_xy, dtype=float)
        if self.world_xy is None:
            self.world_xy = self.pickup_xy.copy()
        else:
            self.world_xy = np.array(self.world_xy, dtype=float)


class PuzzleBotArm:
    """Mini brazo 3 DoF: base rotacional + dos eslabones en plano vertical."""

    def __init__(self, l1: float = 0.10, l2: float = 0.08, l3: float = 0.06) -> None:
        self.l1 = float(l1)
        self.l2 = float(l2)
        self.l3 = float(l3)
        self.q = np.zeros(3, dtype=float)

    def forward_kinematics(self, q: Optional[np.ndarray] = None) -> np.ndarray:
        """Calcula la posición 3D del efector final [x, y, z]."""
        if q is not None:
            self.q = np.asarray(q, dtype=float)
        q1, q2, q3 = self.q
        radial = self.l2 * math.cos(q2) + self.l3 * math.cos(q2 + q3)
        x = math.cos(q1) * radial
        y = math.sin(q1) * radial
        z = self.l1 + self.l2 * math.sin(q2) + self.l3 * math.sin(q2 + q3)
        return np.array([x, y, z], dtype=float)

    def inverse_kinematics(self, p_des: np.ndarray) -> Tuple[np.ndarray, bool]:
        """IK geométrica cerrada para alcanzar p_des = [x, y, z]."""
        x, y, z = np.asarray(p_des, dtype=float)
        q1 = math.atan2(y, x)

        radial = math.hypot(x, y)
        z2 = z - self.l1
        c3 = (radial * radial + z2 * z2 - self.l2**2 - self.l3**2) / (2.0 * self.l2 * self.l3)

        reachable = True
        if c3 < -1.0 or c3 > 1.0:
            reachable = False
            c3 = float(np.clip(c3, -1.0, 1.0))

        q3 = -math.acos(c3)
        q2 = math.atan2(z2, radial) - math.atan2(self.l3 * math.sin(q3), self.l2 + self.l3 * math.cos(q3))
        q = np.array([q1, q2, q3], dtype=float)
        return q, reachable

    def jacobian(self, q: Optional[np.ndarray] = None) -> np.ndarray:
        """Jacobiano analítico 3x3 del efector respecto a q = [q1, q2, q3]."""
        if q is not None:
            self.q = np.asarray(q, dtype=float)
        q1, q2, q3 = self.q

        radial = self.l2 * math.cos(q2) + self.l3 * math.cos(q2 + q3)
        dr_dq2 = -self.l2 * math.sin(q2) - self.l3 * math.sin(q2 + q3)
        dr_dq3 = -self.l3 * math.sin(q2 + q3)
        dz_dq2 = self.l2 * math.cos(q2) + self.l3 * math.cos(q2 + q3)
        dz_dq3 = self.l3 * math.cos(q2 + q3)

        c1 = math.cos(q1)
        s1 = math.sin(q1)

        return np.array(
            [
                [-s1 * radial, c1 * dr_dq2, c1 * dr_dq3],
                [c1 * radial, s1 * dr_dq2, s1 * dr_dq3],
                [0.0, dz_dq2, dz_dq3],
            ],
            dtype=float,
        )

    def force_to_torque(self, f_tip: np.ndarray, q: Optional[np.ndarray] = None) -> np.ndarray:
        """Mapea fuerza del efector a torques articulares con tau = J^T f."""
        J = self.jacobian(q)
        return J.T @ np.asarray(f_tip, dtype=float)

    def cartesian_line(self, p_start: np.ndarray, p_goal: np.ndarray, n_steps: int = 20) -> List[np.ndarray]:
        """Interpolación lineal cartesiana simple."""
        p_start = np.asarray(p_start, dtype=float)
        p_goal = np.asarray(p_goal, dtype=float)
        n_steps = max(2, int(n_steps))
        return [((1.0 - a) * p_start + a * p_goal) for a in np.linspace(0.0, 1.0, n_steps)]

    def build_action_sequence(
        self,
        p_goal: np.ndarray,
        contact_force: np.ndarray,
        n_steps: int = 20,
        hold_steps: int = 8,
    ) -> Dict[str, List[np.ndarray]]:
        """
        Construye una secuencia cartesiana + IK + torques.

        Regresa un diccionario con:
        - points: trayectoria cartesiana
        - q_traj: trayectoria articular
        - tau_traj: torques por paso
        """
        p_start = self.forward_kinematics()
        points = self.cartesian_line(p_start, p_goal, n_steps=n_steps)
        q_traj: List[np.ndarray] = []
        tau_traj: List[np.ndarray] = []

        for p in points:
            q, _ = self.inverse_kinematics(p)
            q_traj.append(q)
            tau_traj.append(self.force_to_torque(np.zeros(3), q))

        q_hold = q_traj[-1].copy()
        for _ in range(max(1, int(hold_steps))):
            q_traj.append(q_hold.copy())
            tau_traj.append(self.force_to_torque(contact_force, q_hold))
            points.append(p_goal.copy())

        return {"points": points, "q_traj": q_traj, "tau_traj": tau_traj}


# ── Random-Forest velocity planner ──────────────────────────────────────────

def build_rf_feature_vector(
    pos: np.ndarray,
    heading: float,
    target_pos: np.ndarray,
    close_objects: List[np.ndarray],
    target_angle: float,
    max_objects: int = 5,
    influence_radius: float = 1.0,
) -> np.ndarray:
    """
    Build a fixed-length feature vector for the Random-Forest model.
 
    Parameters
    ----------
    pos              : (2,) current XY position of the robot.
    heading          : current robot heading [radians] (pose[2]).
    target_pos       : (2,) desired goal XY position.
    close_objects    : list of (2,) arrays – XY positions of nearby obstacles.
                       At most `max_objects` are used; extras are ignored,
                       missing entries are zero-padded.
    target_angle     : desired heading at the goal [radians].
    max_objects      : maximum number of obstacle slots in the feature vector.
    influence_radius : radius used to weight obstacle repulsion features [m].
 
    Returns
    -------
    features : (7 + max_objects * 4,) float64 array
        Layout:
          [0-1]  delta-to-goal (dx, dy) in world frame
          [2]    distance to goal
          [3]    heading error to goal direction  (wrapped to [-pi, pi])
          [4]    heading error to target_angle    (wrapped to [-pi, pi])
          [5]    cos(heading)  – robot orientation (avoids angle discontinuity)
          [6]    sin(heading)
          then for each obstacle slot i (0 … max_objects-1):
            [7+4i]   rel_x
            [8+4i]   rel_y
            [9+4i]   distance (clamped at influence_radius)
            [10+4i]  weight  = max(0, 1 - dist/influence_radius)
    """
    pos = np.asarray(pos, dtype=float)
    heading = float(heading)
    target_pos = np.asarray(target_pos, dtype=float)
 
    delta = target_pos - pos
    dist_goal = float(np.linalg.norm(delta))
    goal_direction = math.atan2(float(delta[1]), float(delta[0]))
    heading_error_to_goal = (goal_direction - heading + math.pi) % (2 * math.pi) - math.pi
    heading_error_to_target_angle = (float(target_angle) - heading + math.pi) % (2 * math.pi) - math.pi
 
    core = np.array([
        delta[0], delta[1],
        dist_goal,
        heading_error_to_goal,
        heading_error_to_target_angle,
        math.cos(heading),
        math.sin(heading),
    ], dtype=float)
 
    # Obstacle features (sorted by distance, closest first)
    obs_feats = np.zeros(max_objects * 4, dtype=float)
    if close_objects:
        diffs = [np.asarray(o, dtype=float) - pos for o in close_objects]
        dists = [float(np.linalg.norm(d)) for d in diffs]
        # Sort by distance
        pairs = sorted(zip(dists, diffs), key=lambda x: x[0])
        for slot, (d, diff) in enumerate(pairs[:max_objects]):
            w = max(0.0, 1.0 - d / influence_radius)
            obs_feats[slot * 4 + 0] = diff[0]
            obs_feats[slot * 4 + 1] = diff[1]
            obs_feats[slot * 4 + 2] = min(d, influence_radius)
            obs_feats[slot * 4 + 3] = w
 
    return np.concatenate([core, obs_feats])


def generate_rf_training_data(
    n_samples: int = 5000,
    max_objects: int = 5,
    influence_radius: float = 1.0,
    max_v: float = 0.45,
    max_w: float = 2.1,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic training data using a unicycle potential-field expert.
 
    The expert computes a desired Cartesian velocity from attractive + repulsive
    fields, then projects it directly onto unicycle (v, w) commands:
 
      v = forward speed, scaled by cos(heading_error) so the robot slows when
          it needs to turn sharply.
      w = proportional heading-error controller toward the desired direction,
          saturated at max_w.
 
    This matches exactly how the robot will consume the model outputs, so the
    forest learns to close the loop in (v, w) space from the start — no
    post-hoc Cartesian-to-unicycle conversion is needed at inference time.
 
    Returns
    -------
    X : (n_samples, feature_dim) feature matrix
    y : (n_samples, 2) targets  [v, w]
    """
    rng = np.random.default_rng(seed)
 
    X_rows: List[np.ndarray] = []
    y_rows: List[np.ndarray] = []
 
    for _ in range(9*n_samples//10):
        pos = rng.uniform(-5.0, 5.0, size=2)
        heading = rng.uniform(-math.pi, math.pi)
        target_pos = rng.uniform(-5.0, 5.0, size=2)
        target_angle = rng.uniform(-math.pi, math.pi)
        n_obs = rng.integers(0, max_objects + 1)
        close_objects = [rng.uniform(-1.0, 1.0, size=2) + pos for _ in range(n_obs)]
 
        # --- Potential-field expert policy (Cartesian) ---
        delta = target_pos - pos
        dist_goal = np.linalg.norm(delta) + 1e-6
 
        # Attractive: smooth tanh ramp so speed naturally drops to zero at goal
        att_speed = max_v * math.tanh(2.0 * dist_goal)
        att = delta / dist_goal * att_speed
 
        # Repulsive: push away from close obstacles
        rep = np.zeros(2)
        for obs in close_objects:
            diff = pos - np.asarray(obs)
            d = np.linalg.norm(diff) + 1e-6
            if d < influence_radius:
                rep += (diff / d) * (1.0 / d - 1.0 / influence_radius) * 0.3
 
        v_cartesian = att + rep
        spd = np.linalg.norm(v_cartesian)
        if spd > max_v:
            v_cartesian = v_cartesian / spd * max_v
 
        # --- Project onto unicycle (v, w) ---
        if spd > 1e-4:
            desired_heading = math.atan2(float(v_cartesian[1]), float(v_cartesian[0]))
        else:
            desired_heading = float(target_angle)
 
        # Near goal: switch to final-angle alignment, stop moving
        if dist_goal < 0.08:
            desired_heading = float(target_angle)
            v_cartesian = np.zeros(2)
 
        heading_err = (desired_heading - heading + math.pi) % (2 * math.pi) - math.pi
        v_cmd = float(np.clip(
            np.linalg.norm(v_cartesian) * max(0.0, math.cos(heading_err)),
            0.0, max_v,
        ))
        w_cmd = float(np.clip(2.5 * heading_err, -max_w, max_w))
 
        feat = build_rf_feature_vector(
            pos, heading, target_pos, close_objects, target_angle,
            max_objects=max_objects, influence_radius=influence_radius,
        )
        X_rows.append(feat)
        y_rows.append(np.array([v_cmd, w_cmd], dtype=float))

    for _ in range(n_samples//10):
        pos = rng.uniform(-5.0, 5.0, size=2)
        heading = rng.uniform(-math.pi, math.pi)
        target_pos = pos
        target_angle = rng.uniform(-math.pi, math.pi)
        n_obs = rng.integers(0, max_objects + 1)
        close_objects = [rng.uniform(-1.0, 1.0, size=2) + pos for _ in range(n_obs)]
 
        # --- Potential-field expert policy (Cartesian) ---
        delta = target_pos - pos
        dist_goal = np.linalg.norm(delta) + 1e-6
 
        # Attractive: smooth tanh ramp so speed naturally drops to zero at goal
        att_speed = max_v * math.tanh(2.0 * dist_goal)
        att = delta / dist_goal * att_speed
 
        # Repulsive: push away from close obstacles
        rep = np.zeros(2)
        for obs in close_objects:
            diff = pos - np.asarray(obs)
            d = np.linalg.norm(diff) + 1e-6
            if d < influence_radius:
                rep += (diff / d) * (1.0 / d - 1.0 / influence_radius) * 0.3
 
        v_cartesian = att + rep
        spd = np.linalg.norm(v_cartesian)
        if spd > max_v:
            v_cartesian = v_cartesian / spd * max_v
 
        # --- Project onto unicycle (v, w) ---
        if spd > 1e-4:
            desired_heading = math.atan2(float(v_cartesian[1]), float(v_cartesian[0]))
        else:
            desired_heading = float(target_angle)
 
        # Near goal: switch to final-angle alignment, stop moving
        if dist_goal < 0.08:
            desired_heading = float(target_angle)
            v_cartesian = np.zeros(2)
 
        heading_err = (desired_heading - heading + math.pi) % (2 * math.pi) - math.pi
        v_cmd = float(np.clip(
            np.linalg.norm(v_cartesian) * max(0.0, math.cos(heading_err)),
            0.0, max_v,
        ))
        w_cmd = float(np.clip(2.5 * heading_err, -max_w, max_w))
 
        feat = build_rf_feature_vector(
            pos, heading, target_pos, close_objects, target_angle,
            max_objects=max_objects, influence_radius=influence_radius,
        )
        X_rows.append(feat)
        y_rows.append(np.array([v_cmd, w_cmd], dtype=float))
 
    return np.vstack(X_rows), np.vstack(y_rows)
 


class RFNavigator:
    """
    Random-Forest velocity planner for a differential-drive robot.

    Usage (training)
    ----------------
    nav = RFNavigator()
    nav.train()           # generates synthetic data and fits the model
    nav.save("rf_nav.pkl")

    Usage (inference)
    -----------------
    nav = RFNavigator.load("rf_nav.pkl")
    vx, vy = nav.predict(pos, vel, target_pos, close_objects, target_angle)
    """

    def __init__(
        self,
        max_objects: int = 5,
        influence_radius: float = 1.0,
        n_estimators: int = 100,
        max_depth: int = 12,
        random_state: int = 42,
    ) -> None:
        if not _RF_AVAILABLE:
            raise ImportError(
                "scikit-learn is required for RFNavigator. "
                "Install it with: pip install scikit-learn"
            )
        self.max_objects = max_objects
        self.influence_radius = influence_radius
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1,
        )
        self.scaler = StandardScaler()
        self._trained = False

    # ------------------------------------------------------------------
    def train(
        self,
        n_samples: int = 5000,
        seed: int = 42,
        X: Optional[np.ndarray] = None,
        y: Optional[np.ndarray] = None,
    ) -> "RFNavigator":
        """
        Fit the Random-Forest model.

        Pass custom (X, y) arrays if you have real trajectory data,
        otherwise synthetic potential-field data is generated automatically.

        Parameters
        ----------
        n_samples : number of synthetic samples (ignored when X/y provided)
        seed      : RNG seed for synthetic data generation
        X         : optional custom feature matrix  (n, feature_dim)
        y         : optional custom velocity targets (n, 2)
        """
        if X is None or y is None:
            print(f"[RFNavigator] Generating {n_samples} synthetic training samples …")
            X, y = generate_rf_training_data(
                n_samples=n_samples,
                max_objects=self.max_objects,
                influence_radius=self.influence_radius,
                seed=seed,
            )
        X_scaled = self.scaler.fit_transform(X)
        print(f"[RFNavigator] Training Random Forest on {X_scaled.shape[0]} samples …")
        self.model.fit(X_scaled, y)
        self._trained = True
        print("[RFNavigator] Training complete.")
        return self

    # ------------------------------------------------------------------
    def predict(
        self,
        pos: np.ndarray,
        heading: float,
        target_pos: np.ndarray,
        close_objects: List[np.ndarray],
        target_angle: float,
    ) -> Tuple[float, float]:
        """
        Predict unicycle commands (v, w) for the current robot state.
 
        Parameters
        ----------
        pos           : (2,) current XY position
        heading       : current robot heading [radians] (pose[2])
        target_pos    : (2,) goal XY position
        close_objects : list of (2,) obstacle positions (can be empty)
        target_angle  : desired heading at goal [radians]
 
        Returns
        -------
        v : float – linear velocity  [m/s],  clipped to [0, max_v]
        w : float – angular velocity [rad/s], clipped to [-max_w, max_w]
        """
        if not self._trained:
            raise RuntimeError("RFNavigator must be trained before calling predict().")
        feat = build_rf_feature_vector(
            pos, heading, target_pos, close_objects, target_angle,
            max_objects=self.max_objects,
            influence_radius=self.influence_radius,
        )
        feat_scaled = self.scaler.transform(feat.reshape(1, -1))
        out = self.model.predict(feat_scaled)[0]
        v = float(np.clip(out[0], 0.0, 0.45))
        w = float(np.clip(out[1], -2.1, 2.1))
        return v, w

    # ------------------------------------------------------------------
    def save(self, path: str) -> None:
        """Persist model + scaler to disk via joblib."""
        joblib.dump({"model": self.model, "scaler": self.scaler,
                     "max_objects": self.max_objects,
                     "influence_radius": self.influence_radius}, path)
        print(f"[RFNavigator] Saved to {path}")

    @classmethod
    def load(cls, path: str) -> "RFNavigator":
        """Restore a previously saved RFNavigator."""
        data = joblib.load(path)
        nav = cls(
            max_objects=data["max_objects"],
            influence_radius=data["influence_radius"],
        )
        nav.model = data["model"]
        nav.scaler = data["scaler"]
        nav._trained = True
        print(f"[RFNavigator] Loaded from {path}")
        return nav


# ── end Random-Forest navigator ──────────────────────────────────────────────


class PuzzleBot2D:
    """Robot móvil diferencial pequeño con un mini brazo 3 DoF."""

    def __init__(
        self,
        name: str,
        assigned_box: str,
        safe_pose: Tuple[float, float, float],
        wheel_radius: float = 0.05,
        axle_length: float = 0.19,
        body_length: float = 0.26,
        body_width: float = 0.18,
    ) -> None:
        self.name = str(name)
        self.assigned_box = str(assigned_box)
        self.safe_pose = np.array(safe_pose, dtype=float)
        self.pose = self.safe_pose.copy()

        self.wheel_radius = float(wheel_radius)
        self.axle_length = float(axle_length)
        self.body_length = float(body_length)
        self.body_width = float(body_width)
        self.max_v = 0.45
        self.max_w = 2.1
        self.route_lane_x = 12.82

        self.arm = PuzzleBotArm()
        self.path: List[np.ndarray] = [self.pose[:2].copy()]

        # RF-based velocity planner (set via attach_rf_navigator or trained inline)
        self.rf_navigator: Optional[RFNavigator] = None
        # Current 2-D velocity estimated from last integrate() call
        self._velocity: np.ndarray = np.zeros(2, dtype=float)

        self.state = "mounted"
        self.deployed = False
        self.task_complete = False
        self.carrying_box: Optional[str] = None
        self.current_action: Optional[Dict[str, List[np.ndarray]]] = None
        self.action_index = 0
        self.current_tau = np.zeros(3, dtype=float)
        self.route_queue: List[np.ndarray] = []

        self.logs = {
            "time": [],
            "x": [],
            "y": [],
            "theta": [],
            "state": [],
            "tau_norm": [],
            "tau_q1": [],
            "tau_q2": [],
            "tau_q3": [],
        }

    @staticmethod
    def wrap_to_pi(angle: float) -> float:
        return (angle + math.pi) % (2.0 * math.pi) - math.pi

    def rot2(self, theta: float) -> np.ndarray:
        c = math.cos(theta)
        s = math.sin(theta)
        return np.array([[c, -s], [s, c]], dtype=float)

    def body_corners_world(self) -> np.ndarray:
        """Esquinas del cuerpo para dibujarlo como robot de dos ruedas."""
        hx = 0.5 * self.body_length
        hy = 0.5 * self.body_width
        corners_body = np.array(
            [[+hx, +hy], [+hx, -hy], [-hx, -hy], [-hx, +hy], [+hx, +hy]],
            dtype=float,
        )
        return (self.rot2(self.pose[2]) @ corners_body.T).T + self.pose[:2]

    def wheel_points_world(self) -> Tuple[np.ndarray, np.ndarray]:
        """Centros aproximados de las dos ruedas en vista superior."""
        left = self.pose[:2] + self.rot2(self.pose[2]) @ np.array([0.0, +0.5 * self.body_width], dtype=float)
        right = self.pose[:2] + self.rot2(self.pose[2]) @ np.array([0.0, -0.5 * self.body_width], dtype=float)
        return left, right

    def front_point(self) -> np.ndarray:
        """Punto frontal geométrico del robot."""
        return self.pose[:2] + self.rot2(self.pose[2]) @ np.array([0.5 * self.body_length + 0.06, 0.0], dtype=float)

    def grasp_point_world(self) -> np.ndarray:
        """Punto aproximado donde el brazo sostiene la caja durante el transporte."""
        return self.pose[:2] + self.rot2(self.pose[2]) @ np.array([0.5 * self.body_length + 0.11, 0.0], dtype=float)

    def integrate(self, dt: float, v: float, w: float) -> None:
        """Integra el modelo diferencial."""
        theta_mid = self.pose[2] + 0.5 * w * dt
        vx = v * math.cos(theta_mid)
        vy = v * math.sin(theta_mid)
        self.pose[0] += vx * dt
        self.pose[1] += vy * dt
        self.pose[2] = self.wrap_to_pi(self.pose[2] + w * dt)
        self._velocity = np.array([vx, vy], dtype=float)
        self.path.append(self.pose[:2].copy())

    def goto_pose_controller(self, goal_pose: np.ndarray) -> Tuple[float, float]:
        """Control proporcional simple hacia una pose objetivo."""
        dx = goal_pose[0] - self.pose[0]
        dy = goal_pose[1] - self.pose[1]
        dist = float(np.hypot(dx, dy))
        desired_heading = math.atan2(dy, dx)
        heading_error = self.wrap_to_pi(desired_heading - self.pose[2])
        final_heading_error = self.wrap_to_pi(goal_pose[2] - self.pose[2])

        if dist > 0.08:
            v = min(self.max_v, 1.0 * dist) * max(0.2, math.cos(heading_error))
            w = float(np.clip(2.5 * heading_error, -self.max_w, self.max_w))
        else:
            v = 0.0
            w = float(np.clip(2.0 * final_heading_error, -self.max_w, self.max_w))
        return float(v), float(w)

    def reached_pose(self, goal_pose: np.ndarray, pos_tol: float = 0.08, ang_tol: float = 0.18) -> bool:
        """Verdadero si el robot está suficientemente cerca de su pose objetivo."""
        pos_ok = np.linalg.norm(self.pose[:2] - goal_pose[:2]) <= pos_tol
        ang_ok = abs(self.wrap_to_pi(self.pose[2] - goal_pose[2])) <= ang_tol
        return bool(pos_ok and ang_ok)

    def mount_on_anymal(self, world_pose: np.ndarray) -> None:
        """Coloca al PuzzleBot sobre el dorso del ANYmal durante el transporte."""
        self.pose = np.asarray(world_pose, dtype=float).copy()
        self.path[-1] = self.pose[:2].copy()
        self.state = "mounted"
        self.route_queue = []

    def update_idle_log(self, time_value: float) -> None:
        """Agrega una muestra al log incluso si el robot no se está moviendo."""
        self.logs["time"].append(float(time_value))
        self.logs["x"].append(float(self.pose[0]))
        self.logs["y"].append(float(self.pose[1]))
        self.logs["theta"].append(float(self.pose[2]))
        self.logs["state"].append(self.state)
        self.logs["tau_norm"].append(float(np.linalg.norm(self.current_tau)))
        self.logs["tau_q1"].append(float(self.current_tau[0]))
        self.logs["tau_q2"].append(float(self.current_tau[1]))
        self.logs["tau_q3"].append(float(self.current_tau[2]))

    def _start_arm_pick(self) -> None:
        target_local = np.array([0.16, 0.0, 0.035], dtype=float)
        contact_force = np.array([0.0, 0.0, -5.0], dtype=float)
        self.current_action = self.arm.build_action_sequence(target_local, contact_force, n_steps=16, hold_steps=8)
        self.action_index = 0
        self.state = "arm_pick"

    def _start_arm_place(self, stack_level: int) -> None:
        target_local = np.array([0.17, 0.0, 0.05 + 0.05 * stack_level], dtype=float)
        contact_force = np.array([0.0, 0.0, -4.0], dtype=float)
        self.current_action = self.arm.build_action_sequence(target_local, contact_force, n_steps=18, hold_steps=10)
        self.action_index = 0
        self.state = "arm_place"

    def _step_arm_sequence(self) -> bool:
        """Ejecuta una muestra de la secuencia del brazo. Devuelve True si termina."""
        if self.current_action is None:
            return True

        idx = min(self.action_index, len(self.current_action["q_traj"]) - 1)
        q = self.current_action["q_traj"][idx]
        tau = self.current_action["tau_traj"][idx]
        self.arm.q = q.copy()
        self.current_tau = tau.copy()
        self.action_index += 1
        done = self.action_index >= len(self.current_action["q_traj"])
        if done:
            self.current_action = None
            self.action_index = 0
        return done

    def _make_lane_route(self, target_pose: np.ndarray, lane_x: Optional[float] = None) -> List[np.ndarray]:
        """Construye una ruta que pasa por un carril lateral para evitar el ANYmal."""
        target_pose = np.asarray(target_pose, dtype=float).copy()
        lane_x = self.route_lane_x if lane_x is None else float(lane_x)

        waypoints: List[np.ndarray] = []
        current = self.pose.copy()

        if abs(current[0] - lane_x) > 0.08:
            yaw = 0.0 if lane_x >= current[0] else math.pi
            waypoints.append(np.array([lane_x, current[1], yaw], dtype=float))
            current = waypoints[-1]

        if abs(target_pose[1] - current[1]) > 0.10:
            heading = math.pi / 2.0 if target_pose[1] >= current[1] else -math.pi / 2.0
            waypoints.append(np.array([lane_x, target_pose[1], heading], dtype=float))

        waypoints.append(target_pose)
        return waypoints

    def _follow_route(self, dt: float) -> bool:
        """Sigue la cola de waypoints de la ruta actual."""
        if not self.route_queue:
            return True

        goal = self.route_queue[0]
        v, w = self.goto_pose_controller(goal)
        self.integrate(dt, v, w)

        final_leg = len(self.route_queue) == 1
        pos_tol = 0.08 if final_leg else 0.10
        ang_tol = 0.18 if final_leg else 0.45
        if self.reached_pose(goal, pos_tol=pos_tol, ang_tol=ang_tol):
            self.pose = goal.copy()
            self.path.append(self.pose[:2].copy())
            self.route_queue.pop(0)
        return len(self.route_queue) == 0

    def attach_rf_navigator(self, navigator: "RFNavigator") -> None:
        """Attach a trained RFNavigator to this robot."""
        self.rf_navigator = navigator

    def update_deployment(
        self,
        dt: float,
        dismount_pose: np.ndarray,
        target_pose: np.ndarray,
        time_value: float,
        close_objects: Optional[List[np.ndarray]] = None,
    ) -> None:
        """
        Baja del ANYmal por un costado y se mueve a su zona segura.

        If an RFNavigator is attached (via attach_rf_navigator) it is used to
        compute the desired 2-D velocity at each time step during the
        "deploying" phase.  The predicted [vx, vy] is then converted to
        differential-drive commands (v, w) and fed into integrate().

        When no RFNavigator is attached the original waypoint-following
        controller (_follow_route) is used as a fallback, so the robot keeps
        working even without a trained model.

        Parameters
        ----------
        dt             : simulation time step [s]
        dismount_pose  : (3,) pose [x, y, theta] where the robot lands after
                         dismounting the ANYmal.
        target_pose    : (3,) goal pose [x, y, theta] (safe/deployment zone).
        time_value     : simulation clock value for logging.
        close_objects  : optional list of (2,) XY positions of nearby
                         obstacles, passed to the RF model.  If None an
                         empty list is used.
        """
        dismount_pose = np.asarray(dismount_pose, dtype=float)
        target_pose = np.asarray(target_pose, dtype=float)
        close_objects = close_objects or []

        if self.state == "mounted":
            self.pose = dismount_pose.copy()
            self.path.append(self.pose[:2].copy())
            self.state = "deploying"
            # Pre-build the lane route even when using the RF planner so we
            # have a fallback and can check route completion easily.
            self.route_queue = self._make_lane_route(target_pose)

        if self.state == "deploying":
            if self.rf_navigator is not None and self.rf_navigator._trained:
                # ── RF-based velocity planning ──────────────────────────────
                target_angle = float(target_pose[2])

                # Ask the model for the desired 2-D velocity
                v, w = self.rf_navigator.predict(
                        pos=self.pose[:2], 
                    heading=self.pose[2],
                    target_pos=target_pose[:2],
                    close_objects=close_objects,
                    target_angle=target_pose[2]
                )

                self.integrate(dt, v, w)
                self.current_tau = np.zeros(3, dtype=float)

                # Check goal reached
                if self.reached_pose(target_pose):
                    self.pose = target_pose.copy()
                    self.path.append(self.pose[:2].copy())
                    self.state = "idle"
                    self.deployed = True
            else:
                # ── Fallback: original waypoint controller ──────────────────
                done = self._follow_route(dt)
                self.current_tau = np.zeros(3, dtype=float)
                if done:
                    self.pose = target_pose.copy()
                    self.path.append(self.pose[:2].copy())
                    self.state = "idle"
                    self.deployed = True

        self.update_idle_log(time_value)

    def update_task(
        self,
        dt: float,
        time_value: float,
        box: SmallBox,
        stack_goal_xy: np.ndarray,
        stack_level: int,
        lane_x: Optional[float] = None,
        close_objects: Optional[List[np.ndarray]] = None,
    ) -> None:
        """
        Ejecuta la tarea completa de un PuzzleBot para una sola caja:
        ir por caja -> usar brazo -> llevar a stack -> usar brazo -> ir a zona segura.
        """
        lane_x = self.route_lane_x if lane_x is None else float(lane_x)
        pickup_pose = np.array([box.pickup_xy[0], box.pickup_xy[1] - 0.22, math.pi / 2.0], dtype=float)
        stack_pose = np.array([stack_goal_xy[0], stack_goal_xy[1] - 0.24, math.pi / 2.0], dtype=float)

        close_objects = close_objects or []

        if self.state == "idle":
            self.state = "to_pickup"
            self.route_queue = self._make_lane_route(pickup_pose, lane_x=lane_x)

        if self.state == "to_pickup":
            if self.rf_navigator is not None and self.rf_navigator._trained:
                # ── RF-based velocity planning ──────────────────────────────
                target_angle = float(pickup_pose[2])

                # Ask the model for the desired 2-D velocity
                v, w = self.rf_navigator.predict(
                    pos=self.pose[:2], 
                    heading=self.pose[2],
                    target_pos=pickup_pose[:2],
                    close_objects=close_objects,
                    target_angle=pickup_pose[2]
                )

                self.integrate(dt, v, w)
                self.current_tau = np.zeros(3, dtype=float)

                # Check goal reached
                if self.reached_pose(pickup_pose):
                    self.pose = pickup_pose.copy()
                    self.path.append(self.pose[:2].copy())
                    self._start_arm_pick()
            else:
                # ── Fallback: original waypoint controller ──────────────────
                done = self._follow_route(dt)
                self.current_tau = np.zeros(3, dtype=float)
                if done:
                    self.pose = pickup_pose.copy()
                    self.path.append(self.pose[:2].copy())
                    self._start_arm_pick()


        elif self.state == "arm_pick":
            finished = self._step_arm_sequence()
            if finished:
                self.carrying_box = box.label
                box.carried_by = self.name
                box.world_xy = self.grasp_point_world().copy()
                self.state = "to_stack"
                self.route_queue = self._make_lane_route(stack_pose, lane_x=lane_x)

        elif self.state == "to_stack":
            if self.rf_navigator is not None and self.rf_navigator._trained:
                # ── RF-based velocity planning ──────────────────────────────
                target_angle = float(stack_pose[2])

                # Ask the model for the desired 2-D velocity
                v, w = self.rf_navigator.predict(
                        pos=self.pose[:2], 
                    heading=self.pose[2],
                    target_pos=stack_pose[:2],
                    close_objects=close_objects,
                    target_angle=stack_pose[2]
                )

                self.integrate(dt, v, w)
                self.current_tau = np.zeros(3, dtype=float)
                
                if self.carrying_box == box.label:
                    box.world_xy = self.grasp_point_world().copy()

                # Check goal reached
                if self.reached_pose(stack_pose):
                    self.pose = stack_pose.copy()
                    self.path.append(self.pose[:2].copy())
                    self._start_arm_place(stack_level)
            else:
                # ── Fallback: original waypoint controller ──────────────────
                done = self._follow_route(dt)
                self.current_tau = np.zeros(3, dtype=float)
                if self.carrying_box == box.label:
                    box.world_xy = self.grasp_point_world().copy()
                if done:
                    self.pose = stack_pose.copy()
                    self.path.append(self.pose[:2].copy())
                    self._start_arm_place(stack_level)

        elif self.state == "arm_place":
            finished = self._step_arm_sequence()
            if self.carrying_box == box.label:
                box.world_xy = self.grasp_point_world().copy()
            if finished:
                self.carrying_box = None
                box.carried_by = None
                box.placed = True
                box.stack_level = int(stack_level)
                box.world_xy = np.asarray(stack_goal_xy, dtype=float).copy()
                self.state = "to_safe"
                self.route_queue = self._make_lane_route(self.safe_pose, lane_x=lane_x)

        elif self.state == "to_safe":
            if self.rf_navigator is not None and self.rf_navigator._trained:
                # ── RF-based velocity planning ──────────────────────────────
                target_angle = float(self.safe_pose[2])

                # Ask the model for the desired 2-D velocity
                v, w = self.rf_navigator.predict(
                        pos=self.pose[:2], 
                    heading=self.pose[2],
                    target_pos=self.safe_pose[:2],
                    close_objects=close_objects,
                    target_angle=self.safe_pose[2]
                )

                self.integrate(dt, v, w)
                self.current_tau = np.zeros(3, dtype=float)
                
                if self.carrying_box == box.label:
                    box.world_xy = self.grasp_point_world().copy()

                # Check goal reached
                if self.reached_pose(self.safe_pose):
                    self.pose = self.safe_pose.copy()
                    self.path.append(self.pose[:2].copy())
                    self.state = "done"
                    self.task_complete = True
            else:
                # ── Fallback: original waypoint controller ──────────────────
                done = self._follow_route(dt)
                self.current_tau = np.zeros(3, dtype=float)
                if done:
                    self.pose = self.safe_pose.copy()
                    self.path.append(self.pose[:2].copy())
                    self.state = "done"
                    self.task_complete = True

        elif self.state == "done":
            self.current_tau = np.zeros(3, dtype=float)

        self.update_idle_log(time_value)
