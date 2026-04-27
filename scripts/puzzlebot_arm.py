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
        self.pose[0] += v * math.cos(theta_mid) * dt
        self.pose[1] += v * math.sin(theta_mid) * dt
        self.pose[2] = self.wrap_to_pi(self.pose[2] + w * dt)
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

    def update_deployment(
        self,
        dt: float,
        dismount_pose: np.ndarray,
        target_pose: np.ndarray,
        time_value: float,
    ) -> None:
        """Baja del ANYmal por un costado y se mueve a su zona segura."""
        dismount_pose = np.asarray(dismount_pose, dtype=float)
        target_pose = np.asarray(target_pose, dtype=float)

        if self.state == "mounted":
            self.pose = dismount_pose.copy()
            self.path.append(self.pose[:2].copy())
            self.state = "deploying"
            self.route_queue = self._make_lane_route(target_pose)

        if self.state == "deploying":
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
    ) -> None:
        """
        Ejecuta la tarea completa de un PuzzleBot para una sola caja:
        ir por caja -> usar brazo -> llevar a stack -> usar brazo -> ir a zona segura.
        """
        lane_x = self.route_lane_x if lane_x is None else float(lane_x)
        pickup_pose = np.array([box.pickup_xy[0], box.pickup_xy[1] - 0.22, math.pi / 2.0], dtype=float)
        stack_pose = np.array([stack_goal_xy[0], stack_goal_xy[1] - 0.24, math.pi / 2.0], dtype=float)

        if self.state == "idle":
            self.state = "to_pickup"
            self.route_queue = self._make_lane_route(pickup_pose, lane_x=lane_x)

        if self.state == "to_pickup":
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
