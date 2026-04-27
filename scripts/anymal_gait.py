
"""
anymal_gait.py

Generador simple de marcha trote para ANYmal en simulación 2D vista desde arriba.
La visualización es plana (x, y, theta), pero cada pata mantiene una cinemática
3D simplificada usando la FK/IK y el Jacobiano analítico indicados en el reto.

Pensado para correr con numpy + matplotlib.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, Tuple, List

import numpy as np


LEG_ORDER = ("LF", "RF", "LH", "RH")
LEG_SIGNS = {
    "LF": +1,
    "LH": +1,
    "RF": -1,
    "RH": -1,
}
SWING_GROUPS = (
    ("LF", "RH"),
    ("RF", "LH"),
)


@dataclass
class LegState:
    """Estado de una pata."""
    q: np.ndarray
    detJ: float
    singular: bool
    reachable: bool


class AnymalGait2D:
    """
    Simulador cinemático simple de ANYmal visto desde arriba.

    El cuerpo se mueve en SE(2): (x, y, theta).
    Cada pata tiene:
      - ancla en el cuerpo (hip)
      - posición del pie en mundo
      - trayectoria de swing con elevación senoidal
      - IK geométrica cerrada
      - Jacobiano analítico y monitoreo de det(J)

    Notas:
    - No es dinámica realista; es una abstracción útil para la fase del reto.
    - La lógica está pensada para integrarse fácil con sim.py.
    """

    def __init__(
        self,
        pose: Tuple[float, float, float] = (0.8, 0.0, 0.0),
        l0: float = 0.0585,
        l1: float = 0.35,
        l2: float = 0.33,
        body_length: float = 0.70,
        body_width: float = 0.34,
        stance_height: float = -0.50,
        lateral_foot_offset: float = 0.040,
        gait_period: float = 0.80,
        step_height: float = 0.08,
        step_gain: float = 0.22,
        detj_threshold: float = 1e-3,
        nominal_speed: float = 0.55,
        nominal_yaw_gain: float = 1.8,
        max_yaw_rate: float = 0.9,
    ) -> None:
        self.pose = np.array(pose, dtype=float)  # x, y, theta
        self.l0 = float(l0)
        self.l1 = float(l1)
        self.l2 = float(l2)

        self.body_length = float(body_length)
        self.body_width = float(body_width)

        self.stance_height = float(stance_height)
        self.lateral_foot_offset = float(lateral_foot_offset)

        self.gait_period = float(gait_period)
        self.half_cycle = 0.5 * self.gait_period
        self.step_height = float(step_height)
        self.step_gain = float(step_gain)
        self.detj_threshold = float(detj_threshold)

        self.nominal_speed = float(nominal_speed)
        self.nominal_yaw_gain = float(nominal_yaw_gain)
        self.max_yaw_rate = float(max_yaw_rate)

        self.phase_time = 0.0
        self.swing_group_index = 0
        self.time = 0.0

        self.body_path: List[np.ndarray] = [self.pose[:2].copy()]
        self.goal_history: List[np.ndarray] = []

        # Anchors de cadera en el marco del cuerpo
        hx = 0.5 * self.body_length * 0.72
        hy = 0.5 * self.body_width
        self.hip_anchors_body: Dict[str, np.ndarray] = {
            "LF": np.array([+hx, +hy, 0.0]),
            "RF": np.array([+hx, -hy, 0.0]),
            "LH": np.array([-hx, +hy, 0.0]),
            "RH": np.array([-hx, -hy, 0.0]),
        }

        # Posición nominal de cada pie con respecto a su hip
        self.nominal_leg_local: Dict[str, np.ndarray] = {}
        self.nominal_body_foot: Dict[str, np.ndarray] = {}
        for leg in LEG_ORDER:
            s = LEG_SIGNS[leg]
            local = np.array([0.0, s * self.lateral_foot_offset, self.stance_height], dtype=float)
            self.nominal_leg_local[leg] = local
            self.nominal_body_foot[leg] = self.hip_anchors_body[leg] + local

        # Estado del pie en mundo
        self.feet_world: Dict[str, np.ndarray] = {}
        for leg in LEG_ORDER:
            self.feet_world[leg] = self.body_to_world(self.nominal_body_foot[leg])

        # Estado de swing
        self.swing_active: Dict[str, bool] = {leg: False for leg in LEG_ORDER}
        self.swing_start_world: Dict[str, np.ndarray] = {}
        self.swing_target_world: Dict[str, np.ndarray] = {}
        for leg in LEG_ORDER:
            self.swing_start_world[leg] = self.feet_world[leg].copy()
            self.swing_target_world[leg] = self.feet_world[leg].copy()

        # Iniciar con primer grupo en swing
        self._start_new_half_cycle(v_cmd=0.0, omega_cmd=0.0)

        # Logs
        self.logs = {
            "time": [],
            "x": [],
            "y": [],
            "theta": [],
            "phase": [],
            "goal_x": [],
            "goal_y": [],
            "min_detJ": [],
            "swing_group": [],
        }
        self.leg_logs = {
            leg: {
                "detJ": [],
                "q1": [],
                "q2": [],
                "q3": [],
                "singular": [],
                "reachable": [],
            }
            for leg in LEG_ORDER
        }

    # ---------------------------------------------------------------------
    # Transformaciones
    # ---------------------------------------------------------------------
    def rot2(self, theta: float) -> np.ndarray:
        """Matriz de rotación 2D."""
        c = math.cos(theta)
        s = math.sin(theta)
        return np.array([[c, -s], [s, c]], dtype=float)

    def body_to_world(self, p_body: np.ndarray) -> np.ndarray:
        """
        Lleva un punto del marco del cuerpo al mundo.
        p_body: [x, y, z]
        """
        p_body = np.asarray(p_body, dtype=float)
        xy = self.rot2(self.pose[2]) @ p_body[:2] + self.pose[:2]
        return np.array([xy[0], xy[1], p_body[2]], dtype=float)

    def world_to_body(self, p_world: np.ndarray) -> np.ndarray:
        """
        Lleva un punto del mundo al marco del cuerpo.
        p_world: [x, y, z]
        """
        p_world = np.asarray(p_world, dtype=float)
        xy = self.rot2(-self.pose[2]) @ (p_world[:2] - self.pose[:2])
        return np.array([xy[0], xy[1], p_world[2]], dtype=float)

    def hip_world(self, leg: str) -> np.ndarray:
        """Posición del hip de una pata en mundo."""
        return self.body_to_world(self.hip_anchors_body[leg])

    # ---------------------------------------------------------------------
    # Kinemática de pata
    # ---------------------------------------------------------------------
    def forward_kinematics_leg(self, q: np.ndarray, leg: str) -> np.ndarray:
        """
        FK de una pata en el marco del hip.

        Ecuaciones del enunciado:
          x = l1 sin q2 + l2 sin(q2 + q3)
          y = s l0 cos q1
          z = -l1 cos q2 - l2 cos(q2 + q3)
        """
        q1, q2, q3 = np.asarray(q, dtype=float)
        s = LEG_SIGNS[leg]
        x = self.l1 * math.sin(q2) + self.l2 * math.sin(q2 + q3)
        y = s * self.l0 * math.cos(q1)
        z = -self.l1 * math.cos(q2) - self.l2 * math.cos(q2 + q3)
        return np.array([x, y, z], dtype=float)

    def inverse_kinematics_leg(self, p_hip: np.ndarray, leg: str) -> Tuple[np.ndarray, bool]:
        """
        IK geométrica cerrada de la pata en el marco del hip.

        Convención elegida:
        - q3 se toma negativo para una rodilla "doblada" hacia atrás
        - q1 usa la rama principal de arccos

        Regresa:
          q, reachable
        """
        x, y, z = np.asarray(p_hip, dtype=float)
        s = LEG_SIGNS[leg]

        reachable = True

        # q1 desde el desplazamiento lateral
        denom = s * self.l0
        if abs(denom) < 1e-12:
            return np.zeros(3), False

        c1 = y / denom
        if c1 < -1.0 or c1 > 1.0:
            reachable = False
            c1 = float(np.clip(c1, -1.0, 1.0))
        q1 = math.acos(c1)

        # q2, q3 desde el problema planar x-z
        r2 = x * x + z * z
        c3 = (r2 - self.l1**2 - self.l2**2) / (2.0 * self.l1 * self.l2)
        if c3 < -1.0 or c3 > 1.0:
            reachable = False
            c3 = float(np.clip(c3, -1.0, 1.0))

        q3 = -math.acos(c3)
        q2 = math.atan2(x, -z) - math.atan2(self.l2 * math.sin(q3), self.l1 + self.l2 * math.cos(q3))

        return np.array([q1, q2, q3], dtype=float), reachable

    def jacobian_leg(self, q: np.ndarray, leg: str) -> np.ndarray:
        """
        Jacobiano analítico de la pata, según el enunciado.
        """
        q1, q2, q3 = np.asarray(q, dtype=float)
        s = LEG_SIGNS[leg]
        c23 = math.cos(q2 + q3)
        s23 = math.sin(q2 + q3)

        return np.array(
            [
                [0.0, self.l1 * math.cos(q2) + self.l2 * c23, self.l2 * c23],
                [-s * self.l0 * math.sin(q1), 0.0, 0.0],
                [0.0, self.l1 * math.sin(q2) + self.l2 * s23, self.l2 * s23],
            ],
            dtype=float,
        )

    def det_jacobian_leg(self, q: np.ndarray, leg: str) -> float:
        """Determinante del Jacobiano."""
        J = self.jacobian_leg(q, leg)
        return float(np.linalg.det(J))

    def leg_state(self, leg: str) -> LegState:
        """
        Calcula estado cinemático actual de una pata a partir del pie en mundo.
        """
        p_body = self.world_to_body(self.feet_world[leg])
        p_hip = p_body - self.hip_anchors_body[leg]
        q, reachable = self.inverse_kinematics_leg(p_hip, leg)
        detJ = self.det_jacobian_leg(q, leg)
        singular = abs(detJ) < self.detj_threshold
        return LegState(q=q, detJ=detJ, singular=singular, reachable=reachable)

    # ---------------------------------------------------------------------
    # Planeación de marcha
    # ---------------------------------------------------------------------
    def desired_command_to_goal(self, goal_xy: np.ndarray) -> Tuple[float, float]:
        """
        Control proporcional simple para mover el cuerpo hacia la meta.
        """
        goal_xy = np.asarray(goal_xy, dtype=float)
        delta = goal_xy - self.pose[:2]
        dist = float(np.linalg.norm(delta))

        if dist < 1e-6:
            return 0.0, 0.0

        desired_heading = math.atan2(delta[1], delta[0])
        heading_error = self.wrap_to_pi(desired_heading - self.pose[2])

        v_cmd = min(self.nominal_speed, 0.55 * dist)
        omega_cmd = np.clip(self.nominal_yaw_gain * heading_error, -self.max_yaw_rate, self.max_yaw_rate)

        # Si está muy desalineado, baja la velocidad de avance
        align = max(0.2, math.cos(heading_error))
        v_cmd *= align
        return v_cmd, float(omega_cmd)

    def _plan_touchdown_world(self, leg: str, v_cmd: float, omega_cmd: float) -> np.ndarray:
        """
        Planea el siguiente punto de apoyo en mundo para una pata en swing.

        Estrategia:
        - predice la pose del cuerpo al final del medio ciclo
        - usa la posición nominal del pie en ese cuerpo futuro
        - adelanta o atrasa un poco según v_cmd
        - si hay singularidad, acorta el paso
        """
        dt_pred = self.half_cycle
        theta_pred = self.pose[2] + omega_cmd * dt_pred
        travel_body = np.array([v_cmd * dt_pred, 0.0])
        travel_world = self.rot2(self.pose[2]) @ travel_body
        pose_pred = np.array([self.pose[0] + travel_world[0], self.pose[1] + travel_world[1], theta_pred])

        nominal = self.nominal_body_foot[leg].copy()
        nominal[0] += self.step_gain * v_cmd

        # Buscar un touchdown que no provoque singularidad fuerte
        scale = 1.0
        for _ in range(8):
            candidate_body = self.nominal_body_foot[leg].copy()
            candidate_body[0] += scale * self.step_gain * v_cmd

            R = np.array(
                [
                    [math.cos(pose_pred[2]), -math.sin(pose_pred[2])],
                    [math.sin(pose_pred[2]),  math.cos(pose_pred[2])],
                ],
                dtype=float,
            )
            world_xy = R @ candidate_body[:2] + pose_pred[:2]
            candidate_world = np.array([world_xy[0], world_xy[1], candidate_body[2]], dtype=float)

            # Revisar IK/detJ del candidato en el cuerpo futuro
            p_hip = candidate_body - self.hip_anchors_body[leg]
            q, reachable = self.inverse_kinematics_leg(p_hip, leg)
            detJ = abs(self.det_jacobian_leg(q, leg))
            if reachable and detJ >= self.detj_threshold * 5.0:
                return candidate_world
            scale *= 0.7

        # Si no encontró algo mejor, usa el nominal futuro
        R = np.array(
            [
                [math.cos(pose_pred[2]), -math.sin(pose_pred[2])],
                [math.sin(pose_pred[2]),  math.cos(pose_pred[2])],
            ],
            dtype=float,
        )
        world_xy = R @ nominal[:2] + pose_pred[:2]
        return np.array([world_xy[0], world_xy[1], nominal[2]], dtype=float)

    def _start_new_half_cycle(self, v_cmd: float, omega_cmd: float) -> None:
        """
        Arranca el grupo diagonal correspondiente.
        """
        active_group = SWING_GROUPS[self.swing_group_index]
        self.swing_active = {leg: (leg in active_group) for leg in LEG_ORDER}

        for leg in active_group:
            self.swing_start_world[leg] = self.feet_world[leg].copy()
            self.swing_target_world[leg] = self._plan_touchdown_world(leg, v_cmd=v_cmd, omega_cmd=omega_cmd)

    # ---------------------------------------------------------------------
    # Integración
    # ---------------------------------------------------------------------
    def update(self, dt: float, goal_xy: np.ndarray) -> Dict[str, LegState]:
        """
        Avanza un paso de simulación.

        Regresa un diccionario con el estado de cada pata.
        """
        dt = float(dt)
        goal_xy = np.asarray(goal_xy, dtype=float)
        self.goal_history.append(goal_xy.copy())

        v_cmd, omega_cmd = self.desired_command_to_goal(goal_xy)

        # Actualizar fase
        previous_phase_time = self.phase_time
        self.phase_time += dt
        self.time += dt

        # Si termina el medio ciclo, cambia grupo swing
        while self.phase_time >= self.half_cycle:
            self.phase_time -= self.half_cycle
            self.swing_group_index = 1 - self.swing_group_index
            self._start_new_half_cycle(v_cmd=v_cmd, omega_cmd=omega_cmd)

        alpha = self.phase_time / self.half_cycle if self.half_cycle > 1e-12 else 0.0
        alpha = float(np.clip(alpha, 0.0, 1.0))

        # Integración simple del cuerpo
        theta_mid = self.pose[2] + 0.5 * omega_cmd * dt
        self.pose[0] += v_cmd * math.cos(theta_mid) * dt
        self.pose[1] += v_cmd * math.sin(theta_mid) * dt
        self.pose[2] = self.wrap_to_pi(self.pose[2] + omega_cmd * dt)

        # Actualizar pies
        for leg in LEG_ORDER:
            if self.swing_active[leg]:
                p0 = self.swing_start_world[leg]
                pf = self.swing_target_world[leg]
                p = (1.0 - alpha) * p0 + alpha * pf
                p[2] = self.stance_height + self.step_height * math.sin(math.pi * alpha)
                self.feet_world[leg] = p
            else:
                # stance: el pie se queda pegado al mundo
                self.feet_world[leg][2] = self.stance_height

        self.body_path.append(self.pose[:2].copy())

        # Estado y logs
        states: Dict[str, LegState] = {leg: self.leg_state(leg) for leg in LEG_ORDER}
        min_det = min(abs(states[leg].detJ) for leg in LEG_ORDER)

        self.logs["time"].append(self.time)
        self.logs["x"].append(self.pose[0])
        self.logs["y"].append(self.pose[1])
        self.logs["theta"].append(self.pose[2])
        self.logs["phase"].append(alpha + self.swing_group_index)
        self.logs["goal_x"].append(goal_xy[0])
        self.logs["goal_y"].append(goal_xy[1])
        self.logs["min_detJ"].append(min_det)
        self.logs["swing_group"].append(self.swing_group_index)

        for leg in LEG_ORDER:
            st = states[leg]
            self.leg_logs[leg]["detJ"].append(st.detJ)
            self.leg_logs[leg]["q1"].append(st.q[0])
            self.leg_logs[leg]["q2"].append(st.q[1])
            self.leg_logs[leg]["q3"].append(st.q[2])
            self.leg_logs[leg]["singular"].append(st.singular)
            self.leg_logs[leg]["reachable"].append(st.reachable)

        return states

    # ---------------------------------------------------------------------
    # Utilidades de dibujo / métricas
    # ---------------------------------------------------------------------
    def body_corners_world(self) -> np.ndarray:
        """Esquinas del rectángulo del cuerpo en mundo, para dibujar."""
        hx = 0.5 * self.body_length
        hy = 0.5 * self.body_width
        corners_body = np.array(
            [
                [+hx, +hy],
                [+hx, -hy],
                [-hx, -hy],
                [-hx, +hy],
                [+hx, +hy],
            ],
            dtype=float,
        )
        corners_world = (self.rot2(self.pose[2]) @ corners_body.T).T + self.pose[:2]
        return corners_world

    def feet_xy(self) -> Dict[str, np.ndarray]:
        """Proyección XY de los pies."""
        return {leg: self.feet_world[leg][:2].copy() for leg in LEG_ORDER}

    def reached_goal(self, goal_xy: np.ndarray, tol: float = 0.15) -> bool:
        """Criterio de llegada."""
        goal_xy = np.asarray(goal_xy, dtype=float)
        return float(np.linalg.norm(goal_xy - self.pose[:2])) <= tol

    @staticmethod
    def wrap_to_pi(angle: float) -> float:
        """Envuelve ángulo a [-pi, pi]."""
        return (angle + math.pi) % (2.0 * math.pi) - math.pi
