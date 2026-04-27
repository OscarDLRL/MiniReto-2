"""
husky_pusher.py

Modelo 2D sencillo del Husky A200 para la fase 1 del mini reto.
El objetivo es detectar/empujar 3 cajas grandes fuera del corredor y dejar
el paso libre para el ANYmal.

No es una simulación dinámica realista; es una abstracción cinemática con:
- skid-steer simplificado
- compensación de deslizamiento mediante factor s
- máquina de estados local por caja
- "LiDAR" 2D muy simple a partir de los obstáculos conocidos
- logs de velocidades comandadas y medidas
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class BoxObstacle:
    """Caja grande a empujar fuera del corredor."""

    center: np.ndarray
    size: np.ndarray
    push_dir: int
    cleared: bool = False

    @property
    def x(self) -> float:
        return float(self.center[0])

    @property
    def y(self) -> float:
        return float(self.center[1])

    @property
    def width(self) -> float:
        return float(self.size[0])

    @property
    def height(self) -> float:
        return float(self.size[1])

    def bounds(self) -> Tuple[float, float, float, float]:
        """Retorna xmin, xmax, ymin, ymax."""
        half = 0.5 * self.size
        return (
            float(self.center[0] - half[0]),
            float(self.center[0] + half[0]),
            float(self.center[1] - half[1]),
            float(self.center[1] + half[1]),
        )

    def is_outside_corridor(self, x0: float, x1: float, y0: float, y1: float) -> bool:
        """Verdadero si la caja ya no intersecta el rectángulo del corredor."""
        xmin, xmax, ymin, ymax = self.bounds()
        overlap_x = not (xmax < x0 or xmin > x1)
        overlap_y = not (ymax < y0 or ymin > y1)
        return not (overlap_x and overlap_y)


class HuskyPusher2D:
    """
    Simulador cinemático simple del Husky A200 para empujar cajas.

    Flujo por caja:
      1) navegar a una pose previa de empuje
      2) alinearse con la dirección de empuje
      3) empujar hasta sacar la caja del corredor
      4) retirarse un poco
      5) continuar con la siguiente caja
    """

    def __init__(
        self,
        pose: Tuple[float, float, float] = (0.7, -1.9, 0.0),
        wheel_radius: float = 0.1651,
        track_width: float = 0.555,
        body_length: float = 0.95,
        body_width: float = 0.62,
        slip_factor: float = 0.85,
        corridor: Tuple[float, float, float, float] = (2.0, 8.0, -1.0, 1.0),
        parking_pose: Tuple[float, float, float] = (1.10, -2.35, 0.0),
        lidar_range: float = 4.0,
        max_v_cmd: float = 0.75,
        max_w_cmd: float = 1.25,
    ) -> None:
        self.pose = np.array(pose, dtype=float)
        self.wheel_radius = float(wheel_radius)
        self.track_width = float(track_width)
        self.body_length = float(body_length)
        self.body_width = float(body_width)
        self.slip_factor = float(slip_factor)
        self.corridor = tuple(float(v) for v in corridor)
        self.parking_pose = np.array(parking_pose, dtype=float)
        self.lidar_range = float(lidar_range)
        self.max_v_cmd = float(max_v_cmd)
        self.max_w_cmd = float(max_w_cmd)

        self.time = 0.0
        self.path = [self.pose[:2].copy()]

        self.current_box_index = 0
        self.state = "goto_prepush"
        self.state_time = 0.0
        self.finished = False
        self.parked_aside = False
        self.last_contact = False
        self.retreat_goal_pose: Optional[np.ndarray] = None

        self.logs = {
            "time": [],
            "x": [],
            "y": [],
            "theta": [],
            "v_cmd": [],
            "w_cmd": [],
            "v_meas": [],
            "w_meas": [],
            "left_w_cmd": [],
            "right_w_cmd": [],
            "left_w_meas": [],
            "right_w_meas": [],
            "state": [],
            "box_index": [],
            "contact": [],
        }

    # ------------------------------------------------------------------
    # Utilidades geométricas
    # ------------------------------------------------------------------
    @staticmethod
    def wrap_to_pi(angle: float) -> float:
        return (angle + math.pi) % (2.0 * math.pi) - math.pi

    def rot2(self, theta: float) -> np.ndarray:
        c = math.cos(theta)
        s = math.sin(theta)
        return np.array([[c, -s], [s, c]], dtype=float)

    def body_corners_world(self) -> np.ndarray:
        """Esquinas del Husky para dibujo."""
        hx = 0.5 * self.body_length
        hy = 0.5 * self.body_width
        corners_body = np.array(
            [[+hx, +hy], [+hx, -hy], [-hx, -hy], [-hx, +hy], [+hx, +hy]],
            dtype=float,
        )
        return (self.rot2(self.pose[2]) @ corners_body.T).T + self.pose[:2]

    def front_point(self) -> np.ndarray:
        """Punto frontal del robot, útil para verificar contacto."""
        return self.pose[:2] + self.rot2(self.pose[2]) @ np.array([0.5 * self.body_length, 0.0])

    # ------------------------------------------------------------------
    # Modelo cinemático skid-steer
    # ------------------------------------------------------------------
    def body_twist_to_wheels(self, v: float, w: float) -> Tuple[float, float]:
        """Convierte v,w en velocidades angulares de lado derecho/izquierdo."""
        omega_r = (v + 0.5 * self.track_width * w) / self.wheel_radius
        omega_l = (v - 0.5 * self.track_width * w) / self.wheel_radius
        return float(omega_l), float(omega_r)

    def wheels_to_body_twist_measured(self, omega_l_cmd: float, omega_r_cmd: float) -> Tuple[float, float, float, float]:
        """
        Aplica deslizamiento al avance lineal y deja casi intacta la rotación.
        Esto sigue la idea de que el factor s afecta sobre todo la traslación.
        """
        v_cmd = 0.5 * self.wheel_radius * (omega_r_cmd + omega_l_cmd)
        w_cmd = self.wheel_radius * (omega_r_cmd - omega_l_cmd) / self.track_width

        v_meas = self.slip_factor * v_cmd
        w_meas = w_cmd

        omega_r_meas = (v_meas + 0.5 * self.track_width * w_meas) / self.wheel_radius
        omega_l_meas = (v_meas - 0.5 * self.track_width * w_meas) / self.wheel_radius
        return float(v_meas), float(w_meas), float(omega_l_meas), float(omega_r_meas)

    def integrate(self, dt: float, v_meas: float, w_meas: float) -> None:
        theta_mid = self.pose[2] + 0.5 * w_meas * dt
        self.pose[0] += v_meas * math.cos(theta_mid) * dt
        self.pose[1] += v_meas * math.sin(theta_mid) * dt
        self.pose[2] = self.wrap_to_pi(self.pose[2] + w_meas * dt)
        self.path.append(self.pose[:2].copy())

    # ------------------------------------------------------------------
    # Planeación local simple
    # ------------------------------------------------------------------
    def _current_box(self, boxes: List[BoxObstacle]) -> Optional[BoxObstacle]:
        if self.current_box_index >= len(boxes):
            self.finished = True
            return None
        return boxes[self.current_box_index]

    def _push_heading(self, box: BoxObstacle) -> float:
        return math.pi / 2.0 if box.push_dir > 0 else -math.pi / 2.0

    def _prepush_pose(self, box: BoxObstacle) -> np.ndarray:
        """Pose desde la cual el Husky inicia el empuje de la caja."""
        heading = self._push_heading(box)
        margin = 0.16
        offset_y = box.push_dir * (0.5 * box.height + 0.5 * self.body_length + margin)
        # Para empujar hacia arriba, se coloca debajo; hacia abajo, arriba.
        pre_y = box.y - offset_y
        return np.array([box.x, pre_y, heading], dtype=float)

    def _retreat_goal(self, box: BoxObstacle) -> np.ndarray:
        heading = self._push_heading(box)
        retreat = 0.55
        return np.array([self.pose[0], self.pose[1] - box.push_dir * retreat, heading], dtype=float)

    def _goto_pose_controller(self, goal_pose: np.ndarray) -> Tuple[float, float]:
        """Control proporcional sencillo a una pose objetivo."""
        dx = goal_pose[0] - self.pose[0]
        dy = goal_pose[1] - self.pose[1]
        dist = float(np.hypot(dx, dy))

        desired_heading = math.atan2(dy, dx)
        heading_error = self.wrap_to_pi(desired_heading - self.pose[2])
        final_heading_error = self.wrap_to_pi(goal_pose[2] - self.pose[2])

        if dist > 0.18:
            v_cmd = np.clip(0.9 * dist, -self.max_v_cmd, self.max_v_cmd)
            v_cmd *= max(0.2, math.cos(heading_error))
            w_cmd = np.clip(2.4 * heading_error, -self.max_w_cmd, self.max_w_cmd)
        else:
            v_cmd = 0.0
            w_cmd = np.clip(2.0 * final_heading_error, -self.max_w_cmd, self.max_w_cmd)

        return float(v_cmd), float(w_cmd)

    def _ready_for_push(self, box: BoxObstacle) -> bool:
        prep = self._prepush_pose(box)
        pos_ok = np.linalg.norm(self.pose[:2] - prep[:2]) < 0.18
        ang_ok = abs(self.wrap_to_pi(self.pose[2] - prep[2])) < 0.12
        return bool(pos_ok and ang_ok)

    def _contact_with_box(self, box: BoxObstacle) -> bool:
        front = self.front_point()
        xmin, xmax, ymin, ymax = box.bounds()
        heading_ok = abs(self.wrap_to_pi(self.pose[2] - self._push_heading(box))) < 0.18

        if box.push_dir > 0:
            close_y = abs(front[1] - ymin) < 0.18
        else:
            close_y = abs(front[1] - ymax) < 0.18

        inside_x = (xmin - 0.15) <= front[0] <= (xmax + 0.15)
        return bool(heading_ok and close_y and inside_x)

    def _push_box_if_contact(self, box: BoxObstacle, dt: float, v_meas: float) -> bool:
        """Si hay contacto, mueve la caja junto con el Husky."""
        contact = self._contact_with_box(box)
        if not contact:
            return False

        # Solo mover sobre el eje y para sacar la caja del corredor.
        dy = box.push_dir * max(0.0, v_meas) * dt
        box.center[1] += dy

        x0, x1, y0, y1 = self.corridor
        if box.is_outside_corridor(x0, x1, y0, y1):
            box.cleared = True
        return True

    def reached_parking_pose(self) -> bool:
        """Verdadero si el Husky ya está en su pose lateral."""
        pos_ok = np.linalg.norm(self.pose[:2] - self.parking_pose[:2]) < 0.22
        ang_ok = abs(self.wrap_to_pi(self.pose[2] - self.parking_pose[2])) < 0.18
        return bool(pos_ok and ang_ok)

    def lidar_scan(self, boxes: List[BoxObstacle], num_beams: int = 90) -> np.ndarray:
        """
        LiDAR 2D super simple.
        Regresa puntos detectados sobre el perímetro de las cajas que estén dentro de rango.
        """
        pts = []
        origin = self.pose[:2]
        for box in boxes:
            xmin, xmax, ymin, ymax = box.bounds()
            xs = np.linspace(xmin, xmax, 8)
            ys = np.linspace(ymin, ymax, 8)
            perimeter = []
            perimeter.extend([(x, ymin) for x in xs])
            perimeter.extend([(x, ymax) for x in xs])
            perimeter.extend([(xmin, y) for y in ys])
            perimeter.extend([(xmax, y) for y in ys])
            for p in perimeter:
                p = np.array(p, dtype=float)
                if np.linalg.norm(p - origin) <= self.lidar_range:
                    pts.append(p)
        if len(pts) == 0:
            return np.zeros((0, 2), dtype=float)
        return np.array(pts, dtype=float)


    def simulate_lidar(self, boxes: List[BoxObstacle]) -> np.ndarray:
        """Alias conveniente para la visualización."""
        return self.lidar_scan(boxes)

    # ------------------------------------------------------------------
    # Loop principal
    # ------------------------------------------------------------------
    def update(self, dt: float, boxes: List[BoxObstacle]) -> dict:
        dt = float(dt)
        self.time += dt
        self.state_time += dt

        box = self._current_box(boxes)
        if box is None:
            if not self.parked_aside:
                self.state = "park_aside"
                v_cmd, w_cmd = self._goto_pose_controller(self.parking_pose)
                if self.reached_parking_pose():
                    self.pose = self.parking_pose.copy()
                    self.path.append(self.pose[:2].copy())
                    self.parked_aside = True
                    self.finished = True
                    v_cmd = w_cmd = 0.0
            else:
                self.finished = True
                v_cmd = w_cmd = 0.0
        else:
            if self.state == "goto_prepush":
                goal = self._prepush_pose(box)
                v_cmd, w_cmd = self._goto_pose_controller(goal)
                if self._ready_for_push(box):
                    self.state = "push"
                    self.state_time = 0.0

            elif self.state == "push":
                # Avanza hacia un punto más allá del corredor manteniendo la orientación
                # de empuje. Esto hace la fase de empuje bastante estable en 2D.
                y_goal = (self.corridor[3] + 0.95) if box.push_dir > 0 else (self.corridor[2] - 0.95)
                push_goal = np.array([box.x, y_goal, self._push_heading(box)], dtype=float)
                v_cmd, w_cmd = self._goto_pose_controller(push_goal)
                v_cmd = max(0.28, v_cmd)

                # Si la caja ya salió, ir a retiro.
                x0, x1, y0, y1 = self.corridor
                if box.is_outside_corridor(x0, x1, y0, y1):
                    box.cleared = True
                    self.state = "retreat"
                    self.state_time = 0.0
                    self.retreat_goal_pose = self._retreat_goal(box)

            elif self.state == "retreat":
                goal = self.retreat_goal_pose if self.retreat_goal_pose is not None else self._retreat_goal(box)
                heading_error = self.wrap_to_pi(self._push_heading(box) - self.pose[2])
                v_cmd = -0.35
                w_cmd = np.clip(2.0 * heading_error, -self.max_w_cmd, self.max_w_cmd)
                if np.linalg.norm(self.pose[:2] - goal[:2]) < 0.12:
                    self.current_box_index += 1
                    self.state = "goto_prepush"
                    self.state_time = 0.0
                    self.retreat_goal_pose = None
                    v_cmd = 0.0
                    w_cmd = 0.0

            else:
                v_cmd = 0.0
                w_cmd = 0.0

        # Limitar y pasar a ruedas.
        v_cmd = float(np.clip(v_cmd, -self.max_v_cmd, self.max_v_cmd))
        w_cmd = float(np.clip(w_cmd, -self.max_w_cmd, self.max_w_cmd))
        omega_l_cmd, omega_r_cmd = self.body_twist_to_wheels(v_cmd, w_cmd)
        v_meas, w_meas, omega_l_meas, omega_r_meas = self.wheels_to_body_twist_measured(omega_l_cmd, omega_r_cmd)

        # Integración y empuje.
        self.integrate(dt, v_meas, w_meas)
        contact = False
        active_box = self._current_box(boxes)
        if active_box is not None and self.state == "push":
            contact = self._push_box_if_contact(active_box, dt, v_meas)
            if active_box.cleared:
                self.state = "retreat"
                self.state_time = 0.0
                self.retreat_goal_pose = self._retreat_goal(active_box)

        self.last_contact = contact

        # Si ya no quedan cajas, el robot pasa a estacionarse a un lado.
        if self._current_box(boxes) is None and self.parked_aside:
            self.finished = True

        # Logs.
        self.logs["time"].append(self.time)
        self.logs["x"].append(self.pose[0])
        self.logs["y"].append(self.pose[1])
        self.logs["theta"].append(self.pose[2])
        self.logs["v_cmd"].append(v_cmd)
        self.logs["w_cmd"].append(w_cmd)
        self.logs["v_meas"].append(v_meas)
        self.logs["w_meas"].append(w_meas)
        self.logs["left_w_cmd"].append(omega_l_cmd)
        self.logs["right_w_cmd"].append(omega_r_cmd)
        self.logs["left_w_meas"].append(omega_l_meas)
        self.logs["right_w_meas"].append(omega_r_meas)
        self.logs["state"].append(self.state)
        self.logs["box_index"].append(self.current_box_index)
        self.logs["contact"].append(contact)

        return {
            "state": self.state,
            "current_box": self.current_box_index,
            "finished": self.finished,
            "contact": contact,
            "lidar": self.lidar_scan(boxes),
            "v_cmd": v_cmd,
            "w_cmd": w_cmd,
            "v_meas": v_meas,
            "w_meas": w_meas,
        }
