"""
coordinator.py

Máquina de estados principal del mini reto.

Orquesta las tres fases pedidas en el PDF:
1) Husky despeja el corredor
2) ANYmal transporta 3 PuzzleBots sobre su dorso, llega a la meta y se orilla
3) Los PuzzleBots se despliegan por un costado y apilan las cajas con time-slotting
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, List, Tuple

import numpy as np

from anymal_gait import AnymalGait2D, LEG_ORDER
from husky_pusher import BoxObstacle, HuskyPusher2D
from puzzlebot_arm import PuzzleBot2D, SmallBox, RFNavigator


DT_DEFAULT = 0.05
CORRIDOR = (2.0, 8.0, -1.0, 1.0)
PARKING_POSE = (1.10, -2.35, 0.0)
ANYMAL_DEST = np.array([11.0, 3.6], dtype=float)
ANYMAL_SIDE_PARK_POSE = np.array([12.02, 3.46, 0.0], dtype=float)
ANYMAL_TRANSPORT_WAYPOINTS = [
    np.array([2.0, 0.0], dtype=float),
    np.array([8.3, 0.0], dtype=float),
    ANYMAL_DEST.copy(),
]
STACK_GOAL = np.array([10.85, 4.10], dtype=float)
STACK_SEQUENCE = ["C", "B", "A"]
PB_ASSIGNMENTS = {"PB1": "C", "PB2": "B", "PB3": "A"}
DEPLOY_LANE_X = 11.45


@dataclass
class MissionSnapshot:
    """Resumen compacto del estado actual de la misión."""

    phase: str
    phase_index: int
    success_flags: Dict[str, bool]
    stack_order: List[str]
    anymal_min_detj: float
    anymal_goal_error: float
    collision_violations: int


class MissionCoordinator:
    """Orquestador principal de todo el escenario."""

    def __init__(self, dt: float = DT_DEFAULT) -> None:
        self.dt = float(dt)
        self.time = 0.0

        self.big_boxes = self._make_big_boxes()
        self.small_boxes = self._make_small_boxes()
        puzzlebotnav = RFNavigator(max_objects=10)
        puzzlebotnav.train(n_samples=10000)


        self.husky = HuskyPusher2D(corridor=CORRIDOR, parking_pose=PARKING_POSE)
        self.anymal = AnymalGait2D(
            l0=0.10,
            max_yaw_rate=0.40,
            nominal_yaw_gain=0.90,
            nominal_speed=0.45,
        )
        self.phase = "HUSKY_CLEAR"
        self.phase_order = [
            "HUSKY_CLEAR",
            "ANYMAL_TRANSPORT",
            "ANYMAL_SIDE_PARK",
            "DEPLOY_PUZZLEBOTS",
            "STACK_SEQUENCE",
            "COMPLETE",
        ]
        self.anymal_wp_idx = 0

        self.payload_offsets_body = {
            "PB1": np.array([-0.10, 0.08], dtype=float),
            "PB2": np.array([0.05, 0.00], dtype=float),
            "PB3": np.array([0.18, -0.08], dtype=float),
        }
        self.dismount_poses = {
            "PB1": np.array([11.62, 3.12, math.pi], dtype=float),
            "PB2": np.array([11.62, 3.44, math.pi], dtype=float),
            "PB3": np.array([11.62, 3.76, math.pi], dtype=float),
        }
        self.safe_poses = {
            "PB1": np.array([12.55, 2.55, math.pi], dtype=float),
            "PB2": np.array([12.55, 4.05, math.pi], dtype=float),
            "PB3": np.array([12.55, 5.15, math.pi], dtype=float),
        }
        self.puzzlebots: List[PuzzleBot2D] = [
            PuzzleBot2D(name=name, assigned_box=box_label, safe_pose=tuple(self.safe_poses[name]))
            for name, box_label in PB_ASSIGNMENTS.items()
        ]
        for pb in self.puzzlebots:
            pb.attach_rf_navigator(puzzlebotnav)

        self.pb_by_box = {pb.assigned_box: pb for pb in self.puzzlebots}

        self._update_payload_mounts()

        self.deploy_index = 0
        self.stack_index = 0
        self.completed_stack_order: List[str] = []
        self.collision_violations = 0
        self.min_distance_history: List[float] = []
        self.status_message = ""
        self.anymal_arrival_recorded = False

        self.success_flags = {
            "husky_corridor_clear": False,
            "husky_parked_aside": False,
            "anymal_arrived": False,
            "anymal_parked_at_side": False,
            "anymal_singularity_ok": True,
            "puzzlebots_collision_free": True,
            "stack_order_ok": False,
            "stack_complete": False,
        }

        self.logs = {
            "time": [],
            "phase": [],
            "phase_index": [],
            "collision_violations": [],
            "min_pb_distance": [],
            "stack_size": [],
            "anymal_goal_error": [],
            "anymal_min_detJ": [],
        }

    def _make_big_boxes(self) -> List[BoxObstacle]:
        return [
            BoxObstacle(center=np.array([3.4, 0.0], dtype=float), size=np.array([0.42, 0.54], dtype=float), push_dir=+1),
            BoxObstacle(center=np.array([4.8, 0.0], dtype=float), size=np.array([0.42, 0.54], dtype=float), push_dir=-1),
            BoxObstacle(center=np.array([6.2, 0.0], dtype=float), size=np.array([0.42, 0.54], dtype=float), push_dir=+1),
        ]

    def _make_small_boxes(self) -> Dict[str, SmallBox]:
        return {
            "A": SmallBox(label="A", color="tab:blue", pickup_xy=np.array([10.10, 2.72], dtype=float)),
            "B": SmallBox(label="B", color="tab:orange", pickup_xy=np.array([10.65, 2.72], dtype=float)),
            "C": SmallBox(label="C", color="tab:green", pickup_xy=np.array([11.18, 2.72], dtype=float)),
        }

    def _update_payload_mounts(self) -> None:
        """Mantiene a los PuzzleBots montados sobre el ANYmal mientras viajan."""
        R = self.anymal.rot2(self.anymal.pose[2])
        for pb in self.puzzlebots:
            offset = self.payload_offsets_body[pb.name]
            xy = R @ offset + self.anymal.pose[:2]
            pose = np.array([xy[0], xy[1], self.anymal.pose[2]], dtype=float)
            if not pb.deployed:
                pb.mount_on_anymal(pose)

    def _phase_idx(self) -> int:
        return self.phase_order.index(self.phase)

    def _compute_puzzlebot_collisions(self) -> None:
        centers = [pb.pose[:2] for pb in self.puzzlebots if pb.deployed]
        if len(centers) < 2:
            self.min_distance_history.append(np.inf)
            return

        min_dist = np.inf
        for i in range(len(centers)):
            for j in range(i + 1, len(centers)):
                d = float(np.linalg.norm(centers[i] - centers[j]))
                min_dist = min(min_dist, d)
                if d < 0.24:
                    self.collision_violations += 1
        self.min_distance_history.append(min_dist)
        self.success_flags["puzzlebots_collision_free"] = self.collision_violations == 0

    def _get_active_deploy_pb(self) -> PuzzleBot2D:
        return self.puzzlebots[self.deploy_index]

    def _get_active_stack_box_label(self) -> str:
        return STACK_SEQUENCE[self.stack_index]

    def _get_active_stack_pb(self) -> PuzzleBot2D:
        return self.pb_by_box[self._get_active_stack_box_label()]

    def _all_small_boxes_stacked(self) -> bool:
        return all(box.placed for box in self.small_boxes.values())

    def anymal_keepout_circles(self) -> List[Tuple[np.ndarray, float]]:
        circles: List[Tuple[np.ndarray, float]] = [(self.anymal.pose[:2].copy(), 0.62)]
        for leg in LEG_ORDER:
            circles.append((self.anymal.feet_world[leg][:2].copy(), 0.17))
        return circles

    def _update_status_message(self, min_detj: float, anymal_goal_error: float) -> None:
        current_box_label = STACK_SEQUENCE[self.stack_index] if self.stack_index < len(STACK_SEQUENCE) else "-"
        self.status_message = (
            f"fase = {self.phase}\n"
            f"t = {self.time:5.2f} s\n"
            f"Husky parked = {self.husky.parked_aside}\n"
            f"ANYmal wp = {min(self.anymal_wp_idx + 1, len(ANYMAL_TRANSPORT_WAYPOINTS))}/{len(ANYMAL_TRANSPORT_WAYPOINTS)}\n"
            f"ANYmal err meta = {anymal_goal_error:.3f} m\n"
            f"ANYmal min|detJ| = {min_detj:.5f}\n"
            f"PB deploy = {sum(pb.deployed for pb in self.puzzlebots)}/{len(self.puzzlebots)}\n"
            f"Caja activa = {current_box_label}\n"
            f"Stack = {'-'.join(self.completed_stack_order) if self.completed_stack_order else '(vacío)'}\n"
            f"Violaciones PB = {self.collision_violations}"
        )

    def get_close_objects(
        self,
        pb: PuzzleBot2D,
        radius: float = 1.5,
    ) -> List[np.ndarray]:
        """
        Return XY positions of all objects within `radius` meters of `pb`.

        Includes: other PuzzleBots, ANYmal body + feet, big boxes, small boxes.
        """
        origin = pb.pose[:2]
        close: List[np.ndarray] = []

        # Other PuzzleBots
        for other in self.puzzlebots:
            if other is pb:
                continue
            pos = other.pose[:2]
            if np.linalg.norm(pos - origin) <= radius:
                close.append(pos.copy())

        # ANYmal body center + each foot
        for center, _ in self.anymal_keepout_circles():
            if np.linalg.norm(center - origin) <= radius:
                close.append(center.copy())

        # Big boxes (Husky obstacles)
        for box in self.big_boxes:
            pos = box.center[:2]
            if np.linalg.norm(pos - origin) <= radius:
                close.append(pos.copy())

        # Small boxes (only if not yet placed / being carried)
        for box in self.small_boxes.values():
            if box.world_xy is not None and box.label !=  self._get_active_stack_box_label():
                pos = np.asarray(box.world_xy[:2])
                if np.linalg.norm(pos - origin) <= radius:
                    close.append(pos.copy())

        return close

    def update(self) -> MissionSnapshot:
        """Avanza una muestra de la misión completa."""
        self.time += self.dt

        min_detj = self.anymal.logs["min_detJ"][-1] if self.anymal.logs["min_detJ"] else np.inf
        anymal_goal_error = float(np.linalg.norm(ANYMAL_DEST - self.anymal.pose[:2]))

        if self.phase == "HUSKY_CLEAR":
            self.husky.update(self.dt, self.big_boxes)
            self._update_payload_mounts()
            if all(box.cleared for box in self.big_boxes):
                self.success_flags["husky_corridor_clear"] = True
            if self.husky.parked_aside:
                self.success_flags["husky_parked_aside"] = True
                self.phase = "ANYMAL_TRANSPORT"
            for pb in self.puzzlebots:
                pb.update_idle_log(self.time)

        elif self.phase == "ANYMAL_TRANSPORT":
            current_goal = ANYMAL_TRANSPORT_WAYPOINTS[self.anymal_wp_idx]
            states = self.anymal.update(self.dt, current_goal)
            min_detj = min(abs(states[leg].detJ) for leg in LEG_ORDER)
            anymal_goal_error = float(np.linalg.norm(ANYMAL_DEST - self.anymal.pose[:2]))
            self.success_flags["anymal_singularity_ok"] = self.success_flags["anymal_singularity_ok"] and (min_detj >= self.anymal.detj_threshold)
            self._update_payload_mounts()

            goal_tol = 0.14 if self.anymal_wp_idx == len(ANYMAL_TRANSPORT_WAYPOINTS) - 1 else 0.22
            if self.anymal.reached_goal(current_goal, tol=goal_tol):
                if self.anymal_wp_idx < len(ANYMAL_TRANSPORT_WAYPOINTS) - 1:
                    self.anymal_wp_idx += 1
                else:
                    self.success_flags["anymal_arrived"] = anymal_goal_error < 0.15
                    self.anymal_arrival_recorded = True
                    self.phase = "ANYMAL_SIDE_PARK"
            for pb in self.puzzlebots:
                pb.update_idle_log(self.time)

        elif self.phase == "ANYMAL_SIDE_PARK":
            park_goal = ANYMAL_SIDE_PARK_POSE[:2]
            states = self.anymal.update(self.dt, park_goal)
            min_detj = min(abs(states[leg].detJ) for leg in LEG_ORDER)
            self.success_flags["anymal_singularity_ok"] = self.success_flags["anymal_singularity_ok"] and (min_detj >= self.anymal.detj_threshold)
            self._update_payload_mounts()
            if self.anymal.reached_goal(park_goal, tol=0.16):
                self.success_flags["anymal_parked_at_side"] = True
                self.phase = "DEPLOY_PUZZLEBOTS"
            for pb in self.puzzlebots:
                pb.update_idle_log(self.time)

        elif self.phase == "DEPLOY_PUZZLEBOTS":
            active_pb = self._get_active_deploy_pb()
            close_objects = self.get_close_objects(active_pb, radius=1.5)  
            active_pb.update_deployment(
                self.dt,
                self.dismount_poses[active_pb.name],
                self.safe_poses[active_pb.name],
                self.time,
                close_objects,                                              
            )
            for pb in self.puzzlebots:
                if pb is not active_pb:
                    pb.update_idle_log(self.time)
            self._compute_puzzlebot_collisions()
            if active_pb.deployed:
                self.deploy_index += 1
                if self.deploy_index >= len(self.puzzlebots):
                    self.phase = "STACK_SEQUENCE"
                    print("iniciando stack")

        elif self.phase == "STACK_SEQUENCE":
            box_label = self._get_active_stack_box_label()
            active_pb = self._get_active_stack_pb()
            close_objects = self.get_close_objects(active_pb, radius=1.5)  
            box = self.small_boxes[box_label]
            active_pb.update_task(self.dt, self.time, box, STACK_GOAL, self.stack_index, lane_x=DEPLOY_LANE_X, close_objects=close_objects)
            if box.carried_by is not None:
                box.world_xy = active_pb.grasp_point_world().copy()

            for pb in self.puzzlebots:
                if pb is not active_pb:
                    pb.update_idle_log(self.time)
            self._compute_puzzlebot_collisions()
            if active_pb.task_complete and box.placed:
                if box.label not in self.completed_stack_order:
                    self.completed_stack_order.append(box.label)
                self.stack_index += 1
                print("Puzzlebot Finished!")
                if self.stack_index >= len(STACK_SEQUENCE):
                    self.success_flags["stack_complete"] = self._all_small_boxes_stacked()
                    self.success_flags["stack_order_ok"] = self.completed_stack_order == STACK_SEQUENCE
                    self.phase = "COMPLETE"
                    print("Al Puzzlebots Done!")

        elif self.phase == "COMPLETE":
            self.success_flags["stack_complete"] = self._all_small_boxes_stacked()
            self.success_flags["stack_order_ok"] = self.completed_stack_order == STACK_SEQUENCE
            self.success_flags["puzzlebots_collision_free"] = self.collision_violations == 0
            for pb in self.puzzlebots:
                pb.update_idle_log(self.time)
            self._compute_puzzlebot_collisions()

        if self.anymal.logs["min_detJ"]:
            min_detj = float(self.anymal.logs["min_detJ"][-1])
        anymal_goal_error = float(np.linalg.norm(ANYMAL_DEST - self.anymal.pose[:2]))
        self._update_status_message(min_detj=min_detj, anymal_goal_error=anymal_goal_error)

        self.logs["time"].append(self.time)
        self.logs["phase"].append(self.phase)
        self.logs["phase_index"].append(self._phase_idx())
        self.logs["collision_violations"].append(self.collision_violations)
        self.logs["min_pb_distance"].append(self.min_distance_history[-1] if self.min_distance_history else np.inf)
        self.logs["stack_size"].append(len(self.completed_stack_order))
        self.logs["anymal_goal_error"].append(anymal_goal_error)
        self.logs["anymal_min_detJ"].append(min_detj)

        return MissionSnapshot(
            phase=self.phase,
            phase_index=self._phase_idx(),
            success_flags=self.success_flags.copy(),
            stack_order=self.completed_stack_order.copy(),
            anymal_min_detj=float(min_detj),
            anymal_goal_error=anymal_goal_error,
            collision_violations=int(self.collision_violations),
        )

    def is_finished(self) -> bool:
        return self.phase == "COMPLETE"
