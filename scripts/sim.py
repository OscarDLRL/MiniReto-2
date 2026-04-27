"""
sim.py

Simulación 2D completa del mini reto:
- Husky despeja el corredor y se hace a un lado
- ANYmal avanza recto hasta la meta, luego se orilla en la zona de trabajo
- Los PuzzleBots bajan por un costado del ANYmal
- Los PuzzleBots apilan C-B-A uno por uno sin cruzar por las patas del ANYmal
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle, Rectangle

from anymal_gait import LEG_ORDER
from coordinator import (
    ANYMAL_DEST,
    ANYMAL_SIDE_PARK_POSE,
    ANYMAL_TRANSPORT_WAYPOINTS,
    CORRIDOR,
    DEPLOY_LANE_X,
    MissionCoordinator,
    PARKING_POSE,
    STACK_GOAL,
)

DT = 0.05
MAX_FRAMES = 3200


def build_world(ax: plt.Axes) -> None:
    ax.set_title("Mini reto 2D - Husky + ANYmal + PuzzleBots")
    ax.set_xlim(-0.6, 13.35)
    ax.set_ylim(-3.2, 5.9)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")

    start_zone = Rectangle((0.0, -1.4), 1.6, 2.8, fill=False, linestyle="--", linewidth=1.5)
    corridor = Rectangle((CORRIDOR[0], CORRIDOR[2]), CORRIDOR[1] - CORRIDOR[0], CORRIDOR[3] - CORRIDOR[2], fill=False, linewidth=2.0)
    work_zone = Rectangle((9.6, 2.2), 2.2, 2.2, fill=False, linestyle="--", linewidth=1.5)
    husky_park_zone = Rectangle((0.1, -2.85), 1.8, 0.9, fill=False, linestyle=":", linewidth=1.5)
    anymal_side_zone = Rectangle((11.70, 3.00), 1.35, 1.0, fill=False, linestyle=":", linewidth=1.3)

    ax.add_patch(start_zone)
    ax.add_patch(corridor)
    ax.add_patch(work_zone)
    ax.add_patch(husky_park_zone)
    ax.add_patch(anymal_side_zone)

    ax.text(0.12, 1.62, "Zona de inicio", fontsize=10)
    ax.text(4.0, 1.28, "Corredor", fontsize=10)
    ax.text(9.76, 4.56, "Zona de trabajo", fontsize=10)
    ax.text(0.16, -1.92, "Park Husky", fontsize=9)
    ax.text(11.72, 4.10, "ANYmal a un costado", fontsize=8)
    ax.text(10.45, 4.95, "Carril interno PB", fontsize=8)

    wp = np.vstack(ANYMAL_TRANSPORT_WAYPOINTS)
    ax.plot(wp[:, 0], wp[:, 1], "--", linewidth=1.4, label="Ruta ANYmal")
    ax.scatter(wp[:, 0], wp[:, 1], s=28)
    ax.scatter([ANYMAL_SIDE_PARK_POSE[0]], [ANYMAL_SIDE_PARK_POSE[1]], s=40, marker="x", label="Park ANYmal")
    ax.scatter([PARKING_POSE[0]], [PARKING_POSE[1]], s=36, marker="x", label="Park Husky")
    ax.scatter([STACK_GOAL[0]], [STACK_GOAL[1]], s=50, marker="*", label="Pila C-B-A")
    ax.legend(loc="upper left")



def plot_results(mission: MissionCoordinator) -> None:
    """Genera gráficas al terminar la animación."""
    husky_logs = mission.husky.logs
    anymal_logs = mission.anymal.logs
    mission_logs = mission.logs

    # 1) Velocidades del Husky
    fig1, ax1 = plt.subplots(figsize=(10, 4.6))
    ax1.plot(husky_logs["time"], husky_logs["v_cmd"], label="v cmd")
    ax1.plot(husky_logs["time"], husky_logs["v_meas"], label="v meas")
    ax1.plot(husky_logs["time"], husky_logs["w_cmd"], label="w cmd")
    ax1.plot(husky_logs["time"], husky_logs["w_meas"], label="w meas")
    ax1.set_title("Husky: velocidades comandadas y medidas")
    ax1.set_xlabel("t [s]")
    ax1.set_ylabel("velocidad")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # 2) Trayectorias XY
    fig2, ax2 = plt.subplots(figsize=(7.6, 6.2))
    build_world(ax2)
    hx = np.asarray(mission.husky.path, dtype=float)
    ax2.plot(hx[:, 0], hx[:, 1], label="Husky path")
    ax2.scatter(hx[0, 0], hx[0, 1], marker="o", s=40, label="Husky start")
    ax2.scatter(hx[-1, 0], hx[-1, 1], marker="x", s=50, label="Husky end")
    ax = np.asarray(mission.anymal.body_path, dtype=float)
    ax2.plot(ax[:, 0], ax[:, 1], label="ANYmal path")
    ax2.scatter(ax[0, 0], ax[0, 1], marker="o", s=40, label="ANYmal start")
    ax2.scatter(ax[-1, 0], ax[-1, 1], marker="x", s=50, label="ANYmal end")
    for pb in mission.puzzlebots:
        p = np.asarray(pb.path, dtype=float)
        if len(p) > 1:
            ax2.plot(p[:, 0], p[:, 1], label=f"{pb.name} path", alpha=0.85)
    ax2.set_title("Trayectorias XY de la misión")
    ax2.legend(loc="best", fontsize=8)

    # 3) Singularidades del ANYmal + fase
    fig3, ax3 = plt.subplots(figsize=(10, 4.6))
    ax3.plot(anymal_logs["time"], anymal_logs["min_detJ"], label="min |detJ|")
    ax3.axhline(mission.anymal.detj_threshold, linestyle="--", label="umbral detJ")
    ax3.set_title("ANYmal: monitoreo de singularidades")
    ax3.set_xlabel("t [s]")
    ax3.set_ylabel("|detJ|")
    ax3.grid(True, alpha=0.3)
    ax3.legend()

    # 4) Métricas globales de la misión
    fig4, ax4 = plt.subplots(figsize=(10, 4.8))
    ax4.plot(mission_logs["time"], mission_logs["stack_size"], label="cajas apiladas")
    ax4.plot(mission_logs["time"], mission_logs["anymal_goal_error"], label="error ANYmal")
    ax4.plot(mission_logs["time"], mission_logs["min_pb_distance"], label="dist min PB-PB")
    ax4.plot(mission_logs["time"], mission_logs["collision_violations"], label="violaciones colisión")
    ax4.set_title("Métricas globales de la misión")
    ax4.set_xlabel("t [s]")
    ax4.grid(True, alpha=0.3)
    ax4.legend()

    # 5) Torques de los brazos de los PuzzleBots
    fig5, ax5 = plt.subplots(figsize=(10, 4.8))
    for pb in mission.puzzlebots:
        ax5.plot(pb.logs["time"], pb.logs["tau_norm"], label=f"{pb.name} ||tau||")
    ax5.set_title("PuzzleBots: norma del torque del brazo")
    ax5.set_xlabel("t [s]")
    ax5.set_ylabel("||tau||")
    ax5.grid(True, alpha=0.3)
    ax5.legend()

    plt.show()


def run_demo() -> None:
    mission = MissionCoordinator(dt=DT)

    fig, ax = plt.subplots(figsize=(12.2, 7.2))
    build_world(ax)

    husky_body_line, = ax.plot([], [], linewidth=2.1)
    husky_heading_line, = ax.plot([], [], linewidth=1.9)
    husky_path_line, = ax.plot([], [], linewidth=1.2)
    lidar_points, = ax.plot([], [], ".", markersize=3, alpha=0.65)

    anymal_body_line, = ax.plot([], [], linewidth=2.1)
    anymal_heading_line, = ax.plot([], [], linewidth=1.9)
    anymal_path_line, = ax.plot([], [], linewidth=1.3)

    foot_artists = {}
    hip_artists = {}
    leg_lines = {}
    foot_labels = {}
    keepout_patches = []
    for _ in range(5):
        circ = Circle((0.0, 0.0), radius=0.1, fill=False, linestyle=":", alpha=0.25)
        ax.add_patch(circ)
        keepout_patches.append(circ)

    for leg in LEG_ORDER:
        foot_artists[leg], = ax.plot([], [], marker="o", markersize=6.0, linestyle="None")
        hip_artists[leg], = ax.plot([], [], marker="s", markersize=4.0, linestyle="None")
        leg_lines[leg], = ax.plot([], [], linewidth=1.1)
        foot_labels[leg] = ax.text(0.0, 0.0, leg, fontsize=8)

    big_box_patches = []
    for _ in mission.big_boxes:
        patch = Rectangle((0.0, 0.0), 1.0, 1.0, alpha=0.55)
        ax.add_patch(patch)
        big_box_patches.append(patch)

    small_box_patches = {}
    small_box_text = {}
    for label, box in mission.small_boxes.items():
        patch = Rectangle((0.0, 0.0), box.size_xy[0], box.size_xy[1], alpha=0.85)
        ax.add_patch(patch)
        small_box_patches[label] = patch
        small_box_text[label] = ax.text(box.world_xy[0], box.world_xy[1], label, ha="center", va="center", fontsize=9)

    pb_body_lines = {}
    pb_wheels_left = {}
    pb_wheels_right = {}
    pb_paths = {}
    pb_labels = {}
    payload_markers = {}
    for pb in mission.puzzlebots:
        pb_body_lines[pb.name], = ax.plot([], [], linewidth=1.8)
        pb_wheels_left[pb.name], = ax.plot([], [], marker="o", markersize=4.5, linestyle="None")
        pb_wheels_right[pb.name], = ax.plot([], [], marker="o", markersize=4.5, linestyle="None")
        pb_paths[pb.name], = ax.plot([], [], linewidth=1.0, alpha=0.8)
        pb_labels[pb.name] = ax.text(0.0, 0.0, pb.name, fontsize=8)
        payload_markers[pb.name], = ax.plot([], [], marker="D", markersize=5.0, linestyle="None")

    status_text = ax.text(
        0.02,
        0.98,
        "",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=10,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.88),
    )

    def update_husky_artists() -> None:
        husky = mission.husky
        corners = husky.body_corners_world()
        husky_body_line.set_data(corners[:, 0], corners[:, 1])
        front = husky.front_point()
        husky_heading_line.set_data([husky.pose[0], front[0]], [husky.pose[1], front[1]])
        path = np.asarray(husky.path, dtype=float)
        husky_path_line.set_data(path[:, 0], path[:, 1])

        lidar = husky.simulate_lidar(mission.big_boxes)
        if lidar.size > 0:
            lidar_points.set_data(lidar[:, 0], lidar[:, 1])
        else:
            lidar_points.set_data([], [])

    def update_anymal_artists() -> None:
        anymal = mission.anymal
        corners = anymal.body_corners_world()
        anymal_body_line.set_data(corners[:, 0], corners[:, 1])
        head = anymal.rot2(anymal.pose[2]) @ np.array([0.5 * anymal.body_length, 0.0]) + anymal.pose[:2]
        anymal_heading_line.set_data([anymal.pose[0], head[0]], [anymal.pose[1], head[1]])
        body_path = np.asarray(anymal.body_path, dtype=float)
        anymal_path_line.set_data(body_path[:, 0], body_path[:, 1])

        for patch, (center, radius) in zip(keepout_patches, mission.anymal_keepout_circles()):
            patch.center = (center[0], center[1])
            patch.set_radius(radius)

        for leg in LEG_ORDER:
            hip = anymal.hip_world(leg)
            foot = anymal.feet_world[leg]
            hip_artists[leg].set_data([hip[0]], [hip[1]])
            foot_artists[leg].set_data([foot[0]], [foot[1]])
            leg_lines[leg].set_data([hip[0], foot[0]], [hip[1], foot[1]])
            foot_labels[leg].set_position((foot[0] + 0.04, foot[1] + 0.04))

    def update_big_boxes() -> None:
        for patch, box in zip(big_box_patches, mission.big_boxes):
            xmin = box.center[0] - 0.5 * box.size[0]
            ymin = box.center[1] - 0.5 * box.size[1]
            patch.set_xy((xmin, ymin))
            patch.set_width(box.size[0])
            patch.set_height(box.size[1])
            if box.cleared:
                patch.set_alpha(0.22)
                patch.set_linestyle("--")
            else:
                patch.set_alpha(0.65)
                patch.set_linestyle("-")

    def update_small_boxes() -> None:
        for label, box in mission.small_boxes.items():
            patch = small_box_patches[label]
            xmin = box.world_xy[0] - 0.5 * box.size_xy[0]
            ymin = box.world_xy[1] - 0.5 * box.size_xy[1]
            patch.set_xy((xmin, ymin))
            patch.set_width(box.size_xy[0])
            patch.set_height(box.size_xy[1])
            patch.set_facecolor(box.color)
            patch.set_alpha(0.9 if not box.placed else 0.55)
            small_box_text[label].set_position((box.world_xy[0], box.world_xy[1]))

    def update_puzzlebots() -> None:
        for pb in mission.puzzlebots:
            corners = pb.body_corners_world()
            pb_body_lines[pb.name].set_data(corners[:, 0], corners[:, 1])
            left, right = pb.wheel_points_world()
            pb_wheels_left[pb.name].set_data([left[0]], [left[1]])
            pb_wheels_right[pb.name].set_data([right[0]], [right[1]])
            path = np.asarray(pb.path, dtype=float)
            pb_paths[pb.name].set_data(path[:, 0], path[:, 1])
            pb_labels[pb.name].set_position((pb.pose[0] + 0.06, pb.pose[1] + 0.08))

            if not pb.deployed:
                payload_markers[pb.name].set_data([pb.pose[0]], [pb.pose[1]])
            else:
                payload_markers[pb.name].set_data([], [])

    def init():
        update_husky_artists()
        update_anymal_artists()
        update_big_boxes()
        update_small_boxes()
        update_puzzlebots()
        status_text.set_text("")
        return [
            husky_body_line,
            husky_heading_line,
            husky_path_line,
            lidar_points,
            anymal_body_line,
            anymal_heading_line,
            anymal_path_line,
            status_text,
            *big_box_patches,
            *small_box_patches.values(),
            *small_box_text.values(),
            *foot_artists.values(),
            *hip_artists.values(),
            *leg_lines.values(),
            *foot_labels.values(),
            *keepout_patches,
            *pb_body_lines.values(),
            *pb_wheels_left.values(),
            *pb_wheels_right.values(),
            *pb_paths.values(),
            *pb_labels.values(),
            *payload_markers.values(),
        ]

    finished_state = {"done": False}

    def update(_frame: int):
        if not finished_state["done"]:
            snap = mission.update()
            if snap.phase == "COMPLETE":
                finished_state["done"] = True

        update_husky_artists()
        update_anymal_artists()
        update_big_boxes()
        update_small_boxes()
        update_puzzlebots()
        status_text.set_text(mission.status_message)

        return [
            husky_body_line,
            husky_heading_line,
            husky_path_line,
            lidar_points,
            anymal_body_line,
            anymal_heading_line,
            anymal_path_line,
            status_text,
            *big_box_patches,
            *small_box_patches.values(),
            *small_box_text.values(),
            *foot_artists.values(),
            *hip_artists.values(),
            *leg_lines.values(),
            *foot_labels.values(),
            *keepout_patches,
            *pb_body_lines.values(),
            *pb_wheels_left.values(),
            *pb_wheels_right.values(),
            *pb_paths.values(),
            *pb_labels.values(),
            *payload_markers.values(),
        ]

    _anim = FuncAnimation(
        fig,
        update,
        frames=None,
        init_func=init,
        interval=int(DT * 1000),
        blit=False,
        repeat=False,
    )

    plt.show()
    plot_results(mission)


if __name__ == "__main__":
    run_demo()
