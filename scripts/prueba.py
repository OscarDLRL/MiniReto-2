# prueba.py

import numpy as np
import matplotlib.pyplot as plt
from main import PuzzleBot, simulate_puzzlebot, plot_puzzlebot_trajectory
from puzzlebot_arm import PuzzleBotArm 


# ── Funciones existentes ────────────────────────────────────────────────────

def prueba_linea_recta():
    print("\n=== PRUEBA 1: Línea recta ===")
    bot = PuzzleBot(r=0.05, L=0.19)
    log = simulate_puzzlebot(bot, wR_func=lambda t: 10.0, wL_func=lambda t: 10.0, T=3.0, dt=0.01)
    x, y, theta = bot.get_pose()
    print(f"Pose final -> x={x:.3f}, y={y:.3f}, theta={theta:.3f} rad")
    plot_puzzlebot_trajectory(log, title="Prueba PuzzleBot - Línea recta", save_path="prueba_recta.png")


def prueba_arco():
    print("\n=== PRUEBA 2: Arco ===")
    bot = PuzzleBot(r=0.05, L=0.19)
    log = simulate_puzzlebot(bot, wR_func=lambda t: 10.0, wL_func=lambda t: 8.0, T=5.0, dt=0.01)
    x, y, theta = bot.get_pose()
    print(f"Pose final -> x={x:.3f}, y={y:.3f}, theta={theta:.3f} rad")
    plot_puzzlebot_trajectory(log, title="Prueba PuzzleBot - Arco", save_path="prueba_arco.png")


def prueba_giro():
    print("\n=== PRUEBA 3: Giro sobre su eje ===")
    bot = PuzzleBot(r=0.05, L=0.19)
    log = simulate_puzzlebot(bot, wR_func=lambda t: 5.0, wL_func=lambda t: -5.0, T=2.0, dt=0.01)
    x, y, theta = bot.get_pose()
    print(f"Pose final -> x={x:.3f}, y={y:.3f}, theta={theta:.3f} rad")
    plot_puzzlebot_trajectory(log, title="Prueba PuzzleBot - Giro en su eje", save_path="prueba_giro.png")


def prueba_espiral():
    print("\n=== PRUEBA 4: Espiral ===")
    bot = PuzzleBot(r=0.05, L=0.19)
    log = simulate_puzzlebot(bot, wR_func=lambda t: 10.0, wL_func=lambda t: 5.0 + 0.3 * t, T=10.0, dt=0.01)
    x, y, theta = bot.get_pose()
    print(f"Pose final -> x={x:.3f}, y={y:.3f}, theta={theta:.3f} rad")
    plot_puzzlebot_trajectory(log, title="Prueba PuzzleBot - Espiral", save_path="prueba_espiral.png")


# ── Nueva función: brazo ────────────────────────────────────────────────────

def prueba_brazo():
    print("\n=== PRUEBA 5: PuzzleBotArm ===")
    arm = PuzzleBotArm(l1=0.10, l2=0.08, l3=0.06)

    # --- FK en configuración home ---
    q_home = np.array([0.0, np.pi / 6, np.pi / 4])
    p_home = arm.forward_kinematics(q_home)
    print(f"FK home -> p={p_home}")

    # --- IK: recuperar q desde p_home ---
    q_ik = arm.inverse_kinematics(p_home)
    p_check = arm.forward_kinematics(q_ik)
    print(f"IK check -> p={p_check}  (error={np.linalg.norm(p_check - p_home):.2e})")

    # --- Jacobiano y torques ---
    arm.forward_kinematics(q_home)
    J = arm.jacobian(q_home)
    print(f"Jacobiano:\n{J}")
    f_tip = np.array([0.0, 0.0, -5.0])
    tau = arm.force_to_torque(f_tip)
    print(f"Torques para f={f_tip} -> tau={tau}")

    # --- grasp_box hacia un objetivo alcanzable ---
    box_pos = p_home + np.array([0.02, 0.01, -0.02])
    try:
        result = arm.grasp_box(box_pos, grip_force=5.0)
        print(f"grasp_box OK  final_q={np.round(result['final_q'], 4)}")
        print(f"  torques aplicados: {np.round(result['joint_torques'], 4)}")
    except ValueError as e:
        print(f"grasp_box ERROR: {e}")

    # --- Gráfica de trayectoria cartesiana ---
    traj = np.array(result["trajectory_positions"])
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], "o-", label="trayectoria")
    ax.scatter(*p_home, color="green", s=60, label="inicio")
    ax.scatter(*box_pos, color="red",  s=60, label="box_pos")
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    ax.set_title("PuzzleBotArm – trayectoria grasp")
    ax.legend()
    plt.tight_layout()
    plt.savefig("prueba_brazo.png")
    print("Gráfica guardada en prueba_brazo.png")


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    prueba_linea_recta()
    prueba_arco()
    prueba_giro()
    prueba_espiral()
    prueba_brazo()
    plt.show()


if __name__ == "__main__":
    main()