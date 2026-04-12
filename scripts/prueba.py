# prueba.py

import matplotlib.pyplot as plt
from main import PuzzleBot, simulate_puzzlebot, plot_puzzlebot_trajectory


def prueba_linea_recta():
    print("\n=== PRUEBA 1: Línea recta ===")
    bot = PuzzleBot(r=0.05, L=0.19)

    log = simulate_puzzlebot(
        bot,
        wR_func=lambda t: 10.0,
        wL_func=lambda t: 10.0,
        T=3.0,
        dt=0.01
    )

    x, y, theta = bot.get_pose()
    print(f"Pose final -> x={x:.3f}, y={y:.3f}, theta={theta:.3f} rad")

    plot_puzzlebot_trajectory(
        log,
        title="Prueba PuzzleBot - Línea recta",
        save_path="prueba_recta.png"
    )


def prueba_arco():
    print("\n=== PRUEBA 2: Arco ===")
    bot = PuzzleBot(r=0.05, L=0.19)

    log = simulate_puzzlebot(
        bot,
        wR_func=lambda t: 10.0,
        wL_func=lambda t: 8.0,
        T=5.0,
        dt=0.01
    )

    x, y, theta = bot.get_pose()
    print(f"Pose final -> x={x:.3f}, y={y:.3f}, theta={theta:.3f} rad")

    plot_puzzlebot_trajectory(
        log,
        title="Prueba PuzzleBot - Arco",
        save_path="prueba_arco.png"
    )


def prueba_giro():
    print("\n=== PRUEBA 3: Giro sobre su eje ===")
    bot = PuzzleBot(r=0.05, L=0.19)

    log = simulate_puzzlebot(
        bot,
        wR_func=lambda t: 5.0,
        wL_func=lambda t: -5.0,
        T=2.0,
        dt=0.01
    )

    x, y, theta = bot.get_pose()
    print(f"Pose final -> x={x:.3f}, y={y:.3f}, theta={theta:.3f} rad")

    plot_puzzlebot_trajectory(
        log,
        title="Prueba PuzzleBot - Giro en su eje",
        save_path="prueba_giro.png"
    )


def prueba_espiral():
    print("\n=== PRUEBA 4: Espiral ===")
    bot = PuzzleBot(r=0.05, L=0.19)

    log = simulate_puzzlebot(
        bot,
        wR_func=lambda t: 10.0,
        wL_func=lambda t: 5.0 + 0.3 * t,
        T=10.0,
        dt=0.01
    )

    x, y, theta = bot.get_pose()
    print(f"Pose final -> x={x:.3f}, y={y:.3f}, theta={theta:.3f} rad")

    plot_puzzlebot_trajectory(
        log,
        title="Prueba PuzzleBot - Espiral",
        save_path="prueba_espiral.png"
    )


def main():
    prueba_linea_recta()
    prueba_arco()
    prueba_giro()
    prueba_espiral()
    plt.show()


if __name__ == "__main__":
    main()