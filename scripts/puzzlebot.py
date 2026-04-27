from puzzlebot_arm import PuzzleBotArm
import numpy as np
import matplotlib.pyplot as plt

# PARTE 1:  PuzzleBot  (robot diferencial)                  

class PuzzleBot:
    """
    Robot diferencial de 2 ruedas (PuzzleBot Manchester Robotics / Tec de Monterrey).

    Modelo cinematico:
        v     = r/2 * (wR + wL)
        omega = r/L * (wR - wL)

    Atributos:
        r      : radio de la rueda [m]
        L      : distancia entre ruedas (track width) [m]
        x, y   : posicion del robot en el marco mundo [m]
        theta  : orientacion del robot [rad]
    """

    def __init__(self, r=0.05, L=0.19, x=0.0, y=0.0):
        # Parametros fisicos (calibrados con el robot real)
        self.r = r              # Radio de la rueda [m]
        self.L = L              # Distancia entre ruedas [m]

        # Estado interno: pose (x, y, theta) en marco mundo
        self.x = x            # Posicion X [m]
        self.y = y            # Posicion Y [m]
        self.theta = 0.0        # Orientacion [rad] en [-pi, pi]
        self.arm = PuzzleBotArm()

        # Limites fisicos del robot (proteccion)
        self.v_max = 0.8        # Velocidad lineal maxima [m/s]
        self.omega_max = 3.0    # Velocidad angular maxima [rad/s]

    def start_task(self, dt): 
        """
        EJECUTAR ACCION -> desplegar
        Recibe intervalo de actualizacon (delta time)
        
        Regrea arreglo de numpy con las posiciones en x y y despues de realizar la tarea 
        o None en caso de error o singularidad

        """
        return None

    def forward_kinematics(self, wR, wL):
        """
        Cinematica directa: de velocidades de rueda a (v, omega) del robot.

        Entradas:
            wR : velocidad angular rueda derecha [rad/s]
            wL : velocidad angular rueda izquierda [rad/s]

        Retorna:
            (v, omega) : velocidad lineal [m/s] y angular [rad/s]

        Complejidad: O(1) - 3 operaciones aritmeticas
        """
        # Velocidad lineal = promedio de velocidades de rueda * radio
        v = self.r / 2.0 * (wR + wL)

        # Velocidad angular = diferencia de velocidades escalada por geometria
        omega = self.r / self.L * (wR - wL)

        return v, omega

    def inverse_kinematics(self, v, omega):
        """
        Cinematica inversa: de (v, omega) deseados a velocidades de rueda.

        Entradas:
            v     : velocidad lineal deseada [m/s]
            omega : velocidad angular deseada [rad/s]

        Retorna:
            (wR, wL) : velocidades angulares de ruedas [rad/s]

        Complejidad: O(1) - 4 operaciones aritmeticas

        Uso tipico: el planificador da (v, omega); este metodo traduce
        a referencias que los PIDs de cada motor deben seguir.
        """
        # Saturar velocidades deseadas a los limites fisicos
        v = np.clip(v, -self.v_max, self.v_max)
        omega = np.clip(omega, -self.omega_max, self.omega_max)

        # Cinematica inversa: resolver el sistema lineal 2x2
        #   wR + wL = 2v/r
        #   wR - wL = omega*L/r
        wR = (2.0 * v + omega * self.L) / (2.0 * self.r)
        wL = (2.0 * v - omega * self.L) / (2.0 * self.r)

        return wR, wL

    def update_pose(self, v, omega, dt):
        """
        Integra la pose del robot por Euler explicito.

        Entradas:
            v     : velocidad lineal actual [m/s]
            omega : velocidad angular actual [rad/s]
            dt    : paso de tiempo [s] (tipicamente 0.01 s = 100 Hz)

        Modifica self.x, self.y, self.theta in-place.

        Complejidad: O(1) por paso
        Error de integracion: O(dt) - Euler explicito
        """
        # Actualizar posicion (traslacion en la direccion actual)
        self.x += v * np.cos(self.theta) * dt
        self.y += v * np.sin(self.theta) * dt

        # Actualizar orientacion
        self.theta += omega * dt

        # Normalizar theta al rango [-pi, pi] para evitar acumulacion
        self.theta = np.arctan2(np.sin(self.theta), np.cos(self.theta))

    def get_pose(self):
        """Retorna la pose actual como tupla (x, y, theta)."""
        return (self.x, self.y, self.theta)

    def reset(self, x=0.0, y=0.0, theta=0.0):
        """Resetea el estado del robot a una pose dada."""
        self.x, self.y, self.theta = x, y, theta


def simulate_puzzlebot(bot, wR_func, wL_func, T=5.0, dt=0.01):
    """
    Simula el movimiento del PuzzleBot dadas funciones de velocidad de rueda.

    Entradas:
        bot     : instancia de PuzzleBot
        wR_func : funcion wR(t) -> velocidad rueda derecha [rad/s]
        wL_func : funcion wL(t) -> velocidad rueda izquierda [rad/s]
        T       : tiempo total de simulacion [s]
        dt      : paso de tiempo [s]

    Retorna:
        dict con claves:
            't'     : array de tiempos [s]
            'x','y','theta' : trayectoria de la pose
            'wR','wL'       : trayectoria de velocidades de rueda [rad/s]
            'v','omega'     : trayectoria de velocidades del cuerpo
    """
    bot.reset()
    n_steps = int(T / dt)
    log = {k: np.zeros(n_steps) for k in ['t', 'x', 'y', 'theta',
                                          'wR', 'wL', 'v', 'omega']}

    for i in range(n_steps):
        t = i * dt
        wR = wR_func(t)
        wL = wL_func(t)
        v, omega = bot.forward_kinematics(wR, wL)
        bot.update_pose(v, omega, dt)

        log['t'][i] = t
        log['x'][i] = bot.x
        log['y'][i] = bot.y
        log['theta'][i] = bot.theta
        log['wR'][i] = wR
        log['wL'][i] = wL
        log['v'][i] = v
        log['omega'][i] = omega

    return log


def plot_puzzlebot_trajectory(log, title="PuzzleBot - Trayectoria y Actuadores",
                              save_path=None):
    """
    Grafica la trayectoria XY y las velocidades de los actuadores del PuzzleBot.

    Los actuadores del PuzzleBot son los 2 motores DC (rueda R y rueda L).

    Genera una figura con 4 subplots:
        1. Trayectoria XY en el plano
        2. Velocidades angulares de las ruedas (actuadores) vs tiempo
        3. Velocidades del cuerpo (v, omega) vs tiempo
        4. Orientacion theta vs tiempo
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle(title, fontsize=14, fontweight='bold')

    # Subplot 1: trayectoria XY
    ax = axes[0, 0]
    ax.plot(log['x'], log['y'], 'b-', linewidth=2, label='Trayectoria')
    ax.plot(log['x'][0], log['y'][0], 'go', markersize=10, label='Inicio')
    ax.plot(log['x'][-1], log['y'][-1], 'rs', markersize=10, label='Fin')
    # Flechas de orientacion cada cierto intervalo
    step = max(1, len(log['t']) // 20)
    for i in range(0, len(log['t']), step):
        dx = 0.05 * np.cos(log['theta'][i])
        dy = 0.05 * np.sin(log['theta'][i])
        ax.arrow(log['x'][i], log['y'][i], dx, dy,
                 head_width=0.02, head_length=0.02, fc='orange', ec='orange')
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_title('Trayectoria en el plano XY')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='datalim')

    # Subplot 2: velocidades de los actuadores (ruedas)
    ax = axes[0, 1]
    ax.plot(log['t'], log['wR'], 'b-', linewidth=2, label=r'$\omega_R$ (rueda der.)')
    ax.plot(log['t'], log['wL'], 'r-', linewidth=2, label=r'$\omega_L$ (rueda izq.)')
    ax.set_xlabel('Tiempo [s]')
    ax.set_ylabel('Velocidad angular [rad/s]')
    ax.set_title('Actuadores: velocidades de ruedas')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Subplot 3: velocidades del cuerpo
    ax = axes[1, 0]
    ax2 = ax.twinx()
    l1 = ax.plot(log['t'], log['v'], 'g-', linewidth=2, label='v [m/s]')
    l2 = ax2.plot(log['t'], log['omega'], 'm-', linewidth=2, label=r'$\omega$ [rad/s]')
    ax.set_xlabel('Tiempo [s]')
    ax.set_ylabel('Velocidad lineal v [m/s]', color='g')
    ax2.set_ylabel(r'Velocidad angular $\omega$ [rad/s]', color='m')
    ax.tick_params(axis='y', labelcolor='g')
    ax2.tick_params(axis='y', labelcolor='m')
    ax.set_title('Velocidades del cuerpo')
    lines = l1 + l2
    ax.legend(lines, [l.get_label() for l in lines], loc='best')
    ax.grid(True, alpha=0.3)

    # Subplot 4: orientacion
    ax = axes[1, 1]
    ax.plot(log['t'], np.degrees(log['theta']), 'k-', linewidth=2)
    ax.set_xlabel('Tiempo [s]')
    ax.set_ylabel(r'$\theta$ [deg]')
    ax.set_title('Orientacion del robot')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  -> Figura guardada en {save_path}")
    return fig


def demo_puzzlebot():
    """Ejemplos de movimiento del PuzzleBot con graficacion."""
    print("=" * 70)
    print("DEMO PuzzleBot")
    print("=" * 70)

    bot = PuzzleBot(r=0.05, L=0.19)

    # Ejemplo 1: avanzar en linea recta
    print("\n[1] Avanzar en linea recta (wR = wL = 10 rad/s)")
    log1 = simulate_puzzlebot(bot,
                              wR_func=lambda t: 10.0,
                              wL_func=lambda t: 10.0,
                              T=3.0)
    print(f"    Pose final: x={log1['x'][-1]:.3f}, y={log1['y'][-1]:.3f}, "
          f"theta={np.degrees(log1['theta'][-1]):.1f} deg")
    plot_puzzlebot_trajectory(log1, title="PuzzleBot Ejemplo 1: Linea Recta",
                              save_path="puzzlebot_ej1_recta.png")

    # Ejemplo 2: movimiento en arco
    print("\n[2] Movimiento en arco (wR=10, wL=8 rad/s)")
    log2 = simulate_puzzlebot(bot,
                              wR_func=lambda t: 10.0,
                              wL_func=lambda t: 8.0,
                              T=5.0)
    print(f"    Pose final: x={log2['x'][-1]:.3f}, y={log2['y'][-1]:.3f}, "
          f"theta={np.degrees(log2['theta'][-1]):.1f} deg")
    plot_puzzlebot_trajectory(log2, title="PuzzleBot Ejemplo 2: Arco",
                              save_path="puzzlebot_ej2_arco.png")

    # Ejemplo 3: espiral (velocidades variables en el tiempo)
    print("\n[3] Espiral (wL crece con el tiempo)")
    log3 = simulate_puzzlebot(bot,
                              wR_func=lambda t: 10.0,
                              wL_func=lambda t: 5.0 + 0.3 * t,
                              T=10.0)
    print(f"    Pose final: x={log3['x'][-1]:.3f}, y={log3['y'][-1]:.3f}, "
          f"theta={np.degrees(log3['theta'][-1]):.1f} deg")
    plot_puzzlebot_trajectory(log3, title="PuzzleBot Ejemplo 3: Espiral",
                              save_path="puzzlebot_ej3_espiral.png")

    # Ejemplo 4: giro en el propio eje
    print("\n[4] Giro sobre el propio eje (wR = -wL)")
    log4 = simulate_puzzlebot(bot,
                              wR_func=lambda t: 5.0,
                              wL_func=lambda t: -5.0,
                              T=2.0)
    print(f"    Pose final: x={log4['x'][-1]:.3f}, y={log4['y'][-1]:.3f}, "
          f"theta={np.degrees(log4['theta'][-1]):.1f} deg")
    plot_puzzlebot_trajectory(log4, title="PuzzleBot Ejemplo 4: Giro en su eje",
                              save_path="puzzlebot_ej4_giro.png")

