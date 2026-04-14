################################################################################
#                                                                              #
#                    PARTE 2:  Husky A200  (skid-steer 4 ruedas)               #
#                                                                              #
################################################################################
import matplotlib as mpl
import numpy as np

class HuskyA200:
    """
    Husky A200 de Clearpath Robotics - Robot skid-steer de 4 ruedas.

    El skid-steer NO tiene mecanismo de direccion: el giro se logra
    variando la velocidad entre los dos lados (como un tanque).
    Esto implica deslizamiento lateral durante el giro.
    """

    def __init__(self, r=0.1651, B=0.555, mass=50.0):
        # --- Parametros geometricos ---
        self.r = r              # Radio de la rueda [m]
        self.B = B              # Distancia efectiva entre lados [m]
        # B debe calibrarse experimentalmente ya que el skid-steer
        # tiene un "centro instantaneo de rotacion" (ICR) virtual

        # --- Parametros inerciales ---
        self.mass = mass        # Masa [kg] sin payload
        self.payload_max = 75.0 # Payload maximo [kg]

        # --- Estado (pose y velocidades) ---
        self.x, self.y, self.theta = 0.0, 0.0, 0.0
        self.v, self.omega = 0.0, 0.0

        # --- Terreno actual (afecta deslizamiento) ---
        self.terrain = "asphalt"
        self._slip_factors = {
            "asphalt": 1.00,    # Superficie ideal
            "grass":   0.85,    # Pasto cortado
            "gravel":  0.78,    # Grava suelta
            "sand":    0.65,    # Arena
            "mud":     0.50,    # Lodo
        }

    def start_task(self, dt): 
        """
        EJECUTAR ACCION -> mover objetos
        Recibe intervalo de actualizacon (delta time)
        
        Regrea arreglo de numpy con las posiciones en x y y despues de realizar la tarea 
        o None en caso de error o singularidad

        """
        return None

    def forward_kinematics(self, wR1, wR2, wL1, wL2):
        """
        Cinematica directa para skid-steer de 4 ruedas.

        Entradas:
            wR1, wR2 : velocidades ruedas derechas (frontal, trasera) [rad/s]
            wL1, wL2 : velocidades ruedas izquierdas [rad/s]

        Retorna:
            (v, omega) : velocidades del cuerpo del robot

        Complejidad: O(1)
        """
        # Promediar cada lado: el modelo asume ruedas del mismo lado
        # giran a la misma velocidad (idealizacion)
        avg_R = (wR1 + wR2) / 2.0
        avg_L = (wL1 + wL2) / 2.0

        # Aplicar factor de deslizamiento del terreno actual
        slip = self._slip_factors.get(self.terrain, 0.8)

        # Velocidades del cuerpo
        v = self.r / 2.0 * (avg_R + avg_L) * slip
        omega = self.r / self.B * (avg_R - avg_L)

        return v, omega

    def inverse_kinematics(self, v, omega):
        """
        Cinematica inversa: (v, omega) -> velocidades de las 4 ruedas.
        Se asume que las ruedas de cada lado giran a la misma velocidad.
        """
        wR = (2.0 * v + omega * self.B) / (2.0 * self.r)
        wL = (2.0 * v - omega * self.B) / (2.0 * self.r)
        return wR, wR, wL, wL   # (wR1, wR2, wL1, wL2)

    def update_pose(self, v, omega, dt):
        """Integra la pose usando el metodo del punto medio (midpoint)."""
        # Orientacion en el punto medio del intervalo (mas preciso que Euler)
        theta_mid = self.theta + omega * dt / 2.0

        self.x += v * np.cos(theta_mid) * dt
        self.y += v * np.sin(theta_mid) * dt
        self.theta += omega * dt
        self.theta = np.arctan2(np.sin(self.theta), np.cos(self.theta))

        # Guardar velocidades para el siguiente ciclo
        self.v, self.omega = v, omega

    def set_terrain(self, terrain_name):
        """Cambia el terreno actual (afecta factor de deslizamiento)."""
        self.terrain = terrain_name

    def get_pose(self):
        """Retorna la pose actual como tupla (x, y, theta)."""
        return (self.x, self.y, self.theta)

    def reset(self, x=0.0, y=0.0, theta=0.0):
        """Resetea el estado del robot."""
        self.x, self.y, self.theta = x, y, theta
        self.v, self.omega = 0.0, 0.0


def simulate_husky(husky, wheel_funcs, T=5.0, dt=0.01):
    """
    Simula el movimiento del Husky A200.

    Entradas:
        husky       : instancia de HuskyA200
        wheel_funcs : tupla (wR1_func, wR2_func, wL1_func, wL2_func)
                      cada una funcion de t -> rad/s
        T           : tiempo de simulacion [s]
        dt          : paso de tiempo [s]

    Retorna:
        dict con trayectorias de pose, 4 velocidades de rueda, y velocidades del cuerpo.
    """
    husky.reset()
    wR1_func, wR2_func, wL1_func, wL2_func = wheel_funcs
    n_steps = int(T / dt)
    log = {k: np.zeros(n_steps) for k in
           ['t', 'x', 'y', 'theta',
            'wR1', 'wR2', 'wL1', 'wL2', 'v', 'omega']}

    for i in range(n_steps):
        t = i * dt
        wR1 = wR1_func(t)
        wR2 = wR2_func(t)
        wL1 = wL1_func(t)
        wL2 = wL2_func(t)
        v, omega = husky.forward_kinematics(wR1, wR2, wL1, wL2)
        husky.update_pose(v, omega, dt)

        log['t'][i] = t
        log['x'][i] = husky.x
        log['y'][i] = husky.y
        log['theta'][i] = husky.theta
        log['wR1'][i] = wR1
        log['wR2'][i] = wR2
        log['wL1'][i] = wL1
        log['wL2'][i] = wL2
        log['v'][i] = v
        log['omega'][i] = omega

    return log


def plot_husky_trajectory(log, title="Husky A200 - Trayectoria y Actuadores",
                          save_path=None):
    """
    Grafica la trayectoria y las velocidades de los 4 actuadores del Husky.

    Los actuadores del Husky son los 4 motores DC (FL, FR, RL, RR).

    Genera una figura con 4 subplots:
        1. Trayectoria XY con flechas de orientacion
        2. Velocidades angulares de las 4 ruedas (actuadores)
        3. Velocidades del cuerpo (v, omega)
        4. Orientacion theta vs tiempo
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle(title, fontsize=14, fontweight='bold')

    # --- Subplot 1: trayectoria XY ---
    ax = axes[0, 0]
    ax.plot(log['x'], log['y'], 'b-', linewidth=2, label='Trayectoria')
    ax.plot(log['x'][0], log['y'][0], 'go', markersize=10, label='Inicio')
    ax.plot(log['x'][-1], log['y'][-1], 'rs', markersize=10, label='Fin')
    step = max(1, len(log['t']) // 20)
    for i in range(0, len(log['t']), step):
        dx = 0.3 * np.cos(log['theta'][i])
        dy = 0.3 * np.sin(log['theta'][i])
        ax.arrow(log['x'][i], log['y'][i], dx, dy,
                 head_width=0.08, head_length=0.08, fc='orange', ec='orange')
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_title('Trayectoria en el plano XY')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='datalim')

    # --- Subplot 2: velocidades de las 4 ruedas (actuadores) ---
    ax = axes[0, 1]
    ax.plot(log['t'], log['wR1'], 'b-', linewidth=2, label=r'$\omega_{R1}$ (FR)')
    ax.plot(log['t'], log['wR2'], 'b--', linewidth=2, label=r'$\omega_{R2}$ (RR)')
    ax.plot(log['t'], log['wL1'], 'r-', linewidth=2, label=r'$\omega_{L1}$ (FL)')
    ax.plot(log['t'], log['wL2'], 'r--', linewidth=2, label=r'$\omega_{L2}$ (RL)')
    ax.set_xlabel('Tiempo [s]')
    ax.set_ylabel('Velocidad angular [rad/s]')
    ax.set_title('Actuadores: 4 ruedas del Husky')
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)

    # --- Subplot 3: velocidades del cuerpo ---
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

    # --- Subplot 4: orientacion ---
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


def demo_husky():
    """Ejemplos de movimiento del Husky A200 con graficacion."""
    print("=" * 70)
    print("DEMO Husky A200")
    print("=" * 70)

    husky = HuskyA200()

    # --- Ejemplo 1: avance recto sobre asfalto ---
    print("\n[1] Avance recto sobre asfalto")
    husky.set_terrain("asphalt")
    log1 = simulate_husky(husky,
                          wheel_funcs=(lambda t: 3.0, lambda t: 3.0,
                                       lambda t: 3.0, lambda t: 3.0),
                          T=4.0)
    print(f"    Pose final: x={log1['x'][-1]:.3f}, y={log1['y'][-1]:.3f}")
    plot_husky_trajectory(log1, title="Husky Ejemplo 1: Recto en asfalto",
                          save_path="husky_ej1_recto.png")

    # --- Ejemplo 2: giro sobre pasto (con slip) ---
    print("\n[2] Giro sobre pasto (slip=0.85)")
    husky.set_terrain("grass")
    log2 = simulate_husky(husky,
                          wheel_funcs=(lambda t: 4.0, lambda t: 4.0,
                                       lambda t: 1.5, lambda t: 1.5),
                          T=6.0)
    print(f"    Pose final: x={log2['x'][-1]:.3f}, y={log2['y'][-1]:.3f}, "
          f"theta={np.degrees(log2['theta'][-1]):.1f} deg")
    plot_husky_trajectory(log2, title="Husky Ejemplo 2: Giro en pasto",
                          save_path="husky_ej2_giro.png")

    # --- Ejemplo 3: trayectoria en S (cambio de direccion de giro) ---
    print("\n[3] Trayectoria en S (cambio de direccion de giro)")
    husky.set_terrain("asphalt")

    def wheel_S(t):
        """Genera comando en S: primero gira a la derecha, luego a la izquierda."""
        if t < 3.0:
            return 4.0, 2.0     # giro derecha
        else:
            return 2.0, 4.0     # giro izquierda

    def wR_s(t): return wheel_S(t)[0]
    def wL_s(t): return wheel_S(t)[1]

    log3 = simulate_husky(husky,
                          wheel_funcs=(wR_s, wR_s, wL_s, wL_s),
                          T=6.0)
    print(f"    Pose final: x={log3['x'][-1]:.3f}, y={log3['y'][-1]:.3f}")
    plot_husky_trajectory(log3, title="Husky Ejemplo 3: Trayectoria en S",
                          save_path="husky_ej3_S.png")

    # --- Ejemplo 4: comparacion de terrenos (misma entrada, diferente slip) ---
    print("\n[4] Comparacion de 4 terrenos con la misma entrada")
    terrains = ["asphalt", "grass", "gravel", "sand"]
    fig, ax = plt.subplots(figsize=(8, 7))
    for terrain in terrains:
        husky.set_terrain(terrain)
        log = simulate_husky(husky,
                             wheel_funcs=(lambda t: 4.0, lambda t: 4.0,
                                          lambda t: 2.0, lambda t: 2.0),
                             T=6.0)
        slip = husky._slip_factors[terrain]
        ax.plot(log['x'], log['y'], linewidth=2,
                label=f"{terrain} (slip={slip})")
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_title('Husky A200: comparacion de terrenos (misma entrada)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='datalim')
    plt.tight_layout()
    plt.savefig("husky_ej4_terrenos.png", dpi=150, bbox_inches='tight')
    print("  -> Figura guardada en husky_ej4_terrenos.png")

