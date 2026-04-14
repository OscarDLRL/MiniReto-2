# ANYmal gait: funciones y obetos necesarios para mover el ANYmal

import numpy as np
from enum import Enum

class ANYmalLeg:
    """
    Una pata del ANYmal con 3 articulaciones (HAA, HFE, KFE).

    Convenciones:
        - Marco de referencia en el hombro (hip) de la pata
        - Eje X: hacia adelante
        - Eje Y: lateral (positivo = hacia afuera)
        - Eje Z: hacia arriba
        - side = +1 para patas izquierdas (LF, LH)
        - side = -1 para patas derechas (RF, RH)

    Las 3 articulaciones (actuadores) son:
        q1 = HAA (Hip Abduction/Adduction)
        q2 = HFE (Hip Flexion/Extension)
        q3 = KFE (Knee Flexion/Extension)
    """

    def __init__(self, name, l0=0.0585, l1=0.35, l2=0.33, side=1):
        # --- Identificacion ---
        self.name = name            # 'LF', 'RF', 'LH', 'RH'

        # --- Longitudes de eslabones (valores tipicos ANYmal) ---
        self.l0 = l0                # Longitud del brazo HAA [m]
        self.l1 = l1                # Longitud del muslo (thigh) [m]
        self.l2 = l2                # Longitud de la espinilla (shank) [m]

        # --- Lado (afecta signo de y) ---
        self.side = side            # +1 = izq, -1 = der

        # --- Estado articular ---
        self.q = np.zeros(3)        # [q_HAA, q_HFE, q_KFE] [rad]

        # --- Limites articulares (seguridad) ---
        self.q_min = np.array([-0.72, -9.42, -2.69])    # [rad]
        self.q_max = np.array([ 0.49,  9.42, -0.03])    # [rad]

    def forward_kinematics(self, q=None):
        """
        FK analitica: (q1, q2, q3) -> posicion del pie [x, y, z].

        Entradas:
            q : array [q1, q2, q3] [rad]. Si es None, usa self.q

        Retorna:
            p : np.array([x, y, z]) posicion del pie en marco hombro [m]

        Complejidad: O(1) - 6 trig + 8 mult/sum
        """
        if q is not None:
            self.q = np.asarray(q, dtype=float)
        q1, q2, q3 = self.q

        # Coordenada X (adelante/atras)
        # Contribuciones del muslo (l1) y espinilla (l2)
        x = self.l1 * np.sin(q2) + self.l2 * np.sin(q2 + q3)

        # Coordenada Y (lateral)
        # El lado (+1/-1) determina direccion; solo depende de q1 (HAA)
        y = self.side * self.l0 * np.cos(q1)

        # Coordenada Z (arriba/abajo) - negativo porque cuelga del hombro
        z = -self.l1 * np.cos(q2) - self.l2 * np.cos(q2 + q3)

        return np.array([x, y, z])

    def inverse_kinematics(self, p_des):
        """
        IK geometrica: posicion deseada del pie -> angulos articulares.

        Entrada:
            p_des : np.array([x, y, z]) posicion deseada [m]

        Retorna:
            q : np.array([q1, q2, q3]) angulos articulares [rad]

        Asume configuracion "rodilla hacia atras" (q3 < 0).
        Complejidad: O(1) - ~20 operaciones trig
        """
        x, y, z = p_des

        # --- Paso 1: HAA (q1) - abduccion de la cadera ---
        # Proyeccion en el plano YZ
        r_yz_sq = y**2 + z**2 - self.l0**2
        r_yz = np.sqrt(max(r_yz_sq, 1e-9))
        q1 = np.arctan2(y, -z) - np.arctan2(self.side * self.l0, r_yz)

        # --- Paso 2: KFE (q3) - angulo de la rodilla (ley de cosenos) ---
        r_sq = x**2 + z**2
        D = (r_sq - self.l1**2 - self.l2**2) / (2.0 * self.l1 * self.l2)
        D = np.clip(D, -1.0, 1.0)       # Proteger arccos de NaN
        q3 = -np.arccos(D)              # Negativo = rodilla atras

        # --- Paso 3: HFE (q2) - flexion de la cadera ---
        alpha = np.arctan2(x, -z)
        beta = np.arctan2(self.l2 * np.sin(-q3),
                          self.l1 + self.l2 * np.cos(q3))
        q2 = alpha - beta

        return np.array([q1, q2, q3])

    def jacobian(self, q=None):
        """
        Jacobiano analitico 3x3: dp_pie/dq.

        J[i,j] = derivada parcial de la coord. i del pie respecto a q_j.

        Retorna:
            J : np.ndarray 3x3

        Uso: control de impedancia, deteccion de singularidades.
        Complejidad: O(1) - 6 trig + 12 mult
        """
        if q is None:
            q = self.q
        q1, q2, q3 = q
        J = np.zeros((3, 3))

        # Fila 0: dx/dq (la coord x no depende de q1)
        J[0, 0] = 0.0
        J[0, 1] = self.l1 * np.cos(q2) + self.l2 * np.cos(q2 + q3)
        J[0, 2] = self.l2 * np.cos(q2 + q3)

        # Fila 1: dy/dq (la coord y solo depende de q1)
        J[1, 0] = -self.side * self.l0 * np.sin(q1)
        J[1, 1] = 0.0
        J[1, 2] = 0.0

        # Fila 2: dz/dq
        J[2, 0] = 0.0
        J[2, 1] = self.l1 * np.sin(q2) + self.l2 * np.sin(q2 + q3)
        J[2, 2] = self.l2 * np.sin(q2 + q3)

        return J

    def is_singular(self, q=None, tol=1e-3):
        """Retorna True si la pata esta cerca de una singularidad."""
        return abs(np.linalg.det(self.jacobian(q))) < tol

class ANYmalState(Enum):
    IDLE = 0
    RUNNING = 1
    ERROR = -1

class ANYmal:
    """
    Cuadrupedo ANYmal completo con 4 patas y 12 DoF.

    Estructura:
        - 4 instancias de ANYmalLeg (LF, RF, LH, RH)
        - Base flotante con 6 DoF (no controlados directamente)
    """

    LEG_NAMES = ['LF', 'RF', 'LH', 'RH']

    def __init__(self, x, y):
        # --- Las 4 patas ---
        # LF = Left Front, RF = Right Front, LH = Left Hind, RH = Right Hind
        self.legs = {
            'LF': ANYmalLeg('LF', side=+1),
            'RF': ANYmalLeg('RF', side=-1),
            'LH': ANYmalLeg('LH', side=+1),
            'RH': ANYmalLeg('RH', side=-1),
        }

        # --- Estado del cuerpo (base flotante) ---
        self.base_pos = np.array([0.0, 0.0, 0.45])  # Altura nominal
        self.base_R = np.eye(3)                      # Matriz rotacion base

        # --- Parametros dinamicos ---
        self.mass = 30.0        # Masa total [kg]
        self.n_legs = 4
        self.n_dof_legs = 12    # 3 DoF * 4 patas

        # --- Parametros de la Simulacion ---
        self.state = ANYmalState.IDLE
        self.posx = x
        self.posy = y

    def start_task(self, dt=0.01): 
        """
        EJECUTAR ACCION -> caminar
        Recibe intervalo de actualizacon (delta time)
        
        Regrea arreglo de numpy con las posiciones en x y y despues de realizar la tarea 
        o None en caso de error o singularidad

        """
        return None

    def get_all_joint_angles(self):
        """Retorna un array de 12 elementos con todos los angulos articulares."""
        return np.concatenate([self.legs[name].q for name in self.LEG_NAMES])

    def set_all_joint_angles(self, q12):
        """Establece los 12 angulos articulares. q12 debe tener 12 elementos."""
        q12 = np.asarray(q12)
        assert q12.shape == (12,), f"Se esperan 12 angulos, se recibieron {q12.shape}"
        for i, name in enumerate(self.LEG_NAMES):
            self.legs[name].q = q12[3*i : 3*(i+1)].copy()

    def get_all_foot_positions(self):
        """Retorna un dict con la posicion 3D de cada pie en su marco de hombro."""
        return {name: self.legs[name].forward_kinematics()
                for name in self.LEG_NAMES}


def simulate_anymal_motion(anymal, joint_func, T=2.0, dt=0.005):
    """
    Simula una trayectoria articular del ANYmal.

    Entradas:
        anymal     : instancia de ANYmal
        joint_func : funcion joint_func(t) -> array de 12 elementos [rad]
                     (los 12 angulos articulares como funcion del tiempo)
        T          : tiempo total de simulacion [s]
        dt         : paso de tiempo [s]

    Retorna:
        dict con:
            't'      : array de tiempos [s]
            'q'      : array (n_steps, 12) con todos los angulos
            'feet'   : dict name -> array (n_steps, 3) con posicion del pie
    """
    n_steps = int(T / dt)
    log = {
        't': np.zeros(n_steps),
        'q': np.zeros((n_steps, 12)),
        'feet': {name: np.zeros((n_steps, 3)) for name in ANYmal.LEG_NAMES},
    }

    for i in range(n_steps):
        t = i * dt
        q12 = joint_func(t)
        anymal.set_all_joint_angles(q12)
        foot_positions = anymal.get_all_foot_positions()

        log['t'][i] = t
        log['q'][i, :] = q12
        for name in ANYmal.LEG_NAMES:
            log['feet'][name][i, :] = foot_positions[name]

    return log
