import numpy as np

class PuzzleBotArm:
    """Mini brazo planar de 3 DoF montado sobre un PuzzleBot.

    Configuracion: base rotacional (q1) + 2 eslabones en plano vertical (q2, q3).
    """

    def __init__(self, l1=0.10, l2=0.08, l3=0.06):
        self.l1, self.l2, self.l3 = l1, l2, l3
        self.q = np.zeros(3)

    def forward_kinematics(self, q=None):
        """TODO: calcular posicion (x, y, z) del efector final."""
        if q is not None:
            self.q = q
        # IMPLEMENTAR
        pass

    def inverse_kinematics(self, p_des):
        """TODO: IK geometrica cerrada -> (q1, q2, q3)."""
        # IMPLEMENTAR
        pass

    def jacobian(self, q=None):
        """TODO: Jacobiano 3x3 analitico."""
        # IMPLEMENTAR
        pass

    def force_to_torque(self, f_tip):
        """Mapea fuerza en el efector a torques articulares: tau = J^T * f."""
        J = self.jacobian()
        return J.T @ f_tip

    def grasp_box(self, box_pos, grip_force=5.0):
        """TODO: mover el efector a box_pos y aplicar fuerza de grip.

        Pasos:
        1. Generar trayectoria cartesiana desde pose actual a box_pos.
        2. Para cada punto, resolver IK.
        3. Detectar contacto y aplicar grip_force vertical con tau = J^T f.
        """
        # IMPLEMENTAR
        pass