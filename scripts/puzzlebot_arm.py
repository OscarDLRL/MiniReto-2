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

        q1, q2, q3 = self.q

        r = self.l2 * np.cos(q2) + self.l3 * np.cos(q2 + q3)
        x = r * np.cos(q1)
        y = r * np.sin(q1)
        z = self.l1 + self.l2 * np.sin(q2) + self.l3 * np.sin(q2 + q3)

        return np.array([x, y, z])

    def inverse_kinematics(self, p_des):
        """TODO: IK geometrica cerrada -> (q1, q2, q3)."""
        x, y, z = p_des

        q1 = np.arctan2(y, x)

        r = np.sqrt(x**2 + y**2)
        z_rel = z - self.l1

        D = (r**2 + z_rel**2 - self.l2**2 - self.l3**2) / (2 * self.l2 * self.l3)

        if D < -1 or D > 1:
            raise ValueError("Punto fuera del espacio de trabajo.")

        q3 = np.arctan2(np.sqrt(1 - D**2), D)

        q2 = np.arctan2(z_rel, r) - np.arctan2(
            self.l3 * np.sin(q3),
            self.l2 + self.l3 * np.cos(q3)
        )

        return np.array([q1, q2, q3])

    def jacobian(self, q=None):
        """TODO: Jacobiano 3x3 analitico."""
        if q is not None:
            self.q = q

        q1, q2, q3 = self.q

        r = self.l2 * np.cos(q2) + self.l3 * np.cos(q2 + q3)
        dr_dq2 = -self.l2 * np.sin(q2) - self.l3 * np.sin(q2 + q3)
        dr_dq3 = -self.l3 * np.sin(q2 + q3)

        J = np.array([
            [-r * np.sin(q1), np.cos(q1) * dr_dq2, np.cos(q1) * dr_dq3],
            [ r * np.cos(q1), np.sin(q1) * dr_dq2, np.sin(q1) * dr_dq3],
            [0, self.l2 * np.cos(q2) + self.l3 * np.cos(q2 + q3), self.l3 * np.cos(q2 + q3)]
        ])

        return J

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
        p0 = self.forward_kinematics()
        N = 10
        traj = [p0 + (i / N) * (box_pos - p0) for i in range(1, N + 1)]

        q_traj = []
        for p in traj:
            q_new = self.inverse_kinematics(p)
            self.q = q_new
            q_traj.append(q_new)

        f_tip = np.array([0.0, 0.0, -grip_force])
        tau = self.force_to_torque(f_tip)

        return {
            "trajectory_positions": traj,
            "trajectory_joints": q_traj,
            "final_q": self.q,
            "applied_force": f_tip,
            "joint_torques": tau
        }