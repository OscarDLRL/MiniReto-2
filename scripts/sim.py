from coordinator import Coordinator
import matplotlib as plt

if __name__ == "__main__":
    print("\n" + "#" * 70)
    print("#   ROBOTS MOVILES TERRESTRES - Ejecutando todos los demos")
    print("#" * 70 + "\n")

    coordinator = Coordinator(0.005)

# CAMBIAR PARA SIMULAR EL MODULO ADECUADO (HUSKY; PUZZLEBOTS, O ANYMAL)
    coordinator.start_sim()

    print("\n" + "=" * 70)
    print("Todos los demos ejecutados. Revisa las imagenes *.png generadas.")
    print("=" * 70)

    # Mostrar todas las figuras al final
    #plt.show()
