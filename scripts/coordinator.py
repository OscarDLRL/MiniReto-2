from anymal_gait import ANYmal
from husky_pusher import HuskyA200
from puzzlebot import PuzzleBot 
from puzzlebot_arm import PuzzleBotArm

class Coordinator:
    def __init__(self, dt):
        self.puzzlebot1 = PuzzleBot()
        self.puzzlebot2 = PuzzleBot()
        self.puzzlebot3 = PuzzleBot()
        self.anymal = ANYmal(0.0,0.0)
        self.husky = HuskyA200()

        self.dt = dt

    def start_sim(self):
        huskysim = self.husky.start_task(self.dt)
        if huskysim is None:
            print("Algo salio mal en el calculo de trayectoria del Husky")
            return

        anymalsim = self.anymal.start_task(self.dt)
        if anymalsim is None:
            print("Algo salio mal en el calculo de trayectoria del ANYmal")
            return

        puzzlebot1sim = self.puzzlebot1.start_task(self.dt)
        if puzzlebot1sim is None:
            print("Algo salio mal en el calculo de trayectoria del primer puzzlebot")
            return

        puzzlebot2sim = self.puzzlebot2.start_task(self.dt)
        if puzzlebot2sim is None:
            print("Algo salio mal en el calculo de trayectoria del segundo puzzlebot")
            return

        puzzlebot3sim = self.puzzlebot3.start_task(self.dt)
        if puzzlebot3sim is None:
            print("Algo salio mal en el calculo de trayectoria del tercer puzzlebot")
            return
        
        print ("Simulaciones completadas")

        #TODO Graficar las simulaciones y guardarlas en la carpeta de results 

    def simulate_husky(self):
        huskysim = self.husky.start_task(self.dt)
        if huskysim is None:
            print("Algo salio mal en el calculo de trayectoria del Husky")
            return

        #TODO Graficar las simulaciones y guardarlas en la carpeta de results 
    
    def simulate_anymal(self):
        anymalsim = self.anymal.start_task(self.dt)
        if anymalsim is None:
            print("Algo salio mal en el calculo de trayectoria del ANYmal")
            return

        #TODO Graficar las simulaciones y guardarlas en la carpeta de results 

    def simulate_puzzlebot1(self):
        puzzlebot1sim = self.puzzlebot1.start_task(self.dt)
        if puzzlebot1sim is None:
            print("Algo salio mal en el calculo de trayectoria del primer puzzlebot")
            return
        #TODO Graficar las simulaciones y guardarlas en la carpeta de results 

    def simulate_puzzlebot2(self):
        puzzlebot2sim = self.puzzlebot2.start_task(self.dt)
        if puzzlebot2sim is None:
            print("Algo salio mal en el calculo de trayectoria del segundo puzzlebot")
            return
        #TODO Graficar las simulaciones y guardarlas en la carpeta de results 

    def simulate_puzzlebot3(self):
        puzzlebot3sim = self.puzzlebot3.start_task(self.dt)
        if puzzlebot3sim is None:
            print("Algo salio mal en el calculo de trayectoria del tercer puzzlebot")
            return
        #TODO Graficar las simulaciones y guardarlas en la carpeta de results 
