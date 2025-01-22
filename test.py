import exp2
from exp2_1 import *
import matplotlib.pyplot as plt

class Test():
    def task4_1(self):
        print('###### Testing Task 4.1 ######')
        a = Interpolation()
        a.plotBinomial()
        a.plotParabola()
        a.plotLn()
        a.plotLn(finer=True)

    def task4_2(self):
        print('###### Testing Task 4.2 ######')
        a = Interpolation()
        a.plotGradBi()
        a.plotGradPa()
        a.plotGradLn()

    def task5_2(self):
        print('###### Testing Task 5.2 ######')
        b = Integration()
        #b.exactIntg(1, 3, d=1)
        b.intg_polynomial_order(1, 3, 1)
        b.quadrature(1, 3, n=2, d=1)
        b.quadrature(1, 3, n=3, d=1)
        b.quadrature(1, 3, n=4, d=1)

        #b.exactIntg(1, 3, d=2)
        b.intg_polynomial_order(1, 3, 2)
        b.quadrature(1, 3, n=2, d=2)
        b.quadrature(1, 3, n=3, d=2)
        b.quadrature(1, 3, n=4, d=2)

        #b.exactIntg(1, 3, d=3)
        b.intg_polynomial_order(1, 3, 3)
        b.quadrature(1, 3, n=2, d=3)
        b.quadrature(1, 3, n=3, d=3)
        b.quadrature(1, 3, n=4, d=3)

        #b.exactIntg(1, 3, d=4)
        b.intg_polynomial_order(1, 3, 4)
        b.quadrature(1, 3, n=2, d=4)
        b.quadrature(1, 3, n=3, d=4)
        b.quadrature(1, 3, n=4, d=4)

        #b.exactIntg(1, 3, d=5)
        b.intg_polynomial_order(1, 3, 5)
        b.quadrature(1, 3, n=2, d=5)
        b.quadrature(1, 3, n=3, d=5)
        b.quadrature(1, 3, n=4, d=5)

        #b.exactIntg(1, 3, d=6)
        b.intg_polynomial_order(1, 3, 6)
        b.quadrature(1, 3, n=2, d=6)
        b.quadrature(1, 3, n=3, d=6)
        b.quadrature(1, 3, n=4, d=6)

        #b.exactIntg(1, 3, d=7)
        b.intg_polynomial_order(1, 3, 7)
        b.quadrature(1, 3, n=2, d=7)
        b.quadrature(1, 3, n=3, d=7)
        b.quadrature(1, 3, n=4, d=7)

    def task5_3(self):
        print('###### Testing Task 5.3 ######')
        LinearSys().oneElement()

    def task6_2_3(self):
        print('###### Testing Task 6.2 + 6.3 ######')
        LinearSys().assembleLinear()

    def task7_2(self):
        print('###### Testing Task 7.2 ######')
        print('1. Testing spring system with one Dirichlet BC on the first node:')
        N = 3 # Number of elements

        r1 = LinearSys().solveLinear(N=N, u1=0, F=12, c=10)
        r2 = LinearSys().solveLinear(N=N, u1=1.6, F=12, c=10)
        r3 = LinearSys().solveLinear(N=N, u1=3.2, F=12, c=10)
        nodes = np.linspace(1, N, N)
        plt.figure()
        plt.plot(nodes, r1, 'ro-', label='u1=0 mm')
        plt.plot(nodes, r2, 'bo-', label='u1=1.6 mm')
        plt.plot(nodes, r3, 'go-', label='u1=3.2 mm')
        plt.xlabel('nodes')
        plt.ylabel('displacement (mm)')
        plt.legend(prop={'size': 15})
        plt.xticks(nodes, fontsize=15)
        plt.yticks(fontsize=15)
        plt.show()

        print('\n2. Testing spring system with Dirichlet BC at both sides:')
        N = 7 # Number of elements
        rr1 = LinearSys().solveLinear(N=N, u1=-0.6, uend=1.6, F=12, c=1)
        rr2 = LinearSys().solveLinear(N=N, u1=0, uend=1.6, F=12, c=1)
        rr3 = LinearSys().solveLinear(N=N, u1=0.6, uend=0.6, F=12, c=1)
        nodes = np.linspace(1, N, N)
        plt.figure()
        plt.plot(nodes, rr1, 'ro-', label='u1=-0.6, u4=1.6')
        plt.plot(nodes, rr2, 'bo-', label='u1=0, u4=1.6')
        plt.plot(nodes, rr3, 'go-', label='u1=0.6, u4=0.6')
        plt.xlabel('nodes')
        plt.ylabel('displacement (mm)')
        plt.xticks(nodes, fontsize=15)
        plt.yticks(fontsize=15)
        plt.legend(prop={'size': 15})
        plt.show()

    def task8_1(self):
        print('###### Testing Task 8.1 ######\n')
        N = 6 # number of nodes

        BCs = np.array([[0, None, None, None, None, None],
                        [15, None, None, None, None, None],
                        [45, None, None, None, None, None],
                        [15, None, None, None, None, 30],
                        [15, None, None, 15, None, None],
                        [0, None, None, None, 30, None]])

        stress_fields, x_points = LinearSys().postProcess(BC_1=BCs, A=25, E=210000, L=50, F=100, N=N)

        print("stress_fields:")
        print(stress_fields)

        """
        # plot some stress fields in a figure for comparison
        plt.figure()
        ax = plt.plot(x_points, stress_fields[0], '-o', label='u1 = 0 mm', markersize=10)
        plt.plot(x_points, stress_fields[1], '-o', label='u1 = 15 mm')
        plt.plot(x_points, stress_fields[3], '-o', label='u1 = 15 mm, u6 = 30 mm')
        plt.plot(x_points, stress_fields[4], '-o', label='u1 = 15 mm, u4 = 15 mm')
        plt.xlabel('nodes')
        plt.ylabel('stress (Pa)')
        plt.legend()
        plt.show()
        """


if __name__ == '__main__':
    # Test().task4_1()
    # Test().task4_2()
    # Test().task5_2()
    #  Test().task5_3()
    # Test().task6_2_3()
    # Test().task7_2()
    Test().task8_1()
