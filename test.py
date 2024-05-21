import numpy as np

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
        ls = LinearSys()
        E = 210000
        A = 25
        Le = 50
        Ke = ls.element_stiffness_matrix(E, A, Le=Le)
        #print("Element stiffness matrix:\n", Ke)

        #K = ls.assembleLinear(Ke=Ke, N=1, u1 = 0.0, uend = 50.0, f_b = 0, f_s = 5, to_print=False)
        #print("Global stiffness matrix:\n", K)

        d = ls.solveLinear(Ke=Ke, N=1, u1 = 0.0, uend = None, F = 5)
        print("Solved displacement vector:\n", d)

    def task6_2_3(self):
        print('###### Testing Task 6.2 + 6.3 ######')
        # Assuming 10 elements, then c = E*A/Le = 0.2*25/5 = 1
        ls = LinearSys()
        E = 0.2
        A = 25
        L = 50
        N = 10
        Le = L/N
        Ke = ls.element_stiffness_matrix(E, A, Le=Le)
        print("Element stiffness matrix:\n", Ke)

        K = ls.assembleLinear(Ke=Ke, N=N, u1=0.0, uend=50.0, f_b=0, f_s=5, to_print=True)
        print("Global stiffness matrix:\n", K)

    def task7_2(self):
        print('###### Testing Task 7.2 ######')
        print('1. Testing spring system with one Dirichlet BC on the first node:')
        ls = LinearSys()
        E = 210000
        A = 25
        N = 4
        L = 50
        Le = L/N
        Ke = ls.element_stiffness_matrix(E, A, Le=Le)

        r1 = ls.solveLinear(N=3, Ke=Ke, u1=0, F=12)
        r2 = ls.solveLinear(N=3, Ke=Ke, u1=0.6e-2, F=12)
        r3 = ls.solveLinear(N=3, Ke=Ke, u1=1.2e-2, F=12)
        nodes = np.linspace(0, 3, 4)
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
        rr1 = LinearSys().solveLinear(N=4, Ke=Ke, u1=-0.6, uend=1.6, F=12, to_print=False)
        rr2 = LinearSys().solveLinear(N=4, Ke=Ke, u1=0, uend=1.6, F=12, to_print=False)
        rr3 = LinearSys().solveLinear(N=4, Ke=Ke, u1=0.6, uend=0.6, F=12, to_print=False)
        nodes = np.linspace(1, 4, 5)
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
        stress_fields, x_points, x_elements = LinearSys().postProcess(BC_1=np.array([[0, None], [15, None], [30, None], [15,15], [15,30], [30,60]])\
                                , A=25, E=210000, L=50, F=10, N=6 )

        # plot some stress fields in a figure for comparison
        plt.figure()
        print("x_points:", x_points)
        print("stress_fields:", stress_fields)

        ax = plt.plot(x_elements, stress_fields[0], '-o', label='u1 = 0 mm, u6 = None', markersize=10)
        plt.plot(x_elements, stress_fields[1], '-o', label='u1 = 15 mm, u6 = None')
        plt.plot(x_elements, stress_fields[4], '-o', label='u1 = 15mm, u6 = 30 mm')
        plt.plot(x_elements, stress_fields[5], '-o', label='u1 = 15 mm, u6 = 30 mm')
        plt.xlabel('elements')
        plt.ylabel('stress (Pa)')
        plt.legend()
        plt.show()


if __name__ == '__main__':
    # Test().task4_1()
    # Test().task4_2()
    # Test().task5_2()
    # Test().task5_3()
    # Test().task6_2_3()
    # Test().task7_2()
    Test().task8_1()

