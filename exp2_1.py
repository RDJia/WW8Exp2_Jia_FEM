import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg


class func():
    '''
    3 functions of choice.
    '''

    def f_Binomial(self, x):
        fx = 2 * np.square(x) + 1
        return fx

    def f_Parabola(self, x):
        fx = -1 * np.square(x) + 5
        return fx

    def f_ln(self, x):
        fx = np.log(x)
        return fx

    def f_polynomial_order(self, x, order=3):
        fx = 2 * np.power(x, order) + 1
        return fx


class Interpolation():
    def __init__(self):
        self.x = np.array([1, 2, 3, 4, 5, 6, 7])
        self.y_bi = func().f_Binomial(self.x)
        self.y_pa = func().f_Parabola(self.x)
        self.y_ln = func().f_ln(self.x)

    def evaluationPoints(self, x1, x2, N=2, exclude=False):
        '''
        Create evenly distributed evaluation points.
        :param x1:  float/int, lower limit
        :param x2:  float/int, upper limit
        :param N:   int, Number of evaluation points
        :param exclude:     boolean, =True to exclude original points
        :return random_points:
                    list, x coordinates for evaluation points
        '''
        points = np.linspace(x1, x2, N + 2)
        if exclude == True:
            points = np.delete(points, [0, -1])

        return points

    def plot_interpolated_curve(self, x_points, y_points, lbl=''):
        '''
        Plot interpolated curves given x and y points.
        :param x_points: ndarray, x points
        :param y_points: ndarray, y points
        :param lbl: string, indicates the original function for this curve.
        '''
        plt.figure()

        plt.plot(x_points, y_points, label=lbl, linewidth=3)

        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.legend(prop={'size': 15})
        plt.show()

    def lagrange(self, x1, x2, y1, y2, xpoints):
        '''
        Create Lagrange function of 1 element (2 points)
        Approximate the values using random created points.
        :param x1, x2:  float/int, x coordinates of two points
        :param y1, y2:  float/int, y coordinates of two points
        :param xpoints: list/array, random evaluation points
        :return y_lagr: list/array, y values for evaluation points
        '''
        x1_full, x2_full = np.full(len(xpoints), x1), np.full(len(xpoints), x2)
        y_lagr = y1 * (xpoints - x2_full) / (x1_full - x2_full) + y2 * (xpoints - x1_full) / (x2_full - x1_full)

        return y_lagr

    def gradient(self, x1, x2, y1, y2):
        '''
        Calculate finite gradient for an element.
        :param x1, x2:  float/int, x coordinates of two points
        :param y1, y2:  float/int, y coordinates of two points
        :return grad: float, gradient of this element
        '''
        grad = (y2 - y1) / (x2 - x1)
        return grad

    def plotBinomial(self):
        '''
        Draw binomial curve and plot the interpolation points.
        '''
        plt.figure()

        # plot each element
        for element in range(np.shape(self.x)[0] - 1):
            x1, x2 = self.x[element], self.x[element + 1]  # x coordinates of two points
            y1, y2 = self.y_bi[element], self.y_bi[element + 1]  # y coordinates of two points

            x_points = self.evaluationPoints(x1, x2)  # evaluation points
            y_lagr = self.lagrange(x1, x2, y1, y2, x_points)  # generate y values by Lagrange

            # for labels of plot
            if element == 0:
                lbl_f = 'original curve'
                lbl_p = 'original points'
                lbl_e = 'evaluation points'
            else:
                lbl_f, lbl_p, lbl_e = None, None, None

            # plot the original function curve
            x_for_func = np.linspace(self.x[element], self.x[element + 1], 50)
            y_for_func = func().f_Binomial(x_for_func)
            plt.plot(x_for_func, y_for_func, c='black', label=lbl_f)
            plt.xticks(fontsize=15)
            plt.yticks(fontsize=15)
            # plot original points
            plt.scatter([self.x[element], self.x[element] + 1], [self.y_bi[element], self.y_bi[element + 1]], c='red',
                        marker='o', label=lbl_p, s=100)
            # plot evaluation points
            plt.scatter(x_points, y_lagr, c='blue', marker='x', label=lbl_e, s=100)

        plt.legend(prop={'size': 15})
        plt.show()

        # plot approximated function graph
        self.plot_interpolated_curve(self.x, self.y_bi, lbl='Binomial, element size = 1')

    def plotParabola(self):
        '''
        Draw parabola curve and plot the interpolation points.
        '''
        plt.figure()

        for element in range(np.shape(self.x)[0] - 1):
            x1, x2 = self.x[element], self.x[element + 1]
            y1, y2 = self.y_pa[element], self.y_pa[element + 1]

            x_points = self.evaluationPoints(x1, x2)
            y_lagr = self.lagrange(x1, x2, y1, y2, x_points)

            if element == 0:
                lbl_f = 'original curve'
                lbl_p = 'original points'
                lbl_e = 'evaluation points'
            else:
                lbl_f, lbl_p, lbl_e = None, None, None

            x_for_func = np.linspace(self.x[element], self.x[element + 1], 50)
            y_for_func = func().f_Parabola(x_for_func)
            plt.plot(x_for_func, y_for_func, c='black', label=lbl_f)
            plt.xticks(fontsize=15)
            plt.yticks(fontsize=15)

            plt.scatter([self.x[element], self.x[element] + 1], [self.y_pa[element], self.y_pa[element + 1]], c='red',
                        marker='o', label=lbl_p, s=100)
            plt.scatter(x_points, y_lagr, c='blue', marker='x', label=lbl_e, s=100)

        plt.legend(prop={'size': 15})
        plt.show()

        self.plot_interpolated_curve(self.x, self.y_pa, lbl='Parabola, element size = 1')

    def plotLn(self, finer=False):
        '''
        Draw natural log curve and plot the interpolation points.
        :param finer: boolean, True: element length = 0.5, otherwise 1
        '''
        plt.figure()
        if finer == True:
            x_finer = np.array([0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7])
        else:
            x_finer = self.x
        y_ln_finer = func().f_ln(x_finer)

        for element in range(np.shape(x_finer)[0] - 1):
            x1, x2 = x_finer[element], x_finer[element + 1]
            y1, y2 = y_ln_finer[element], y_ln_finer[element + 1]

            x_points = self.evaluationPoints(x1, x2)
            y_lagr = self.lagrange(x1, x2, y1, y2, x_points)

            if element == 0:
                lbl_f = 'original curve'
                lbl_p = 'original points'
                lbl_e = 'evaluation points'
            else:
                lbl_f, lbl_p, lbl_e = None, None, None

            x_for_func = np.linspace(x_finer[element], x_finer[element + 1], 50)
            y_for_func = func().f_ln(x_for_func)
            plt.plot(x_for_func, y_for_func, c='black', label=lbl_f)
            plt.xticks(fontsize=15)
            plt.yticks(fontsize=15)

            plt.scatter([x1, x2], [y1, y2], c='red',
                        marker='o', label=lbl_p, s=100)
            plt.scatter(x_points, y_lagr, c='blue', marker='x', label=lbl_e, s=100)

        plt.legend(prop={'size': 15})
        plt.show()

        if finer == True:
            self.plot_interpolated_curve(x_finer, y_ln_finer, lbl='Logarithm, element size = 0.5')
        else:
            self.plot_interpolated_curve(x_finer, y_ln_finer, lbl='Logarithm, element size = 1')

    def plotGrad(self, x_points, y_points, lbl='', eva=True, colo='blue'):
        '''
        Plot gradient graph and evaluation points.
        :param x_points: ndarray, x coordinates of original functional points
        :param x_points: ndarray, y coordinates of original functional points
        :param lbl: label name (function type)
        :param eva: boolean, whether show "evaluation points" label
        :param colo: string, color of the graph
        '''
        last_grad = 0

        for element in range(np.shape(x_points)[0] - 1):
            x1, x2 = x_points[element], x_points[element + 1]
            y1, y2 = y_points[element], y_points[element + 1]

            evalPoint = self.evaluationPoints(x1, x2, N=1, exclude=True)
            grad = self.gradient(x1, x2, y1, y2)

            if element == 0:
                plt.plot([x1, x2], [grad, grad], c=colo, label=lbl)
                plt.plot([x1, x1], [grad, grad], c=colo)
            else:
                plt.plot([x1, x2], [grad, grad], c=colo)
                plt.plot([x1, x1], [last_grad, grad], c=colo)

            if eva == True:
                if element == 0:
                    lbl_e = 'evaluation points e_size = 1'
                else:
                    lbl_e = ''
                plt.scatter(evalPoint, grad, c='red', marker='o', label=lbl_e)
            else:
                if element == 0:
                    lbl_e = 'evaluation points e_size = 0.3'
                else:
                    lbl_e = ''
                plt.scatter(evalPoint, grad, c='blue', marker='o', label=lbl_e)
            last_grad = grad

    def plotGradBi(self, finer=False):
        '''
        Plot finite gradient of binomial function, with evaluation points.
        :param finer: boolean, True: element length = 0.5, otherwise 1
        '''
        plt.figure(figsize=(8, 6))

        # x_finer = np.array([1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7])
        x_finer = np.linspace(1, 7, 20)
        x_org = self.x
        y_finer = func().f_Binomial(x_finer)
        y_org = func().f_Binomial(x_org)

        self.plotGrad(x_points=x_finer, y_points=y_finer, lbl='element size = 0.3', eva=False)
        self.plotGrad(x_points=x_org, y_points=y_org, lbl='element size = 1', colo='red')

        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.legend(prop={'size': 15})
        plt.show()

    def plotGradPa(self, finer=False):
        '''
        Plot gradient of parabola function, with evaluation points.
        :param finer: boolean, True: element length = 0.5, otherwise 1
        '''
        plt.figure(figsize=(8, 6))

        # x_finer = np.array([1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7])
        x_finer = np.linspace(1, 7, 20)
        x_org = self.x
        y_finer = func().f_Parabola(x_finer)
        y_org = func().f_Parabola(x_org)

        self.plotGrad(x_points=x_finer, y_points=y_finer, lbl='element size = 0.3', eva=False)
        self.plotGrad(x_points=x_org, y_points=y_org, lbl='element size = 1', colo='red')

        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.legend(prop={'size': 15})
        plt.show()

    def plotGradLn(self):
        '''
        Plot gradient of logn function, with evaluation points.
        '''
        plt.figure(figsize=(8, 6))

        # x_finer = np.array([1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7])
        x_finer = np.linspace(1, 7, 20)
        x_org = self.x
        y_finer = func().f_ln(x_finer)
        y_org = func().f_ln(x_org)

        self.plotGrad(x_points=x_finer, y_points=y_finer, lbl='element size = 0.3', eva=False)
        self.plotGrad(x_points=x_org, y_points=y_org, lbl='element size = 1', colo='red')

        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.legend(prop={'size': 15})
        plt.show()


class Integration():
    def quadrature(self, x1, x2, n, d):
        '''
        Do quadrature integration.
        :param x1:   float,  starting point of integral
        :param x2:   float,  ending point of integral
        :param n:    int,    degree of Newton-Cotes formula
        :param d:    int,    degree of polynomial
        :return quad_integration:   float, result of quadrature integration
        '''
        quad_integration = 0

        para_zeta, para_lambda = self.NewtonCotes(x1, x2, n)  # get lambda & zeta
        f_zeta = func().f_polynomial_order(para_zeta, order=d)  # get f(zeta)

        for i in range(n):
            quad_integration += para_lambda[i] * f_zeta[i]  # do summation
        quad_integration = quad_integration * (x2 - x1)

        print(f'integral of n={n}, d={d}:  ', quad_integration)
        return quad_integration

    def NewtonCotes(self, x1, x2, n):
        '''
        Create lambda and zeta given degree n.
        :param x1:  float,  starting point of integral
        :param x2:  float,  ending point of integral
        :param n:   int, degree of Newton-Cotes formula
        :return para_zeta:  ndarray, zeta values
        :return para_lambda:  ndarray, lambda values
        '''
        if n == 2:
            para_zeta = [x1, x2]
            para_lambda = [0.5, 0.5]
        elif n == 3:
            para_zeta = [x1, 0.5 * (x1 + x2), x2]
            para_lambda = [1 / 6, 4 / 6, 1 / 6]
        elif n == 4:
            para_zeta = [x1, (2 * x1 + x2) / 3, (x1 + 2 * x2) / 3, x2]
            para_lambda = [1 / 8, 3 / 8, 3 / 8, 1 / 8]

        return para_zeta, para_lambda

    def intg_polynomial_order(self, x1, x2, order=3):
        '''
        Return analytical solution of integral of the polynomial function.
        :param x1, x2: float, starting/ending point of the integral
        :param order: int, order of polynomial
        :return intg: function value(s) of given point(s)
        '''
        od = order + 1
        intg = 2 / od * (x2 ** od - x1 ** od) + x2 - x1

        print(f'=====real area of polynomial order d={order}:', intg, '=====')
        return intg


class LinearSys():

    def oneElement(self, x1=0, x2=50, E=210000, A=25, l=50, f_b=0, f_s=5):
        '''
        Special case for Task 5_3: one element with BC on first node (x=0, u=0).
        Test 1 element given following values.
        :param x1, x2:  float,  starting/ending position (mm)
        :param E:  float,  E-Modul (N/mm^2)
        :param A:  float,  cross section area (mm^2)
        :param l:  float,  element length (mm)
        :param f_b, f_s:  float,  force on starting/ending point (N)
        '''

        Ke = E * A * 2 * l / (x1 - x2) ** 2
        u2 = (f_b + f_s) / Ke

        print('displacement of ending point ≈', np.round(u2, 7))

    def assembleLinear(self, c=1, N=5, BCs=None, f_b=-5, f_s=5, to_print=True):
        '''
        * Assembly routine of stiffness matrix for linear elements.
        * Adjusted to support boundary conditions (BCs) on any node.
        :param c:  float, stiffness multiplier.
        :param N:  int, number of nodes.
        :param BCs:  list/array, displacement boundary conditions for each node (None if no BC).
        :param f_b, f_s:  float, force on starting/ending point (right hand side) (N).
        :param to_print:  boolean, whether to print the parameters.
        :return K: ndarray, global stiffness matrix.
        '''
        K = np.zeros((N, N))
        for ele in range(N - 1):
            K[ele][ele] += 1
            K[ele][ele + 1] -= 1
            K[ele + 1][ele] -= 1
            K[ele + 1][ele + 1] += 1
            if to_print:
                print(f'--------# element {ele} #:--------')
                print('###Global stiffness matrix### :\n', K)

        # Create displacement vector `u` based on BCs
        u = np.full((N, 1), 'unknown')
        if BCs is not None:
            for i, bc in enumerate(BCs):
                if bc is not None:
                    u[i] = bc

        # Right-hand side
        rhs = np.zeros((N, 1))
        rhs[0], rhs[-1] = f_b, f_s

        if to_print:
            print('###displacement vector### :\n', u)
            print('--------# left hand side of the equation #--------')
            print(f'{c} * {K}\n · {u}')
            print('--------# right hand side of the equation #--------')
            print(f'f_b + f_s = \n{rhs}')

        print('###K*c### :\n', K * c)

        return K * c, u, rhs

    def calc_c(self, A, E, L):
        '''
        Calculate c value (multiplier for stiffness matrix).
        :param A: float, section area. (mm^2)
        :param E: float, E-Modulus. (N/mm^2)
        :param L: float, length of the bar. (mm)
        :return c: float, c value.
        '''
        c = A * E / L
        return c

    def permutationMat(self, N, uend=None):
        '''
        * Create a permutation matrix.
        * The value on the first node is assumed to be known. (Dirichlet BC)
        :param N: int, number of nodes.
        :param uend: float, = None if no Dirichlet BC is applied on the last node.
        :return P: ndarray, permutation matrix.
        '''
        if uend is None:  # P for case with only 1 Dirichlet BC at node 1.
            P = np.zeros([N - 1, N])
            for row in range(N - 1):
                for col in range(N):
                    if row + 1 == col:
                        P[row, col] = 1
        else:  # P for Dirichlet BCs at both sides
            P = np.zeros([N - 2, N])
            for row in range(N - 2):
                for col in range(N):
                    if row + 1 == col:
                        P[row, col] = 1
        return P

    def deleteRow(self, originMat, uend=None, axis=0):
        '''
        * Delete row(s)/column(s) in the linear equations where
          the nodal values are already given by Dirichlet BCs.
        * The value on the first node is assumed to be known. (Dirichlet BC)
        :param originMat: ndarray, original matrix.
        :param uend: float, = None if no Dirichlet BC is applied on the last node.
        :param axis: int, = 0 to remove row, = 1 to remove column
        :return reducedMat: ndarray, reduced matrix.
        '''
        # print('originMat: ', originMat)
        # print('shape:', np.shape(originMat))
        reducedMat = np.delete(originMat, 0, axis=axis)
        # print('reducedMat:', reducedMat)
        if uend is not None:  # for Dirichlet BCs at both sides.
            reducedMat = np.delete(reducedMat, -1, axis=axis)
        return reducedMat

    def solveLinear(self, N=6, BCs=None, F=-5, c=1, to_print=True):
        '''
        * Solve linear equations for given conditions, with arbitrary displacement BCs.
        :param N:  int, number of nodes.
        :param BCs:  list/array, displacement boundary conditions for each node (None if no BC).
        :param F:  float, force applied on the last node.
        :param c:  float, multiplier of global stiffness matrix.
        :param to_print:  boolean, whether to print the parameters.
        :return d:  ndarray, solution, displacement matrix.
        '''
        if to_print:
            print('==============================')
            print(f'Testing: N = {N}, BCs = {BCs}, F = {F}, c = {c}')

        # Create global stiffness matrix
        K, u, rhs = self.assembleLinear(N=N, BCs=BCs, f_b=0, f_s=F, c=c, to_print=to_print)

        # Identify nodes with Dirichlet BCs
        constrained_nodes = [i for i, bc in enumerate(BCs) if bc is not None]

        # Remove rows and columns corresponding to constrained nodes
        Kr = np.delete(K, constrained_nodes, axis=0)
        Kr = np.delete(Kr, constrained_nodes, axis=1)

        # Reduced right-hand side
        fr = np.delete(rhs, constrained_nodes, axis=0)

        # Adjust fr to account for constraints
        for node in constrained_nodes:
            fr -= np.delete(K[:, node].reshape(-1, 1), constrained_nodes, axis=0) * BCs[node]

        # Solve reduced system
        dr = scipy.linalg.solve(Kr, fr)
        d = np.zeros((N, 1))
        for i, bc in enumerate(BCs):
            if bc is not None:
                d[i] = bc

        # Fill res. in the displacement vector
        unconstrained_nodes = [i for i in range(N) if i not in constrained_nodes]
        for i, node in enumerate(unconstrained_nodes):
            d[node] = dr[i]

        if to_print:
            print('=== Displacement vector ===\n', d)
            print('=== Reduced stiffness matrix Kr ===\n', Kr)
            print('=== Reduced RHS vector fr ===\n', fr)
            print('=== Solved displacement vector dr ===\n', dr)

        print("******d*******")
        print(d)
        return d

    def postProcess(self, BC_1, A, E, L, F, N=6):
        '''
        * Post process:
          Calculate and plot stress tensors under different BCs.
        :param BC_1: ndarray, boundary condition array for all nodes (None if no BC on a node).
        :param A: float, section area. (mm^2)
        :param E: float, E-Modulus. (N/mm^2)
        :param L: float, length of the bar. (mm)
        :param F: float, force applied on the end.
        :param N: int, number of elements.
        :return stress_fields: ndarray, stress fields based on different BCs.
        :return x_points: ndarray, indicating node positions.
        '''
        d_1 = []  # displacements
        N_BCs = len(BC_1)  # number of BC sets
        x_points = np.linspace(0, L, N)  # Node positions

        print(f'Number of BCs = {N_BCs}, Number of elements = {N}')

        # Solve for displacements under each set of BCs
        for BCs in BC_1:
            d_1.append(self.solveLinear(N=N, BCs=BCs, F=F, c=A * E / L, to_print=True))

        stress_fields = []  # List of all stress fields

        for idx, (BCs, d) in enumerate(zip(BC_1, d_1)):
            grad_1 = np.gradient(d.squeeze(), x_points)  # Gradient (strain)
            stress = E * grad_1  # Stress calculation
            stress_fields.append(stress)

            print(f'BC: {BCs}\nDisplacement: {d.T}\nStress: {stress}')

            # Plot results
            plt.figure(figsize=(10, 8), dpi=80)

            # Stress field
            ax_sf = plt.subplot(122)
            ax_sf.plot(x_points, stress, 'g-o', label='Stress Field')
            ax_sf.set_title('Stress Field (Pa)')
            ax_sf.set_xlabel('Position (mm)')
            ax_sf.set_ylabel('Stress (Pa)')
            ax_sf.legend()

            # Displacement
            ax_d = plt.subplot(221)
            ax_d.plot(x_points, d, '-o', label='Displacement')
            ax_d.set_title(f'Displacement (u1={BCs[0]}, uend={BCs[-1]})')
            ax_d.set_xlabel('Position (mm)')
            ax_d.set_ylabel('Displacement (mm)')
            ax_d.legend()

            # Gradient (strain)
            ax_g = plt.subplot(223)
            ax_g.plot(x_points, grad_1, 'r-o', label='Gradient')
            ax_g.set_title('Gradient (Strain)')
            ax_g.set_xlabel('Position (mm)')
            ax_g.set_ylabel('Gradient')
            ax_g.legend()

            # Overall title
            plt.suptitle(f'Boundary Conditions: {BCs}\nN = {N}, F = {F} N, c = {A * E / L}')
            plt.tight_layout()
            plt.show()

        return stress_fields, x_points


if __name__ == '__main__':
    BCs = np.array([[0, None, None, None, None, None],
                    [15, None, None, None, None, None],
                    [45, None, None, None, None, None],
                    [15, None, None, None, None, 30],
                    [15, None, None, 15, None, None],
                    [0, None, None, None, 30, None]])
    stress_fields, x_points = LinearSys().postProcess(BC_1=BCs, A=25, E=210000, L=50, F=10, N=6)





