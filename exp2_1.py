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
        self.x = np.array([1,2,3,4,5,6,7])
        self.y_bi = func().f_Binomial(self.x)
        self.y_pa = func().f_Parabola(self.x)
        self.y_ln = func().f_ln(self.x)

    def evaluationPoints(self,x1, x2, N = 2, exclude=False):
        '''
        Create evenly distributed evaluation points.
        :param x1:  float/int, lower limit
        :param x2:  float/int, upper limit
        :param N:   int, Number of evaluation points
        :param exclude:     boolean, =True to exclude original points
        :return random_points:
                    list, x coordinates for evaluation points
        '''
        points = np.linspace(x1, x2, N+2)
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
        y_lagr = y1 * (xpoints - x2_full)/(x1_full - x2_full) + y2 * (xpoints - x1_full)/(x2_full - x1_full)

        return y_lagr

    def gradient(self, x1, x2, y1, y2):
        '''
        Calculate finite gradient for an element.
        :param x1, x2:  float/int, x coordinates of two points
        :param y1, y2:  float/int, y coordinates of two points
        :return grad: float, gradient of this element
        '''
        grad = (y2-y1) / (x2-x1)
        return grad

    def plotBinomial(self):
        '''
        Draw binomial curve and plot the interpolation points.
        '''
        plt.figure()

        # plot each element
        for element in range(np.shape(self.x)[0] - 1):
            x1, x2 = self.x[element], self.x[element+1]         # x coordinates of two points
            y1, y2 = self.y_bi[element], self.y_bi[element+1]   # y coordinates of two points

            x_points = self.evaluationPoints(x1, x2)            # evaluation points
            y_lagr = self.lagrange(x1, x2, y1, y2, x_points)    # generate y values by Lagrange

            # for labels of plot
            if element == 0:
                lbl_f = 'original curve'
                lbl_p = 'original points'
                lbl_e = 'evaluation points'
            else:
                lbl_f, lbl_p, lbl_e = None, None, None

            # plot the original function curve
            x_for_func = np.linspace(self.x[element], self.x[element+1], 50)
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

        for element in range(np.shape(x_finer)[0]-1):
            x1, x2 = x_finer[element], x_finer[element+1]
            y1, y2 = y_ln_finer[element], y_ln_finer[element+1]

            x_points = self.evaluationPoints(x1, x2)
            y_lagr = self.lagrange(x1, x2, y1, y2, x_points)

            if element == 0:
                lbl_f = 'original curve'
                lbl_p = 'original points'
                lbl_e = 'evaluation points'
            else:
                lbl_f, lbl_p, lbl_e = None, None, None

            x_for_func = np.linspace(x_finer[element], x_finer[element+1], 50)
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
            x1, x2 = x_points[element], x_points[element+1]
            y1, y2 = y_points[element], y_points[element+1]

            evalPoint = self.evaluationPoints(x1, x2, N=1, exclude=True)
            grad = self.gradient(x1, x2, y1, y2)

            if element == 0:
                plt.plot([x1,x2], [grad, grad], c=colo, label=lbl)
                plt.plot([x1,x1], [grad, grad], c=colo)
            else:
                plt.plot([x1,x2], [grad, grad], c=colo)
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

        #x_finer = np.array([1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7])
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

        #x_finer = np.array([1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7])
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

        #x_finer = np.array([1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7])
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

        para_zeta, para_lambda = self.NewtonCotes(x1, x2, n)    # get lambda & zeta
        f_zeta = func().f_polynomial_order(para_zeta, order=d)  # get f(zeta)

        for i in range(n):
            quad_integration += para_lambda[i] * f_zeta[i]      # do summation
        quad_integration = quad_integration * (x2 - x1)

        print(f'integral of n={n}, d={d}:  ', quad_integration)
        return quad_integration

    @staticmethod
    def NewtonCotes(x1, x2, n):
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
            para_zeta = [x1, 0.5*(x1+x2), x2]
            para_lambda = [1/6, 4/6, 1/6]
        elif n == 4:
            para_zeta = [x1, (2*x1+x2)/3, (x1+2*x2)/3, x2]
            para_lambda = [1/8, 3/8, 3/8, 1/8]

        return para_zeta, para_lambda

    def intg_polynomial_order(self, x1, x2, order=3):
        '''
        Return analytical solution of integral of the polynomial function.
        :param x1, x2: float, starting/ending point of the integral
        :param order: int, order of polynomial
        :return intg: function value(s) of given point(s)
        '''
        od = order + 1
        intg = 2/od * (x2**od - x1**od) + x2 - x1

        print(f'=====real area of polynomial order d={order}:', intg, '=====')
        return intg

class LinearSys():
    def linear_shape_functions(self, xi):
        """
        linear shape function
        :param xi: Evaluation point in the reference domain [-1, 1]
                   xi=0: center, xi=-1: left end, xi=1: right end.
        :return: Array of shape functions [N1, N2]
        """
        N1 = 0.5 * (1 - xi)
        N2 = 0.5 * (1 + xi)
        return np.array([N1, N2])

    @staticmethod
    def linear_shape_function_derivatives(Le):
        '''
        Calculate the derivatives of linear shape functions in the reference domain.
        :param Le: float.
        :return: Array of shape function derivatives [dN1_dxi, dN2_dxi]
        '''
        dN1_dxi = -1/Le
        dN2_dxi = 1/Le
        return np.array([dN1_dxi, dN2_dxi])

    def element_stiffness_matrix(self, E, A, Le):
        """
        Assemble element stiffness matrix based on linear shape func and numerical integration.
        :param E, A, Le: float, Young’s Modulus, sectional area and element length.
        :return: Element stiffness matrix Ke
        """
        B = self.linear_shape_function_derivatives(Le).reshape(1,2)
        # print("B:\n", B)
        # print(B.shape)
        # print("B.T:\n", B.T)
        #print(B.T.shape)
        Ke = E * A * (B.T @ B) * Le

        return Ke

    def assembleLinear(self, Ke=None, N = 5, u1 = 1.0, uend = 5.0, f_b = -5, f_s = 5, to_print = True):
        '''
        * Task 6.2: assembly routine of stiffness matrix for linear elements.
        * The element stiffness matrix Ke = np.array([[1,-1],[-1,1]])
          indicates a linear relationship of values between neighbouring elements.
        * The global stiffness matrix is assembled by adding up element
          stiffness matricesin diagonal direction.
        :param Ke:  ndarray, element stiffness matrix.
        :param N:  int, number of elements.
        :param u1:      float, displacement on first node (Dirichlet BC)
        :param uend:    float, displacement on last node (Dirichlet BC)
                        = None if no Dirichlet BC on it
        :param f_b, f_s:  float,  force on starting/ending point (right hand side) (N)
        :param to_print:    boolean, whether to print the parameters
        :return K: ndarray, global stiffness matrix.
        '''
        # Check element stiffness matrix.
        if Ke is None:
            raise ValueError("The parameter Ke (element stiffness matrix) must be provided.")
        else:
            print("Assembling global stiffness matrix with Ke=\n", Ke)
        # fill the global stiffness matrix
        K = np.zeros((N+1, N+1))
        for ele in range(N):
            # Global node indices for the current element
            node1 = ele
            node2 = ele + 1

            # Assemble the local Ke into the global K
            K[node1, node1] += Ke[0, 0]
            K[node1, node2] += Ke[0, 1]
            K[node2, node1] += Ke[1, 0]
            K[node2, node2] += Ke[1, 1]

            if to_print == True:
                print(f'--------# element {ele} #:--------')
                print('###Globall stiffness matrix### :\n', K)

        # generate displacement vector u
        u = np.full((N+1,1),'unknown')
        u[0] = u1
        if uend is not None:
            u[-1] = uend
        # right hand side
        rhs = np.zeros((N+1,1))
        rhs[0], rhs[-1] = f_b, f_s

        if to_print == True:
            print('###displacement vector### :\n', u)
            print('--------# left hand side of the equation #--------')
            print(f'{Ke}\n · {u}')
            print('--------# right hand side of the equation #--------')
            print(f'f_b + f_s = \n{rhs}')

        return K

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

    def permutationMat(self, N, uend = None):
        '''
        * Create a permutation matrix.
        * The value on the first node is assumed to be known. (Dirichlet BC)
        :param N: int, number of nodes.
        :param uend: float, = None if no Dirichlet BC is applied on the last node.
        :return P: ndarray, permutation matrix.
        '''
        if uend is None:        # P for case with only 1 Dirichlet BC at node 1.
            P = np.zeros([N-1, N])
            for row in range(N-1):
                for col in range(N):
                    if row + 1 == col:
                        P[row,col] = 1
        else:                   # P for Dirichlet BCs at both sides
            P = np.zeros([N - 2, N])
            for row in range(N-2):
                for col in range(N):
                    if row + 1 == col:
                        P[row,col] = 1
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
        # print('reducedMat after deleting first element:', reducedMat)
        if uend is not None:  # for Dirichlet BCs at both sides.
            reducedMat = np.delete(reducedMat, -1, axis=axis)
            # print('reducedMat after potentially deleting last element:', reducedMat)
        return reducedMat

    def solveLinear(self, N=6, Ke=None, u1=0, uend=None, F=12, to_print=True):
        '''
        * Solve linear equations for given conditions,
          acquire displacement result on the nodes.
        * Reduced matrices are created by deleting
          rows and columns.
        :param N:       int, number of elements
        :param Ke:  ndarray, element stiffness matrix.
        :param u1:      float, displacement on first node (Dirichlet BC)
        :param uend:    float, displacement on last node (Dirichlet BC)
                        = None if no Dirichlet BC on it
        :param F:       float, force applied on the last node
        :param to_print:    boolean, whether to print the parameters
        :return d:      ndarray, solution, displacement matrix
        '''
        if to_print == True:
            print('==============================')
            print('Testing: N =', N, ', u1 =', u1, ', uend =', uend, ', F =', F, ', Ke =\n', Ke)

        # Create global stiffness matrix.
        K = self.assembleLinear(N=N, Ke=Ke, u1=u1, uend=uend, f_b=0, f_s=F, to_print=False)

        # Modify the stiffness matrix and load vector to account for Dirichlet BCs
        Kr = self.deleteRow(K, uend=uend, axis=1)
        Kr = self.deleteRow(Kr, uend=uend, axis=0)

        d_dirichlet = np.zeros([N + 1, 1])
        d_dirichlet[0] = u1
        if uend is not None:
            d_dirichlet[-1] = uend

        # reduced right hand side
        f = np.zeros([N + 1, 1])
        f[-1] = F
        if uend is not None:
            f[0] -= K[0, -1] * uend
            f[-1] -= K[-1, 0] * u1

        fr = self.deleteRow(f, uend=uend, axis=0)

        # calculate displacement vector
        dr = scipy.linalg.solve(Kr, fr)

        if uend is not None:
            d = np.zeros([N + 1, 1])
            for element in range(N - 1):
                d[element + 1] = dr[element]
            d[0] = u1
            d[-1] = uend
        else:
            d = np.insert(dr, 0, 0, axis=0)
            d += u1

        # If both u1 and uend are defined, adjust the intermediate nodes linearly
        if uend is not None:
            for i in range(1, N):
                d[i] = u1 + i * (uend - u1) / N
        print('=== d ===\n', d)
        return d

    def postProcess(self, BC_1, A, E, L, F, N ):
        '''
        Post process:
        Calculate and plot stress tensors under different BCs.
        :param BC_1: ndarray, boundary condition
        :param A: float, section area. (mm^2)
        :param E: float, E-Modulus. (N/mm^2)
        :param L: float, length of the bar. (mm)
        :param F: float, force applied on the end.
        :param N: int, number of elements.
        :return stress_fields: ndarray, stress fields based on different BCs.
        :return x_points: ndarray, indicating
        '''
        d_1 = []    # displacements
        num_bc = np.shape(BC_1)[0]  # number of BCs
        print('num_bc=',num_bc)
        # Assemble the system based on given BC, solve the displacements.
        for BC in range(N):
            Ke = self.element_stiffness_matrix(E=E, A=A, Le=L/N)
            d_1.append(self.solveLinear(N=N, Ke=Ke, u1=BC_1[BC][0], uend=BC_1[BC][1], F=F, to_print=False))
        print('---all elements of d_1:---\n', d_1)

        stress_fields = []  # list of all stress fields
        x_points = np.linspace(1, N+1, N+1) # nodes
        x_elements = np.linspace(1, N, N)  # elements
        # Calculate gradient for on displacement results based on different BCs.
        for BC in range(num_bc):
            print('BC:', BC)
            grad_1 = np.zeros(N+1)
            d_element = np.zeros(N+1)      # local displacement in element
            for i in range(N):
                print('i:', i)
                print('N:', N)
                print('grad_1[i + 1]:', grad_1[i + 1])
                print('d_1[BC][i + 1]:', d_1[BC][i + 1])
                grad_1[i + 1] = d_1[BC][i + 1] - d_1[BC][i][0]
                d_element[i + 1] = d_1[BC][i + 1] - d_1[BC][i][0]
            grad_1[0] = grad_1[1]
            d_element[0] = d_element[1]

            this_stress_field = [] # stress field under this particular BC

            for node in range(N):
                #print('grad_1[node]:', grad_1[node])
                #print('d_1[BC][node][0]:', d_1[BC][node][0])
                # Simplify the dot product of 1*2 times 2*1, since in an element,
                # the local displacement on the first node is 0.
                this_stress_field.append(E * grad_1[node] * d_element[node] * 1000000)

            stress_fields.append(this_stress_field)

            print(f'{BC + 1}. result of u1 = {BC_1[BC][0]} mm, uend = {BC_1[BC][1]} mm \n')
            print('*** displacement ***: \n', d_1[BC].T)
            print('***  gradient ***: \n', grad_1)
            print('*** stress field ***: \n', this_stress_field, '\n')

            # Plot results
            plt.figure(figsize=(10, 8), dpi=80)

            ax_sf = plt.subplot(122)
            ax_sf.plot(this_stress_field,'g-o')
            ax_sf.set_title('stress field (Pa)')
            ax_sf.xaxis.tick_top()

            ax_d = plt.subplot(221)
            ax_d.plot(x_points, d_1[BC], '-o')
            #ax_d.set_ylim(-1, 90)
            ax_d.set_title(f'displacement  of u1 = {BC_1[BC][0]} mm, uend = {BC_1[BC][1]} mm')
            ax_d.set_ylabel('displacement (mm)')

            ax_g = plt.subplot(223)
            ax_g.plot(x_points, grad_1, 'r-o')
            #ax_g.set_ylim(0, 0.08)
            ax_g.set_title(f'gradient of u1 = {BC_1[BC][0]} mm, uend = {BC_1[BC][1]} mm')
            ax_g.set_ylabel('gradient')

            plt.suptitle(f'u1 = {BC_1[BC][0]} mm, uend = {BC_1[BC][1]} mm\n N = 6, F = 12 N, c = 1')
            plt.show()

        print("stress_fields shape:", len(stress_fields))
        print("x_points shape:", x_points.shape)
        return stress_fields, x_points, x_elements

            



