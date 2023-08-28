# Importing library
import sympy as sym   #SymPy is a Python library for symbolic mathematics.

# Declaring variables
x, y, z = sym.symbols('x y z')

# expression of which we have to find derivative
exp = x**3 * y + y**3 + z

# Differentiating exp with respect to x
derivative1_x = sym.diff(exp, x)
print('derivative w.r.t x: ',
	derivative1_x)

# Differentiating exp with respect to y
derivative1_y = sym.diff(exp, y)
print('derivative w.r.t y: ',
	derivative1_y)

# Indefinite integration of cos(x) w.r.t. dx
integral1 = sym.integrate(sym.cos(x), x)
print('indefinite integral of cos(x): ',
	  integral1)

