#!/usr/bin/env python3.9
# -*- coding:utf-8 -*-
from sympy.physics.quantum import TensorProduct
import sympy as sp
import numpy as np
import sys
from colorama import Fore, Style

print('Prepare unitaries')
# print('All modules imported')

try:
	pars = sys.argv[1]

	doIneedY      = pars[0] == '1'
	doIknowI      = pars[1] == '1'
	doIknowSC     = pars[2] == '1'
	doIknow3β     = pars[3] == '1'
	avoidSC       = pars[4] == '1'
	ignore_phases = pars[5] == '1'
except IndexError:
	doIneedY      = False
	doIknowI      = False
	doIknowSC     = False
	doIknow3β     = True
	avoidSC       = True
	ignore_phases = False


if doIknow3β:
	x = sp.Symbol("3β/2", real=True)
else:
	x = sp.Symbol("3β"  , real=True)/2

if doIknowSC:
	s = sp.sin(x)
	c = sp.cos(x)
else:
	s, c = sp.symbols('s c', real=True)

# U = unitary2(np.sqrt(1/2), np.sqrt(1/2), y, i)
if avoidSC:
	# c81 = 5 + 4 * sp.cos(2*x)
	# c45 = 7 + 2 * sp.cos(2*x)
	c81 = sp.Symbol('5+4cos(3β)', positive=True)
	c45 = sp.Symbol('7+2cos(3β)', positive=True)
else:
	c81 = 1+8*c*c
	c45 = 5+4*c*c

if doIneedY:
	y = sp.Symbol("√ⅈ")
if doIknowI:
	i = sp.I
else:
	if doIneedY:
		i = y * y
	else:
		i = sp.Symbol("ⅈ")

np.set_printoptions(suppress=True,linewidth=sys.maxsize,threshold=sys.maxsize)


# 1 / y = y * y * y * y * y * y * y
# y * y * y * y = - 1
def my_simplify(expr):
	expr = sp.sympify(expr)
	# expr.subs(sp.sqrt(32*c**4 + 44*c**2 + 5), sp.sqrt(4*c**2 + 5) * sp.sqrt(8*c**2 + 1))
	expr = expr.subs(9-8*s**2, 1 + 8*c**2)
	expr = expr.subs(9-4*s**2, 5 + 4*c**2)
	if not doIknowSC:
		expr = expr.subs(s**2, 1 - c**2)
	if doIneedY:
		if doIknowI:
			expr = expr.subs(sp.conjugate(y), -y*i)
			expr = expr.subs(y**2, i)
			expr = expr.subs(y**3, i*y)
		else:
			expr = expr.subs(sp.conjugate(y), -y**3)
			# expr = expr.subs(1/y, y**7)
			expr = expr.subs(y**4, -1)
			expr = sp.simplify(expr).subs(y**4, -1)
	else:
		if not doIknowI:
			expr = expr.subs(sp.conjugate(i), -i)
			expr = expr.subs(i**2, -1)
	if not doIknowSC:
		expr = sp.simplify(expr).subs(s**2, 1 - c**2)
	if avoidSC:
		expr = sp.simplify(expr).subs(c81, 5 + 4 * sp.cos(2*x)).subs(c45, 7 + 2 * sp.cos(2*x))
	return sp.simplify(expr)

def conj(expr):
	return my_simplify(sp.conjugate(expr))
	# if doIknowI:
	# 	return my_simplify(expr.subs(y, - y**3).subs(i, - i))
	# else:
	# 	return my_simplify(expr.subs(y, - y**3))


def check_equal(Expr1,Expr2): # from https://stackoverflow.com/questions/37112738/sympy-comparing-expressions
	if Expr1==None or Expr2==None:
		return(False)
	if Expr1.free_symbols!=Expr2.free_symbols:
		return(False)
	vars = Expr1.free_symbols
	your_values=np.random.random(len(vars))
	Expr1_num=Expr1
	Expr2_num=Expr2
	for symbol,number in zip(vars, your_values):
		Expr1_num=Expr1_num.subs(symbol, sp.Float(number))
		Expr2_num=Expr2_num.subs(symbol, sp.Float(number))
	real = sp.re(Expr1_num)
	imag = sp.im(Expr1_num)
	Expr1_num=np.float64(real) + 1j * np.float64(imag)
	real = sp.re(Expr2_num)
	imag = sp.im(Expr2_num)
	Expr2_num=np.float64(real) + 1j * np.float64(imag)
	if not np.allclose(Expr1_num,Expr2_num):
		return(False)
	if (Expr1.equals(Expr2)):
		return(True)
	else:
		return(False)

def compare_matrices(M1, M2, size = 6):
	bools = []
	for i1 in range(size):
		row = []
		for i2 in range(size):
			row.append(check_equal(M1[i1, i2], M2[i1, i2]))
		bools.append(row)
	return np.array(bools, dtype=bool)

def visual_compare(M1, M2):
	str1 = sp.pretty(M1)
	str2 = sp.pretty(M2)
	for line1, line2 in zip(str1.split('\n'), str2.split('\n')):
		print(Fore.WHITE if line1 == line2 else Fore.RED, line1, line2)
	print(Fore.RESET)


def is_one_efficient(expr): # derived from https://stackoverflow.com/questions/37112738/sympy-comparing-expressions
	if expr == None:
		return False
	if expr.free_symbols:
		return False
	vars = expr.free_symbols
	your_values=np.random.random(len(vars))
	expr_num=expr
	for symbol,number in zip(vars, your_values):
		expr_num=expr_num.subs(symbol, sp.Float(number))
	expr_num=complex(expr_num)
	if not np.allclose(expr_num, 1):
		return(False)
	return expr.equals(sp.S.One)

def is_one(expr_raw):
	expr = my_simplify(expr_raw)
	if expr.free_symbols:
		vars = expr.free_symbols
		your_values=np.random.random(len(vars))
		expr_num=expr
		for symbol,number in zip(vars, your_values):
			expr_num=expr_num.subs(symbol, sp.Float(number))
		real = sp.re(expr_num)
		imag = sp.im(expr_num)
		expr_num=np.float64(real) + 1j * np.float64(imag)
		return np.allclose(expr_num, 1)
	real = sp.re(expr)
	imag = sp.im(expr)
	expr = np.float64(real) + 1j * np.float64(imag)
	return np.isclose(expr, 1)

def is_zero(expr_raw):
	expr = my_simplify(expr_raw)
	if expr.free_symbols:
		vars = expr.free_symbols
		your_values=np.random.random(len(vars))
		expr_num=expr
		for symbol,number in zip(vars, your_values):
			expr_num=expr_num.subs(symbol, sp.Float(number))
		real = sp.re(expr_num)
		imag = sp.im(expr_num)
		expr_num=np.float64(real) + 1j * np.float64(imag)
		return np.allclose(expr_num, 0)
	real = sp.re(expr)
	imag = sp.im(expr)
	expr = np.float64(real) + 1j * np.float64(imag)
	return np.isclose(expr, 0)

def assert_unitarity(M, size=2):
	eye = M.H @ M
	for i0 in range(size):
		for i1 in range(size):
			assert is_one(eye[i0, i1]) if i0 == i1 else is_zero(eye[i0, i1]) or ignore_phases, ("Matrix is not unitary", M, my_simplify(M.H), my_simplify(eye))


def print_mat(M):
	lx, ly     = M.shape
	longest    = [2] * ly
	texts      = []
	print_text = ""
	for x in range(lx):
		row = []
		for y in range(ly):
			text = str(M[x,y])
			row.append(text)
			longest[y] = max(longest[y], len(text))
		texts.append(row)
	for x in range(lx):
		for y in range(ly):
			texts[x][y] = ' '* (longest[y] - len(texts[x][y])) + texts[x][y]
	print('\n'.join(', '.join(row) for row in texts))

def texify(M):
	# M    = M   .subs(i,	sp.Symbol('i \\cdot'))
	M    = M   .subs(s,	sp.Symbol('\\sin(\\frac{3\\beta}2)'))
	M    = M   .subs(c,	sp.Symbol('\\cos(\\frac{3\\beta}2)'))
	# M    = M   .subs(c45, sp.Symbol('\\left(7 + 2 \cos(3\\beta)\\right)'))
	# M    = M   .subs(c81, sp.Symbol('\\left(5 + 4 \cos(3\\beta)\\right)'))
	teXt = sp  .latex(M)
	teXt = teXt.replace(sp.latex(my_simplify(i)),	'i')
	teXt = teXt.replace(sp.latex(my_simplify(c45)), '\\left(7 + 2 \cos(3\\beta)\\right)')
	teXt = teXt.replace(sp.latex(my_simplify(c81)), '\\left(5 + 4 \cos(3\\beta)\\right)')
	teXt = teXt.replace('\\cos{\\left(2 \\cdot 3β/2 \\right)}', '\\cos(3\\beta)')
	teXt = teXt.replace('\\sin{\\left(2 \\cdot 3β/2 \\right)}', '\\sin(3\\beta)')
	teXt = teXt.replace('7+2cos(3β)', '7 + 2 \\cos(3\\beta)')
	teXt = teXt.replace('5+4cos(3β)', '5 + 4 \\cos(3\\beta)')
	teXt = teXt.replace('\\cos{\\left(3 \\cdot 3β/2 \\right)}', '\\cos(\\frac92 \\beta)')
	teXt = teXt.replace('\\sin{\\left(3 \\cdot 3β/2 \\right)}', '\\sin(\\frac92 \\beta)')
	teXt = teXt.replace('0', '\\textcolor{empty}{0}')
	teXt = teXt.replace('1', '\\textcolor{irrelevant}{1}')
	teXt = teXt.replace('\\cos(3\\beta)', 'c')
	teXt = teXt.replace('\\sin(3\\beta)', 's')
	teXt = teXt.replace('\\cos(\\frac{3\\beta}2)', "c'")
	teXt = teXt.replace('\\sin(\\frac{3\\beta}2)', "s'")
	teXt = teXt.replace('\\left[\\begin{matrix}', '\\begin{pmatrix}\n\t')
	teXt = teXt.replace('\\\\', '\\\\\n\t')
	teXt = teXt.replace('\\end{matrix}\\right]', '\n\\end{pmatrix}') #\n\\cdot\n\\\\\\cdot')
	teXt = teXt.replace('\\end{matrix}\\right]', '\n\\end{pmatrix}\n\\cdot\\\\&\\cdot') # \n&\\cdot&')
	print(teXt)
	# return teXt

def unitary2(diag, offdiag, divisor_sq = 1, phase_10 = i, phase_00 = 1, phase_11 = sp.nan, phase_01 = sp.nan):
	# if not (isinstance(phase_00, float) or isinstance(phase_00, int)):
	try:
		phases  = sp.Matrix(divisor_sq)
		divisor_sq = 1
	except TypeError:
		try:
			phases = sp.Matrix(phase_10)
		except TypeError:
			phases = sp.Matrix([[phase_00, phase_01], [phase_10, phase_11]])
	if phases[1, 1] == sp.nan:
		phases[1, 1] = - conj(phases[0, 0])
	if phases[0, 1] == sp.nan:
		phases[0, 1] = + conj(phases[1, 0])
	
	assert sp.sympify(diag      ).is_real
	assert sp.sympify(offdiag   ).is_real
	assert sp.sympify(divisor_sq).is_positive or divisor_sq == c45 or divisor_sq == c81
	assert is_one((diag * diag + offdiag * offdiag) / divisor_sq) or (avoidSC and not doIknowSC), ("Invalid unitary matrix: ", my_simplify(diag * diag + offdiag * offdiag), my_simplify(divisor_sq))
	for phase in phases.vec():
		assert is_one(phase * conj(phase)), (phase, conj(phase), phase * conj(phase))

	U = phases / sp.sqrt(divisor_sq)
	U[0,0] *= diag
	U[0,1] *= offdiag
	U[1,0] *= offdiag
	U[1,1] *= diag
	if doIknowSC or not avoidSC:
		assert_unitarity(U)
	return U

def two_level(i0, i1, U, size = 6):
	M = sp.matrices.eye(size)
	M[i0, i0] = U[0, 0]
	M[i0, i1] = U[0, 1]
	M[i1, i0] = U[1, 0]
	M[i1, i1] = U[1, 1]
	return M



# two_levels = []
# phases = [
# 	[-   1,    -   i],
# 	[-   i,    -   1]]
# two_levels.append(two_level(2, 5, unitary2(c,					s,					1,					phases)))
# two_levels.append(two_level(1, 4, unitary2(c,					s,					1,					+ i)))
# if avoidSC:
# 	two_levels.append(two_level(0, 3, unitary2(2 + sp.cos(2*x),	sp.sin(2*x),		c81,				+ i)))
# else:
# 	two_levels.append(two_level(0, 3, unitary2(1+2*c*c,			2*s*c,				c81,				+ i)))
# two_levels.append(two_level(0, 4, unitary2(sp.sqrt(c81),		2*s,				c45,				- i)))
# two_levels.append(two_level(0, 5, unitary2(sp.sqrt(c45),		2*s,				9,					- i)))

# two_levels.append(two_level(1, 3, unitary2(sp.sqrt(c81),		2*s,				c45,				- i)))
# two_levels.append(two_level(1, 4, unitary2(8*c+sp.cos(3*x),		sp.sin(3*x),		(c81)**2,			- i)))
# two_levels.append(two_level(1, 5, unitary2(sp.sqrt(c81),		2*s,				c45,				+ i)))

# phases = [
# 	[-   1,    +   i],
# 	[-   i,    +   1]]
# two_levels.append(two_level(2, 3, unitary2(sp.sqrt(c45),		2*s,				9,					+ i,		- 1)))
# phases = [
# 	[+   1,    +   i],
# 	[-   i,    -   1]]
# two_levels.append(two_level(2, 4, unitary2(sp.sqrt(c81),		2*s,				c45,				- i)))
# phases = [
# 	[+   1,    -   i],
# 	[+   i,    -   1]]
# two_levels.append(two_level(2, 5, unitary2(3*c,					s,					1+8*c*c,			+ i)))

two_levels = []
two_levels.append(two_level(2, 5, unitary2(c,					s)))
two_levels.append(two_level(1, 4, unitary2(c,					s)))
phases = [
	[+   1,    +   i],
	[+   i,    +   1]]
if avoidSC:
	two_levels.append(two_level(0, 3, unitary2(2 + sp.cos(2*x),	sp.sin(2*x),		c81,		phases)))
else:
	two_levels.append(two_level(0, 3, unitary2(1+2*c*c,			2*s*c,				c81,		phases)))
two_levels.append(two_level(0, 4, unitary2(sp.sqrt(c81),		- 2*s,				c45			)))
two_levels.append(two_level(0, 5, unitary2(sp.sqrt(c45),		- 2*s,				9			)))
two_levels.append(two_level(1, 3, unitary2(sp.sqrt(c81),		2*s,				c45			)))
two_levels.append(two_level(1, 4, unitary2(8*c+sp.cos(3*x),		- sp.sin(3*x),		(c81)**2	)))
two_levels.append(two_level(1, 5, unitary2(sp.sqrt(c81),		2*s,				c45			)))
two_levels.append(two_level(2, 3, unitary2(sp.sqrt(c45),		- 2*s,				9			)))
two_levels.append(two_level(2, 4, unitary2(sp.sqrt(c81),		- 2*s,				c45			)))
two_levels.append(two_level(2, 5, unitary2(3*c,					s,					1+8*c*c		)))

print('Unitaries calculated')

# reference = []
# phases = [
# 	[-y*i, +y*i],
# 	[+y  , +y  ]]
# reference.append(two_level(2, 5, unitary2(c,						s,												phases)))
# reference.append(two_level(1, 4, unitary2(c,						s,												phases)))
# reference.append(two_level(0, 3, unitary2(1+2*c*c,					2*s*c,					1+8*c*c,				+ i)))
# reference.append(two_level(0, 4, unitary2(sp.sqrt(1+8*c*c),		2*s,					5+4*c*c,				+ y)))
# reference.append(two_level(0, 5, unitary2(sp.sqrt(5+4*c*c),		2*s,					9,						+ y)))
# reference.append(two_level(1, 3, unitary2(sp.sqrt(1+8*c*c),		2*s,					5+4*c*c,				- i,		+ y)))
# reference.append(two_level(1, 4, unitary2((5+4*c*c)*c,				(1-4*c*c)*s,			(1+8*c*c)**2,			- y)))
# reference.append(two_level(1, 5, unitary2(sp.sqrt(1+8*c*c),		2*s,					5+4*c*c,				- y)))
# reference.append(two_level(2, 3, unitary2(sp.sqrt(5+4*c*c),		2*s,					9,						- y,		- 1)))
# reference.append(two_level(2, 4, unitary2(sp.sqrt(1+8*c*c),		2*s,					5+4*c*c,				+ y,		- y,	+ y*i)))
# phases = [
# 	[+  1, -  i],
# 	[-y  , +y*i]]
# reference.append(two_level(2, 5, unitary2(3*c,						s,						1+8*c*c,				phases)))

# print(two_levels == reference)
# for m in range(len(reference)):
# 	print(two_levels[m] == reference[m])


# for tl in two_levels:
# 	sp.pprint(my_simplify(tl))
# 	print("\n")

A = [
	[sp.cos(x*2)-1, sp.sin(x*2)*i],
	[sp.sin(x*2)*i, sp.cos(x*2)-1],
]
reference_unitary = sp.Matrix([[sp.matrices.ones(3, 3) * v for v in row] for row in A]) / 3 + sp.eye(6)
# reference_unitary = sp.physics.quantum.TensorProduct(A, sp.matrices.ones(3, 3)) / 3 + sp.eye(6)


the_unitary = sp.matrices.eye(6)
for unitary in two_levels:
	if doIknowI:
		the_unitary =  the_unitary @ unitary
	else:
		the_unitary = (the_unitary @ unitary).subs(i**2, -1)
	texify(unitary)

# if not doIknowSC:
# 	the_unitary = my_simplify(the_unitary.subs(s, sp.sin(x)).subs(c, sp.cos(x)))
# 	# doIknowSC   = True

# simplified = my_simplify(my_simplify(3 * the_unitary))

# comp = compare_matrices(simplified / 3, reference_unitary)

# # sp.pprint(3 * reference_unitary)
# sp.pprint(simplified)
# print(comp)

# # current_unitary = 3 * reference_unitary
# # agree = 0
# # for unitary in two_levels:
# # 	if agree:
# # 		agree -= 1
# # 	else:
# # 		input("Press Enter to continue...")
# # 		sp.pprint(current_unitary)
# # 		sp.pprint(unitary)
# # 	last_unitary    = current_unitary
# # 	current_unitary = my_simplify(unitary.H @ current_unitary)

# # input("Press Enter to continue...")
# # sp.pprint(current_unitary)
# # sp.pprint(unitary)


