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

	doIknowI      = pars[0] == '1'
	ignore_phases = pars[1] == '1'
	avoidSC       = pars[4] == '1'
except IndexError:
	# doIneedY      = False
	doIknowI      = True
	# doIknowSC     = True
	# doIknow3β     = True
	avoidSC       = True
	ignore_phases = False


x = sp.Symbol("3β", real=True)

s = sp.sin(x)
c = sp.cos(x)

# U = unitary2(np.sqrt(1/2), np.sqrt(1/2), y, i)

if avoidSC:
	c81 = sp.Symbol('5+4cos(3β)', positive=True)
	c45 = sp.Symbol('7+2cos(3β)', positive=True)
else:
	c81 = 5 + 4*c
	c45 = 7 + 2*c

if doIknowI:
	i = sp.I
else:
	i = sp.Symbol("ⅈ")

p_f = c + i*s # e^+i3β
cpf = c - i*s # e^-i3β

np.set_printoptions(suppress=True,linewidth=sys.maxsize,threshold=sys.maxsize)


# 1 / y = y * y * y * y * y * y * y
# y * y * y * y = - 1
def my_simplify(expr):
	expr = sp.sympify(expr)
	if not doIknowI:
		expr = expr.subs(sp.conjugate(i), -i)
		expr = expr.subs(i**2, -1)
	if avoidSC:
		expr = sp.simplify(expr).subs(c81, 5 + 4 * c).subs(c45, 7 + 2 * c)
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
	lz, ly     = M.shape
	longest    = [2] * ly
	texts      = []
	print_text = ""
	for z in range(lz):
		row = []
		for y in range(ly):
			text = str(M[z,y])
			row.append(text)
			longest[y] = max(longest[y], len(text))
		texts.append(row)
	for z in range(lz):
		for y in range(ly):
			texts[z][y] = ' '* (longest[y] - len(texts[z][y])) + texts[z][y]
	print('\n'.join(', '.join(row) for row in texts))

def texify(M):
	# M    = M   .subs(i,	sp.Symbol('i \\cdot'))
	M    = M   .subs(p_f,	sp.Symbol('\\exp(+3i\\beta)'))
	M    = M   .subs(cpf,	sp.Symbol('\\exp(-3i\\beta)'))
	M    = M   .subs(s,	sp.Symbol('\\sin(3\\beta)'))
	M    = M   .subs(c,	sp.Symbol('\\cos(3\\beta)'))
	teXt = sp  .latex(M)
	teXt = teXt.replace(sp.latex(my_simplify(i)),	'i')
	teXt = teXt.replace(sp.latex(my_simplify(c45)), '\\left(7 + 2 \cos(3\\beta)\\right)')
	teXt = teXt.replace(sp.latex(my_simplify(c81)), '\\left(5 + 4 \cos(3\\beta)\\right)')
	teXt = teXt.replace('e^{+ i 3β}', '\\exp(+ 3 i \\beta)')
	teXt = teXt.replace('e^{- i 3β}', '\\exp(- 3 i \\beta)')
	# teXt = teXt.replace('\\sin{\\left(2 \\cdot 3β/2 \\right)}', '\\sin(3\\beta)')
	# teXt = teXt.replace('7+2cos(3β)', '7 + 2 \\cos(3\\beta)')
	# teXt = teXt.replace('5+4cos(3β)', '5 + 4 \\cos(3\\beta)')
	# teXt = teXt.replace('\\cos{\\left(3 \\cdot 3β/2 \\right)}', '\\cos(\\frac92 \\beta)')
	# teXt = teXt.replace('\\sin{\\left(3 \\cdot 3β/2 \\right)}', '\\sin(\\frac92 \\beta)')
	teXt = teXt.replace('0', '\\textcolor{empty}{0}')
	teXt = teXt.replace('1', '\\textcolor{irrelevant}{1}')
	teXt = teXt.replace('\\cos(3\\beta)', 'c')
	teXt = teXt.replace('\\sin(3\\beta)', 's')
	teXt = teXt.replace('\\cos(\\frac{3\\beta}2)', "c'")
	teXt = teXt.replace('\\sin(\\frac{3\\beta}2)', "s'")
	teXt = teXt.replace('\\left[\\begin{matrix}', '\\begin{pmatrix}\n\t')
	teXt = teXt.replace('\\\\', '\\\\\n\t')
	teXt = teXt.replace('\\end{matrix}\\right]', '\n\\end{pmatrix}\n&\\cdot&') # \n\\cdot\\\\&\\cdot')
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
	
	assert sp.sympify(diag      ).is_real or True
	assert sp.sympify(offdiag   ).is_real or True
	assert sp.sympify(divisor_sq).is_positive or divisor_sq == c45 or divisor_sq == c81
	assert is_one((diag * conj(diag) + offdiag * conj(offdiag)) / divisor_sq) or avoidSC, ("Invalid unitary matrix: ", my_simplify(diag * diag + offdiag * offdiag), my_simplify(divisor_sq))
	for phase in phases.vec():
		assert is_one(phase * conj(phase)), (phase, conj(phase), phase * conj(phase))

	U = phases / sp.sqrt(divisor_sq)
	U[0,0] *= diag
	U[1,0] *= offdiag
	U[0,1] *= conj(offdiag)
	U[1,1] *= conj(diag)
	if not avoidSC:
		assert_unitarity(U)
	return U

def two_level(i0, i1, U, size = 6):
	M = sp.matrices.eye(size)
	M[i0, i0] = U[0, 0]
	M[i0, i1] = U[0, 1]
	M[i1, i0] = U[1, 0]
	M[i1, i1] = U[1, 1]
	return M

two_levels = []
two_levels.append(two_level(0, 1, unitary2(p_f + 2,					p_f - 1,				c45,		1),		3))
two_levels.append(two_level(0, 2, unitary2(sp.sqrt(c45),			p_f - 1,				9,			1),		3))
two_levels.append(two_level(1, 2, sp.Matrix([[-(2*p_f + 1), 1-p_f], [1-p_f, -(2+p_f)]]) / sp.sqrt(c45),			3))
print('Unitaries calculated')


# for tl in two_levels:
# 	sp.pprint(my_simplify(tl))
# 	print("\n")

reference_unitary = sp.matrices.ones(3, 3) * (p_f - 1) / 3 + sp.eye(3)
# reference_unitary = sp.physics.quantum.TensorProduct(A, sp.matrices.ones(3, 3)) / 3 + sp.eye(6)


the_unitary = sp.matrices.eye(3)
for unitary in two_levels:
	if doIknowI:
		the_unitary =  the_unitary @ unitary
	else:
		the_unitary = (the_unitary @ unitary).subs(i**2, -1)
	texify(unitary)

simplified = my_simplify(3 * the_unitary)

comp = compare_matrices(simplified / 3, reference_unitary, size=3)

# sp.pprint(3 * reference_unitary)
sp.pprint(simplified)
print(comp)

# current_unitary = 3 * reference_unitary
# agree = 0
# for unitary in two_levels:
# 	if agree:
# 		agree -= 1
# 	else:
# 		input("Press Enter to continue...")
# 		sp.pprint(current_unitary)
# 		sp.pprint(unitary)
# 	last_unitary    = current_unitary
# 	current_unitary = my_simplify(unitary.H @ current_unitary)

# input("Press Enter to continue...")
# sp.pprint(current_unitary)
# sp.pprint(unitary)


