# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.9.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Requirements for Sturm's Theorem
# ### Note: All Polynomials are Univariate
# We want to be able to use Sturm's theorem which requires:
# - Bounds
# - Evaluation of Polynomials
# - Derivatives
# - Polynomial Division
#
# Bounds will be found using Cauchy's Bound Theorem. We also need the polynomial to be "square-free". This means that all roots of the polynomial have multiplicity of one. If a root occurs more than once, it will also appear in the derivative. Observe
#
# \begin{equation*}
# (x - 1)^2 = x^2 - 2x + 1 \\ 
# \frac{d}{dx}(x^2 -2x + 1) = 2x - 2 = 2(x - 1)
# \end{equation*}
#
# We want to remove all repeated roots. We can do this by finding $\frac{f}{\textrm{gcd}(f, f')}$. This then expresses $f$ without the repeated roots. However, there are other cases to consider. How does this work with other roots? Or when there are no repeated roots. 
#
# \begin{equation*}
# (x - 1)(x + 3) = x^2 + 2x -3 \\
# \frac{d}{dx}(x^2 +2x -3) = 2x + 2 = 2(x + 1)
# \end{equation*}
#
# Now before division of the initial function we first need to find $\textrm{gcd}(x^2 + 2x - 3, 2x + 2) = 2$ - found using Euclid's Algorithm. Since there is no multiple root, the greatest common divisor is a scalar, and this method allows us to restate the equation as square-free.
#
# This adds a new criteria for our function:
# - Euclid's Algorithm for Greatest Common Divisor (which will use polynomial division)
#
# Finally, for evaluating Sturm's theorem, we need to be able to count the number of sign changes for the Sturm series. This will be the last function required:
# - Count Number of Sign Changes in Series
#
# Additionally, since we are dealing with representation of polynomials, we need to consider the order of this list. We only plan to reduce the degree (differentiation or division), so we will store these in a list with the largest degree, $n$, first and store each coefficent in the $i$th position.

# %%
from fractions import Fraction
from copy import deepcopy


# %%
def cauchy_bound(coef):
    '''This uses the Cauchy Bound Theorem to identify the range that a polynomial with coefficents coef
    could have real roots. More explicitly, it finds the point at which the leading term dominates over
    the other terms.
    
    :param coef: list-like of coefficents of a polynomial in descending order of degree
    :return: tuple of two elements, a lower bound and an upper bound'''
    
    # get the absolute maximum coef
    max_abs = abs(coef[0])
    for c in coef:
        if abs(c) > max_abs:
            max_abs = abs(c)
    # find the point at which the leading term overtakes the maximum coef
    M = (abs(Fraction(max_abs)) / abs(Fraction(coef[0]))) + 1
    return -M, M


# %%
def horners_method(coef, x):
    '''This evaluates a polynomial of coefficents coef at point x.
    
    :param coef: list-like of coefficents of a polynomial in descending order of degree
    :param x: real number to evaluate polynomial at
    :return: float result of evaluation'''
    
    out = 0
    for c in coef[:-1]:
        out += c
        out *= x
    out += coef[-1]
    return out


# %%
def differentiate(coef):
    '''This finds the derivative of a polynomial with coefficents coef by applying the power rule.
    
    :param coef: list-like of coefficents of a polynomial in descending order of degree
    :return: list-like of coefficents of derivative in descending order of degree'''
    
    degree = len(coef) - 1
    out = []
    for i in range(degree):
        out.append(Fraction(coef[i]) * (degree - i))
    return out


# %%
def poly_div(f, g):
    '''Divide polynomial f by polynomial g. Returns the result and remainder.
    
    :param f: list-like of coefficents of a polynomial in descending order of degree
    :param g: list-like of coefficents of a polynomial in descending order of degree
    :return: tuple of two lists of polynomial coeffients - the result and the remainder'''
    
    degree_f = len(f) - 1
    degree_g = len(g) - 1
    assert f[0] != 0, "Must have nonzero leading coefficent"
    assert g[0] != 0, "Must have nonzero leading coefficent"
    assert degree_f >= degree_g, "Must divide by lesser degree"
    
    # create output list of coef of the correct degree 
    # and a list for the remainder
    out = [0 for i in range(degree_f - degree_g + 1)]
    rem = deepcopy(f)
    # use a while loop to handle special cases
    for i in range(degree_f - degree_g + 1):
        # find the coef of the ith term in f divided by the leading term of g
        d = Fraction(rem[i]) / Fraction(g[0])
        out[i] = d
        # update subtraction
        for j in range(len(g)):
            rem[i + j] -= d * Fraction(g[j])

    # f has been updated in-place to represent the remainder
    # we need to remove the front-zeros
    try:
        while abs(rem[0]) < 1e-16:
            rem.pop(0)
    except IndexError:
        # we have no remainder so f is empty and throws an index error
        rem = [0]
        
    return out, rem


# %%
def gcd(f, g):
    '''Using Euclid's algorithm, we find the gcd between two polynomials. Returns a list of coefficents
    to a polynomial.
    
    :param f: list-like of coefficents of a polynomial in descending order of degree
    :param g: list-like of coefficents of a polynomial in descending order of degree
    :return: list-like of coefficents of a polynomial in descending order of degree'''
    
    r = [f, g]
    i = 0
    while abs(r[i + 1][0]) > 1e-16:
        r.append(poly_div(r[i], r[i + 1])[1])
        i += 1
    
    return r[-2]


# %%
def sturm_seq(coef):
    '''This creates a list of polynomials which represent those in the sturm sequence as 
    defined by Sturm's theorem
    
    :param coef: list-like of coefficents of a square-free polynomial in descending order of degree
    :return: list of lists of coefficents'''
    
    out = [coef, differentiate(coef)]
    next_term = [c * -1 for c in poly_div(out[0], out[1])[1]]
    i = 1
    while abs(next_term[0]) > 1e-16:
        out.append(next_term)
        next_term = [c * -1 for c in poly_div(out[i], out[i + 1])[1]]
        i += 1
    return out


# %%
def is_opposite(a, b):
    return (a * b) <= 0


# %%
def count_changes(a):
    '''Given list a, count the number of sign changes'''
    
    testable = []
    for c in a:
        if abs(c) > 1e-16:
            testable.append(c)
    
    num_changes = 0
    for i in range(1, len(testable)):
        if is_opposite(a[i - 1], a[i]):
            num_changes += 1
        
    return num_changes


# %%
def sturm_seq_eval(seq, xi):
    evaluated = [horners_method(x, xi) for x in seq]
    return count_changes(evaluated)


# %%
def determinant(A):
    # done by row reduction rather than cofactor expansion
    m = len(A)
    n = len(A[0])
    out = deepcopy(A)
    # convert to upper triangular matrix
    for t in range(m):
        base_row = out[t]
        for i in range(t + 1, m):
            factor = (-1 * Fraction(A[i][t])) / Fraction(A[t][t])
            for j in range(n):
                out[i][j] += factor*base_row[j]
    # calculate determinant
    det = 1
    for d in range(min(m, n)):
        det *= out[d][d]
    return det


# %%
def transpose(A):
    m = len(A)
    n = len(A[0])
    A_prime = []
    for i in range(n):
        row = []
        for j in range(m):
            row.append(A[j][i])
        A_prime.append(row)
    return A_prime


# %%
def resulatant(f, g):
    degree_f = len(f) - 1
    degree_g = len(g) - 1
    # create matrix
    sylvester_matrix = []
    m = n = degree_f + degree_g
    # add coefficenets of polynomial f
    for i in range(degree_g):
        row = [0 for j in range(n)]
        j = i
        while (j - i < degree_f) and (j < n):
            row[j] = f[j - i]
        sylvester_matrix.append(row)
    # add coeffiecents of polynomial g
    for i in range(degree_f):
        row = [0 for j in range(n)]
        j = i
        while (j - i < degree_g) and (j < n):
            row[j] = g[j - i]
    # find transpose of sylvester matrix
    linear_map = transpose(sylvester_matrix)
    # find determinant of map for resultant of two polynomials
    det = determinant(linear_map)
    
    return det            


# %%
def discriminant(coef):
    degree = len(coef) - 1
    sign = 1 if ((n * (n - 1) / 2) % 2 == 0) else -1
    factor = Decimal(sign) / Decimal(coef[0])
    res = resultant(coef, differentiate(coef))
    return factor * res


# %%
# f = (x - 1)^2 (x - 3)(x + 4) = x^4 - x^3 - 13x^2 + 25x - 12
f = [1, -1 , -13, 25, -12]
num_ints = 8 # number of intervals
precision = 1e-15

# remove multiple roots
f_star = poly_div(f, gcd(f, differentiate(f)))[0]

# find Sturm sequence for function
seq = sturm_seq(f_star)

# find the bounds of our roots and step up inital intervals
lb, ub = cauchy_bound(f_star)
step = (ub - lb) / num_ints
num_roots = sturm_seq_eval(seq, lb) - sturm_seq_eval(seq, ub)
next_ints = [[lb + (i*step), lb + ((i+1)*step)] for i in range(num_ints)]
intervals = next_ints

while max([abs(x[1] - x[0]) for x in intervals]) > precision:
    intervals = next_ints

    # get root counts for intervals
    root_counts = []
    for x in intervals:
        root_counts.append(sturm_seq_eval(seq, x[0]) - sturm_seq_eval(seq, x[1]))

    # find smaller intervals 
    next_ints = []
    for i in range(1, len(root_counts)):
        if root_counts[i] > 0:
            a = intervals[i][0]
            b = intervals[i][1]
            step = (b - a) / num_ints
            next_ints += [[a + (i*step), a + ((i+1)*step)] for i in range(num_ints + 1)]
    #print([(x[0], float(x[1][0]), float(x[1][1])) for x in zip(root_counts, intervals)])
    #print()
    
# find our candiadate intervals
candidates = []
for j in range(len(root_counts)):
    if root_counts[j] != 0:
        candidates.append((root_counts[j], intervals[j][0], intervals[j][1]))

# %%
[float(x) for x in f_star]

# %%
sturm_seq_eval(seq, -3)

# %%
candidates

# %%
for i in range(len(candidates)):
    print(f"""\t\item The interval $(\\frac{{{candidates[i][1].numerator}}}{{{candidates[i][1].denominator}}}, \\frac{{{candidates[i][2].numerator}}}{{{candidates[i][2].denominator}}}) \\approx ({float(candidates[i][1])}, {float(candidates[i][2])})$ has {candidates[i][0]} root.""")

# %%
root_counts

# %%
[float(x) for x in f_star]

# %%
[(x[0], float(x[1][0]), float(x[1][1])) for x in zip(root_counts, intervals)]

# %%
horners_method(f_star, 1.3846153846153844)

# %%
horners_method(f_star, 2.0816659994661326)

# %%
[horners_method(x, 2.0816659994661326) for x in seq]

# %%
seq

# %%
lb, ub = (-3, 3)
num_ints = 3
step = (ub - lb) / num_ints
intervals = [lb + (i*step) for i in range(num_ints)] + [ub]

# %%
root_counts = [sturm_seq_eval(temp, x) for x in intervals]
next_ints = []
for i in range(1, len(root_counts)):
    r1 = root_counts[i - 1]
    r2 = root_counts[i]
    if (r1 - r2) > 0:
        a = intervals[i - 1]
        b = intervals[i]
        next_ints.append((a, (b + a) / 2, b))

# %%
next_ints

# %%
temp

# %%
temp = sturm_seq([1, -1])
[sturm_seq_eval(temp, x) for x in intervals]

# %%
[horners_method(x, -3) for x in sturm_chain([1, -1])]

# %%
count_changes([horners_method(x, -3) for x in sturm_chain([1, -1])])

# %%
count_changes([1, 2, -1, 0, -3, 1])

# %%
poly_div([1, -2, 1], [1, -1])

# %%
gcd([1, -2, 1], [1, -1])

# %%
1 <= 1e-16

# %%
poly_div([1, -2, 1], [1, -1])

# %%
poly_div([1, -2, 1], [1, -2, 1])

# %%
gcd([1, -2, 1], [1, -1])

# %%
gcd([1, -2, 1], [1, -2, 1])

# %%
x = [1, -2, 1]

# %%
differentiate(x)

# %%
poly_div(x, gcd(x, differentiate(x)))

# %%
x = [1, -5, 7, -3]
x_star = poly_div(x, gcd(x, differentiate(x)))
x_star

# %%
import numpy as np

# %%
horners_method(x_star[0], 1)

# %%
gcd([1, -5, 7, -3], [3, -10, 7])

# %%
poly_div([3, -10, 7], [1, -1])

# %%
x = [1, -3, 3, -1]
den = gcd(x, differentiate(x))
den

# %%
poly_div(x, den)

# %%
poly_div(x, differentiate(x))

# %%
