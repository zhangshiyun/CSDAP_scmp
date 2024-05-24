'''
IS-Dmin computation script
inherent.* are the Inherent structure file with lammps dump format.
Hai-Bin Yu 
haibinyu@hust.edu.cn
'''
import numpy as np
import os
import glob
import re
from scipy.optimize import linear_sum_assignment
fn_ref = 'inherent.0'
fn_screen = "screen2.txt"
fp_screep = open(fn_screen, "w")
use_saved_cost_matrix =  not True
save_cost_matrix = not True
save_adjuested_config = True

use_only_selected = not  True
select_type = 1

files = glob.glob("inherent.*")
files.remove(fn_ref)

'''sort by the index
should make sure that this work properly
this is the basis for pair-compare
'''
files.sort(key=lambda l: int(re.findall('\d+', l)[0]))
print(files)

def load_lmp_dump_data(fname):
    lines = open(fname).readlines()
    NAtom = int(lines[3])
    #Boundary length for x,y,z
    x1,x2= list(map(float,lines[5].split())); LX=np.abs(x1-x2)
    y1,y2= list(map(float,lines[6].split())); LY=np.abs(y1-y2)
    z1,z2= list(map(float,lines[7].split())); LZ=np.abs(z1-z2)
    data = np.loadtxt(lines[9:NAtom+9])
    if use_only_selected: 
        data = data[np.where(data[:,1]==select_type)]
    return LX, LY, LZ, data

def apply_PBC(M,L):
    M = np.where(M>L/2.0, M-L, M)
    M = np.where(M<-1*L/2.0, M+L, M)
    return M

def vector_to_matrix_function(A,B, L):
    MA= np.array([A for k in range(B.size)])
    MB= np.array([B for k in range(A.size)]).transpose()
    MC=MA-MB
    MC = apply_PBC(MC, L)
    MC2= MC**2.0
    return MC2

LX0, LY0, LZ0, data0 = load_lmp_dump_data(fn_ref)
x0 = data0[:,2]; y0 = data0[:,3]; z0 = data0[:,4]

result=np.zeros(len(files))

for f_idx, f in enumerate(files):
    fn_calc = f
    label = fn_calc.split(".")[-1]
    f_cost= "cost_matrix.%s" % label
    f_cost_with_ext = f_cost + ".npy"
    
    LX, LY, LZ, data = load_lmp_dump_data(fn_calc)
    
    if os.path.isfile(f_cost_with_ext) and use_saved_cost_matrix:
        cost = np.load(f_cost_with_ext)
    else:
        x= data[:,2]; y =data[:,3]; z = data[:,4]
        MX= vector_to_matrix_function(x,x0,LX)
        MY= vector_to_matrix_function(y,y0,LY)
        MZ= vector_to_matrix_function(z,z0,LZ)
        M_dis2 = MX+MY+MZ
        if save_cost_matrix:  np.save(f_cost,M_dis2)
    cost=M_dis2
    row_ind, col_ind = linear_sum_assignment(cost)
    new_data= data[col_ind]
    N = len(data)
    new_data[:,0] = np.arange(1, N+1)
    if save_adjuested_config:
        headstring=open(fn_calc).readlines()[:9]
        fp_res=open('adjusted.%s'%label,'w')
        headstring[3]='%s\n' %len(new_data)
        fp_res.writelines(headstring)
        np.savetxt(fp_res, new_data,fmt="%d %d %lf %lf %lf")
        fp_res.close()
    s = cost[row_ind, col_ind].mean()
    s_perAtom = np.sqrt(s)
    print("%s %lf" % (label, s_perAtom))
    fp_screep.write("%s %lf\n" % (label, s_perAtom))
    fp_screep.flush()
    '''
    The following line is used for pair compare 
    e.g., 0-1,1-2,..
    '''
    x0 = x; y0 = y; z0 = z
    result[f_idx] = s_perAtom
fp_screep.close()
print('Result: Meanvalue = %.4f  Std =%.4f, Nsamp = %d' %(result.mean(),
result.std(), len(result)) )
    
