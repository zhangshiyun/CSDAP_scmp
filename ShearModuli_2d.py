#!/usr/bin/python
# coding = utf-8
#This module is part of an analysis package

Authorinfo = """
             ------------------Name: Shiyun Zhang--------------
             --------------Email: zsy12@mail.ustc.edu.cn-----------
             """

docstr = """
         Calculate shear modulus of disordered solids from Hessian Matrix
         
         Based on Maloney's method

         2d systems

         """

import numpy as np
#import pandas as pd
import scipy.sparse as sparse
from scipy.sparse import coo_matrix
from scipy.sparse import csr_matrix
from scipy.sparse import linalg
from scipy.sparse.linalg import cg
import time
import freud # use freud to get the neighborlist

def solve_matrix_arpack_lanczos(sm,k0):
    """ Diagonalize matrix using arpack with k0 values """
    vals,vecs = linalg.eigsh(sm,k=k0,which='SA',tol=1e-10) 
    return vals,vecs

def solve_matrix_arpack_arnoldi(sm,k0):
    """ Diagonalize matrix using arpack with k0 values """
    vals,vecs = linalg.eigs(sm,k=k0,which='SM',tol=1e-10)
    return vals,vecs

def solve_matrix_lobpcg(sm,x_nk):
    """ Diagonalize matrix using lobpcg """
    vals,vecs = linalg.lobpcg(sm,X = x_nk,largest=False,tol=1e-6)
    return vals,vecs

def solve_matrix_lapack(dym):
    """ Diagonalize matrix using lapack """    
    vals, vecs = np.linalg.eigh(dym)
    return vals, vecs

def calc_dynamic_matrix_harmonic(xydfile,lattice_lx,alpha=2.):

    """ Dynamic (Hessain) matrix of a given solid with harmonic potentital

        Modify the vr vrr if using another potential

        Sparse matrix to save the memory
    """

    #xyd0 = pd.read_csv(xydfile,sep='\t',header=None)
    #xyd = xyd0.values
    #print xyd.shape
    xyd0 = np.loadtxt(xydfile)

    lx = xyd0[0,0]
    ly = xyd0[0,1]

    xyd = xyd0[1:,:]
    natom = len(xyd)

    for i in range(len(xyd)):
        iround1 = round( xyd[i,0] / lx)
        iround2 = round( xyd[i,1] / ly)
        xyd[i,0] = xyd[i,0] - iround1 * lx
        xyd[i,1] = xyd[i,1] - iround2 * ly

    box = freud.box.Box(lx,ly,is2D=True)
    xyz = box.wrap(xyd)

    aq = freud.locality.AABBQuery(box, xyz)
    cnlist = aq.query(xyz, {'r_max': 1.8}).toNeighborList()

   # dym = np.zeros((2*natom,2*natom))
    diag = np.zeros(4*natom)
    row = []
    col = []
    data = []

    Xi = np.zeros(2*natom) # Xi represents \partial U^2 / (\partial r \partial \gamma)
    born_term = 0
    v = lx*ly

    for i in range(natom):
        ri = xyd[i,:]
        #for j in particle_new_list[i]:
        tlist = cnlist.point_indices[cnlist.query_point_indices==i]
        for j in tlist:
            if i>=j:
                continue
            rj = xyd[j,:]
            rij = calc_dra(ri,rj,lx,ly)
            vec_r_ij = rj[:2] - ri[:2]
            dij = xyd[i,2] + xyd[j,2]
            if rij > dij:
                continue

            vr = -(1. - rij/dij)**(alpha - 1) / dij
            vrr = (alpha - 1.)*(1. - rij/dij)**(alpha - 2.) / dij**2

            vec_r_ij[0] = vec_r_ij[0] - round(vec_r_ij[0]/lx)*lx
            vec_r_ij[1] = vec_r_ij[1] - round(vec_r_ij[1]/ly)*ly
            xij = vec_r_ij[0] 
            yij = vec_r_ij[1] 
            
            rx = xij / rij
            ry = yij / rij

            rxx = 1./rij - xij**2 / rij**3
            ryy = 1./rij - yij**2 / rij**3

            rxy =        - xij*yij / rij**3

            res = rx * yij
            resx = rxx * yij
            resy = rxy * yij
            reses = rxx * yij**2

            mies1 = - ( vrr*rx*res + vr*resx )
            mies2 = - ( vrr*ry*res + vr*resy )

            mee   = vrr*res*res + vr*reses

            Xi[2*i] += mies1 
            Xi[2*i+1] += mies2 

            Xi[2*j] -= mies1 
            Xi[2*j+1] -= mies2 

            born_term += mee

           
            mij = np.zeros((2,2))
            mij[0,0] = - (vrr*rx**2 + vr*rxx)
            mij[1,1] = - (vrr*ry**2 + vr*ryy)

            mij[0,1] = - (vrr*rx*ry + vr*rxy)
            mij[1,0] = mij[0,1]

            for row_index in range(2*i,2*i+2):
                for col_index in range(2*j,2*j+2):
                    row.append(row_index)
                    col.append(col_index)

                    t_row = row_index - 2*i
                    t_col = col_index - 2*j
                    data.append(mij[t_row,t_col])

            
            #   symmetry
            for row_index in range(2*j,2*j+2):
                for col_index in range(2*i,2*i+2):
                    row.append(row_index)
                    col.append(col_index)

                    t_row = row_index - 2*j
                    t_col = col_index - 2*i
                    data.append(mij[t_row,t_col])


            # diag i
            diag[4*i]   = diag[4*i] - mij[0,0]
            diag[4*i+1] = diag[4*i+1] - mij[0,1]
            diag[4*i+2] = diag[4*i+2] - mij[1,0]
            diag[4*i+3] = diag[4*i+3] - mij[1,1]

            #dym[2*i,2*i] -= mij[0,0] 
            #dym[2*i+1,2*i]   -= mij[0,1]
            #dym[2*i,2*i+1]   -= mij[1,0]
            #dym[2*i+1,2*i+1]     -= mij[1,1]

            #diag j 
            diag[4*j]   = diag[4*j] - mij[0,0]
            diag[4*j+1] = diag[4*j+1] - mij[0,1]
            diag[4*j+2] = diag[4*j+2] - mij[1,0]
            diag[4*j+3] = diag[4*j+3] - mij[1,1]

            #dym[2*j,2*j] -= mij[0,0] 
            #dym[2*j+1,2*j]   -= mij[0,1]
            #dym[2*j,2*j+1]   -= mij[1,0]
            #dym[2*j+1,2*j+1]     -= mij[1,1]

    for i in range(natom):
        row.append(2*i)
        col.append(2*i)
        data.append(diag[4*i])

        row.append(2*i)
        col.append(2*i+1)
        data.append(diag[4*i+1])

        row.append(2*i+1)
        col.append(2*i)
        data.append(diag[4*i+2])

        row.append(2*i+1)
        col.append(2*i+1)
        data.append(diag[4*i+3])

    arrayrow = np.array(row)
    arraycol = np.array(col)
    arraydata = np.array(data)
    dym = csr_matrix((arraydata,(arrayrow,arraycol)),shape=(2*natom,2*natom))

    # return dymanic matrix, extended hessian term, born term, nonaffine velocity and neighborlist
    return dym, Xi, born_term,v, cnlist

def calc_dra(ri,rj,lx,ly):
    rij = rj[:2] - ri[:2]
    rij[0] = rij[0] - round(rij[0]/lx)*lx
    rij[1] = rij[1] - round(rij[1]/ly)*ly
    drij = np.sqrt(rij[0]**2 + rij[1]**2 )
    return drij

def shear_xq_bondG(filename,nonaffine_v,cnlist):
    """ measure the bond-level moduli by linear approximation """
    alpha = 2

    savetmp = []

    xyd0 = np.loadtxt(filename)

    lx = xyd0[0,0]
    ly = xyd0[0,1]

    v = lx*ly

    xyd = xyd0[1:,:]
    natom = len(xyd)

    xyd1 = xyd.copy()
    strain = 1.e-9
    xyd1[:,0] = xyd1[:,0] + strain * xyd1[:,1] 

    for atom_i in range(natom):
        xyd1[atom_i,0] += strain*nonaffine_v[2*atom_i]
        xyd1[atom_i,1] += strain*nonaffine_v[2*atom_i+1]

    for atom_i in range(natom):
        tlist = cnlist.point_indices[cnlist.query_point_indices==atom_i]
        for atom_j in tlist:
            if atom_i>=atom_j:
                continue
            drij = calc_dra(xyd[atom_i,:],xyd[atom_j,:],lx,ly)
            dij  = xyd[atom_i,2] + xyd[atom_j,2]
            if drij < dij:

                #before
                rij0 = xyd[atom_j,:2] - xyd[atom_i,:2] 
                cory0 = round( rij0[1] / ly )
                rij0[0] -= cory0 * 0 * ly

                rij0[0] -= round(rij0[0] / lx) * lx
                rij0[1] -= round(rij0[1] / ly) * ly

                rij2_0 = np.sum(rij0**2)
                drij_0 = np.sqrt(rij2_0)

                gij_0 = (1.0 - drij_0/dij)**(alpha-1) * drij_0 / dij / rij2_0  /  (- v) * rij0[0] * rij0[1]

                #after
                rij1 = xyd1[atom_j,:2] - xyd1[atom_i,:2] 
                cory1 = round( rij1[1] / ly )
                rij1[0] -= cory1 * strain * ly

                rij1[0] -= round(rij1[0] / lx) * lx
                rij1[1] -= round(rij1[1] / ly) * ly

                rij2_1 = np.sum(rij1**2)
                drij_1 = np.sqrt(rij2_1)

                gij_1 = (1.0 - drij_1/dij)**(alpha-1) * drij_1 / dij / rij2_1  /  (- v) * rij1[0] * rij1[1]

                savetmp.append([atom_i,atom_j, (gij_1 - gij_0)/strain, lx, ly, xyd[atom_i,0] + 0.5*rij0[0], xyd[atom_i,1] + 0.5*rij0[1]])
    
    return np.array(savetmp)

def main_getG(filename,savename):
    start = time.time()
    dym, Xi, born_term,v,cnlist = calc_dynamic_matrix_harmonic(filename,43)
    end1 = time.time()
    print("making dynamic matrix costs %f sec" %(end1 - start))
    #sparse.save_npz('dym_4d.npz',dym)
    #k0 = 100
    ##evals,evecs = solve_matrix_arpack_arnoldi(dym,k0)
    #evals,evecs = solve_matrix_arpack_lanczos(dym,k0)
    #end = time.time()
    #print("solving matrix costs %f sec" % (end - end1))
    #np.savetxt('eigen_4d_harm.dat',evals)


    end = time.time()

    nonaffine_v, exit_code = cg(dym, -Xi) 
    g = (born_term + np.dot(Xi, nonaffine_v)) / v

    end1 = time.time()
    print(g)
    print("CG calc G costs %f sec" % (end1 - end))

    bondG = shear_xq_bondG(filename, nonaffine_v,cnlist)
    end = time.time()
    print("calc bond G costs %f sec" % (end - end1))

    print(bondG[:,2].sum())

    np.savetxt(savename, bondG)

if __name__=='__main__':
    #filename3 = '1024_zero_strain_istart_69_7859_P_1_config.dat'
    filename3 = '../config_xyd/xydconfig_N16384_356.dat'
    savename = 'N16384_bondG.dat'

    main_getG(filename3,savename)

