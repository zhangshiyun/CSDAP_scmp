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

         3d systems

         """

import numpy as np
#import pandas as pd
import scipy.sparse as sparse
from scipy.sparse import coo_matrix
from scipy.sparse import csr_matrix
from scipy.sparse import linalg
from scipy.sparse.linalg import cg
import time
import freud



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
    lz = xyd0[0,2]
    xyd = xyd0[1:,:]
    natom = len(xyd)

    for i in range(len(xyd)):
        iround1 = round( xyd[i,0] / lx)
        iround2 = round( xyd[i,1] / ly)
        iround3 = round( xyd[i,2] / lz)
        xyd[i,0] = xyd[i,0] - iround1 * lx
        xyd[i,1] = xyd[i,1] - iround2 * ly
        xyd[i,2] = xyd[i,2] - iround3 * lz

    box = freud.box.Box(Lx=lx,Ly=ly,Lz=lz)
    xyz = box.wrap(xyd[:,:3])

    aq = freud.locality.AABBQuery(box, xyz)
    cnlist = aq.query(xyz,{'r_max': 1.8}).toNeighborList()

   # dym = np.zeros((2*natom,2*natom))
    diag = np.zeros(9*natom)
    row = []
    col = []
    data = []

    #particle_new_list = make_lattice(xyd,lx,ly,lz,lattice_lx)

    Xi = np.zeros(3*natom)
    born_term = 0
    v = lx*ly*lz

    for i in range(natom-1):
        ri = xyd[i,:]
        #for j in particle_new_list[i]:
        for j in range(i+1,natom):
            if i>=j:
                continue
            rj = xyd[j,:]
            rij = calc_dra(ri,rj,lx,ly,lz)
            vec_r_ij = rj[:3] - ri[:3]
            dij = xyd[i,3] + xyd[j,3]
            if rij > dij:
                continue

            vr = -(1. - rij/dij)**(alpha - 1) / dij
            vrr = (alpha - 1.)*(1. - rij/dij)**(alpha - 2.) / dij**2

            vec_r_ij[0] = vec_r_ij[0] - round(vec_r_ij[0]/lx)*lx
            vec_r_ij[1] = vec_r_ij[1] - round(vec_r_ij[1]/ly)*ly
            vec_r_ij[2] = vec_r_ij[2] - round(vec_r_ij[2]/lz)*lz
            xij = vec_r_ij[0] 
            yij = vec_r_ij[1] 
            zij = vec_r_ij[2] 
            
            rx = xij / rij
            ry = yij / rij
            rz = zij / rij

            rxx = 1./rij - xij**2 / rij**3
            ryy = 1./rij - yij**2 / rij**3
            rzz = 1./rij - zij**2 / rij**3
            rxy =        - xij*yij / rij**3
            rxz =        - xij*zij / rij**3
            ryz =        - yij*zij / rij**3

            res = rx * zij
            resx = rxx * zij
            resy = rxy * zij
            resz = rxz * zij
            reses = rxx * zij**2

            mies1 = - ( vrr*rx*res + vr*resx )
            mies2 = - ( vrr*ry*res + vr*resy )
            mies3 = - ( vrr*rz*res + vr*resz )

            mee   = vrr*res*res + vr*reses

            Xi[3*i] += mies1 
            Xi[3*i+1] += mies2 
            Xi[3*i+2] += mies3 
            Xi[3*j] -= mies1 
            Xi[3*j+1] -= mies2 
            Xi[3*j+2] -= mies3 
            born_term += mee

           
            mij = np.zeros((3,3))
            mij[0,0] = - (vrr*rx**2 + vr*rxx)
            mij[1,1] = - (vrr*ry**2 + vr*ryy)
            mij[2,2] = - (vrr*rz**2 + vr*rzz)
            mij[0,1] = - (vrr*rx*ry + vr*rxy)
            mij[1,0] = mij[0,1]
            mij[0,2] = - (vrr*rx*rz + vr*rxz)
            mij[2,0] = mij[0,2]
            mij[1,2] = - (vrr*ry*rz + vr*ryz)
            mij[2,1] = mij[1,2]


            row.append(3*i)
            col.append(3*j)
            data.append(mij[0,0])

            row.append(3*i+1)
            col.append(3*j)
            data.append(mij[1,0])

            row.append(3*i)
            col.append(3*j+1)
            data.append(mij[0,1])

            row.append(3*i+1)
            col.append(3*j+1)
            data.append(mij[1,1])

            row.append(3*i+2)
            col.append(3*j)
            data.append(mij[2,0])

            row.append(3*i)
            col.append(3*j+2)
            data.append(mij[0,2])

            row.append(3*i+2)
            col.append(3*j+1)
            data.append(mij[2,1])

            row.append(3*i+1)
            col.append(3*j+2)
            data.append(mij[1,2])

            row.append(3*i+2)
            col.append(3*j+2)
            data.append(mij[2,2])

            #dym[2*i,2*j] = mij[0,0] 
            #dym[2*i+1,2*j]   = mij[1,0]
            #dym[2*i,2*j+1]   = mij[0,1]
            #dym[2*i+1,2*j+1]     = mij[1,1]
            
            #   symmetry
            row.append(3*j)
            col.append(3*i)
            data.append(mij[0,0])

            row.append(3*j+1)
            col.append(3*i)
            data.append(mij[1,0])

            row.append(3*j)
            col.append(3*i+1)
            data.append(mij[0,1])

            row.append(3*j+1)
            col.append(3*i+1)
            data.append(mij[1,1])

            row.append(3*j+2)
            col.append(3*i)
            data.append(mij[2,0])

            row.append(3*j)
            col.append(3*i+2)
            data.append(mij[0,2])

            row.append(3*j+2)
            col.append(3*i+1)
            data.append(mij[2,1])

            row.append(3*j+1)
            col.append(3*i+2)
            data.append(mij[1,2])

            row.append(3*j+2)
            col.append(3*i+2)
            data.append(mij[2,2])

            #dym[2*j,2*i] = mij[0,0] 
            #dym[2*j,2*i+1]   = mij[0,1]
            #dym[2*j+1,2*i]   = mij[1,0]
            #dym[2*j+1,2*i+1]     = mij[1,1]

            # diag i
            diag[9*i] = diag[9*i] - mij[0,0]
            diag[9*i+1] = diag[9*i+1] - mij[0,1]
            diag[9*i+2] = diag[9*i+2] - mij[1,0]
            diag[9*i+3] = diag[9*i+3] - mij[1,1]
            diag[9*i+4] = diag[9*i+4] - mij[0,2]
            diag[9*i+5] = diag[9*i+5] - mij[2,0]
            diag[9*i+6] = diag[9*i+6] - mij[1,2]
            diag[9*i+7] = diag[9*i+7] - mij[2,1]
            diag[9*i+8] = diag[9*i+8] - mij[2,2]

            #dym[2*i,2*i] -= mij[0,0] 
            #dym[2*i+1,2*i]   -= mij[0,1]
            #dym[2*i,2*i+1]   -= mij[1,0]
            #dym[2*i+1,2*i+1]     -= mij[1,1]

            #diag j 
            diag[9*j] = diag[9*j] - mij[0,0]
            diag[9*j+1] = diag[9*j+1] - mij[0,1]
            diag[9*j+2] = diag[9*j+2] - mij[1,0]
            diag[9*j+3] = diag[9*j+3] - mij[1,1]
            diag[9*j+4] = diag[9*j+4] - mij[0,2]
            diag[9*j+5] = diag[9*j+5] - mij[2,0]
            diag[9*j+6] = diag[9*j+6] - mij[1,2]
            diag[9*j+7] = diag[9*j+7] - mij[2,1]
            diag[9*j+8] = diag[9*j+8] - mij[2,2]

            #dym[2*j,2*j] -= mij[0,0] 
            #dym[2*j+1,2*j]   -= mij[0,1]
            #dym[2*j,2*j+1]   -= mij[1,0]
            #dym[2*j+1,2*j+1]     -= mij[1,1]

    for i in range(natom):
        row.append(3*i)
        col.append(3*i)
        data.append(diag[9*i])

        row.append(3*i)
        col.append(3*i+1)
        data.append(diag[9*i+1])

        row.append(3*i+1)
        col.append(3*i)
        data.append(diag[9*i+2])

        row.append(3*i+1)
        col.append(3*i+1)
        data.append(diag[9*i+3])

        row.append(3*i)
        col.append(3*i+2)
        data.append(diag[9*i+4])

        row.append(3*i+2)
        col.append(3*i)
        data.append(diag[9*i+5])

        row.append(3*i+1)
        col.append(3*i+2)
        data.append(diag[9*i+6])

        row.append(3*i+2)
        col.append(3*i+1)
        data.append(diag[9*i+7])

        row.append(3*i+2)
        col.append(3*i+2)
        data.append(diag[9*i+8])


    arrayrow = np.array(row)
    arraycol = np.array(col)
    arraydata = np.array(data)
    dym = csr_matrix((arraydata,(arrayrow,arraycol)),shape=(3*natom,3*natom))
    return dym, Xi, born_term,v, cnlist

def calc_dra(ri,rj,lx,ly,lz):
    rij = rj[:3] - ri[:3]
    rij[0] = rij[0] - round(rij[0]/lx)*lx
    rij[1] = rij[1] - round(rij[1]/ly)*ly
    rij[2] = rij[2] - round(rij[2]/lz)*lz
    drij = np.sqrt(rij[0]**2 + rij[1]**2 + rij[2]**2)
    return drij

def shear_xq_bondG(filename,nonaffine_v,cnlist):
    """ measure the bond-level moduli by linear approximation """

    alpha = 2

    savetmp = []

    xyd0 = np.loadtxt(filename)

    lx = xyd0[0,0]
    ly = xyd0[0,1]
    lz = xyd0[0,2]

    v = lx*ly*lz

    xyd = xyd0[1:,:]
    natom = len(xyd)

    xyd1 = xyd.copy()
    strain = 1.e-9
    xyd1[:,0] = xyd1[:,0] + strain * xyd1[:,2] 

    for atom_i in range(natom):
        xyd1[atom_i,0] += strain*nonaffine_v[3*atom_i]
        xyd1[atom_i,1] += strain*nonaffine_v[3*atom_i+1]
        xyd1[atom_i,2] += strain*nonaffine_v[3*atom_i+2]

    #for atom_i in range(natom-1):
    #    for atom_j in range(atom_i+1, natom):
    for atom_i in range(natom):
        tlist = cnlist.point_indices[cnlist.query_point_indices==atom_i]
        for atom_j in tlist:
            if atom_i>=atom_j:
                continue
            drij = calc_dra(xyd[atom_i,:],xyd[atom_j,:],lx,ly,lz)
            dij  = xyd[atom_i,3] + xyd[atom_j,3]
            if drij < dij:

                #before
                rij0 = xyd[atom_j,:3] - xyd[atom_i,:3] 
                cory0 = round( rij0[2] / lz )
                rij0[0] -= cory0 * 0 * lz

                rij0[0] -= round(rij0[0] / lx) * lx
                rij0[1] -= round(rij0[1] / ly) * ly
                rij0[2] -= round(rij0[2] / lz) * lz

                rij2_0 = np.sum(rij0**2)
                drij_0 = np.sqrt(rij2_0)

                gij_0 = (1.0 - drij_0/dij)**(alpha-1) * drij_0 / dij / rij2_0  /  (- v) * rij0[0] * rij0[2]

                #after
                rij1 = xyd1[atom_j,:3] - xyd1[atom_i,:3] 
                cory1 = round( rij1[2] / lz )
                rij1[0] -= cory1 * strain * lz

                rij1[0] -= round(rij1[0] / lx) * lx
                rij1[1] -= round(rij1[1] / ly) * ly
                rij1[2] -= round(rij1[2] / lz) * lz

                rij2_1 = np.sum(rij1**2)
                drij_1 = np.sqrt(rij2_1)

                gij_1 = (1.0 - drij_1/dij)**(alpha-1) * drij_1 / dij / rij2_1  /  (- v) * rij1[0] * rij1[2]

                savetmp.append([atom_i,atom_j, (gij_1 - gij_0)/strain, lx, ly,lz, xyd[atom_i,0] + 0.5*rij0[0], xyd[atom_i,1] + 0.5*rij0[1], xyd[atom_i,2] + 0.5*rij0[2]])
    
    return np.array(savetmp)

def main_getG(filename,savename):
    start = time.time()
    dym, Xi, born_term,v,cnlist = calc_dynamic_matrix_harmonic(filename,43)
    end1 = time.time()
    print("making dynamic matrix costs %f sec" %(end1 - start))

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
    filename3 = '1024_zero_strain_istart_39_203824_P_1_config.dat'
    savename = 'N1024_bondG.dat'

    main_getG(filename3,savename)

    a = np.loadtxt('1024_zero_strain_istart_39_203824_Pstep_1_bonds.dat')
    print(a[:,2].sum())

