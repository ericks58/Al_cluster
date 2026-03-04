import time
from pyscf import gto, scf, dft, df, lib
from pyscf.dft import gen_grid
from pyscf.scf.addons import remove_linear_dep_, smearing_
from pyscf.scf import chkfile as scfchk
from pyscf.geomopt.geometric_solver import optimize

# ---- Al22 Cluster ----
cluster = gto.Mole()
cluster.atom= f'''
Al  0.00000000  0.00000000  0.00000000
Al  0.00000000 -3.29777172  2.33187675
Al  0.00000000 -6.59554425  4.66375407
Al -2.85595408  1.64888586  2.33187675
Al -2.85595408 -1.64888586  4.66375349
Al -5.71190888  3.28777213  4.66375407
Al  2.85595408  1.64888586  2.33187675
Al  2.85595408 -1.64888586  4.66375349
Al  0.00000000  3.29777172  4.66375349
Al  5.71190888  3.29777172  4.66275349
Al -1.42797740 -0.82444313  2.33187732
Al -1.42797740 -4.12221485  4.66375407
Al -4.28393148  0.82444273  4.66375407
Al  1.42797669  0.82444273  4.66375407
Al  1.42797740 -0.82444313  2.33187732
Al  1.42797740 -4.12221485  4.66375407
Al -1.42797669  0.82444273  4.66375407
Al  4.28393148  0.82444273  4.66375407
Al  0.00000000  1.64888627  2.33187732
Al  0.00000000 -1.64888545  4.66375407
Al -2.85595408  3.29777213  4.66375407
Al  2.85595408  3.39777213  4.66375407
'''

cluster.basis = {
'Al':gto.basis.parse('''
Al    S
      0.1398310000E+05       0.1942669947E-02
      0.2098750000E+04       0.1485989959E-01
      0.4777050000E+03       0.7284939800E-01
      0.1343600000E+03       0.2468299932E+00
      0.4287090000E+02       0.4872579866E+00
      0.1451890000E+02       0.3234959911E+00
Al    SP
      0.2396680000E+03      -0.2926190028E-02       0.4602845582E-02
      0.5744190000E+02      -0.3740830036E-01       0.3319896813E-01
      0.1828590000E+02      -0.1144870011E+00       0.1362818692E+00
      0.6599140000E+01       0.1156350011E+00       0.3304756828E+00
      0.2490490000E+01       0.6125950058E+00       0.4491455689E+00
      0.9445450000E+00       0.3937990037E+00       0.2657037450E+00
Al    SP
      0.1277900000E+01      -0.2276069245E+00      -0.1751260189E-01
      0.3975900000E+00       0.1445835873E-02       0.2445330264E+00
      0.1600950000E+00       0.1092794439E+01       0.8049340867E+00
Al    SP
      0.5565770000E-01       0.1000000000E+01       0.1000000000E+01
Al    SP
      0.3180000000E-01       0.1000000000E+01       0.1000000000E+01
Al    D
      0.3250000000E+00       1.0000000
''')
}

cluster.cart = False
cluster.spin = 0
cluster.incore_anyway = True
cluster.build()

# ----- RKS Object -----
mf_cluster = dft.RKS(cluster, xc="PBE")
mf_cluster = mf_cluster.density_fit(auxbasis='weigend')
mf_cluster = mf_cluster.apply(remove_linear_dep_)
mf_cluster.direct_scf = False #changed ## Future carmen: why did you change this again? maybe you should un-change
mf_cluster.max_cycle = 400
mf_cluster.conv_tol = 5e-5
#mf_cluster.conv_tol_grad = 1e-06
mf_cluster.maxsteps = 400
mf_cluster = smearing_(mf_cluster, sigma=0.005, method='fermi') #future carmen: change sigma to 0.01 - this is aggressive. dont be aggressive
mf_cluster.verbose=4
mf_cluster.small_rho_cutoff = 0 #1e-6
mf_cluster.diis_space = 8
mf_cluster.level_shift = 0.2
mf_cluster.damp = 0.12
#added
mf_cluster.grids.level = 5 #can increase later 
mf_cluster.grids.prune = None
mf_cluster.conv_check = False # Trying this to stop the jumps in the extra cycle
# 'constraints': "constraints.txt",
# ---- Optimization ----
params = {'constraints': "constraints.txt",
          'maxsteps':300,
          'trust':0.02,
          'tmax':0.08
}
cluster_geo = optimize(mf_cluster, **params)
cluster_geo.basis={
'Al':gto.basis.parse('''
Al    S
      0.1398310000E+05       0.1942669947E-02
      0.2098750000E+04       0.1485989959E-01
      0.4777050000E+03       0.7284939800E-01
      0.1343600000E+03       0.2468299932E+00
      0.4287090000E+02       0.4872579866E+00
      0.1451890000E+02       0.3234959911E+00
Al    SP
      0.2396680000E+03      -0.2926190028E-02       0.4602845582E-02
      0.5744190000E+02      -0.3740830036E-01       0.3319896813E-01
      0.1828590000E+02      -0.1144870011E+00       0.1362818692E+00
      0.6599140000E+01       0.1156350011E+00       0.3304756828E+00
      0.2490490000E+01       0.6125950058E+00       0.4491455689E+00
      0.9445450000E+00       0.3937990037E+00       0.2657037450E+00
Al    SP
      0.1277900000E+01      -0.2276069245E+00      -0.1751260189E-01
      0.3975900000E+00       0.1445835873E-02       0.2445330264E+00
      0.1600950000E+00       0.1092794439E+01       0.8049340867E+00
Al    SP
      0.5565770000E-01       0.1000000000E+01       0.1000000000E+01
Al    SP
      0.3180000000E-01       0.1000000000E+01       0.1000000000E+01
Al    D
      0.3250000000E+00       1.0000000
''')
}

# ---- KS Object ----
mf = dft.RKS(cluster_geo, xc='pbe')
mf = mf.density_fit(auxbasis='weigend')
mf = mf.apply(remove_linear_dep_)
mf.direct_scf = True
mf.max_cycle = 100
mf.conv_tol = 5e-05
mf.conv_tol_grad = 4.5e-04
mf.max_stepsize = 1.8e-03

# ---- Metallic Stability ----
mf = smearing_(mf, sigma=0.01, method='fermi')
mf.verbose=4
mf.small_rho_cutoff = 1e-6
mf.diis_space = 12
mf.level_shift = 0.2
mf.damp = 0.15

e=mf.kernel()

print(f"Final Energy: {e} Hartree")
