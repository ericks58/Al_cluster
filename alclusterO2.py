import time
from pyscf import gto, scf, dft, df, lib
from pyscf.dft import gen_grid
from pyscf.scf.addons import remove_linear_dep_, smearing_
from pyscf.scf import chkfile as scfchk

d = 2.4
Z = d + 4.95613 

# ---- Al22 Cluster ----
cluster = gto.Mole()
cluster.atom= f'''
Al  -0.000042  -0.000018   0.000003
Al  -0.000145  -3.297372   2.331977
Al   0.196699  -5.720882   3.547310
Al  -2.855845   1.648264   2.332039
Al  -2.473921  -1.467455   4.685477
Al  -4.838758   2.996456   3.600710
Al   2.856253   1.649024   2.331851
Al   2.889648  -1.476585   4.681459
Al   0.204234   3.195891   4.702956
Al   5.047590   2.887405   3.127090
Al  -1.427854  -0.824381   2.331527
Al  -1.171459  -3.723129   4.706021
Al  -3.768714   0.788984   4.703314
Al   1.512936   0.842753   4.678985
Al   1.428318  -0.824151   2.331955
Al   1.620995  -3.813934   4.780576
Al  -1.128028   0.881825   4.956129
Al   4.231842   0.734298   4.486379
Al   0.000109   1.648844   2.331860
Al   0.225180  -1.471083   4.940156
Al  -2.455819   3.259885   4.797607
Al   2.794915   3.243950   4.530983
O    0.331970  -0.052030   {Z}
O    1.445740  -0.710940   {Z}
'''

cluster.basis = {
     'O': gto.basis.parse('''
O     S
      0.5484671660E+04       0.1831074430E-02
      0.8252349460E+03       0.1395017220E-01
      0.1880469580E+03       0.6844507810E-01
      0.5296450000E+02       0.2327143360E+00
      0.1689757040E+02       0.4701928980E+00
      0.5799635340E+01       0.3585208530E+00
O    SP
      0.1553961625E+02      -0.1107775495E+00       0.7087426823E-01
      0.3599933586E+01      -0.1480262627E+00       0.3397528391E+00
      0.1013761750E+01       0.1130767015E+01       0.7271585773E+00
O    SP
      0.2700058226E+00       0.1000000000E+01       0.1000000000E+01
O    SP
      0.8450000000E-01       0.1000000000E+01       0.1000000000E+01
O    D
      0.8000000000E+00       1.0000000
'''),
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
#keep integrals in RAM
cluster.spin = 2
cluster.incore_anyway = True
cluster.build()

# ---------- RKS object ----------
mf = dft.UKS(cluster, xc='PBE')
mf = mf.density_fit(auxbasis='weigend')   # enable RI-J first
mf = mf.apply(remove_linear_dep_)
mf.direct_scf = True
mf.max_cycle = 300
mf.conv_tol = 5e-05
# Metallic stability
mf = smearing_(mf, sigma=0.005, method='fermi')
mf.verbose = 4
# Coarse grid
mf.grids.level = 5
mf.grids.prune = gen_grid.treutler_prune
mf.small_rho_cutoff = 1e-6
mf.diis_space  = 12
mf.level_shift = 0.2
mf.damp        = 0.15
mf.conv_check = False
mf.chkfile =f'./al{d}.chk'
e=mf.kernel()


#----------- DM as Initial Guess ----------
mf = dft.UKS(cluster, xc='PBE')
mf = mf.density_fit(auxbasis='weigend')   # enable RI-J first
mf = mf.apply(remove_linear_dep_)
mf.direct_scf = True
mf.max_cycle = 500
mf.conv_tol = 5e-05
# Metallic stability
mf = smearing_(mf, sigma=0.01, method='fermi')
mf.verbose = 4
# Coarse grid
mf.grids.level = 3
mf.grids.prune = gen_grid.treutler_prune
mf.small_rho_cutoff = 1e-6
mf.diis_space  = 12
mf.level_shift = 0.2
mf.damp        = 0.15

al, scf_dict = scfchk.load_scf(f'./al{d}.chk')
mo_coeff = scf_dict['mo_coeff']
mo_occ = scf_dict['mo_occ']
mo_energy = scf_dict['mo_energy']
dm = mf.make_rdm1(mo_coeff, mo_occ)
mf.conv_check = False
mf.chkfile = f'./al{d}.chk'
e_chk=mf.kernel(dm)

print(f'For distance {d} the energy was {e_chk} Hartree')