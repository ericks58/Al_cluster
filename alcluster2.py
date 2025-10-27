import time
from pyscf import gto, scf, dft, df, lib
from pyscf.dft import gen_grid
from pyscf.scf.addons import remove_linear_dep_, smearing_
from pyscf.geomopt.geometric_solver import optimize

def calcbondlength(b):
	o1 = [0.00060,0.00087]
	o2 = [0.00060,0.00087]
	dist = b/2
	o1[1] += dist
	o2[1] -= dist
	return o1, o2

bonds = np.linspace(1,3.0,20)
Energies_z = []
#dist = np.linspace(1.8,3.0,5)
dist = np.linspace(1.8,3.0,20)
for d in dist:

	Energies_xy=[]
	for b in bonds:
		o1, o2 = calcbondlength(b)
		x1, y1 = o1
		x2, y2 = o2
		Z = 6.99563081 + d
		print(f'Distance: {d} and Z position: {Z}')
		#create a cluster of Al atoms
		cluster = gto.Mole()
		cluster.atom=f'''
		Al -0.00000250 -0.00000393 0.00001014
		Al -0.00001035 -3.29776820 2.33188593
		Al 0.00000271 -6.59554088 4.66375465
		Al 0.00000028 -9.89330895 6.99562804
		Al -2.85594425 1.64887744 2.33186800
		Al -2.85591982 -1.64885021 4.66368245
		Al -2.85593638 -4.94665766 6.99559652
		Al -5.71191948 3.29776330 4.66375756
		Al -5.71190301 0.00001144 6.99572982
		Al -8.56786193 4.94665550 6.99566040
		Al 2.85594197 1.64887618 2.33187480
		Al 2.85591102 -1.64887528 4.66370830
		Al 2.85594192 -4.94666228 6.99571612
		Al 0.00001981 3.29772932 4.66369971
		Al 0.00060073 0.00087430 7.08173027
		Al -2.85595414 4.94665273 6.99557199
		Al 5.71189902 3.29777569 4.66375847
		Al 5.71191052 0.00000270 6.99558158
		Al 2.85596266 4.94665432 6.99570886
		Al 8.56785950 4.94665750 6.99566468
		Al -1.42795146 -0.82442424 2.33184736
		Al -1.42795662 -4.12222691 4.66369219
		Al -1.42796566 -7.41996734 6.99554687
		Al -4.28395196 0.82445890 4.66368915
		Al -4.28391682 -2.47332891 6.99564618
		Al -7.13986904 2.47332671 6.99562289
		Al 1.39853070 0.80786659 4.64936506
		Al 1.37497802 -2.39385703 6.98880837
		Al -1.38602334 2.38796034 6.98876427
		Al 4.13188901 2.38572918 7.08500115
		Al 1.42794054 -0.82442531 2.33185228
		Al 1.42797217 -4.12225477 4.66372583
		Al 1.42796919 -7.41996686 6.99559881
		Al -1.39766764 0.80801006 4.64917618
		Al -1.37429335 -2.39381234 6.98919873
		Al -4.13160802 2.38590683 7.08539647
		Al 4.28392726 0.82446590 4.66371573
		Al 4.28393084 -2.47332504 6.99566058
		Al 1.38698526 2.38808693 6.98844954
		Al 7.13987318 2.47331999 6.99559005
		Al -0.00000735 1.64884644 2.33186748
		Al 0.00054946 -1.61391839 4.64888546
		Al 0.00016015 -4.77110212 7.08658975
		Al -2.85598391 3.29776543 4.66371662
		Al -2.76063549 0.00729729 6.98930221
		Al -5.71189328 4.94664828 6.99559320
		Al 2.85599081 3.29778309 4.66372928
		Al 2.76137067 0.00754474 6.98858236
		Al -0.00000098 4.94665289 6.99566063
		Al 5.71189800 4.94664993 6.99563465
		O {x1} {y1}  {Z}
		O {x2} {y2}  {Z}
		'''

		cluster.basis='6-31G'
		#spherical gaussians
		cluster.cart = False
		#keep integrals in RAM
		cluster.incore_anyway = True
		cluster.spin = 2
		cluster.build()

		# ---------- RKS object ----------
		mf = dft.UKS(cluster, xc='PBE')
		mf = mf.density_fit(auxbasis='weigend')   # enable RI-J first
		mf = mf.apply(remove_linear_dep_)
		mf.direct_scf = True
		mf.max_cycle = 200
		mf.conv_tol = 5e-05
		mf.conv_tol_grad = 4.5e-04
		# Metallic stability
		mf = smearing_(mf, sigma=0.01, method='fermi')
		mf.verbose = 4
		mf.small_rho_cutoff = 1e-6
		mf.diis_space  = 12
		mf.level_shift = 0.2
		mf.damp        = 0.15
		mf.verbose=4
		# KS Object
		mf.grids.level = 5
		mf.verbose=4
		e=mf.kernel()
		Energies_xy.append(e)
	Energies_z.append(Energies_xy)
	print("---------------- Loop Finished ------------------")


with open('./energy.dat', 'w') as f:
	f.write{'energies for distance in lists of bond length: ', Energies_z)
	f.write('\n', 'Distances:', d)
	f.write('\n', 'Bond Lengths:', bonds)

