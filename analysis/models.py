import numpy as np
from scipy.signal import find_peaks as find_peaks

### HELPER FUNCTIONS
def loss_biological(j1, j2, alpha1=1E-5, alpha2=1E-2):
        """Computes scalarized loss including genetic constraint and product production"""
        loss = alpha1*j1 + alpha2*j2
        return j1, j2, loss

def activation(x, k, theta, n):
    return (k*(x/theta)**n)/(1+(x/theta)**n)

def repression(x, k, theta, n):
    return k/(1+(x/theta)**n)
    
def nonlinearity(x, kc, km):
    return (kc*x)/(km+x)

def reversible_michaelismenten(x, y, vm, keq, kmx, kmy):
    return (vm*(x - (y/keq)))/(x + kmx*(1+(y/kmy)))

def hilleqn(x, vm, n, km):
    return (vm*x**n)/(km**n + x**n)

def michaelismenten_substrateactivation(x, vm, km, a, ka):
    vm_eff = vm * (1+ (a*x)/(ka + x))
    return (vm_eff*x)/(km  + x)

def michaelismenten(x, vm, km):
    return (vm*x)/(km+x)

def promoteractivation(x, r, kd):
    return r/(1+(x/kd))

def is_oscillatory(solution, beta = 10E3):
    #oscillations include >3 peaks with peak-peak distance standard deviation < 1
    for k in range(1, solution.values.y.shape[1]-2):
        peaks = find_peaks(solution.values.y[:, k])[0]
        dists = [j-i for i, j in zip(peaks[:-1], peaks[1:])]
        if len(peaks) > 3 and np.std(dists) < 1:
            return 1*beta
        else: return 0

### TOY MODEL
def toy_model(t, y, ydot, params):
    kc=12.; km=10.; lam=1.93E-4; Vin=1.; e0=0.0467
    T = 1; E = 2; X = 2
    A, W = params
    ydot[0] = Vin - lam*y[0] - e0*nonlinearity(y[0], kc, km) - y[2]*nonlinearity(y[0], kc, km)
    ydot[1] = y[2]*nonlinearity(y[0], kc, km) - y[3]*nonlinearity(y[1], kc, km) - lam*y[1]
    for e in range(E):
        ydot[e+X] = -lam*y[e+X] + np.sum(A[e]*np.array([activation(y[T], W[e][2], W[e][1], W[e][0]), repression(y[T], W[e][2], W[e][1], W[e][0]), W[e][2]]))
    ydot[E+X] = (Vin - y[X+1]*nonlinearity(y[X-1], kc, km))**2 #J1
    ydot[E+X+1] = np.sum([np.sum(A[e]*np.array([activation(y[T], W[e][2], W[e][1], W[e][0]), repression(y[T], W[e][2], W[e][1], W[e][0]), W[e][2]])) for e in range(E)]) #J2

### GLUCARIC ACID MODEL
def glucaric_acid(t, y, ydot, params):

    lam = 2.7778E-05
    v_pts = 0.1656
    vm_pgi = 0.8751
    keq_pgi = 0.3
    km_pgi_g6p = 0.28
    km_pgi_f6p = 0.147
    vm_zwf = 0.0853
    km_zwf_g6p = 0.1
    vm_pfk = 2.615
    km_pfk_f6p = 0.16
    n_pfk = 3
    vm_ino1 = 0.2616
    km_ino1_g6p = 1.18
    vm_t_mi = 0.045
    km_t_mi = 15
    vm_miox = 0.2201
    km_miox_mi = 24.7
    a_miox = 5.4222
    ka_miox_mi = 20


    g6p, f6p, mi, ino1, miox, j1, j2 = y

    A, W = params 

    n_ino1, theta_ino1, k_ino1 = W[0]
    n_miox, theta_miox, k_miox = W[1]

    v_pgi = reversible_michaelismenten(g6p, f6p, vm_pgi, keq_pgi, km_pgi_g6p, km_pgi_f6p)
    v_zwf = michaelismenten(g6p, vm_zwf, km_zwf_g6p)
    v_pfk = hilleqn(f6p, vm_pfk, n_pfk, km_pfk_f6p)
    v_ino1 = ino1 * michaelismenten(g6p, vm_ino1, km_ino1_g6p)
    v_tm = michaelismenten(mi, vm_t_mi, km_t_mi)
    v_miox = miox * michaelismenten_substrateactivation(mi, vm_miox, km_miox_mi, a_miox, ka_miox_mi)

    u_ino1_mi = np.sum(A[0]*np.array([activation(mi, k_ino1, theta_ino1, n_ino1), repression(mi, k_ino1, theta_ino1, n_ino1), k_ino1]))
    u_miox_mi = np.sum(A[1]*np.array([activation(mi, k_miox, theta_miox, n_miox), repression(mi, k_miox, theta_miox, n_miox), k_miox]))

    ydot[0] = v_pts - v_zwf - v_pgi - lam*g6p
    ydot[1] = v_pgi + 0.5*v_zwf - v_pfk - lam*f6p
    ydot[2] = v_ino1 - v_tm - v_miox - lam*mi
    ydot[3] = u_ino1_mi  - lam*ino1
    ydot[4] = u_miox_mi - lam*miox
    ydot[5] = (v_pts - v_miox)**2
    ydot[6] = u_ino1_mi + u_miox_mi

### FATTY ACID MODEL
def fa_openloop(t, y, ydot, params):
    FFA, tesA, j1, j2 = y
    mu = 3.85E-4
    k_tesA = 100.
    r_lac = params
    ydot[0] = tesA*k_tesA - mu*FFA
    ydot[1] = r_lac - mu*tesA
    ydot[2] = tesA*k_tesA 
    ydot[3] = r_lac

def fa_openloopintermediate(t, y, ydot, params):
    FFA, tesA, CAR, j1, j2 = y
    mu = 3.85E-4
    k_car = 2.83E-4
    k_tesA = 100
    r_lac, r_bad = params
    ydot[0] = tesA * k_tesA - mu * FFA - CAR * k_car
    ydot[1] = r_lac - mu*tesA
    ydot[2] = r_bad - mu*CAR
    ydot[3] = tesA * k_tesA - CAR * k_car
    ydot[4] = r_bad + r_lac

def fa_negativegeneloop(t, y, ydot, params):
    FFA, tesA, tetR, j1, j2 = y
    mu = 3.85E-4
    kd_tetR = 3.0E-8
    k_tesA = 105.25
    r_tl, r_tl_tetR = params
    ydot[0] = tesA * k_tesA - mu * FFA
    ydot[1] = promoteractivation(tetR, r_tl, kd_tetR) - mu * tesA
    ydot[2] = promoteractivation(tetR, r_tl_tetR, kd_tetR) - mu * tetR
    ydot[3] = tesA * k_tesA
    ydot[4] = promoteractivation(tetR, r_tl, kd_tetR) + promoteractivation(tetR, r_tl_tetR, kd_tetR)

def fa_negativemetabolicloop(t, y, ydot, params):
    FFA, tesA, j1, j2 = y
    k_tesA = 77.75
    mu = 3.85E-4
    r_fl_prime, ki = params
    ydot[0] = k_tesA * tesA - mu*FFA
    ydot[1] = promoteractivation(FFA, r_fl_prime, ki) - mu * tesA
    ydot[2] = k_tesA * tesA
    ydot[3] = promoteractivation(FFA, r_fl_prime, ki) 

def fa_layerednegativemetabolicloop(t, y, ydot, params):
    FFA,  tesA , tetR, j1, j2 = y
    k_tesA = 230.9
    kd_tetR = 3.85E-8
    kd_fadR_FFA = 0.001
    k2 = 138.50
    mu = 3.85E-4
    r_tl, r_ar2 = params
    ydot[0] = tesA * k_tesA - mu * FFA
    ydot[1] = promoteractivation(tetR, r_tl, kd_tetR) - mu*tesA
    ydot[2] = promoteractivation(k2, r_ar2, (1+(FFA/kd_fadR_FFA))) - mu*tetR
    ydot[3] = tesA * k_tesA
    ydot[4] = promoteractivation(tetR, r_tl, kd_tetR) + promoteractivation(k2, r_ar2, (1+(FFA/kd_fadR_FFA)))


### P-AMINOSTYRENE MODEL
def p_aminostyrene(t, y, ydot, params):
    #Parse input parameters
    chorismate, pa1, pa2, pa3, paf, paca_int, paca_ext, promoter1, papA_mrna, papA_uf, papA, papB_mrna, papB_uf, papB, papC_mrna, papC_uf, papC, deaminase, promoter2, laao_mrna, laao_uf, laao, promoter3, eff_mrna, eff_uf, eff, j1, j2 = y
    architecture, thetas, ks, perturbs = params
    theta_paf_prom1, theta_paf_prom2, theta_paf_prom3 = thetas[0]
    theta_paca_prom1, theta_paca_prom2, theta_paca_prom3 = thetas[1]
    k_paf_papA, k_paf_papB, k_paf_papC, k_paf_prom2, k_paf_prom3 = ks[0]
    k_paca_papA, k_paca_papB, k_paca_papC, k_paca_prom2, k_paca_prom3 = ks[1]
    n = 2 #Fix n based on dimerization

    #Cellular constants
    chorismate_production_rate = 1100. #range [2E2, 2E3]
    deaminase_production_rate =  1E1 #range [1E0, 1E2]are 
    mrna_degradation_rate = 3E-3 #range [3E-4, 3E-2]
    protein_degradation_rate = 2E-4 #range [2E-5, 2E-3]
    protein_folding_rate = 2E0 #range [2E-1, 2E1] 
    dilution_rate = 5.79E-4
    dna_duplication_rate = 5.78E-4
    avogadro = 6.0221408e+23
    cell_volume = 2.5E-15

    #Toxicity factor
    #Perturbed parameters
    ta = perturbs[0] #range [1E-4, 1E-3]
    tp = perturbs[1] #range [1E1, 1E2]
    #Unperturbed parameters
    ki = 5E-5 #range [1E-5, 1E-4]
    tl = 50 #range [1E1, 1E2]
    toxicity_factor = 1/(1 + (paca_int/(ki/ta) + eff/(ki/tp) + laao/(ki/tl)))
    
    pap_mrnalength = 3400
    eff_mrnalength = 2900
    laao_mrnalength = 1600
    ribosome_elongation = 20
    tsn_init = 2E-1

    #Enzyme kinetic parameters
    enzyme_kcat = 5E0 #range [5E-1, 5E1]
    enzyme_km = 1E-6 #range [1E-7, 1E-5]
    papA_kcat = 0.2975
    papA_km = 0.056
    papB_kcat = 39
    papB_km = 0.38 
    papC_kcat = 20.44
    papC_km = 0.555
    laao_kcat = 1.29
    laao_km = 10.82
    deaminase_kcat = enzyme_kcat
    deaminase_km = enzyme_km
    efflux_rate = 275. #range [5E1, 5E2]

    #Kinetic pathway
    chorismate_biosynthesis = chorismate_production_rate * toxicity_factor
    deaminase_biosynthesis = deaminase_production_rate * toxicity_factor
    papA_catalyzed_biosynthesis = papA_kcat * papA * ((chorismate / avogadro) / cell_volume) / (papA_km + ((chorismate / avogadro) / cell_volume)) * toxicity_factor
    papB_catalyzed_biosynthesis = papB_kcat * ((pa1 / avogadro) / cell_volume) / (papB_km + ((pa1 / avogadro) / cell_volume)) * papB * toxicity_factor
    papC_catalyzed_biosynthesis = papC_kcat * papC * ((pa2 / avogadro) / cell_volume) / (papC_km + ((pa2 / avogadro) / cell_volume)) * toxicity_factor
    deaminase_catalyzed_biosynthesis = deaminase_kcat * deaminase * ((pa3 / avogadro) / cell_volume) / (deaminase_km + ((pa3 / avogadro) / cell_volume)) * toxicity_factor
    laao_catalyzed_biosynthesis = laao_kcat * laao * ((paf / avogadro) / cell_volume) / (laao_km + ((paf / avogadro) / cell_volume)) * toxicity_factor
    paca_external_efflux = eff * ((paca_int / avogadro) / cell_volume) * efflux_rate * toxicity_factor

    papA_mrna_txn = np.sum(architecture[0]*np.array([activation(paf, k_paf_papA, theta_paf_prom1, n), activation(paca_int, k_paca_papA, theta_paca_prom1, n), repression(paf, k_paf_papA, theta_paf_prom1, n), repression(paca_int, k_paca_papA, theta_paca_prom1, n), k_paf_papA]))
    papB_mrna_txn = np.sum(architecture[0]*np.array([activation(paf, k_paf_papB, theta_paf_prom1, n), activation(paca_int, k_paca_papB, theta_paca_prom1, n), repression(paf, k_paf_papB, theta_paf_prom1, n), repression(paca_int, k_paca_papB, theta_paca_prom1, n), k_paf_papB]))
    papC_mrna_txn = np.sum(architecture[0]*np.array([activation(paf, k_paf_papC, theta_paf_prom1, n), activation(paca_int, k_paca_papC, theta_paca_prom1, n), repression(paf, k_paf_papC, theta_paf_prom1, n), repression(paca_int, k_paca_papC, theta_paca_prom1, n), k_paf_papC]))
    laao_mrna_txn = np.sum(architecture[1]*np.array([activation(paf, k_paf_prom2, theta_paf_prom2, n), activation(paca_int, k_paca_prom2, theta_paca_prom2, n), repression(paf, k_paf_prom2, theta_paf_prom2, n), repression(paca_int, k_paca_prom2, theta_paca_prom2, n), k_paf_prom2]))
    eff_mrna_txn = np.sum(architecture[2]*np.array([activation(paf, k_paf_prom3, theta_paf_prom3, n), activation(paca_int, k_paca_prom3, theta_paca_prom3, n), repression(paf, k_paf_prom3, theta_paf_prom3, n), repression(paca_int, k_paca_prom3, theta_paca_prom3, n), k_paf_prom3]))

    papA_translation_rate = ((papA_mrna)/(tsn_init + (pap_mrnalength/ribosome_elongation)))
    papB_translation_rate = ((papB_mrna)/(tsn_init + (pap_mrnalength/ribosome_elongation)))
    papC_translation_rate = ((papC_mrna)/(tsn_init + (pap_mrnalength/ribosome_elongation)))
    laao_translation_rate = ((laao_mrna)/(tsn_init + (laao_mrnalength/ribosome_elongation))) 
    eff_translation_rate = ((eff_mrna)/(tsn_init + (eff_mrnalength/ribosome_elongation))) 

    ydot[0] = chorismate_biosynthesis - papA_catalyzed_biosynthesis - chorismate*dilution_rate #chorismate
    ydot[1] = papA_catalyzed_biosynthesis - papB_catalyzed_biosynthesis - pa1*dilution_rate #pa1
    ydot[2] = papB_catalyzed_biosynthesis - papC_catalyzed_biosynthesis - pa2*dilution_rate #pa2
    ydot[3] = papC_catalyzed_biosynthesis - deaminase_catalyzed_biosynthesis - pa3*dilution_rate #pa3
    ydot[4] = deaminase_catalyzed_biosynthesis -  laao_catalyzed_biosynthesis - paf*dilution_rate - paf*1.40E-5 #range [1.4E-6, 1.4E-4] #paf
    ydot[5] = laao_catalyzed_biosynthesis - paca_external_efflux - paca_int*dilution_rate #paca_int 
    ydot[6] = paca_external_efflux #paca_ext
    ydot[7] = promoter1*dna_duplication_rate - promoter1*dilution_rate #promoter1
    ydot[8] = papA_mrna_txn - papA_mrna * dilution_rate - papA_mrna * mrna_degradation_rate * toxicity_factor #papA_mrna
    ydot[9] = papA_translation_rate * toxicity_factor - papA_uf * protein_folding_rate * toxicity_factor - papA_uf * dilution_rate - papA_uf * protein_degradation_rate  * toxicity_factor #papA_uf 
    ydot[10] = papA_uf * protein_folding_rate * toxicity_factor - papA * dilution_rate - papA*protein_degradation_rate*toxicity_factor  #papA 
    ydot[11] = papB_mrna_txn - papB_mrna * dilution_rate - papB_mrna * mrna_degradation_rate * toxicity_factor #papB_mrna
    ydot[12] = papB_translation_rate * toxicity_factor - papB_uf * protein_folding_rate * toxicity_factor - papB_uf * dilution_rate - papB_uf * protein_degradation_rate  * toxicity_factor #papB_uf
    ydot[13] = papB_uf * protein_folding_rate * toxicity_factor - papB * dilution_rate - papB*protein_degradation_rate*toxicity_factor  #papB
    ydot[14] = papC_mrna_txn - papC_mrna * dilution_rate - papC_mrna * mrna_degradation_rate * toxicity_factor  #papC_mrna
    ydot[15] = papC_translation_rate * toxicity_factor - papC_uf * protein_folding_rate * toxicity_factor - papC_uf * dilution_rate - papC_uf * protein_degradation_rate  * toxicity_factor #papC_uf
    ydot[16] = papC_uf * protein_folding_rate * toxicity_factor - papC * dilution_rate - papC*protein_degradation_rate*toxicity_factor #papC
    ydot[17] = deaminase_biosynthesis - deaminase * dilution_rate #deaminase
    ydot[18] = promoter2*dna_duplication_rate - promoter2*dilution_rate #promoter2
    ydot[19] = laao_mrna_txn - laao_mrna * dilution_rate - laao_mrna * mrna_degradation_rate * toxicity_factor  #laao_mrna
    ydot[20] = laao_translation_rate * toxicity_factor - laao_uf * protein_folding_rate * toxicity_factor - laao_uf * dilution_rate - laao_uf * protein_degradation_rate  * toxicity_factor #laao_uf
    ydot[21] = laao_uf * protein_folding_rate * toxicity_factor - laao * dilution_rate - laao*protein_degradation_rate*toxicity_factor  #laao
    ydot[22] = promoter3*dna_duplication_rate - promoter3*dilution_rate #promoter3
    ydot[23] = eff_mrna_txn - eff_mrna * dilution_rate - eff_mrna * mrna_degradation_rate * toxicity_factor  #eff_mrna
    ydot[24] = eff_translation_rate * toxicity_factor - eff_uf * protein_folding_rate * toxicity_factor - eff_uf * dilution_rate - eff_uf * protein_degradation_rate  * toxicity_factor #eff_uf
    ydot[25] = eff_uf * protein_folding_rate * toxicity_factor - eff * dilution_rate - eff*protein_degradation_rate*toxicity_factor  #eff

    #J1 and J2
    ydot[26] = (chorismate_biosynthesis - paca_external_efflux)**2
    ydot[27] = papA_mrna_txn + papB_mrna_txn + papC_mrna_txn + laao_mrna_txn + eff_mrna_txn