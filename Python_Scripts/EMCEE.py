#########################################################################
#    use this link https://emcee.readthedocs.io/en/stable/tutorials/line/
#########################################################################
#%%
import pandas as pd
import numpy as np
import emcee
import corner
import matplotlib.pyplot as plt

from Load_Data import load_catalogs
catalogs = load_catalogs()
cat_fortin = catalogs['cat_fortin']

filtered_df = cat_fortin.dropna(subset=['RV', 'Period', 'Mo', 'Mx'])

G = 6.67430e-11            # m^3 kg^-1 s^-2
Msun = 1.98847e30          # kg
day = 86400                # s

K_err_kms = 3  

priors = [
    (1.0, 10.0),
    # (2.0, 15.0),
    # (5.0, 20.0)
]

def mass_function(M2, M1_obs, inclination_rad):
    sin_i = np.sin(inclination_rad)
    numerator = (M2 * sin_i)**3
    denominator = (M1_obs + M2)**2
    return numerator / denominator

def log_likelihood(theta, K_obs, P_obs, M1_obs, K_err):
    M2, cos_i = theta
    if M2 <= 0 or not (0 < cos_i < 1):
        return -np.inf
    i_rad = np.arccos(cos_i)
    f_M = mass_function(M2 * Msun, M1_obs * Msun, i_rad)
    K_pred = ((2 * np.pi * G * f_M) / P_obs)**(1/3)
    return -0.5 * ((K_obs - K_pred)**2 / K_err**2 + np.log(2 * np.pi * K_err**2))

def log_prior(theta, M2_min, M2_max):
    M2, cos_i = theta
    if M2_min < M2 < M2_max and 0 < cos_i < 1:
        return 0.0
    return -np.inf

def log_posterior(theta, K_obs, P_obs, M1_obs, K_err, M2_min, M2_max):
    lp = log_prior(theta, M2_min, M2_max)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, K_obs, P_obs, M1_obs, K_err)

def compute_semi_major_axis(M2_array, M1_obs, P_obs):
    total_mass = (M1_obs + M2_array) * Msun
    a_cubed = (G * total_mass * (P_obs)**2) / (4 * np.pi**2)
    return (a_cubed**(1/3)) / (6.957e8)  

for prior_index, (M2_min, M2_max) in enumerate(priors):
    print(f"\n=== Prior {prior_index+1}: M2 ∈ [{M2_min}, {M2_max}] M_sun ===")

    results = []

    for index, row in filtered_df.iterrows():
        P_obs_days = row['Period']
        K_obs_kms = row['RV']
        M1_obs_Msun = row['Mo']
        Mx_true = row['Mx']

        P_obs = P_obs_days * day
        K_obs = K_obs_kms * 1000
        K_err = K_err_kms * 1000
        M1_obs = M1_obs_Msun

        ndim = 2
        nwalkers = 50
        nsteps = 3000
        initial_guess = [np.mean([M2_min, M2_max]), 0.5]
        pos = initial_guess + 1e-4 * np.random.randn(nwalkers, ndim)

        sampler = emcee.EnsembleSampler(
            nwalkers, ndim, log_posterior,
            args=(K_obs, P_obs, M1_obs, K_err, M2_min, M2_max)
        )
        sampler.run_mcmc(pos, nsteps, progress=False)

        samples = sampler.get_chain(discard=1000, thin=10, flat=True)
        M2_samples = samples[:, 0]
        cos_i_samples = samples[:, 1]
        i_samples_deg = np.degrees(np.arccos(cos_i_samples))

        a_samples_Rsun = compute_semi_major_axis(M2_samples, M1_obs, P_obs)

        M2_median = np.median(M2_samples)
        i_median = np.median(i_samples_deg)
        a_median = np.median(a_samples_Rsun)

        rel_diff = np.abs(M2_median - Mx_true) / Mx_true

        results.append({
            'System_Index': index,
            'M2_median_Msun': M2_median,
            'i_median_deg': i_median,
            'a_median_Rsun': a_median,
            'Mx': Mx_true,
            'Rel_Diff_M2_vs_Mx': rel_diff
        })

    results_df = pd.DataFrame(results)
    results_df.set_index('System_Index', inplace=True)
    print(results_df[['M2_median_Msun', 'i_median_deg', 'a_median_Rsun', 'Mx', 'Rel_Diff_M2_vs_Mx']])













#%%

#########################################################################
#                           TRYING DIFFERENT COMBINATION OF PRIORS
#########################################################################
import numpy as np
import pandas as pd
import emcee

G = 6.67430e-11  # m^3 kg^-1 s^-2
Msun = 1.98847e30
Rsun = 6.957e8
day = 86400

mass_priors = {
    'uniform': lambda M2: 0.0 if 0.1 < M2 < 50 else -np.inf,
    'log_uniform': lambda M2: -np.log(M2) if 0.1 < M2 < 50 else -np.inf,
    'gaussian_centered_10': lambda M2: -0.5*((M2-10)/2)**2 if 0.1 < M2 < 50 else -np.inf,
    'gaussian_ns': lambda M2: -0.5*((M2-1.4)/0.3)**2 if 0.1 < M2 < 50 else -np.inf,
    'gaussian_bh': lambda M2: -0.5*((M2-8)/2)**2 if 0.1 < M2 < 50 else -np.inf,
    'salpeter': lambda M2: -2.35*np.log(M2) if 0.1 < M2 < 50 else -np.inf,
    'log_uniform_soft_cut': lambda M2: -np.log(M2 + 0.5) if 0.1 < M2 < 50 else -np.inf,
    'mixture_ns_bh': lambda M2: np.log(
        0.5 * np.exp(-0.5*((M2-1.4)/0.3)**2) +
        0.5 * np.exp(-0.5*((M2-8)/2)**2)
    ) if 0.1 < M2 < 50 else -np.inf
}

inclination_priors = {
    'isotropic': lambda cos_i: np.log(np.sin(np.arccos(cos_i)) + 1e-10) if 0 < cos_i < 1 else -np.inf,
    'uniform_cos': lambda cos_i: 0.0 if 0 < cos_i < 1 else -np.inf,
}
a_priors = {
    'uniform': lambda a: 0.0 if 1 < a < 1000 else -np.inf,
    'log_uniform': lambda a: -np.log(a) if 1 < a < 1000 else -np.inf,
}

results_dict = {}
print(len(filtered_df))
for m_name, mass_prior in mass_priors.items():
    for i_name, inc_prior in inclination_priors.items():
        for a_name, a_prior in a_priors.items():
            label = f'{m_name}_{i_name}_{a_name}'
            results = []

            for _, row in filtered_df.iterrows():
                try:
                    Mx = row['Mo']
                    P = row['Period']
                    K = row['RV']
                    e = row.get('Eccentricity', 0.0)

                    P_sec = P * day
                    K_m_s = K * 1000
                    Mx_kg = Mx * Msun

                    def log_prior(theta):
                        M2, cos_i, a = theta
                        if not (0 < cos_i < 1):
                            return -np.inf
                        if not (0.1 < M2 < 50):
                            return -np.inf
                        if not (1 < a < 1000):
                            return -np.inf

                        mass_val = mass_prior(M2)
                        inc_val = inc_prior(cos_i)
                        a_val = a_prior(a)

                        if not (np.isfinite(mass_val) and np.isfinite(inc_val) and np.isfinite(a_val)):
                            return -np.inf

                        total = mass_val + inc_val + a_val
                        if not np.isfinite(total):
                            return -np.inf

                        return total


                    def log_likelihood(theta):
                        M2, cos_i, a = theta
                        if not (0 < cos_i < 1):
                            return -np.inf
                        if M2 <= 0 or a <= 0:
                            return -np.inf

                        i = np.arccos(cos_i)
                        denom = (Mx_kg + M2 * Msun)**2
                        if denom == 0:
                            return -np.inf

                        f_M = (M2 * Msun * np.sin(i))**3 / denom
                        if f_M < 0:
                            return -np.inf

                        K_model = (2 * np.pi * G * f_M / P_sec)**(1/3)
                        if np.isnan(K_model) or np.isinf(K_model):
                            return -np.inf

                        return -0.5 * ((K_m_s - K_model)**2 / (3000)**2)


                    
                    
                    def log_posterior(theta):
                        lp = log_prior(theta)
                        return lp + log_likelihood(theta) if np.isfinite(lp) else -np.inf

                    ndim, nwalkers = 3, 30
                    p0 = []
                    for _ in range(nwalkers):
                        M2 = np.random.uniform(1, 10)  
                        cos_i = np.random.uniform(0.01, 0.99)  
                        a = np.random.uniform(5, 100) 
                        p0.append(np.array([M2, cos_i, a]))


                    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior)
                    sampler.run_mcmc(p0, 1000, progress=False)

                    samples = sampler.get_chain(discard=200, thin=10, flat=True)
                    M2_med = np.median(samples[:, 0])
                    i_med = np.degrees(np.arccos(np.median(samples[:, 1])))
                    a_med = np.median(samples[:, 2])
                    rel_diff = (M2_med - Mx) / Mx

                    results.append({
                        'Name': row['Main_ID'],
                        'Mx': Mx,
                        'M2_median_Msun': M2_med,
                        'i_median_deg': i_med,
                        'a_median_Rsun': a_med,
                        'Rel_Diff': rel_diff
                    })

                except Exception as e:
                    print(f"Error en fila {row['Main_ID'] if 'Main_ID' in row else 'desconocido'}: {e}")
                    continue

            
            df = pd.DataFrame(results)
            print("Columns in df:", df.columns)
            print("Length of results:", len(results))

            df["Rel_Diff"] = np.abs(df["M2_median_Msun"] - df["Mx"]) / df["Mx"]
            results_dict[label] = df
            print(f'\nPriors: {label}, Media ΔRelativa = {df["Rel_Diff"].mean():.3f}')
#%%
results_dict['gaussian_centered_10_isotropic_uniform']


#%%
#########################################################################
#                           Now for a cuadratic function
#########################################################################import numpy as np
import numpy as np
import matplotlib.pyplot as plt
import emcee
import corner

grado = 8
N = 2
lim_sup = 10
lim_inf = -10
lim_inf_theta = -10
lim_sup_theta = 10

ndim = grado + 1 
np.random.seed(42)
theta_true = np.random.uniform(lim_inf_theta, lim_sup_theta, size=ndim)

x = np.sort(np.random.uniform(lim_inf, lim_sup, N))
yerr = 1.0
y = np.polyval(theta_true, x) + np.random.normal(0, yerr, size=N)

def log_likelihood(theta, x, y, yerr):
    model = np.polyval(theta, x)
    sigma2 = yerr ** 2
    return -0.5 * np.sum((y - model) ** 2 / sigma2 + np.log(2 * np.pi * sigma2))

def log_prior(theta):
    if np.all((lim_inf_theta < theta) & (theta < lim_sup_theta)):
        return 0.0
    return -np.inf

def log_posterior(theta, x, y, yerr):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, x, y, yerr)

nwalkers = 2 * ndim + 4
initial_pos = theta_true + 1e-4 * np.random.randn(nwalkers, ndim)
sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior, args=(x, y, yerr))

sampler.run_mcmc(initial_pos, 3000, progress=True)

flat_samples = sampler.get_chain(discard=1000, thin=15, flat=True)
labels = [f"$\\theta_{i}$" for i in range(ndim)]

fig = corner.corner(flat_samples, labels=labels, truths=theta_true)
plt.show()

x0 = np.linspace(lim_inf, lim_sup, 200)
plt.errorbar(x, y, yerr=yerr, fmt=".k", capsize=0, label="datos")
inds = np.random.randint(len(flat_samples), size=100)
for ind in inds:
    sample = flat_samples[ind]
    y_sample = np.polyval(sample, x0)
    plt.plot(x0, y_sample, color="C1", alpha=0.1)

plt.plot(x0, np.polyval(theta_true, x0), "k", lw=2, label="verdadero")
plt.legend()
plt.xlabel("x")
plt.ylabel("y")
plt.title(f"Ajuste polinomial de grado {grado}")
plt.show()
