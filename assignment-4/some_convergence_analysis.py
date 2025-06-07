import numpy as np
import matplotlib.pyplot as plt
from simulator import MM1QueueSimulator, confidence_interval

def analyze_convergence(lambda_mu_pairs, simulation_time=5000, replications=100):
    results = []
    
    for arrival_rate, service_rate in lambda_mu_pairs:
        rho = arrival_rate / service_rate
        print(f"Analyzing λ={arrival_rate}, μ={service_rate}, ρ={rho:.3f}\n")
                    
        theoretical_avg = rho / (1 - rho)
        
        #eseguiamo le independent replications
        empirical_averages = []
        
        for _ in range(replications):
            simulator = MM1QueueSimulator(arrival_rate, service_rate, simulation_time)
            simulator.simulate()
            
            empirical_avg = simulator.compute_average()
            empirical_averages.append(empirical_avg)
                    
        emp_mean = np.mean(empirical_averages)
        emp_std = np.std(empirical_averages)
        
        #relative errors and confidence interval
        relative_error = abs(emp_mean - theoretical_avg) / theoretical_avg * 100
        
        ci_lower, ci_upper = confidence_interval(empirical_averages)
        
        theoretical_in_ci = ci_lower <= theoretical_avg <= ci_upper
        
        results.append({
            'lambda': arrival_rate,
            'mu': service_rate,
            'rho': rho,
            'theoretical': theoretical_avg,
            'empirical_mean': emp_mean,
            'empirical_std': emp_std,
            'relative_error': relative_error,
            'ci_lower': float(ci_lower),
            'ci_upper': float(ci_upper),
            'theoretical_in_ci': theoretical_in_ci,
        })
        
        print(f"Theoretical value: {theoretical_avg:.3f}")
        print(f"Empirical value: {emp_mean:.3f} (standard deviation: {emp_std:.3f}) (errore relativo: {relative_error:.2f}%)")
        print(f"Confidence interval: [{float(ci_lower), float(ci_upper)}] (theoretical value inside CI: {theoretical_in_ci})\n")
    
    return results

def plot_convergence_analysis(results):
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Estrai dati per i grafici
    rhos = [r['rho'] for r in results]
    theoretical = [r['theoretical'] for r in results]
    empirical = [r['empirical_mean'] for r in results]
    #empirical_warmup = [r['empirical_mean_warmup'] for r in results]
    relative_errors = [r['relative_error'] for r in results]
    #relative_errors_warmup = [r['relative_error_warmup'] for r in results]
    empirical_stds = [r['empirical_std'] for r in results]
    #empirical_stds_warmup = [r['empirical_std_warmup'] for r in results]
    
    # Grafico 1: Valori teorici vs empirici
    axes[0, 0].scatter(theoretical, empirical, alpha=0.7, label='Senza warmup', color='blue')
    #axes[0, 0].scatter(theoretical, empirical_warmup, alpha=0.7, label='Con warmup', color='red')
    axes[0, 0].plot([0, max(theoretical)], [0, max(theoretical)], 'k--', alpha=0.5, label='Linea ideale')
    axes[0, 0].set_xlabel('Valore Teorico')
    axes[0, 0].set_ylabel('Valore Empirico')
    axes[0, 0].set_title('Convergenza: Teorico vs Empirico')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Grafico 2: Errore relativo vs ρ
    axes[0, 1].scatter(rhos, relative_errors, alpha=0.7, label='Senza warmup', color='blue')
    #axes[0, 1].scatter(rhos, relative_errors_warmup, alpha=0.7, label='Con warmup', color='red')
    axes[0, 1].set_xlabel('ρ (Utilizzazione)')
    axes[0, 1].set_ylabel('Errore Relativo (%)')
    axes[0, 1].set_title('Errore Relativo vs Utilizzazione')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Grafico 3: Deviazione standard vs ρ
    axes[1, 0].scatter(rhos, empirical_stds, alpha=0.7, label='Senza warmup', color='blue')
    #axes[1, 0].scatter(rhos, empirical_stds_warmup, alpha=0.7, label='Con warmup', color='red')
    axes[1, 0].set_xlabel('ρ (Utilizzazione)')
    axes[1, 0].set_ylabel('Deviazione Standard')
    axes[1, 0].set_title('Variabilità vs Utilizzazione')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Grafico 4: Intervalli di confidenza
    x_pos = range(len(results))
    axes[1, 1].errorbar(x_pos, empirical, 
                       yerr=[[emp - ci_low for emp, ci_low in zip(empirical, [r['ci_lower'] for r in results])],
                             [ci_up - emp for emp, ci_up in zip(empirical, [r['ci_upper'] for r in results])]], 
                       fmt='o', alpha=0.7, label='Senza warmup', color='blue', capsize=5)
    #axes[1, 1].errorbar([x + 0.1 for x in x_pos], empirical_warmup, 
                       #yerr=[[emp - ci_low for emp, ci_low in zip(empirical_warmup, [r['ci_lower_warmup'] for r in results])],
                        #     [ci_up - emp for emp, ci_up in zip(empirical_warmup, [r['ci_upper_warmup'] for r in results])]], 
                       #fmt='s', alpha=0.7, label='Con warmup', color='red', capsize=5)
    axes[1, 1].scatter(x_pos, theoretical, color='green', marker='*', s=100, label='Teorico', zorder=5)
    axes[1, 1].set_xlabel('Configurazione')
    axes[1, 1].set_ylabel('Numero medio di pacchetti')
    axes[1, 1].set_title('Intervalli di Confidenza')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # Aggiungi etichette per le configurazioni
    labels = [f'λ={r["lambda"]}, μ={r["mu"]}\nρ={r["rho"]:.2f}' for r in results]
    axes[1, 1].set_xticks(x_pos)
    axes[1, 1].set_xticklabels(labels, rotation=45, ha='right')
    
    plt.tight_layout()
    plt.show()

def print_summary_table(results):
    print("\n" + "="*120)
    print("Summary table")
    print("="*120)
    print(f"{'λ':>6} {'μ':>6} {'ρ':>8} {'Teorico':>10} {'Empirico':>12} {'Err%':>8} {'CI':>15} {'IC_OK':>8}")
    print("-"*120)
    
    for r in results:
        print(f"{r['lambda']:>6.1f} {r['mu']:>6.1f} {r['rho']:>8.3f} {r['theoretical']:>10.3f} "
              f"{r['empirical_mean']:>12.3f} {r['relative_error']:>8.2f}  "
              f"[{r['ci_lower']:.3f},{r['ci_upper']:.3f}]{' '*(15-len(f'[{r['ci_lower']:.3f},{r['ci_upper']:.3f}]'))}"
              f"{str(r['theoretical_in_ci']):>8}"
            )

if __name__ == "__main__":

    lambda_mu_pairs = [
        #bassa utilizzazione
        (0.5, 2.0),   # ρ = 0.25
        (1.0, 2.0),   # ρ = 0.50
        (1.0, 1.5),   # ρ = 0.67
        
        #media utilizzazione
        (1.5, 2.0),   # ρ = 0.75
        (1.8, 2.0),   # ρ = 0.90
        
        #alta utilizzazione (vicino all'instabilità)
        (1.9, 2.0),   # ρ = 0.95
        (1.95, 2.0),  # ρ = 0.975
        (1.99, 2.0),  # ρ = 0.995
        
        #diversi valori assoluti con stesso ρ
        (2.0, 4.0),   #ρ = 0.50
        (3.0, 4.0),   #ρ = 0.75
        (1.5, 6.0),   #ρ = 0.25
        (4.75, 5.0)   #ρ = 0.95

    ]
    
    print("DIFFERENT COMBINATION OF λ AND μ")
    print("="*60)
    
    #analizziamo i vari valori
    results = analyze_convergence(lambda_mu_pairs, simulation_time=500, replications=200)
    
    print_summary_table(results)
    
    #grafici
    plot_convergence_analysis(results)

    exit(0)
    
    # Analisi e discussione
    print("\n" + "="*60)
    print("DISCUSSIONE DEI RISULTATI")
    print("="*60)
    
    print("\n1. EFFETTO DELL'UTILIZZAZIONE (ρ):")
    high_rho_results = [r for r in results if r['rho'] > 0.9]
    low_rho_results = [r for r in results if r['rho'] < 0.7]
    
    if high_rho_results and low_rho_results:
        avg_error_high = np.mean([r['relative_error'] for r in high_rho_results])
        avg_error_low = np.mean([r['relative_error'] for r in low_rho_results])
        avg_std_high = np.mean([r['empirical_std'] for r in high_rho_results])
        avg_std_low = np.mean([r['empirical_std'] for r in low_rho_results])
        
        print(f"   - Alta utilizzazione (ρ > 0.9): errore medio {avg_error_high:.2f}%, std media {avg_std_high:.3f}")
        print(f"   - Bassa utilizzazione (ρ < 0.7): errore medio {avg_error_low:.2f}%, std media {avg_std_low:.3f}")
    
    print("\n2. EFFETTO DEL WARMUP:")
    for r in results:
        improvement = r['relative_error'] - r['relative_error_warmup']
        if improvement > 0.5:  # Miglioramento significativo
            print(f"   - λ={r['lambda']}, μ={r['mu']}: miglioramento di {improvement:.2f}% con warmup")
    
    print("\n3. AFFIDABILITÀ DEGLI INTERVALLI DI CONFIDENZA:")
    ci_success_normal = sum(1 for r in results if r['theoretical_in_ci'])
    ci_success_warmup = sum(1 for r in results if r['theoretical_in_ci_warmup'])
    print(f"   - Senza warmup: {ci_success_normal}/{len(results)} IC contengono il valore teorico")
    print(f"   - Con warmup: {ci_success_warmup}/{len(results)} IC contengono il valore teorico")