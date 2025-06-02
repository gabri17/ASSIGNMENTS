import numpy as np
import matplotlib.pyplot as plt
from exercise1_try import MM1QueueSimulator, confidence_interval

def analyze_convergence(lambda_mu_pairs, simulation_time=5000, replications=100):
    """
    Analizza la convergenza per diverse combinazioni di λ e μ
    """
    results = []
    
    for arrival_rate, service_rate in lambda_mu_pairs:
        print(f"\nAnalizzando λ={arrival_rate}, μ={service_rate}")
        
        # Calcola il valore teorico
        rho = arrival_rate / service_rate
        if rho >= 1:
            print(f"Saltato: ρ={rho:.3f} >= 1 (sistema instabile)")
            continue
            
        theoretical_avg = rho / (1 - rho)
        
        # Esegui simulazioni
        empirical_averages = []
        empirical_averages_warmup = []
        
        for _ in range(replications):
            simulator = MM1QueueSimulator(arrival_rate, service_rate, simulation_time)
            simulator.simulate()
            
            empirical_avg = simulator.compute_average()
            empirical_averages.append(empirical_avg)
            
            empirical_avg_warmup = simulator.compute_average_with_warmup(warmup_time=0.5)
            empirical_averages_warmup.append(empirical_avg_warmup)
        
        # Calcola statistiche
        emp_mean = np.mean(empirical_averages)
        emp_std = np.std(empirical_averages)
        emp_mean_warmup = np.mean(empirical_averages_warmup)
        emp_std_warmup = np.std(empirical_averages_warmup)
        
        # Calcola errori relativi
        relative_error = abs(emp_mean - theoretical_avg) / theoretical_avg * 100
        relative_error_warmup = abs(emp_mean_warmup - theoretical_avg) / theoretical_avg * 100
        
        # Intervalli di confidenza
        ci_lower, ci_upper = confidence_interval(empirical_averages)
        ci_lower_warmup, ci_upper_warmup = confidence_interval(empirical_averages_warmup)
        
        # Controlla se il valore teorico è nell'intervallo di confidenza
        theoretical_in_ci = ci_lower <= theoretical_avg <= ci_upper
        theoretical_in_ci_warmup = ci_lower_warmup <= theoretical_avg <= ci_upper_warmup
        
        results.append({
            'lambda': arrival_rate,
            'mu': service_rate,
            'rho': rho,
            'theoretical': theoretical_avg,
            'empirical_mean': emp_mean,
            'empirical_std': emp_std,
            'empirical_mean_warmup': emp_mean_warmup,
            'empirical_std_warmup': emp_std_warmup,
            'relative_error': relative_error,
            'relative_error_warmup': relative_error_warmup,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'ci_lower_warmup': ci_lower_warmup,
            'ci_upper_warmup': ci_upper_warmup,
            'theoretical_in_ci': theoretical_in_ci,
            'theoretical_in_ci_warmup': theoretical_in_ci_warmup
        })
        
        print(f"ρ = {rho:.3f}")
        print(f"Teorico: {theoretical_avg:.3f}")
        print(f"Empirico: {emp_mean:.3f} ± {emp_std:.3f} (errore relativo: {relative_error:.2f}%)")
        print(f"Empirico (warmup): {emp_mean_warmup:.3f} ± {emp_std_warmup:.3f} (errore relativo: {relative_error_warmup:.2f}%)")
        print(f"Teorico nell'IC: {theoretical_in_ci} (normale), {theoretical_in_ci_warmup} (warmup)")
    
    return results

def plot_convergence_analysis(results):
    """
    Crea grafici per visualizzare l'analisi di convergenza
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Estrai dati per i grafici
    rhos = [r['rho'] for r in results]
    theoretical = [r['theoretical'] for r in results]
    empirical = [r['empirical_mean'] for r in results]
    empirical_warmup = [r['empirical_mean_warmup'] for r in results]
    relative_errors = [r['relative_error'] for r in results]
    relative_errors_warmup = [r['relative_error_warmup'] for r in results]
    empirical_stds = [r['empirical_std'] for r in results]
    empirical_stds_warmup = [r['empirical_std_warmup'] for r in results]
    
    # Grafico 1: Valori teorici vs empirici
    axes[0, 0].scatter(theoretical, empirical, alpha=0.7, label='Senza warmup', color='blue')
    axes[0, 0].scatter(theoretical, empirical_warmup, alpha=0.7, label='Con warmup', color='red')
    axes[0, 0].plot([0, max(theoretical)], [0, max(theoretical)], 'k--', alpha=0.5, label='Linea ideale')
    axes[0, 0].set_xlabel('Valore Teorico')
    axes[0, 0].set_ylabel('Valore Empirico')
    axes[0, 0].set_title('Convergenza: Teorico vs Empirico')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Grafico 2: Errore relativo vs ρ
    axes[0, 1].scatter(rhos, relative_errors, alpha=0.7, label='Senza warmup', color='blue')
    axes[0, 1].scatter(rhos, relative_errors_warmup, alpha=0.7, label='Con warmup', color='red')
    axes[0, 1].set_xlabel('ρ (Utilizzazione)')
    axes[0, 1].set_ylabel('Errore Relativo (%)')
    axes[0, 1].set_title('Errore Relativo vs Utilizzazione')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Grafico 3: Deviazione standard vs ρ
    axes[1, 0].scatter(rhos, empirical_stds, alpha=0.7, label='Senza warmup', color='blue')
    axes[1, 0].scatter(rhos, empirical_stds_warmup, alpha=0.7, label='Con warmup', color='red')
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
    axes[1, 1].errorbar([x + 0.1 for x in x_pos], empirical_warmup, 
                       yerr=[[emp - ci_low for emp, ci_low in zip(empirical_warmup, [r['ci_lower_warmup'] for r in results])],
                             [ci_up - emp for emp, ci_up in zip(empirical_warmup, [r['ci_upper_warmup'] for r in results])]], 
                       fmt='s', alpha=0.7, label='Con warmup', color='red', capsize=5)
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
    """
    Stampa una tabella riassuntiva dei risultati
    """
    print("\n" + "="*120)
    print("TABELLA RIASSUNTIVA DELL'ANALISI DI CONVERGENZA")
    print("="*120)
    print(f"{'λ':>6} {'μ':>6} {'ρ':>8} {'Teorico':>10} {'Empirico':>12} {'Err%':>8} {'Emp(W)':>12} {'Err%(W)':>8} {'IC_OK':>8} {'IC_OK(W)':>10}")
    print("-"*120)
    
    for r in results:
        print(f"{r['lambda']:>6.1f} {r['mu']:>6.1f} {r['rho']:>8.3f} {r['theoretical']:>10.3f} "
              f"{r['empirical_mean']:>12.3f} {r['relative_error']:>8.2f} {r['empirical_mean_warmup']:>12.3f} "
              f"{r['relative_error_warmup']:>8.2f} {str(r['theoretical_in_ci']):>8} {str(r['theoretical_in_ci_warmup']):>10}")

if __name__ == "__main__":
    # Definisci diverse combinazioni di λ e μ per testare l'effetto dell'utilizzazione
    lambda_mu_pairs = [
        # Bassa utilizzazione
        (0.5, 2.0),   # ρ = 0.25
        (1.0, 2.0),   # ρ = 0.50
        (1.0, 1.5),   # ρ = 0.67
        
        # Media utilizzazione
        (1.5, 2.0),   # ρ = 0.75
        (1.8, 2.0),   # ρ = 0.90
        
        # Alta utilizzazione (vicino all'instabilità)
        (1.9, 2.0),   # ρ = 0.95
        (1.95, 2.0),  # ρ = 0.975
        (1.99, 2.0),  # ρ = 0.995
        
        # Diversi valori assoluti con stesso ρ
        (2.0, 4.0),   # ρ = 0.50 (diversi da 1.0, 2.0)
        (3.0, 4.0),   # ρ = 0.75 (diversi da 1.5, 2.0)
    ]
    
    print("ANALISI DELLA CONVERGENZA PER DIVERSE COMBINAZIONI DI λ E μ")
    print("="*60)
    
    # Esegui l'analisi
    results = analyze_convergence(lambda_mu_pairs, simulation_time=10000, replications=200)
    
    # Stampa tabella riassuntiva
    print_summary_table(results)
    
    # Crea grafici
    plot_convergence_analysis(results)
    
    # Analisi e discussione
    print("\n" + "="*60)
    print("DISCUSSIONE DEI RISULTATI")
    print("="*60)
    
    print("\n1. EFFETTO DELL'UTILIZZAZIONE (ρ):")
    high_rho_results = [r for r in results if r['rho'] > 0.9]
    low_rho_results = [r for r in results if r['rho'] < 0.7]
    
    if high_rho_results and low_rho_results:
        avg_error_high = np.mean([r['relative_error_warmup'] for r in high_rho_results])
        avg_error_low = np.mean([r['relative_error_warmup'] for r in low_rho_results])
        avg_std_high = np.mean([r['empirical_std_warmup'] for r in high_rho_results])
        avg_std_low = np.mean([r['empirical_std_warmup'] for r in low_rho_results])
        
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