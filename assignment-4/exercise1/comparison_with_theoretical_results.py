import numpy as np
import matplotlib.pyplot as plt
from simulator import MM1QueueSimulator, confidence_interval
import argparse


#parameters
arrival_rate = 1.0  #λ
service_rate = 2.0  #μ
simulation_time = 700.0  #seconds
replications = 50
warmup = simulation_time / 3

parser = argparse.ArgumentParser(description="M/M/1 Queue Simulation")
parser.add_argument("--arrival_rate", type=float, default=arrival_rate, help=f"Arrival rate (λ, default {arrival_rate})")
parser.add_argument("--service_rate", type=float, default=service_rate, help=f"Service rate (μ, default {service_rate})")
parser.add_argument("--simulation_time", type=float, default=simulation_time, help=f"Time of the simulation (default {simulation_time})")
parser.add_argument("--warmup", type=float, default=warmup, help=f"Time of the simulation (default simulation_time / 3)")
parser.add_argument("--replications", type=int, default=replications, help=f"Number of independent replications we do (default {replications})")
args = parser.parse_args()

arrival_rate = args.arrival_rate
service_rate = args.service_rate
if(service_rate <= arrival_rate):
    print("Cannot continue: system is not stable! Must have μ > λ!")
    exit(0)
simulation_time = args.simulation_time
replications = args.replications
warmup = simulation_time / 3
warmup = args.warmup
print(f"You are running the simulator with an arrival_rate of {arrival_rate}, a service_rate of {service_rate} and a simulation_time of {simulation_time} (warmup {warmup}), for {replications} independent replications\n")

############################SINGULAR CASE: very noise results############################
simulator = MM1QueueSimulator(arrival_rate, service_rate, simulation_time)
simulator.simulate()

rho = arrival_rate / service_rate
theoretical_avg = rho / (1 - rho)
empirical_average = simulator.average_packets_in_system()

print(f"\nTheoretical average number of packets in system: {theoretical_avg}\n")

print(f"Empirical average number of packets in system: {empirical_average}")
#simulator.plot_cumulative_packets_in_system()
#simulator.plot_packets_in_system()

empirical_average = simulator.average_packets_in_system(warmup)
print(f"[Warmup {warmup}] Empirical average number of packets in system: {empirical_average}")
#simulator.plot_cumulative_packets_in_system(warmup)
#simulator.plot_packets_in_system(warmup)

############################APPLYING ADDITIONAL OUTPUT ANALYSIS: INDEPENDENT REPLICATIONS############################

#We do independent replications
empirical_averages = []
empirical_averages_warmupped = []

all_times = []
all_cumulative_means = []
all_times_now = []
all_cumulative_means_now = []

for j in range(replications):
    simulator = MM1QueueSimulator(arrival_rate, service_rate, simulation_time)
    simulator.simulate()

    empirical_avg = simulator.average_packets_in_system() #average #packets in the system
    empirical_averages.append(empirical_avg)
    empirical_avg = simulator.average_packets_in_system(warmup) #average #packets in the system
    empirical_averages_warmupped.append(empirical_avg)

    if warmup != 0.0:
        filtered_data = [(t, count) for t, count in simulator.packets_in_system if t >= warmup]
        
        if not filtered_data:
            continue  # salta se non ci sono dati dopo warmup
            
        # Calcola tempi relativi (partendo da 0 dopo warmup)
        start_time = filtered_data[0][0]
        times, packets = zip(*[(t - start_time, count) for (_, (t, count)) in enumerate(filtered_data, 1)])

    times_now, packets_now = zip(*simulator.packets_in_system)

    times_now = np.array(times_now)
    times = np.array(times)

    if len(times) > 1:
        durations = np.diff(times, append=times[-1])
        cumulative_area = np.cumsum(durations * np.array(packets))
        cumulative_mean = np.full_like(times, np.nan, dtype=float)
        cumulative_mean[1:] = cumulative_area[1:] / times[1:]    
    
    if len(times_now) > 0:
        durations_now = np.diff(times_now, append=times_now[-1])  # differenze con primo elemento
        
        # Calcolo area cumulativa
        cumulative_area_now = np.cumsum(durations_now * np.array(packets_now))
        
        # Media cumulativa (tempo relativo al warmup)
        cumulative_mean_now = np.full_like(times_now, np.nan, dtype=float)
        cumulative_mean_now = cumulative_area_now / times_now


    all_times.append(times)
    all_cumulative_means.append(cumulative_mean)

    all_times_now.append(times_now)
    all_cumulative_means_now.append(cumulative_mean_now)


# Calcolo tempo massimo tra tutte le repliche (dopo warmup)
max_time = max([t[-1] if len(t) > 0 else 0 for t in all_times_now])
time_grid = np.linspace(0, max_time, num=500)

interpolated_curves = []
for times, means in zip(all_times, all_cumulative_means):
    interp_curve = np.interp(time_grid, times, means)
    interpolated_curves.append(interp_curve)

interpolated_curves = np.array(interpolated_curves)
mean_curve = np.mean(interpolated_curves, axis=0)
warmup_std = np.nanstd(interpolated_curves)

interpolated_curves_now = []
for times, means in zip(all_times_now, all_cumulative_means_now):
    interp_curve = np.interp(time_grid, times, means)
    interpolated_curves_now.append(interp_curve)

interpolated_curves_now = np.array(interpolated_curves_now)
mean_curve_now = np.mean(interpolated_curves_now, axis=0)
no_warmup_std = np.std(interpolated_curves_now)

plt.figure(figsize=(12, 6))
plt.plot(time_grid, mean_curve_now, label='Media cumulativa stimata - no warmup', color='red')
plt.plot(time_grid, mean_curve, label='Media cumulativa stimata', color='blue')
plt.axhline(y=arrival_rate / (service_rate - arrival_rate), linestyle='--', color='black', label='Teorico: ρ/(1−ρ)')
plt.xlabel("Tempo")
plt.ylabel("Numero medio cumulativo di pacchetti nel sistema")
plt.title("Media cumulativa pacchetti nel sistema (independent replications)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

rho = arrival_rate / service_rate
theoretical_avg = rho / (1 - rho)
print(f"\nTheoretical average number of packets in system: {theoretical_avg}\n")

print(f"Empirical average number of packets in system: {np.mean(empirical_averages)} (standard deviation {np.std(empirical_averages)})")
lower_ci, upper_ci = confidence_interval(empirical_averages)
print(f"Confidence Interval: [{lower_ci}, {upper_ci}]\n")

print(f"[Warmup {warmup}]Empirical average number of packets in system: {np.mean(empirical_averages_warmupped)} (standard deviation {np.std(empirical_averages_warmupped)})")
lower_ci, upper_ci = confidence_interval(empirical_averages_warmupped)
print(f"Confidence Interval: [{lower_ci}, {upper_ci}]\n")

print(f"About IR...\nNO WARMUP STD: {no_warmup_std}\nWARMUP STD: {warmup_std}")