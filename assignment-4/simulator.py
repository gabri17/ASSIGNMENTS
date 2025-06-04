import heapq
import random
import matplotlib.pyplot as plt
import argparse
import numpy as np
from scipy.stats import t

"""
Class to characterize an event.

Attributes:
    time: at which time the event occurs
    event_type: 'arrival' or 'departure' (also 'end' to manage the end of a simulation)
"""
class Event:
    def __init__(self, time, event_type):
        self.time = time
        self.event_type = event_type

    def __lt__(self, other):
        return self.time < other.time

"""
Class to perform the simulation.

Attributes:
    arrival_rate: λ
    service_rate: μ
    simulation_time: for how long we make the simulation last
    
    event_queue: store the events scheduled (it's a priority queue where priority is based on time and we use a binary heap to manage it)
    current_time: current time of the simulation
    server_busy: True or False to determine whether is the server is busy or not
    queue_length: we save the number of packets in queue
    packets_in_system: packets in the system at a given time (used to compute average number of packets in the system)
    packets_in_queue: packets in the queue at a given time (used to compute average time in the queue)
    arrival_times_queue: it represents the state of the system (which packets there are in the queue)
    time_in_system: it stores, for each packet, how much time it stayed in the system and how many packets it has in front when entered the system
"""
class MM1QueueSimulator:
    
    def __init__(self, arrival_rate, service_rate, simulation_time):
        self.arrival_rate = arrival_rate
        self.service_rate = service_rate
        self.simulation_time = simulation_time
        self.event_queue = []
        self.current_time = 0
        self.server_busy = False
        self.queue_length = 0
        self.packets_in_system = []
        self.packets_in_queue = []

        self.arrival_times_queue = []
        self.time_in_system = []
    
    """
    Add an event in queue.
    """
    def schedule_event(self, event):
        heapq.heappush(self.event_queue, event)

    """
    Draw from an exponential
    """
    def generate_time(self, rate):
        return random.expovariate(rate)

    """
    Method to handle the simulaton
    """
    def simulate(self):
        #schedule the first arrival and simulation end
        self.schedule_event(Event(self.generate_time(self.arrival_rate), "arrival"))
        self.schedule_event(Event(self.simulation_time, "end"))

        #print(f"Simulation started at time {self.current_time}")

        while self.event_queue:

            event = heapq.heappop(self.event_queue)
            self.current_time = event.time

            if event.event_type == "arrival":
                #print(f"Arrival at time {self.current_time} [current in system {self.queue_length + int(self.server_busy)}]")
                self.handle_arrival()
            elif event.event_type == "departure":
                #print(f"Departure at time {self.current_time} [current in system {self.queue_length + int(self.server_busy)}]")
                self.handle_departure()
            elif event.event_type == "end":
                #print(f"Simulation ended at time {self.current_time} [current in system {self.queue_length + int(self.server_busy)}]. I still could need to handle some last departure events")
                break

            self.packets_in_system.append((self.current_time, self.queue_length + int(self.server_busy)))
            self.packets_in_queue.append((self.current_time, self.queue_length))

        #After the end event, we will have at most some departures
        #Process last departures in the queue
        while self.event_queue:

            event = heapq.heappop(self.event_queue)
            self.current_time = event.time

            if event.event_type == "arrival":
                print(f"SHOULD NOT HAPPEN")
                self.handle_arrival()
            elif event.event_type == "departure":
                #print(f"Departure at time {self.current_time} [current in system {self.queue_length + int(self.server_busy)}]")
                self.handle_departure()
            elif event.event_type == "end":
                print(f"SHOULD NOT HAPPEN")
            
            self.packets_in_system.append((self.current_time, self.queue_length + int(self.server_busy)))
            self.packets_in_queue.append((self.current_time, self.queue_length))

    def handle_arrival(self):

        actual_queue_length = self.queue_length
        in_server = 1 if self.server_busy else 0

        if self.server_busy:
            self.queue_length += 1
        else:
            self.server_busy = True
            self.schedule_event(Event(self.current_time + self.generate_time(self.service_rate), "departure"))
        
        #self.arrival_times_queue.append((self.current_time, actual_queue_length))
        self.arrival_times_queue.append((self.current_time, actual_queue_length + in_server))

        #schedule the next arrival - only if it can arrive until the end of the simualtion
        next_arrival = self.current_time + self.generate_time(self.arrival_rate)
        if next_arrival < self.simulation_time:
            self.schedule_event(Event(next_arrival, "arrival"))

    def handle_departure(self):

        #We take the arrival time of the actual packet we are processing in order to compute how much time it spent in the system
        #We store also how many packets it got in front when it arrived
        arrival_time, packets_in_front = self.arrival_times_queue.pop(0)
        time_in_system = self.current_time - arrival_time
        self.time_in_system.append((time_in_system, packets_in_front))

        if self.queue_length > 0:
            self.queue_length -= 1
            self.schedule_event(Event(self.current_time + self.generate_time(self.service_rate), "departure"))
        else:
            self.server_busy = False
        

    """
    We plot #packets in the system vs time
    """
    def plot_results(self):
        rho = self.arrival_rate / self.service_rate
        theoretical_value = rho / (1 - rho)
        empirical_value = self.compute_average()
        times, packets = zip(*self.packets_in_system)
        plt.plot(times, packets, drawstyle='steps-post', color="red")
        plt.xlabel("Time")
        plt.ylabel("Number of Packets in System")
        plt.title("Number of Packets in System Over Time")
        plt.text(
            0.02, 0.90,
            f"Theoretical value: {theoretical_value:.3f}\nEmpirical value: {empirical_value:.3f}",
            transform=plt.gca().transAxes,
            color='black',
            bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3')
        )
        plt.show()
    
    """
    Method to plot the cumulative average of time spent in the system after each packet.
    """
    def plot_cumulative_avg_times_in_sys(self, warmup=0):
        times = self.times_in_system()

        running_total = 0
        running_average = []
        running_average_no_warmup = []
        running_total_no_warmup = 0

        for i, t in enumerate(times):
            running_total_no_warmup += t
            running_average_no_warmup.append(running_total_no_warmup / (i+1))
            if (i >= len(times)*warmup):
                running_total += t
                running_average.append(running_total / (i-len(times)*warmup+1))
            else:
                running_average.append(0)
        
        plt.plot(running_average, label="Warmup")
        plt.plot(running_average_no_warmup, label="No warmup")
        plt.axhline(1 / (self.service_rate - self.arrival_rate), color='black', linestyle='--', label='Expected: 1/(μ - λ)')
        plt.xlabel("# packets")
        plt.ylabel("Time")
        plt.ylim((0.0, max(max(running_average), max(running_average_no_warmup))*1.1))
        plt.legend(loc="best")
        plt.title("Average time in the system after x packet")
        plt.show()

        #at the beginning a lot of time in the system then we stabilize to theoretical values

    """
    Method to plot the time spent in the system by each packet.
    """
    def plot_times_in_sys(self, warmup=0.5):
        times = self.times_in_system()
        first_element = int(len(times)*warmup)
        exclude_firsts = times[first_element:]
        plt.plot(exclude_firsts, label="Warmup")
        plt.plot(times, label="No warmup", alpha=0.7)
        plt.axhline(1 / (self.service_rate - self.arrival_rate), color='black', linestyle='--', label='Expected: 1/(μ - λ)')
        plt.xlabel("Packet")
        plt.ylabel("Time")
        plt.ylim((0.0, max(max(times), max(times))*1.1))
        plt.legend(loc="best")
        plt.title("Time in the system by each packet")
        plt.show()

        #at the beginning a lot of time in the system then we stabilize to theoretical values

    """
    Method to compute empirically the average number of packets in the system
    """
    def compute_average(self):
        
        total_time = 0
        weighted_sum = 0

        for i in range(len(self.packets_in_system) - 1):
            time_current, packets_current = self.packets_in_system[i]
            time_next, _ = self.packets_in_system[i + 1]

            #time interval
            delta_time = time_next - time_current

            #add to weighted sum
            weighted_sum += packets_current * delta_time

            total_time += delta_time    
        
        average_packets = weighted_sum / total_time
        return average_packets

    """
    Method to compute empirically the average number of packets in the queue
    """
    def compute_average_in_queue(self):
        
        total_time = 0
        weighted_sum = 0

        for i in range(len(self.packets_in_queue) - 1):
            time_current, packets_current = self.packets_in_queue[i]
            time_next, _ = self.packets_in_queue[i + 1]

            #time interval
            delta_time = time_next - time_current

            #add to weighted sum
            weighted_sum += packets_current * delta_time

            total_time += delta_time    
        
        average_packets = weighted_sum / total_time
        return average_packets

    """
    Method to compute empirically the average number of packets in the system, with a warmup period
    """
    def compute_average_with_warmup(self, warmup_time=0.1):
        
        last_event_time, _ = self.packets_in_system[len(self.packets_in_system) - 1]
        warmup_limit = last_event_time * warmup_time
        filtered_data = [(t, p) for t, p in self.packets_in_system if t >= warmup_limit]

        total_time = 0
        weighted_sum = 0

        for i in range(len(filtered_data) - 1):
            time_current, packets_current = filtered_data[i]
            time_next, _ = filtered_data[i + 1]

            #time interval
            delta_time = time_next - time_current

            #add to weighted sum
            weighted_sum += packets_current * delta_time

            total_time += delta_time    
        
        average_packets = weighted_sum / total_time
        
        return average_packets

    """
    Method to compute empirically the average time spent in the system
    """
    def compute_average_time_in_system(self):
        return np.mean([el[0] for el in self.time_in_system])

    """
    A getter for the time spent in the system of all the packets
    """
    def times_in_system(self):
        return [el[0] for el in self.time_in_system]
    
    """
    A getter of the time spent in the system by all the packets and the queue length when they entered
    """
    def averages_time_in_sys_and_queue_length(self):
        return self.time_in_system

"""
A function to compute the CI doing independent replications

Parameters:
    data: vector of metric computed, once for each replication
    confidence_level: the level of confidence to compute

Returns:
    lower bound and upper bound of the confidence interval
"""
def confidence_interval(data, confidence_level=0.95):
    grand_mean = np.mean(data)
    replications = len(data)
    variance_estimator = sum([ ((data[i] - grand_mean)**2) for i in range(replications)])
    variance_estimator = variance_estimator / (replications - 1)

    degrees_of_freedom = replications - 1
    alpha = (1 + confidence_level) / 2
    t_quantile = t.ppf(alpha, degrees_of_freedom)
    lower_bound = grand_mean - t_quantile * np.sqrt(variance_estimator / replications)
    upper_bound = grand_mean + t_quantile * np.sqrt(variance_estimator / replications)
    return lower_bound, upper_bound


if __name__ == "__main__":
    #parameters
    arrival_rate = 1.0  #λ
    service_rate = 2.0  #μ
    simulation_time = 5000.0  #seconds
    replications = 500

    parser = argparse.ArgumentParser(description="M/M/1 Queue Simulation")
    parser.add_argument("--arrival_rate", type=float, default=arrival_rate, help=f"Arrival rate (λ, default {arrival_rate})")
    parser.add_argument("--service_rate", type=float, default=service_rate, help=f"Service rate (μ, default {service_rate})")
    parser.add_argument("--simulation_time", type=float, default=simulation_time, help=f"Time of the simulation (default {simulation_time})")
    parser.add_argument("--replications", type=int, default=replications, help=f"Number of independent replications we do (default {replications})")
    args = parser.parse_args()

    arrival_rate = args.arrival_rate
    service_rate = args.service_rate
    if(service_rate <= arrival_rate):
        print("Cannot continue! Must have μ > λ!")
        exit(0)
    simulation_time = args.simulation_time
    replications = args.replications

    print(f"You are running the simulator with an arrival_rate of {arrival_rate}, a service_rate of {service_rate} and a simulation_time of {simulation_time}, for {replications} independent replications\n")

    #We do independent replications
    empirical_averages = []
    empirical_averages_warmup = []
    empirical_times_in_system = []
    
    all_curves = []
    num_points = int(simulation_time * 10)
    time_grid = np.linspace(0, simulation_time, num_points)

    for j in range(replications):
        simulator = MM1QueueSimulator(arrival_rate, service_rate, simulation_time)
        simulator.simulate()
        #simulator.plot_results()
        #simulator.plot_times_in_sys(warmup=0.8)
        times, packets = zip(*simulator.packets_in_system)
        packets_interp = np.interp(time_grid, times, packets)
        #stimo i valori di pacchetti medi in ogni unita di tempo in time_grid
        #secondo e terzo parametro sono i valori di x e y
        all_curves.append(packets_interp)


        #Comparisons
        empirical_avg = simulator.compute_average() #average #packets in the system
        empirical_averages.append(empirical_avg)
        empirical_avg_warmup = simulator.compute_average_with_warmup(warmup_time=0.5) #average #packets in the system with warmup
        empirical_averages_warmup.append(empirical_avg_warmup)
        empirical_avg_time_in_sys = simulator.compute_average_time_in_system() #average time spent in the system
        empirical_times_in_system.append(empirical_avg_time_in_sys)

    ################
    all_curves = np.array(all_curves)
    mean_curve = np.mean(all_curves, axis=0)
    std_curve = np.std(all_curves, axis=0)

    plt.figure(figsize=(10,6))
    plt.plot(time_grid, mean_curve, label="Media sulle repliche")
    plt.fill_between(time_grid, mean_curve-std_curve, mean_curve+std_curve, color='orange', alpha=0.3, label="±1 std")
    plt.xlabel("Tempo")
    plt.ylabel("Numero di pacchetti nel sistema")
    plt.title("Media temporale del numero di pacchetti nel sistema (sulle repliche)")
    plt.legend()
    plt.show()
    ################

    rho = arrival_rate / service_rate
    theoretical_avg = rho / (1 - rho)
    print(f"\nTheoretical average number of packets in system: {theoretical_avg}\n")
    
    print(f"Empirical average number of packets in system: {np.mean(empirical_averages)} (standard deviation {np.std(empirical_averages)})")
    lower_ci, upper_ci = confidence_interval(empirical_averages)
    print(f"Confidence Interval: [{lower_ci}, {upper_ci}]\n")
    
    print(f"[WITH WARMUP] Empirical average number of packets in system: {np.mean(empirical_averages_warmup)} (standard deviation {np.std(empirical_averages_warmup)})")
    lower_ci, upper_ci = confidence_interval(empirical_averages_warmup)
    print(f"Confidence Interval: [{lower_ci}, {upper_ci}]\n")

    print(f"\nTheoretical average time in system for one packet: {1/(service_rate-arrival_rate)}\n")

    print(f"Empirical average time in system for one packet: {np.mean(empirical_times_in_system)} (standard deviation {np.std(empirical_times_in_system)})")
    lower_ci, upper_ci = confidence_interval(empirical_times_in_system)
    print(f"Confidence Interval: [{lower_ci}, {upper_ci}]\n")

