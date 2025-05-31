import numpy as np
import matplotlib.pyplot as plt
import argparse

#INITIALIZATION OF PARAMETERS

arrival_rate = 1  #lambda
departure_rate = 2 #mu
N = 150

parser = argparse.ArgumentParser(description="Simulazione M/M/1")
parser.add_argument("--arrival_rate", type=float, default=1, help="Arrival rate (lambda, def 1)")
parser.add_argument("--departure_rate", type=float, default=2, help="Departure rate (mu, def 2)")
parser.add_argument("--N", type=int, default=150, help="Events number in queue (def 150)")
args = parser.parse_args()

arrival_rate = args.arrival_rate
departure_rate = args.departure_rate
N = args.N

print(arrival_rate, departure_rate, N)

#function to compute the event queue for M/M/1 queue

def compute_event_queue(arrival_rate, service_rate, N):
    
    #inter-arrival times
    inter_arrival_times = np.random.exponential(1 / arrival_rate, N)
    
    #service times
    service_times = np.random.exponential(1 / service_rate, N)

    #arrival times
    arrival_times = np.cumsum(inter_arrival_times)

    #departure times
    departure_times = np.zeros(N)
    
    for i in range(N):
        
        if i == 0:
            departure_times[i] = arrival_times[i] + service_times[i]
        else:
            departure_times[i] = max(arrival_times[i], departure_times[i - 1]) + service_times[i]
            #depending if server is free when event arrives or it's busy, so it must way to i-1 event to go
        
    arrivals_index = 0
    departure_index = 0
    event_queue = []
    
    while arrivals_index < N and departure_index < N:
        if arrival_times[arrivals_index] < departure_times[departure_index]:
            event_queue.append((float(arrival_times[arrivals_index]), 'a'))
            arrivals_index += 1
        else:
            event_queue.append((float(departure_times[departure_index]), 'd'))
            departure_index += 1

    while arrivals_index < N:
        print("it should not happen")
        arrivals_index += 1

    while departure_index < N:
        event_queue.append((float(departure_times[departure_index]), 'd'))
        departure_index += 1

    return event_queue

#function to simulate the M/M/1 queue with a list of events
#returns:
# 1) average number of packets in the system
# 2) average number of packets in the server
# 3) average number of packets in the queue
# 4) waiting times for each packet
# 5) service times for each packet
def simulate_with_list(events_list, plot=False):

    current_time = 0
    index_event = 0

    server_status = 0
    in_queue = 0
    last_event_time = 0

    area_server = 0
    area_queue = 0

    server_status_history = [0]
    queue_length_history = [0]
    times = [0]
    system_history = [0]
    area_system = 0

    empirical_arrival_times = []
    empirical_service_times = []
    empirical_departure_times = []
    waiting_times = []
    service_times = []

    while index_event < len(events_list):
        event = events_list[index_event]
        index_event += 1

        current_time = event[0]
        type_event = event[1]

        
        area_server += (current_time - last_event_time) * server_status
        area_queue += in_queue * (current_time - last_event_time)
        area_system += in_queue * (current_time - last_event_time) + (current_time - last_event_time) * server_status


        if type_event == 'a':
            empirical_arrival_times.append(current_time)
            if server_status == 0:
                server_status = 1
                empirical_service_times.append(current_time)
            else:
                in_queue += 1
        else:
            empirical_departure_times.append(current_time)
            server_status = 0

            if in_queue != 0:
                server_status = 1
                in_queue -= 1
                empirical_service_times.append(current_time)

        #print(f"AVG in queue {(area_queue/current_time):.3f}, AVG in queue in server {(area_server/current_time):.3f}, AVG in system {(area_system/current_time):3f}")
        
        times.append(current_time)
        server_status_history.append(server_status)
        queue_length_history.append(in_queue)
        system_history.append(server_status+in_queue)

        last_event_time = current_time

    if(plot):
        p = arrival_rate/departure_rate
        print(f"Average number of packets in the system {(p*(1-p)):.3f}")
        print(f"Empirical number of packets in the system {(area_system / current_time):.3f}")

        max_y = max(max(system_history), max(queue_length_history), max(server_status_history))

        plt.figure(figsize=(10, 4))

        plt.subplot(1, 3, 1)
        plt.plot(times, server_status_history, drawstyle='steps-post')
        plt.xlabel("Time")
        plt.ylabel("Server status")
        plt.yticks(np.arange(0, 3, 1)) 
        plt.title("Server Status Over Time")

        plt.subplot(1, 3, 2)
        plt.plot(times, queue_length_history, drawstyle='steps-post', color='orange')
        plt.xlabel("Time")
        plt.ylabel("Queue length")
        plt.yticks(np.arange(0, max_y+2, 1))
        plt.title("Queue Length Over Time")

        plt.subplot(1, 3, 3)
        plt.plot(times, system_history, drawstyle='steps-post', color='green')
        plt.xlabel("Time")
        plt.ylabel("System packets")
        plt.yticks(np.arange(0, max_y+2, 1))
        plt.title("System packets Over Time")


        plt.tight_layout()
        plt.show()

    print(f"AVG in queue {(area_queue/current_time):.3f}, AVG in server {(area_server/current_time):.3f}, AVG in system {(area_system/current_time):3f}")
    
    for i in range(len(empirical_service_times)):
        waiting_times.append(empirical_service_times[i] - empirical_arrival_times[i])
    
    for i in range(len(empirical_service_times)):
        service_times.append(empirical_departure_times[i] - empirical_service_times[i])

    return waiting_times, service_times, area_queue/current_time

replications = 1
averages_waiting = []
averages_services = []
averages_sys = []

p = arrival_rate/departure_rate
theoretical_avg_packets_in_queue = p * (1 - p)


for j in range(replications):
    events_list = compute_event_queue(arrival_rate, departure_rate, N)
    waiting_times, service_times, avg_packets_in_queue = simulate_with_list(events_list)
    system_times = [waiting_times[i] + service_times[i] for i in range(len(waiting_times))]
    averages_sys.append(np.mean(system_times))


print("\n")
print(f"Average thoeretical system time {p/(1-p):.3f}\n")
print(f"Avg in system time with naive estimator {np.mean(system_times):.3f}, st dev {np.std(system_times):.3f}\n")
