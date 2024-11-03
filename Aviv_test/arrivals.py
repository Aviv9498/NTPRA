import numpy as np


def poisson_arrivals(seed, time_steps, arrival_rates, num_flows):
    np.random.seed(seed)
    return np.random.poisson(arrival_rates, (time_steps, num_flows))


class TCPRenoSimulator:
    def __init__(self, num_flows, arrival_rates):
        self.num_flows = num_flows
        self.arrival_rates = arrival_rates
        self.cwnd = np.ones(num_flows)  # Congestion window for each flow, starting with 1 MSS
        self.ssthresh = np.full(num_flows, 64)  # Slow start threshold, set to a higher value
        self.state = np.full(num_flows, 'slow_start')  # Initial state for all flows
        self.dup_acks = np.zeros(num_flows)  # Counter for duplicate ACKs
        self.min_cwnd = 1  # Minimum congestion window to avoid zero arrivals
        self.arrival_matrix = None

    def simulate(self, time_steps):
        self.arrival_matrix = np.zeros((time_steps, self.num_flows))

        for t in range(time_steps):
            arrivals = np.zeros(self.num_flows)

            for flow in range(self.num_flows):
                # Determine the number of arrivals based on the current cwnd
                arrivals[flow] = min(self.cwnd[flow], self.arrival_rates[flow])

                # Handle state transitions and cwnd updates
                if self.state[flow] == 'slow_start':
                    if self.cwnd[flow] < self.ssthresh[flow]:
                        self.cwnd[flow] *= 2  # Exponential growth
                    else:
                        self.state[flow] = 'congestion_avoidance'
                        self.cwnd[flow] += 1  # Transition to linear growth

                elif self.state[flow] == 'congestion_avoidance':
                    self.cwnd[flow] += 1 / self.cwnd[flow]  # Linear growth

                elif self.state[flow] == 'fast_recovery':
                    self.cwnd[flow] += 1  # Inflate the congestion window during fast recovery

                # Handle packet loss (simplified)
                if np.random.rand() < 0.01:  # Assume 1% packet loss rate
                    self.dup_acks[flow] += 1
                    if self.dup_acks[flow] == 3:
                        # Fast retransmit and fast recovery
                        self.ssthresh[flow] = max(self.cwnd[flow] / 2, self.min_cwnd)
                        self.cwnd[flow] = self.ssthresh[flow] + 3  # Inflate cwnd for fast recovery
                        self.state[flow] = 'fast_recovery'
                    elif self.dup_acks[flow] > 3:
                        # During fast recovery
                        self.cwnd[flow] += 1
                    else:
                        self.cwnd[flow] = self.ssthresh[flow]
                        self.state[flow] = 'congestion_avoidance'
                        self.dup_acks[flow] = 0

                # Ensure minimum arrivals
                arrivals[flow] = max(arrivals[flow], self.min_cwnd)

            self.arrival_matrix[t, :] = np.round(arrivals).astype(int)

        return self.arrival_matrix


if __name__ == "__main__":
    # Example usage
    num_flows = 10
    arrival_rates = np.random.poisson(10, num_flows) * 10  # Average arrival rates for each flow
    tcp_reno_sim = TCPRenoSimulator(num_flows, arrival_rates)
    arrival_matrix = tcp_reno_sim.simulate(time_steps=100)

    print(arrival_matrix)





