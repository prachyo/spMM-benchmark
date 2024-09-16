import os
from data_utils.read_output import read_output
import matplotlib.pyplot as plt

dimensions = [2048, 4096, 8192] # tweak this as you wish

def gather_results():
    results = []

    for dim in dimensions:
        density_values, triton_times, triton_blocked_times, cusparse_csr_times, cusparse_bsr_times = read_output(dim)
        result_row = [density_values, triton_times, triton_blocked_times, cusparse_csr_times, cusparse_bsr_times]
        results.append(result_row)
    return results

def plot_results(results):
    # create a new plot for each row in results
    for i in range(len(results)):
        density_values, triton_times, triton_blocked_times, cusparse_times, cusparse_bsr_times = results[i]
        plt.figure()
        plt.plot(density_values, triton_times, label="Triton SpMM")
        plt.plot(density_values, triton_blocked_times, label="Triton Blocked SpMM")
        plt.plot(density_values, cusparse_times, label="cuSPARSE CSR SpMM")
        plt.plot(density_values, cusparse_bsr_times, label="cuSPARSE BSR SpMM")
        plt.xlabel("Density")
        plt.ylabel("Execution Time (s)")
        plt.title(f"M = N = K = {dimensions[i]}")
        plt.legend()
        # save the plot as a png with file name density_vs_time_{dim}.png
        plt.savefig(f"density_vs_time_{dimensions[i]}.png")

def main():
    block_size = 32

    for dim in dimensions:
        # loop over values from 0.0 to 0.2 in steps of 0.02
        for i in range(11):
            density = i / 50
            command = f"python3 spMM_test.py {dim} {density} {block_size} >> dim_{dim}_output.txt"
            os.system(command)

    results = gather_results()
    plot_results(results)

if __name__ == "__main__":
    main()