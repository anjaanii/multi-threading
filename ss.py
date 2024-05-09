import numpy as np
import matplotlib.pyplot as plt
import time
import multiprocessing
import psutil

def generate_random_matrix(size):
    return np.random.rand(size, size)

def multiply_matrices(constant_matrix, num_matrices):
    results = []
    for _ in range(num_matrices):
        random_matrix = generate_random_matrix(constant_matrix.shape[0])
        result = np.dot(random_matrix, constant_matrix)
        results.append(result)
    return results

def parallel_multiplication(constant_matrix, num_matrices):
    num_cores = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes=num_cores)
    results = pool.starmap(multiply_matrices, [(constant_matrix, num_matrices // num_cores)] * num_cores)
    pool.close()
    pool.join()
    return results

def plot_cpu_usage(cpu_usage):
    plt.plot(cpu_usage)
    plt.xlabel('Time (s)')
    plt.ylabel('CPU Usage (%)')
    plt.title('CPU Usage During Matrix Multiplication')
    plt.show()

if __name__ == "__main__":
    constant_matrix = generate_random_matrix(1000)
    num_matrices = 100

    start_time = time.time()
    results = parallel_multiplication(constant_matrix, num_matrices)
    end_time = time.time()
    total_time = end_time - start_time
    print("Total time taken:", total_time, "seconds")

    # Measure CPU usage using psutil
    cpu_usage = []
    for _ in range(int(total_time)):
        cpu_percent = psutil.cpu_percent(interval=1, percpu=True)
        cpu_usage.append(sum(cpu_percent) / len(cpu_percent))

    # Plot CPU usage
    plot_cpu_usage(cpu_usage)
