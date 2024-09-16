import re

def read_output(dim):
    # Open the output file and read it line by line
    density_values = []
    triton_times = []
    triton_blocked_times = []
    cusparse_csr_times = []
    cusparse_bsr_times = []

    file_path = f'./dim_{dim}_output.txt'
    with open(file_path, 'r') as file:
        lines = file.readlines()
        
        # Regex pattern to match density values
        density_pattern = re.compile(r'density\s*=\s*([\d\.]+)')
        
        # Iterate through the lines to extract the relevant information
        for i, line in enumerate(lines):
            # Find the density line using regex and extract the density value
            if "density" in line:
                match = density_pattern.search(line)
                if match:
                    try:
                        density_str = match.group(1)  # Extracted density value
                        density = float(density_str)
                        density_values.append(density)
                        #print(f"Extracted density: {density}")  # Debug print
                    except ValueError as e:
                        print(f"Could not convert density value: {density_str}. Error: {e}")
            
            # Extract execution time for each kernel
            if "Triton SpMM Metrics:" in line:
                try:
                    exec_time_str = lines[i+1].split(':')[1].strip().split()[0]
                    exec_time_str = exec_time_str.replace(',', '').replace('=', '')
                    triton_times.append(float(exec_time_str))
                except ValueError as e:
                    print(f"Could not convert Triton SpMM execution time: {exec_time_str}. Error: {e}")
            
            elif "Triton Blocked SpMM Metrics:" in line:
                try:
                    exec_time_str = lines[i+1].split(':')[1].strip().split()[0]
                    exec_time_str = exec_time_str.replace(',', '').replace('=', '')
                    triton_blocked_times.append(float(exec_time_str))
                except ValueError as e:
                    print(f"Could not convert Triton Blocked SpMM execution time: {exec_time_str}. Error: {e}")
            
            elif "cuSPARSE CSR SpMM Metrics:" in line:
                try:
                    exec_time_str = lines[i+1].split(':')[1].strip().split()[0]
                    exec_time_str = exec_time_str.replace(',', '').replace('=', '')
                    cusparse_csr_times.append(float(exec_time_str))
                except ValueError as e:
                    print(f"Could not convert cuSPARSE CSR SpMM execution time: {exec_time_str}. Error: {e}")
            
            elif "cuSPARSE BSR SpMM Metrics:" in line:
                try:
                    exec_time_str = lines[i+1].split(':')[1].strip().split()[0]
                    exec_time_str = exec_time_str.replace(',', '').replace('=', '')
                    cusparse_bsr_times.append(float(exec_time_str))
                except ValueError as e:
                    print(f"Could not convert cuSPARSE BSR SpMM execution time: {exec_time_str}. Error: {e}")

    # Output to check the parsed values
    #print("Density values:", density_values)
    #print("Triton SpMM times:", triton_times)
    #print("Triton Blocked SpMM times:", triton_blocked_times)
    #print("cuSPARSE SpMM times:", cusparse_times)
    #print("cuSPARSE BSR SpMM times:", cusparse_bsr_times)

    return density_values, triton_times, triton_blocked_times, cusparse_csr_times, cusparse_bsr_times

def main():
    dim = 2048
    density_values, triton_times, triton_blocked_times, cusparse_times, cusparse_bsr_times = read_output(dim)

if __name__ == "__main__":
    main()
