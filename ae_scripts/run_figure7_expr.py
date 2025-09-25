import subprocess
import sys
import os
import time

execution_types = [
    "serverless_vllm",
    "hydraserve_with_single_worker",
    "hydraserve",
    "serverlessllm",
    "serverlessllm_with_cached_model",
]
model_sets = ["0", "1"]
backends = ["a10", "v100"]

# Calculate the total number of experiments
total_experiments = len(execution_types) * (len(model_sets) * len(backends) - 1)
current_experiment = 0

# Main loop to run experiments for each combination of parameters
print("Starting the experiment reproduction script.")
print(f"Total experiments to run: {total_experiments}")
print("-" * 40)

for exec_type in execution_types:
    print(f"Configuring node labels for {exec_type}...")
    env = os.environ.copy()
    env['SHARE'] = '0' if 'serverlessllm' in exec_type else '1'
    subprocess.run(['python', '/root/hydraserve-artifact/scripts/kubernetes/label_nodes.py'], check=True, env=env)
    for model_set in model_sets:
        for backend in backends:
            if backend == "a10" and model_set == "1":
                continue
            current_experiment += 1
            print(f"\n[{current_experiment}/{total_experiments}] Running experiment with settings:")
            print(f"  - Execution Type: {exec_type}")
            print(f"  - Model Set:      {model_set}")
            print(f"  - Backend:        {backend}")
            print("-" * 20)

            # Create a copy of the current environment and set variables
            env = os.environ.copy()
            env['exec_type'] = exec_type
            env['model_set'] = model_set
            env['backend'] = backend
            
            try:
                # Step 1: Start the server
                start_server_cmd = ['sh', './start_server.sh', '0', exec_type, model_set, backend, '0', '0']
                print(f"Executing and waiting for 'start_server.sh' to complete: {' '.join(start_server_cmd)}")
                
                # Using check=True, the script will raise an exception and stop if start_server.sh fails (returns a non-zero code).
                subprocess.run(start_server_cmd, check=True, env=env)
                
                print("'start_server.sh' script has finished. The server process should now be running in the background.")
                time.sleep(3)

                # Step 2: Run the cold start experiment
                # This command is executed only after start_server.sh has completed successfully.
                cold_start_cmd = ['sh', './coldstart.sh', exec_type, model_set, backend]
                print(f"Executing: {' '.join(cold_start_cmd)}")
                subprocess.run(cold_start_cmd, check=True, env=env)
                print("Cold start experiment completed successfully for this setting.")

            except subprocess.CalledProcessError as e:
                # Catch exceptions from failed commands
                print(f"\nERROR: The command '{' '.join(e.cmd)}' failed with exit code {e.returncode}.", file=sys.stderr)
                # Print the command's output to help with debugging
                if e.stdout:
                    print(f"STDOUT:\n{e.stdout}", file=sys.stderr)
                if e.stderr:
                    print(f"STDERR:\n{e.stderr}", file=sys.stderr)
                print("Skipping to the next experiment setting.", file=sys.stderr)
            except FileNotFoundError as e:
                print(f"\nERROR: Command not found: {e.filename}. Please ensure the script exists and has execution permissions.", file=sys.stderr)
                sys.exit(1)
            except KeyboardInterrupt:
                print("\nScript interrupted by user. Exiting.")
                sys.exit(1)


# Final step after all experiments are completed
print("\nAll experiments have been completed.")
print("=" * 40)
print("Step 4: Generating figure figs/figure7.pdf using figure7.py")

try:
    figure_cmd = ['python', 'figure7.py']
    print(f"Executing: {' '.join(figure_cmd)}")
    subprocess.run(figure_cmd, check=True)
    print("\nFigure 'figs/figure7.pdf' generated successfully!")
except Exception as e:
    print(f"\nERROR: An error occurred while generating the figure: {e}", file=sys.stderr)
    sys.exit(1)

print("\nScript finished successfully.")